import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT
from tqdm import tqdm

load_dotenv()


def _extract_short_answer(raw: str) -> str:
    """从模型输出中提取简短答案（用于 F1/BLEU）。优先取 Answer: 后、或「answer is」后、或最后一句短句。"""
    if not raw or not isinstance(raw, str):
        return ""
    raw = raw.strip()
    if not raw:
        return ""
    import re
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("answer:"):
            return line[7:].strip() or line.strip()
        # "The answer is 7 May 2023" / "Answer is ..."
        m = re.search(r"\banswer\s+is\s*[:\-]?\s*(.+)", lower, re.I)
        if m:
            return m.group(1).strip()
    # 取最后一行；若过长则取最后一句话（按 . ! ? 分）
    lines = [s.strip() for s in raw.split("\n") if s.strip()]
    last = lines[-1] if lines else raw
    if len(last) > 120:
        parts = re.split(r"[.!?]+", last)
        parts = [p.strip() for p in parts if p.strip()]
        last = parts[-1] if parts else last[:100]
    return last.strip() if last else raw[:200].strip()


def _get_message_content(response) -> str:
    """从 chat completions 响应中取出 content，兼容 OpenAI / Ollama 等。"""
    if not getattr(response, "choices", None) or not response.choices:
        return ""
    choice = response.choices[0]
    msg = getattr(choice, "message", None)
    if msg is None:
        return ""
    if hasattr(msg, "content"):
        c = msg.content
        if c is not None:
            if isinstance(c, str):
                return c.strip()
            if hasattr(c, "__iter__") and not isinstance(c, str):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return (part.get("text") or "").strip()
                    if isinstance(part, str):
                        return part.strip()
            return str(c).strip()
    if hasattr(msg, "get") and callable(getattr(msg, "get", None)):
        return (msg.get("content") or "").strip()
    # 首次空时打印调试信息
    if not getattr(_get_message_content, "_logged_empty", False):
        try:
            r = response
            info = f"choices[0].message type={type(getattr(r.choices[0], 'message', None))}"
            if hasattr(r.choices[0].message, "content"):
                info += f" content={repr(getattr(r.choices[0].message, 'content', None))[:200]}"
            print("[search] 警告: 无法从响应中取出 content，", info)
            _get_message_content._logged_empty = True
        except Exception:
            _get_message_content._logged_empty = True
    return ""


def _make_answer_client():
    """无 MEM0_API_KEY 时用 Ollama；或 TINKER_* / OPENAI_* 时用 Tinker 等 OpenAI 兼容接口；有 MEM0_API_KEY 则用 OpenAI。"""
    openai_base = os.getenv("TINKER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    openai_key = os.getenv("TINKER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if openai_base and openai_key:
        base = openai_base.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1" if "/v1" not in base else base
        return OpenAI(base_url=base, api_key=openai_key), os.getenv("MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    if os.getenv("MEM0_API_KEY"):
        return OpenAI(), os.getenv("MODEL", "gpt-4o-mini")
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return OpenAI(base_url=base, api_key="ollama"), os.getenv("LLM_MODEL", "llama3.2")


class MemorySearch:
    def __init__(self, output_path="results.json", top_k=10, run_stats=None, mem0_client=None):
        """LoCoMo 仅 vector-only，不开图。mem0_client 可选，用于 add_then_search 复用同一连接避免 Qdrant 锁冲突。"""
        if mem0_client is not None:
            self.mem0_client = mem0_client
            self._use_local = True
        else:
            api_key = os.getenv("MEM0_API_KEY")
            if api_key:
                from mem0 import MemoryClient
                self.mem0_client = MemoryClient(
                    api_key=api_key,
                    org_id=os.getenv("MEM0_ORGANIZATION_ID"),
                    project_id=os.getenv("MEM0_PROJECT_ID"),
                )
                self._use_local = False
            else:
                from src.memzero.local_memory import create_local_memory_with_retry
                self.mem0_client = create_local_memory_with_retry()
                self._use_local = True

        self.top_k = top_k
        self._answer_client, self._answer_model = _make_answer_client()
        openai_base = os.getenv("TINKER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        openai_key = os.getenv("TINKER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if openai_base and openai_key:
            _answer_backend = "OpenAI-compatible (Tinker/Qwen)"
        elif os.getenv("MEM0_API_KEY"):
            _answer_backend = "OpenAI (MEM0_API_KEY set)"
        else:
            _answer_backend = "Ollama"
        print(f"[search] 回答模型后端: {_answer_backend}, model={self._answer_model}")
        self.results = defaultdict(list)
        self.output_path = output_path
        self.run_stats = run_stats if run_stats is not None else {}
        self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                if self._use_local:
                    raw = self.mem0_client.search(query, user_id=user_id, limit=self.top_k)
                else:
                    raw = self.mem0_client.search(
                        query, user_id=user_id, top_k=self.top_k, filter_memories=False
                    )
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        if isinstance(raw, dict) and "results" in raw:
            memories_list = raw["results"]
        else:
            memories_list = raw if isinstance(raw, list) else []

        semantic_memories = [
            {
                "memory": memory["memory"],
                "timestamp": (memory.get("metadata") or {}).get("timestamp", ""),
                "score": round(memory.get("score", 0.0), 2),
            }
            for memory in memories_list
        ]
        return semantic_memories, end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        speaker_1_memories, speaker_1_memory_time = self.search_memory(speaker_1_user_id, question)
        speaker_2_memories, speaker_2_memory_time = self.search_memory(speaker_2_user_id, question)

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            question=question,
        )
        # 必须带一条 user 消息，Ollama/OpenAI 才会生成回复；仅 system 会返回空
        messages = [
            {"role": "system", "content": answer_prompt},
            {"role": "user", "content": f"Reply with ONLY the short answer (e.g. a date, name, or phrase in 5–6 words). No explanation.\n\nQuestion: {question}"},
        ]

        t1 = time.time()
        response = self._answer_client.chat.completions.create(
            model=self._answer_model,
            messages=messages,
            temperature=0.0,
        )
        t2 = time.time()
        response_time = t2 - t1
        if self.run_stats and getattr(response, "usage", None):
            u = response.usage
            self.run_stats["input_tokens"] = self.run_stats.get("input_tokens", 0) + (u.prompt_tokens or 0)
            self.run_stats["output_tokens"] = self.run_stats.get("output_tokens", 0) + (u.completion_tokens or 0)
            self.run_stats["context_window_peak"] = max(
                self.run_stats.get("context_window_peak", 0), u.prompt_tokens or 0
            )
        raw_content = _get_message_content(response)
        answer_text = _extract_short_answer(raw_content)
        # 若仍为空则重试一次（部分 Ollama/API 偶发返回空）
        if not answer_text.strip():
            try:
                response2 = self._answer_client.chat.completions.create(
                    model=self._answer_model,
                    messages=messages,
                    temperature=0.0,
                )
                raw_content = _get_message_content(response2)
                answer_text = _extract_short_answer(raw_content)
                if self.run_stats and getattr(response2, "usage", None):
                    u = response2.usage
                    self.run_stats["input_tokens"] = self.run_stats.get("input_tokens", 0) + (u.prompt_tokens or 0)
                    self.run_stats["output_tokens"] = self.run_stats.get("output_tokens", 0) + (u.completion_tokens or 0)
            except Exception:
                pass
        return (
            answer_text or "",
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            response_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            response_time,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "response_time": response_time,
        }

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path, max_samples=None, max_questions=None):
        with open(file_path, "r") as f:
            data = json.load(f)
        if max_samples is not None:
            data = data[:max_samples]

        stage_start = time.time()
        total_questions_done = 0
        if self.run_stats is not None:
            self.run_stats.setdefault("input_tokens", 0)
            self.run_stats.setdefault("output_tokens", 0)
            self.run_stats.setdefault("context_window_peak", 0)
        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
            ):
                if max_questions is not None and total_questions_done >= max_questions:
                    break
                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                self.results[idx].append(result)
                total_questions_done += 1

                # Save results after each question is processed
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)
            if max_questions is not None and total_questions_done >= max_questions:
                break

        if self.run_stats is not None:
            self.run_stats["search_time_sec"] = time.time() - stage_start
            self.run_stats["n_questions"] = total_questions_done
        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results
