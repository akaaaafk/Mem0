# Usage:
# 用法：
#   # 只跑 add（把对话写入向量库）
#   python eval_locomo_mem0_full.py --method add --data mem0/evaluation/locomo/locomo10.json --outdir ./results/eval_locomo_mem0_full
#
#   # 本地 Ollama LLM + OpenAI 嵌入（需 OPENAI_API_KEY，及可选 LLM_BASE_URL、LLM_MODEL_ID、LLM_API_KEY）
#   # LLM_BASE_URL 不要带 /v1，ollama 客户端请求的是 /api/chat，带 /v1 会 404
#   # 若出现 'ascii' codec 错误，请先执行：set PYTHONUTF8=1（CMD）或 $env:PYTHONUTF8=1（PowerShell）
#   set LLM_BASE_URL=http://localhost:11434
#   set LLM_MODEL_ID=llama3
#   set LLM_API_KEY=API_KEY
#   set OPENAI_API_KEY=sk-...
#   python eval_locomo_mem0_full.py --method both --use-local-llm --max-samples 1 --max-questions 5
#
#   # 只跑 search+评判（需先跑过 add，且 --outdir 与 add 一致以共用向量库）
#   python eval_locomo_mem0_full.py --method search --data mem0/evaluation/locomo/locomo10.json --outdir ./results/eval_locomo_mem0_full
#
#   # 先 add 再 search（一条龙）
#   python eval_locomo_mem0_full.py --method both --data mem0/evaluation/locomo/locomo10.json --outdir ./results/eval_locomo_mem0_full --max-samples 1 --max-questions 20
#
# 依赖：
#   - add 阶段：默认使用你的 AWS Inference Profile（LLM 事实抽取）+ Bedrock Titan Embed（向量化）+ FAISS 本地向量库；需 AWS 凭证。
#   - search 阶段：使用同一 AWS Inference Profile 答题，需 AWS 凭证。
#
# # 只跑 add（写入向量库）
# python eval_locomo_mem0_full.py --method add --data mem0/evaluation/locomo/locomo10.json --outdir ./results/eval_locomo_mem0_full
#
# # 只跑 search + 评判（需先跑过 add，且 --outdir 与 add 一致）
# python eval_locomo_mem0_full.py --method search --data mem0/evaluation/locomo/locomo10.json --outdir ./results/eval_locomo_mem0_full --max-questions 20
#
# # 一条龙：先 add 再 search
# python eval_locomo_mem0_full.py --method both --data mem0/evaluation/locomo/locomo10.json --outdir ./results/eval_locomo_mem0_full --max-samples 1 --max-questions 20 --verbose

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA = os.path.join(_SCRIPT_DIR, "mem0", "evaluation", "locomo", "locomo10.json")
DEFAULT_REGION = "us-east-1"
DEFAULT_ACCOUNT_ID = os.environ.get("BEDROCK_ACCOUNT_ID", "...")
DEFAULT_INFERENCE_PROFILE_ID = os.environ.get("BEDROCK_INFERENCE_PROFILE_ID", "...")
DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
TITAN_EMBED_DIMS = 1024
OPENAI_EMBED_DIMS = 1536

ENV_LLM_BASE_URL = "LLM_BASE_URL"
ENV_LLM_MODEL_ID = "LLM_MODEL_ID"
ENV_LLM_API_KEY = "LLM_API_KEY"
DEFAULT_LLM_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL_ID = "llama3"
DEFAULT_LLM_API_KEY = "API_KEY"

def build_inference_profile_arn(region: str, account_id: str, profile_id: str) -> str:
    return f"arn:aws:bedrock:{region}:{account_id}:application-inference-profile/{profile_id}"

ANSWER_PROMPT_TWO_SPEAKERS = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.),
   calculate the actual date based on the memory timestamp.
6. Always convert relative time references to specific dates, months, or years.
7. Focus only on the content of the memories from both speakers.
8. The answer should be less than 5-6 words.

Memories for user {{speaker_1_user_id}}:
{{speaker_1_memories}}

Memories for user {{speaker_2_user_id}}:
{{speaker_2_memories}}

Question: {{question}}
Answer:
"""

def _session_keys(conv: dict):
    keys = [
        k
        for k in conv
        if isinstance(k, str) and k.startswith("session_") and k.split("_")[-1].isdigit()
        and not k.endswith("_date_time")
    ]
    keys.sort(key=lambda x: int(x.split("_")[1]))
    return keys

def build_messages_for_speaker(conv: dict, speaker_a: str, speaker_b: str, session_key: str):
    chats = conv.get(session_key) or []
    messages_a = []
    messages_b = []
    for turn in chats:
        speaker = turn.get("speaker", "")
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        content = f"{speaker}: {text}"
        if speaker == speaker_a:
            messages_a.append({"role": "user", "content": content})
            messages_b.append({"role": "assistant", "content": content})
        elif speaker == speaker_b:
            messages_a.append({"role": "assistant", "content": content})
            messages_b.append({"role": "user", "content": content})
    return messages_a, messages_b

def _patch_mem0_bedrock_for_inference_profile():
    import mem0.llms.aws_bedrock as aws_bedrock

    _orig_extract = aws_bedrock.extract_provider

    def extract_provider(model: str) -> str:
        if model and model.startswith("arn:"):
            return "anthropic"
        return _orig_extract(model)

    aws_bedrock.extract_provider = extract_provider

    _orig_test = aws_bedrock.AWSBedrockLLM._test_connection

    def _test_connection_skip_arn(self):
        if self.config.model and self.config.model.startswith("arn:"):
            return
        _orig_test(self)

    aws_bedrock.AWSBedrockLLM._test_connection = _test_connection_skip_arn

    _orig_generate = aws_bedrock.AWSBedrockLLM._generate_standard

    def _generate_standard_inference_profile(self, messages, stream=False):
        if self.config.model and self.config.model.startswith("arn:"):
            old_converse = self.client.converse

            def converse_skip_top_p(**kwargs):
                inf = kwargs.get("inferenceConfig") or {}
                if "topP" in inf:
                    inf = {k: v for k, v in inf.items() if k != "topP"}
                    kwargs = {**kwargs, "inferenceConfig": inf}
                return old_converse(**kwargs)

            self.client.converse = converse_skip_top_p
            try:
                return _orig_generate(self, messages, stream=stream)
            finally:
                self.client.converse = old_converse
        return _orig_generate(self, messages, stream=stream)

    aws_bedrock.AWSBedrockLLM._generate_standard = _generate_standard_inference_profile

def run_add(
    data_path: str,
    outdir: str,
    max_samples: int,
    config_path: str | None,
    region: str,
    account_id: str,
    inference_profile_id: str,
    embed_model: str,
    use_local_llm: bool,
) -> None:
    sys.path.insert(0, _SCRIPT_DIR)
    from mem0 import Memory
    from mem0.configs.base import MemoryConfig

    if not use_local_llm:
        inference_arn = build_inference_profile_arn(region, account_id, inference_profile_id)
        _patch_mem0_bedrock_for_inference_profile()
        os.environ["AWS_REGION"] = region

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("data must be a list of items with 'conversation' and 'qa'")

    data = data[:max_samples]
    os.makedirs(outdir, exist_ok=True)
    vector_store_path = os.path.join(outdir, "vector_store")
    os.makedirs(vector_store_path, exist_ok=True)

    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        if "vector_store" in config_dict and "config" in config_dict["vector_store"]:
            config_dict["vector_store"]["config"] = config_dict["vector_store"].get("config") or {}
            config_dict["vector_store"]["config"]["path"] = vector_store_path
            config_dict["vector_store"]["config"]["collection_name"] = "locomo_eval"
        config = MemoryConfig(**config_dict)
    elif use_local_llm:
        llm_base_url = os.environ.get(ENV_LLM_BASE_URL, DEFAULT_LLM_BASE_URL)
        llm_model = os.environ.get(ENV_LLM_MODEL_ID, DEFAULT_LLM_MODEL_ID)
        llm_api_key = os.environ.get(ENV_LLM_API_KEY, DEFAULT_LLM_API_KEY)
        config = MemoryConfig(
            vector_store={
                "provider": "faiss",
                "config": {
                    "path": vector_store_path,
                    "collection_name": "locomo_eval",
                    "embedding_model_dims": OPENAI_EMBED_DIMS,
                },
            },
            llm={
                "provider": "ollama",
                "config": {
                    "model": llm_model,
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": llm_base_url,
                    "api_key": llm_api_key,
                },
            },
            embedder={
                "provider": "openai",
                "config": {
                    "embedding_dims": OPENAI_EMBED_DIMS,
                },
            },
        )
    else:
        config = MemoryConfig(
            vector_store={
                "provider": "faiss",
                "config": {
                    "path": vector_store_path,
                    "collection_name": "locomo_eval",
                    "embedding_model_dims": TITAN_EMBED_DIMS,
                },
            },
            llm={
                "provider": "aws_bedrock",
                "config": {
                    "model": build_inference_profile_arn(region, account_id, inference_profile_id),
                    "temperature": 0.1,
                    "max_tokens": 2000,
                },
            },
            embedder={
                "provider": "aws_bedrock",
                "config": {
                    "model": embed_model,
                    "embedding_dims": TITAN_EMBED_DIMS,
                    "aws_region": region,
                },
            },
        )

    memory = Memory(config=config)

    for idx, item in enumerate(data):
        conv = item.get("conversation") or {}
        speaker_a = conv.get("speaker_a") or "Speaker_A"
        speaker_b = conv.get("speaker_b") or "Speaker_B"
        user_id_a = f"{speaker_a}_{idx}"
        user_id_b = f"{speaker_b}_{idx}"

        for uid in (user_id_a, user_id_b):
            try:
                out = memory.get_all(user_id=uid, limit=10000)
                items = out.get("results") if isinstance(out, dict) else (out if isinstance(out, list) else [])
                for item in items or []:
                    mid = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
                    if mid:
                        memory.delete(mid)
            except Exception:
                pass

        for session_key in _session_keys(conv):
            date_key = f"{session_key}_date_time"
            timestamp = conv.get(date_key) or ""
            messages_a, messages_b = build_messages_for_speaker(conv, speaker_a, speaker_b, session_key)
            if not messages_a:
                continue
            metadata = {"timestamp": timestamp}
            try:
                memory.add(
                    messages_a,
                    user_id=user_id_a,
                    metadata=metadata,
                    infer=True,
                )
            except Exception as e:
                print(f"  [add] sample {idx} {session_key} speaker_a error: {e}", file=sys.stderr)
            try:
                memory.add(
                    messages_b,
                    user_id=user_id_b,
                    metadata=metadata,
                    infer=True,
                )
            except Exception as e:
                print(f"  [add] sample {idx} {session_key} speaker_b error: {e}", file=sys.stderr)

        print(f"  [add] sample {idx} ({speaker_a} / {speaker_b}) done.")

    print(f"Add 完成，向量库目录: {vector_store_path}")

def invoke_bedrock(region: str, model_id: str, prompt: str, max_tokens: int = 256):
    import boto3
    client = boto3.client("bedrock-runtime", region_name=region)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    resp = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    ret = json.loads(resp["body"].read())
    text = ""
    if ret.get("content") and isinstance(ret["content"], list) and len(ret["content"]) > 0:
        text = (ret["content"][0].get("text", "") or "").strip()
    usage = ret.get("usage", {})
    inp = usage.get("input_tokens", usage.get("inputTokens", 0))
    out_tok = usage.get("output_tokens", usage.get("outputTokens", 0))
    return text, inp, out_tok

def _tokenize(text):
    t = str(text).lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ")
    return t.split()

def _f1(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    pred_tokens = set(_tokenize(prediction))
    ref_tokens = set(_tokenize(reference))
    common = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(ref_tokens)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def _bleu1(prediction: str, reference: str) -> float:
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        pred_tokens = nltk.word_tokenize(prediction.lower())
        ref_tokens = [nltk.word_tokenize(reference.lower())]
        return sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    except Exception:
        pt, rt = _tokenize(prediction), _tokenize(reference)
        if not rt:
            return 0.0
        return sum(1 for t in pt if t in rt) / len(rt) if rt else 0.0

def format_memories_for_prompt(results: list) -> str:
    lines = []
    for r in results:
        mem = r.get("memory") or ""
        ts = r.get("updated_at") or r.get("created_at") or ""
        if ts and mem:
            lines.append(f"{ts}: {mem}")
        elif mem:
            lines.append(mem)
    return json.dumps(lines, indent=2) if lines else "[]"

def run_search(
    data_path: str,
    outdir: str,
    max_samples: int,
    max_questions: int | None,
    top_k: int,
    config_path: str | None,
    region: str,
    account_id: str,
    inference_profile_id: str,
    embed_model: str,
    use_local_llm: bool,
    price_input: float,
    price_output: float,
    verbose: bool,
) -> None:
    sys.path.insert(0, _SCRIPT_DIR)
    from mem0 import Memory
    from mem0.configs.base import MemoryConfig

    model_id = build_inference_profile_arn(region, account_id, inference_profile_id)
    if not use_local_llm:
        _patch_mem0_bedrock_for_inference_profile()
        os.environ["AWS_REGION"] = region

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("data must be a list")

    data = data[:max_samples]
    vector_store_path = os.path.join(outdir, "vector_store")
    if not os.path.isdir(vector_store_path):
        raise FileNotFoundError(
            f"向量库目录不存在: {vector_store_path}，请先运行 --method add 并指定相同 --outdir"
        )

    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        if "vector_store" in config_dict and "config" in config_dict["vector_store"]:
            config_dict["vector_store"]["config"] = config_dict["vector_store"].get("config") or {}
            config_dict["vector_store"]["config"]["path"] = vector_store_path
            config_dict["vector_store"]["config"]["collection_name"] = "locomo_eval"
        config = MemoryConfig(**config_dict)
    elif use_local_llm:
        llm_base_url = os.environ.get(ENV_LLM_BASE_URL, DEFAULT_LLM_BASE_URL)
        llm_model = os.environ.get(ENV_LLM_MODEL_ID, DEFAULT_LLM_MODEL_ID)
        llm_api_key = os.environ.get(ENV_LLM_API_KEY, DEFAULT_LLM_API_KEY)
        config = MemoryConfig(
            vector_store={
                "provider": "faiss",
                "config": {
                    "path": vector_store_path,
                    "collection_name": "locomo_eval",
                    "embedding_model_dims": OPENAI_EMBED_DIMS,
                },
            },
            llm={
                "provider": "ollama",
                "config": {
                    "model": llm_model,
                    "ollama_base_url": llm_base_url,
                    "api_key": llm_api_key,
                },
            },
            embedder={
                "provider": "openai",
                "config": {"embedding_dims": OPENAI_EMBED_DIMS},
            },
        )
    else:
        config = MemoryConfig(
            vector_store={
                "provider": "faiss",
                "config": {
                    "path": vector_store_path,
                    "collection_name": "locomo_eval",
                    "embedding_model_dims": TITAN_EMBED_DIMS,
                },
            },
            llm={
                "provider": "aws_bedrock",
                "config": {"model": model_id, "temperature": 0.1, "max_tokens": 2000},
            },
            embedder={
                "provider": "aws_bedrock",
                "config": {
                    "model": embed_model,
                    "embedding_dims": TITAN_EMBED_DIMS,
                    "aws_region": region,
                },
            },
        )

    memory = Memory(config=config)

    results = []
    total_in, total_out = 0, 0
    count = 0
    start_time = time.perf_counter()

    for conv_idx, item in enumerate(data):
        conv = item.get("conversation") or {}
        speaker_a = conv.get("speaker_a") or "Speaker_A"
        speaker_b = conv.get("speaker_b") or "Speaker_B"
        user_id_a = f"{speaker_a}_{conv_idx}"
        user_id_b = f"{speaker_b}_{conv_idx}"
        qa_list = item.get("qa", [])

        for q in qa_list:
            if max_questions is not None and count >= max_questions:
                break
            if str(q.get("category", "")) == "5":
                continue
            question = str(q.get("question") or "")
            raw_answer = q.get("answer", "")
            answer = "" if raw_answer is None else str(raw_answer)

            try:
                out_a = memory.search(question, user_id=user_id_a, limit=top_k)
                out_b = memory.search(question, user_id=user_id_b, limit=top_k)
            except Exception as e:
                print(f"  [search] conv={conv_idx} q error: {e}", file=sys.stderr)
                results.append({
                    "conv_idx": conv_idx, "q_idx": count + 1, "category": q.get("category"),
                    "question": question, "answer": answer, "response": "", "f1": 0.0, "bleu1": 0.0,
                })
                count += 1
                continue

            list_a = out_a.get("results") or []
            list_b = out_b.get("results") or []
            str_a = format_memories_for_prompt(list_a)
            str_b = format_memories_for_prompt(list_b)

            prompt = (
                ANSWER_PROMPT_TWO_SPEAKERS.replace("{{speaker_1_user_id}}", speaker_a)
                .replace("{{speaker_2_user_id}}", speaker_b)
                .replace("{{speaker_1_memories}}", str_a)
                .replace("{{speaker_2_memories}}", str_b)
                .replace("{{question}}", question)
            )

            try:
                response_text, inp, out_tok = invoke_bedrock(region, model_id, prompt)
            except Exception as e:
                print(f"  [bedrock] conv={conv_idx} q error: {e}", file=sys.stderr)
                response_text, inp, out_tok = "", 0, 0

            total_in += inp
            total_out += out_tok
            f1 = _f1(response_text, answer)
            bleu1 = _bleu1(response_text, answer)
            count += 1

            results.append({
                "conv_idx": conv_idx,
                "q_idx": count,
                "category": q.get("category"),
                "question": question,
                "answer": answer,
                "response": response_text,
                "f1": f1,
                "bleu1": bleu1,
            })
            print(f"  [sample {conv_idx} q{count}] cat={q.get('category')} F1={f1:.4f} BLEU1={bleu1:.4f} | Q: {question[:50]}...")
            if verbose and count <= 3:
                print(f"      gold:  {repr(answer)}")
                print(f"      model: {repr(response_text)}")

        if max_questions is not None and count >= max_questions:
            break

    if not results:
        print("没有有效问题结果")
        return

    end_time = time.perf_counter()
    n = len(results)
    avg_f1 = sum(r["f1"] for r in results) / n
    avg_bleu1 = sum(r["bleu1"] for r in results) / n
    n_samples = len(data)
    cost_usd = (total_in / 1e6) * price_input + (total_out / 1e6) * price_output

    print()
    print("=" * 60)
    print("LoCoMo mem0 完整测评（add → search → Bedrock）")
    print("=" * 60)
    print(f"  问题数: {n}")
    print(f"  样本数: {n_samples}")
    print(f"  input_tokens:  {total_in}")
    print(f"  output_tokens: {total_out}")
    print(f"  Experiment Time: {end_time - start_time:.2f} s")
    print(f"  估算费用: ${cost_usd:.4f}")
    print(f"  平均 F1:   {avg_f1:.4f}")
    print(f"  平均 BLEU1: {avg_bleu1:.4f}")
    print("=" * 60)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"results": results, "total_questions": n, "avg_f1": avg_f1, "avg_bleu1": avg_bleu1},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"结果已保存: {out_path}")

    log_path = os.path.join(outdir, "experiment_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "method": "search",
                    "argv": sys.argv,
                    "outdir": os.path.abspath(outdir),
                    "total_questions": n,
                    "avg_f1": avg_f1,
                    "avg_bleu1": avg_bleu1,
                    "input_tokens": total_in,
                    "output_tokens": total_out,
                    "cost_usd": round(cost_usd, 4),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    print(f"实验记录已追加: {log_path}")

def main():
    parser = argparse.ArgumentParser(description="LoCoMo 完整 mem0 测评：add → search → Bedrock")
    parser.add_argument("--method", choices=["add", "search", "both"], default="both", help="add | search | both")
    parser.add_argument("--data", type=str, default=None, help="locomo10.json 路径")
    parser.add_argument("--outdir", type=str, default="./results/eval_locomo_mem0_full", help="输出目录（add/search 共用，向量库在 outdir/vector_store）")
    parser.add_argument("--config", type=str, default=None, help="MemoryConfig 的 JSON 文件（可选）；未指定则用默认 FAISS")
    parser.add_argument("--max-samples", type=int, default=1, help="跑几个 sample")
    parser.add_argument("--max-questions", type=int, default=None, help="search 时最多多少题，默认不限制")
    parser.add_argument("--top-k", type=int, default=30, help="search 时每个 user 取几条记忆")
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--account-id", type=str, default=DEFAULT_ACCOUNT_ID)
    parser.add_argument("--inference-profile-id", type=str, default=DEFAULT_INFERENCE_PROFILE_ID)
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="Bedrock 嵌入模型（仅 add/search 默认配置时使用，如 amazon.titan-embed-text-v2:0）")
    parser.add_argument("--use-local-llm", action="store_true", help="用 Ollama 本地 LLM + OpenAI 嵌入：LLM_BASE_URL/LLM_MODEL_ID/LLM_API_KEY 与 OPENAI_API_KEY 从环境变量读取")
    parser.add_argument("--price-input", type=float, default=3.0)
    parser.add_argument("--price-output", type=float, default=15.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_path = args.data or DEFAULT_DATA
    if not os.path.isfile(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return 1

    if args.method in ("add", "both"):
        run_add(
            data_path,
            args.outdir,
            args.max_samples,
            args.config,
            args.region,
            args.account_id,
            args.inference_profile_id,
            args.embed_model,
            args.use_local_llm,
        )
    if args.method in ("search", "both"):
        run_search(
            data_path,
            args.outdir,
            args.max_samples,
            args.max_questions,
            args.top_k,
            args.config,
            args.region,
            args.account_id,
            args.inference_profile_id,
            args.embed_model,
            args.use_local_llm,
            args.price_input,
            args.price_output,
            args.verbose,
        )

    return 0

if __name__ == "__main__":
    sys.exit(main())
