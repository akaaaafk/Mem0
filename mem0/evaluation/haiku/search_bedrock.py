"""
LoCoMo search phase: mem0 vector search + AWS Bedrock (Claude Haiku) for answer.
Uses same prompt and answer extraction as evaluation run_locomo (memzero) for comparable scores.
Produces mem0_results.json and run_stats compatible with evaluation/score_results.py.
"""
import json
import os
import re
import sys
import time
from collections import defaultdict

from jinja2 import Template

# Reuse official prompt and extractor from evaluation (same as run_locomo / memzero search)
from evaluation.prompts import ANSWER_PROMPT

# Default 0: match evaluation/src/memzero/search.py (Qwen path) — same user text + heuristics; paper Mem0 row uses this style.
# Set LOCOMO_JSON_ANSWER=1 for experimental JSON {"answer":"..."} parsing.
_USE_JSON_ANSWER = os.environ.get("LOCOMO_JSON_ANSWER", "0").strip().lower() not in ("0", "false", "no")


def _json_answer_candidates(s: str) -> list[str]:
    """Collect substrings that might parse as {"answer": "..."}."""
    s = s.strip()
    out: list[str] = []
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.I)
    if m:
        out.append(m.group(1).strip())
    out.append(s)
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        out.append(s[start : end + 1])
    return out


def _try_parse_answer_json(raw: str) -> str | None:
    """If model returns {"answer": "..."} (Bedrock Haiku path), use it — improves F1/BLEU vs line heuristics."""
    if not raw or not isinstance(raw, str):
        return None
    for candidate in _json_answer_candidates(raw):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        val = obj.get("answer")
        if val is None:
            val = obj.get("Answer")
        if val is None:
            continue
        out = str(val).strip()
        if not out:
            continue
        low = out.lower()
        if low in ("none", "n/a", "none.", "n/a.", "null"):
            continue
        return out
    return None


def _extract_short_answer(raw: str) -> str:
    """Prefer JSON {"answer"} (Haiku), else same heuristics as evaluation/src/memzero/search.py."""
    if not raw or not isinstance(raw, str):
        return ""
    raw = raw.strip()
    if not raw:
        return ""
    if _USE_JSON_ANSWER:
        j = _try_parse_answer_json(raw)
        if j:
            return j
    lines = [s.strip() for s in raw.split("\n") if s.strip()]
    for line in lines:
        lower = line.lower()
        if lower.startswith("answer:"):
            out = line[7:].strip() or line.strip()
            if out and out.lower() not in ("none", "n/a", "none.", "n/a."):
                return out
        m = re.search(r"\banswer\s+is\s*[:\-]?\s*(.+)", lower, re.I)
        if m:
            out = m.group(1).strip()
            if out and out.lower() not in ("none", "n/a", "none.", "n/a."):
                return out
    for line in lines:
        if len(line) <= 60 and not re.match(r"^(the|a|an|it|this|that|there)\s", line, re.I):
            if line.lower() not in ("none", "n/a", "none.", "n/a.", "i don't know", "not found"):
                return line
    short_lines = [ln for ln in lines if 2 <= len(ln) <= 80 and ln.lower() not in ("none", "n/a", "i don't know", "not found")]
    if short_lines:
        return min(short_lines, key=len)
    last = lines[-1] if lines else raw
    if len(last) > 100:
        m = re.search(r"(\d{1,3}(?:,\d{3})*)\s*seated", last, re.I)
        if m:
            return m.group(0).strip()
        m = re.search(r"(?:capacity|seat[s]?)\s*(?:of\s*)?(\d{1,3}(?:,\d{3})*)", last, re.I)
        if m:
            return m.group(1) + " seated"
        parts = re.split(r",\s*", last)
        for p in reversed(parts):
            p = p.strip()
            if 5 <= len(p) <= 60:
                p = re.sub(r"[.!?]\s*$", "", p)
                if p:
                    return p
        parts = re.split(r"[.!?]+", last)
        parts = [p.strip() for p in parts if p.strip()]
        last = parts[-1] if parts else last
        if len(last) > 100:
            last = last[:100]
    return last.strip() if last else raw[:200].strip()


def run_search_loop(
    mem0_client,
    data_path: str,
    output_path: str,
    run_stats: dict,
    region: str,
    inference_profile_arn: str,
    top_k: int = 30,
    max_questions: int | None = None,
):
    """
    Load LoCoMo JSON, for each conversation and qa: search both speakers, build prompt,
    invoke Bedrock, collect results. Write mem0_results.json (format expected by score_results.py).
    """
    from .bedrock_client import invoke_bedrock

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("LoCoMo data must be a list of items with 'conversation' and 'qa'")

    results = defaultdict(list)
    total_questions_done = 0
    stage_start = time.time()

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        conv = item.get("conversation") or {}
        if not isinstance(conv, dict):
            conv = {}
        speaker_a = conv.get("speaker_a") or "Speaker_A"
        speaker_b = conv.get("speaker_b") or "Speaker_B"
        user_id_a = f"{speaker_a}_{idx}"
        user_id_b = f"{speaker_b}_{idx}"
        qa_list = item.get("qa", [])

        for q in qa_list:
            if not isinstance(q, dict):
                continue
            if max_questions is not None and total_questions_done >= max_questions:
                break
            if str(q.get("category", "")) == "5":
                continue
            question = str(q.get("question") or "")
            answer = "" if q.get("answer") is None else str(q.get("answer", ""))
            category = q.get("category", -1)
            evidence = q.get("evidence", [])
            adversarial_answer = q.get("adversarial_answer", "")

            try:
                raw_a = mem0_client.search(question, user_id=user_id_a, limit=top_k)
                raw_b = mem0_client.search(question, user_id=user_id_b, limit=top_k)
            except Exception as e:
                print(f"[search_bedrock] conv={idx} q error: {e}", file=sys.stderr)
                results[str(idx)].append({
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "evidence": evidence,
                    "response": "",
                    "adversarial_answer": adversarial_answer,
                    "speaker_1_memories": [],
                    "speaker_2_memories": [],
                    "num_speaker_1_memories": 0,
                    "num_speaker_2_memories": 0,
                    "speaker_1_memory_time": 0,
                    "speaker_2_memory_time": 0,
                    "response_time": 0,
                })
                total_questions_done += 1
                continue

            list_a = raw_a.get("results") if isinstance(raw_a, dict) else (raw_a if isinstance(raw_a, list) else [])
            list_b = raw_b.get("results") if isinstance(raw_b, dict) else (raw_b if isinstance(raw_b, list) else [])
            list_a = list_a or []
            list_b = list_b or []
            def _mem_line(m):
                meta = m.get("metadata") if isinstance(m, dict) else {}
                if not isinstance(meta, dict):
                    meta = {}
                ts = meta.get("timestamp", "")
                text = m.get("memory", m.get("message", "")) if isinstance(m, dict) else ""
                return f"{ts}: {text}"
            def _mem_dict(m):
                meta = m.get("metadata") if isinstance(m, dict) else {}
                if not isinstance(meta, dict):
                    meta = {}
                return {"memory": m.get("memory", m.get("message", "")) if isinstance(m, dict) else "", "timestamp": meta.get("timestamp", "")}
            mem_a = [_mem_line(m) for m in list_a]
            mem_b = [_mem_line(m) for m in list_b]
            speaker_1_memories = [_mem_dict(m) for m in list_a]
            speaker_2_memories = [_mem_dict(m) for m in list_b]

            # Same display names as MemorySearch.answer_question (search.py): first segment before "_"
            sp1_label = user_id_a.split("_")[0]
            sp2_label = user_id_b.split("_")[0]
            template = Template(ANSWER_PROMPT)
            prompt = template.render(
                speaker_1_user_id=sp1_label,
                speaker_2_user_id=sp2_label,
                speaker_1_memories=json.dumps(mem_a, indent=4, ensure_ascii=False),
                speaker_2_memories=json.dumps(mem_b, indent=4, ensure_ascii=False),
                question=question,
            )
            # JSON line: stable parsing + shorter surface form → better F1/BLEU than free-text heuristics.
            if _USE_JSON_ANSWER:
                user_msg = (
                    "The system message already contains the memories and the question.\n"
                    "Reply with ONLY one line of valid JSON (no markdown, no other text). Schema:\n"
                    '{"answer":"<factual phrase only, maximum 6 words; match style of gold answers e.g. dates/names>"}\n'
                    'If memories do not support an answer, use: {"answer":"unknown"}\n'
                    "Rules: no full sentences; no explanation; value is the answer string only."
                )
            else:
                user_msg = (
                    f"Reply with ONLY the short answer (e.g. a date, name, or phrase in 5–6 words). No explanation.\n\n"
                    f"Question: {question}"
                )

            t1 = time.time()
            response_text = ""
            inp, out_tok = 0, 0
            try:
                _default_max = "384" if _USE_JSON_ANSWER else "1024"
                max_out = int(os.environ.get("LOCOMO_BEDROCK_MAX_TOKENS", _default_max))
                for attempt in range(2):
                    raw_text, i, o = invoke_bedrock(
                        region, inference_profile_arn, user_msg, max_tokens=max_out, system=prompt
                    )
                    inp += i
                    out_tok += o
                    response_text = (_extract_short_answer(raw_text) or raw_text or "").strip()
                    if response_text or attempt == 1:
                        break
            except Exception as e:
                print(f"[search_bedrock] bedrock conv={idx} q error: {e}", file=sys.stderr)
                response_text, inp, out_tok = "", 0, 0
            response_time = time.time() - t1

            run_stats["input_tokens"] = run_stats.get("input_tokens", 0) + inp
            run_stats["output_tokens"] = run_stats.get("output_tokens", 0) + out_tok
            run_stats["context_window_peak"] = max(run_stats.get("context_window_peak", 0), inp)

            results[str(idx)].append({
                "question": question,
                "answer": answer,
                "category": category,
                "evidence": evidence,
                "response": response_text,
                "adversarial_answer": adversarial_answer,
                "speaker_1_memories": speaker_1_memories,
                "speaker_2_memories": speaker_2_memories,
                "num_speaker_1_memories": len(speaker_1_memories),
                "num_speaker_2_memories": len(speaker_2_memories),
                "speaker_1_memory_time": 0,
                "speaker_2_memory_time": 0,
                "response_time": response_time,
            })
            total_questions_done += 1

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dict(results), f, indent=4, ensure_ascii=False)

    run_stats["search_time_sec"] = time.time() - stage_start
    run_stats["n_questions"] = total_questions_done
    return dict(results)
