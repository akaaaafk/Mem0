"""
Direct LoCoMo evaluation with AWS Bedrock (Claude Haiku) reading FULL conversations (no mem0).

For each LoCoMo conversation and QA pair:
- Build a prompt containing the full conversation text and the question
- Call Bedrock Haiku once per question
- Use the existing calculate_metrics / calculate_bleu_scores to compute F1 and BLEU1
- Aggregate F1/BLEU1 by category (no accuracy)

Usage (from mem0 repo root):
  python -m evaluation.haiku.run_locomo_direct_haiku --model_arn <inference_profile_or_model_arn>

Environment:
  AWS credentials (profile or env) for Bedrock.
  Optional: AWS_REGION (default us-east-1).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

_haiku_dir = os.path.dirname(os.path.abspath(__file__))
_evals_dir = os.path.dirname(_haiku_dir)
_project_root = os.path.dirname(_evals_dir)
if _evals_dir not in sys.path:
    sys.path.insert(0, _evals_dir)

from dotenv import load_dotenv
from jinja2 import Template

from metrics.utils import calculate_bleu_scores, calculate_metrics
from haiku.bedrock_client import DEFAULT_REGION, invoke_bedrock

load_dotenv(os.path.join(_evals_dir, ".env"))

DEFAULT_LOCOMO_DATA = os.path.join(_evals_dir, "data", "locomo", "locomo10.json")


FULL_DIALOG_PROMPT = """
You are an intelligent assistant tasked with answering questions about a conversation.

# CONTEXT:
You are given the full conversation between two speakers. The conversation is split into turns with turn IDs.
Use ONLY the information in the conversation to answer the question.

# INSTRUCTIONS:
1. Carefully read the entire conversation.
2. If the question asks about a specific event, fact, time, or number, look for direct evidence in the conversation.
3. If multiple turns mention related information, combine them to answer the question.
4. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date
   based on the conversation text when possible.
5. Focus only on the content of the conversation. Do not make up facts that are not supported by the text.
6. The answer should be a short phrase (ideally 5–10 words), not a long explanation.

# APPROACH (Think briefly then answer):
1. Identify the turns that are relevant to the question.
2. Extract the key fact(s) needed to answer.
3. Answer with a concise phrase that directly answers the question.

Conversation:
{{conversation}}

Question: {{question}}

Answer:
"""


def load_locomo(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_conversation_text(item: dict) -> str:
    """LoCoMo item has a 'conversation' field with turns; render to plain text."""
    conv = item.get("conversation", {})
    if not isinstance(conv, dict):
        return ""
    turns = conv.get("turns") or conv.get("dialogue") or conv.get("conversation", [])
    if not isinstance(turns, list):
        return json.dumps(conv, ensure_ascii=False, indent=2)
    lines = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        turn_id = t.get("turn_id", t.get("id", ""))
        speaker = t.get("speaker") or t.get("role") or ""
        text = t.get("utterance") or t.get("text") or ""
        prefix = f"[{turn_id}] " if turn_id != "" else ""
        speaker_prefix = f"{speaker}: " if speaker else ""
        lines.append(f"{prefix}{speaker_prefix}{text}")
    return "\n".join(lines)


def answer_with_haiku(region: str, model_arn: str, conversation: str, question: str) -> str:
    template = Template(FULL_DIALOG_PROMPT)
    system_prompt = template.render(conversation=conversation, question=question)
    # Keep user message very short; all context is in system
    user_msg = f"Answer the question with a short phrase only.\n\nQuestion: {question}"
    text, _, _ = invoke_bedrock(region=region, model_id=model_arn, prompt=user_msg, max_tokens=64, system=system_prompt)
    return (text or "").strip()


def aggregate_by_category(items):
    by_cat = defaultdict(lambda: {"f1": [], "bleu1": []})
    for it in items:
        c = it.get("category", 0)
        by_cat[c]["f1"].append(it["f1"])
        by_cat[c]["bleu1"].append(it["bleu1"])
    return by_cat


def print_summary(by_cat, items):
    print("LoCoMo FULL-DIALOG – Score summary (Haiku, no memory)")
    print("By Category:")
    for c in sorted(by_cat.keys(), key=lambda x: (int(x) if str(x).isdigit() else 0)):
        v = by_cat[c]
        n = len(v["f1"])
        f1_avg = sum(v["f1"]) / n if n else 0.0
        b1_avg = sum(v["bleu1"]) / n if n else 0.0
        print(f"  Category {c}:  n={n}  F1 avg={f1_avg:.4f}  BLEU1 avg={b1_avg:.4f}")
    n_total = len(items)
    f1_overall = sum(i["f1"] for i in items) / n_total if n_total else 0.0
    b1_overall = sum(i["bleu1"] for i in items) / n_total if n_total else 0.0
    print("Overall:")
    print(f"  Number of questions: {n_total}")
    print(f"  Average F1: {f1_overall:.4f}")
    print(f"  Average BLEU1: {b1_overall:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="LoCoMo direct evaluation with Bedrock Haiku (no mem0).")
    parser.add_argument("--data_path", type=str, default=None, help="LoCoMo dataset path (default data/locomo/locomo10.json)")
    parser.add_argument("--output_file", type=str, default=None, help="Where to save per-question results JSON")
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION", DEFAULT_REGION))
    parser.add_argument("--model_arn", type=str, required=True, help="Bedrock model or inference profile ARN for Haiku")
    parser.add_argument("--max_questions", type=int, default=None, help="Optional limit on number of questions")
    args = parser.parse_args()

    data_path = args.data_path or DEFAULT_LOCOMO_DATA
    data_path = os.path.abspath(os.path.normpath(data_path))
    if not os.path.isfile(data_path):
        print(f"[run_locomo_direct_haiku] LoCoMo data not found at {data_path}")
        return 1

    out_path = args.output_file or os.path.join(_haiku_dir, "results", "locomo_haiku", "direct_haiku_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data = load_locomo(data_path)
    results = defaultdict(list)
    flat_items = []
    total_q = 0

    t0 = time.perf_counter()
    for idx, item in enumerate(data):
        conversation_text = build_conversation_text(item)
        qa_list = item.get("qa", [])
        if not isinstance(qa_list, list):
            continue
        for q in qa_list:
            if not isinstance(q, dict):
                continue
            if args.max_questions is not None and total_q >= args.max_questions:
                break
            question = str(q.get("question") or "")
            answer = "" if q.get("answer") is None else str(q.get("answer", ""))
            category = q.get("category", -1)

            try:
                pred = answer_with_haiku(args.region, args.model_arn, conversation_text, question)
            except Exception as e:
                print(f"[run_locomo_direct_haiku] error conv={idx}: {e}", file=sys.stderr)
                pred = ""

            m = calculate_metrics(pred.strip(), answer.strip())
            b = calculate_bleu_scores(pred.strip(), answer.strip())
            f1 = m["f1"]
            bleu1 = b.get("bleu1", 0.0)

            rec = {
                "question": question,
                "answer": answer,
                "response": pred,
                "category": category,
                "f1": f1,
                "bleu1": bleu1,
            }
            results[str(idx)].append(rec)
            flat_items.append(rec)
            total_q += 1

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dict(results), f, indent=2, ensure_ascii=False)
        if args.max_questions is not None and total_q >= args.max_questions:
            break

    elapsed = time.perf_counter() - t0
    by_cat = aggregate_by_category(flat_items)
    print_summary(by_cat, flat_items)
    print(f"[run_locomo_direct_haiku] Done. questions={total_q} time_sec={elapsed:.2f}")
    print(f"[run_locomo_direct_haiku] Results saved to: {os.path.abspath(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

