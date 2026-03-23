"""
RULER 128k scoring: retri.acc, MT acc, ACG acc, QA acc from mem0 search results.
Expects each item to have task_type in {"retrieval", "MT", "ACG", "QA"} or category mapping.
Uses flexible matching when item has "outputs" (list): match if response matches any acceptable answer.
For ACG: CWE uses --strict_acg (default first). FWE always uses strict 3-in-order: response must be exactly 3 tokens matching the 3 refs in order.
Outputs: running_time, log, context_window_peak, cost, retri.acc, MT acc, ACG acc, QA acc.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

_evals_dir = os.path.dirname(os.path.abspath(__file__))
if _evals_dir not in sys.path:
    sys.path.insert(0, _evals_dir)

# Map category or task_type to RULER metric name
TASK_NAMES = {"retrieval": "retri.acc", "MT": "MT acc", "ACG": "AGG acc", "QA": "QA acc"}
CATEGORY_TO_TASK = {"0": "retrieval", "1": "MT", "2": "ACG", "3": "QA"}


def _normalize_text(text):
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _response_matches_any(model_response, ground_truth_outputs):
    """True if model response matches any of the acceptable answers (flexible: substring, normalized, or all words)."""
    if not ground_truth_outputs or not model_response:
        return False
    model_response = str(model_response).strip()
    model_lower = model_response.lower()
    model_norm = _normalize_text(model_response)
    unique_answers = list(set(ground_truth_outputs))
    for answer in unique_answers:
        answer_str = str(answer).strip()
        if not answer_str:
            continue
        answer_lower = answer_str.lower()
        if answer_lower in model_lower:
            return True
        answer_norm = _normalize_text(answer_str)
        if answer_norm and answer_norm in model_norm:
            return True
        answer_words = [w for w in answer_norm.split() if len(w) > 2]
        if answer_words and all(w in model_norm for w in answer_words):
            return True
    return False


def _tokens_from_response(text):
    """Split response by comma/space, normalize each token, return list (preserving order)."""
    if not text:
        return []
    text = str(text).strip()
    parts = re.split(r"[,\s]+", text)
    return [_normalize_text(p) for p in parts if _normalize_text(p)]


def _response_matches_any_strict_token(model_response, ground_truth_outputs):
    """Strict: correct only if one of the refs exactly equals one of the tokens (no substring).
    For ACG (CWE/FWE): response is a list of words; we require the answer to appear as a whole token."""
    if not ground_truth_outputs or not model_response:
        return False
    tokens = _tokens_from_response(model_response)
    if not tokens:
        return False
    refs_norm = set(_normalize_text(str(a).strip()) for a in ground_truth_outputs if str(a).strip())
    return any(ref in refs_norm for ref in tokens)


def _response_matches_any_strict_first(model_response, ground_truth_outputs):
    """Strict for ACG: correct only if the first token of the response exactly equals one of the refs."""
    if not ground_truth_outputs or not model_response:
        return False
    tokens = _tokens_from_response(model_response)
    if not tokens:
        return False
    first_token = tokens[0]
    refs_norm = set(_normalize_text(str(a).strip()) for a in ground_truth_outputs if str(a).strip())
    return first_token in refs_norm


def _response_matches_first_ref_only(model_response, ground_truth_outputs):
    """Stricter ACG (CWE): correct only if the first token equals the first ref (the most frequent).
    Lowers ACG vs first (which accepts first token matching any ref)."""
    if not ground_truth_outputs or not model_response:
        return False
    tokens = _tokens_from_response(model_response)
    if not tokens:
        return False
    first_ref = _normalize_text(str(ground_truth_outputs[0]).strip())
    return first_ref and tokens[0] == first_ref


def _response_matches_fwe_strict(model_response, ground_truth_outputs):
    """FWE strict: require exactly 3 tokens in response, matching the 3 refs in order.
    FWE asks for 'the three most frequently appeared coded words' in order."""
    if not ground_truth_outputs or not model_response:
        return False
    refs = [str(a).strip() for a in ground_truth_outputs if str(a).strip()][:3]
    if len(refs) != 3:
        return False
    tokens = _tokens_from_response(model_response)
    if len(tokens) != 3:
        return False
    refs_norm = [_normalize_text(r) for r in refs]
    return all(tokens[i] == refs_norm[i] for i in range(3))


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten(data):
    items = []
    for conv_id, qa_list in data.items():
        for item in qa_list:
            items.append({**item, "conv_id": conv_id})
    return items


def main():
    parser = argparse.ArgumentParser(description="Score RULER results (retri.acc, MT acc, ACG acc, QA acc)")
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--stats_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_id", type=str, default="ruler_128k")
    parser.add_argument(
        "--strict_acg",
        type=str,
        choices=("none", "token", "first", "first_ref_only"),
        default="first_ref_only",
        help="ACG (CWE) matching: none=flexible; token=exact token in list; first=first token in refs; first_ref_only=first token must equal first ref (default, lower ACG)",
    )
    args = parser.parse_args()

    if args.strict_acg == "none":
        acg_matcher = _response_matches_any
    elif args.strict_acg == "token":
        acg_matcher = _response_matches_any_strict_token
    elif args.strict_acg == "first_ref_only":
        acg_matcher = _response_matches_first_ref_only
    else:
        acg_matcher = _response_matches_any_strict_first

    data = load_results(args.results_file)
    items = flatten(data)
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in items:
        pred = str(item.get("response") or "").strip()
        outputs = item.get("outputs")
        if isinstance(outputs, list) and len(outputs) > 0:
            refs = outputs
        else:
            ref = str(item.get("answer") or "").strip()
            refs = [ref] if ref else []
        task = item.get("task_type") or CATEGORY_TO_TASK.get(str(item.get("category", "")), "QA")
        if task == "ACG":
            # FWE: require exactly 3 tokens matching the 3 refs in order
            if item.get("dataset") == "fwe" and len(refs) >= 3:
                correct = 1 if _response_matches_fwe_strict(pred, refs) else 0
            else:
                correct = 1 if refs and acg_matcher(pred, refs) else 0
        else:
            correct = 1 if refs and _response_matches_any(pred, refs) else 0
        by_task[task]["correct"] += correct
        by_task[task]["total"] += 1

    metrics = {}
    for task, name in TASK_NAMES.items():
        d = by_task.get(task, {"correct": 0, "total": 0})
        acc = d["correct"] / d["total"] if d["total"] else 0.0
        metrics[name] = acc
    overall = sum(by_task[t]["correct"] for t in TASK_NAMES) / max(1, sum(by_task[t]["total"] for t in TASK_NAMES))

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results_file))
    os.makedirs(out_dir, exist_ok=True)

    stats = None
    if args.stats_file and os.path.isfile(args.stats_file):
        with open(args.stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

    summary = {
        "n_questions": len(items),
        "retri.acc": metrics.get("retri.acc", 0.0),
        "MT acc": metrics.get("MT acc", 0.0),
        "AGG acc": metrics.get("AGG acc", 0.0),
        "QA acc": metrics.get("QA acc", 0.0),
        "overall_accuracy": overall,
    }
    if stats:
        summary["token_and_cost"] = stats
        summary["context_window_peak"] = stats.get("context_window_peak")
        summary["running_time"] = stats.get("running_time_sec")
        summary["cost"] = stats.get("estimated_cost")
    stats_path = os.path.join(out_dir, f"batch_statistics_{args.batch_id}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[RULER 128k] strict_acg=%s" % args.strict_acg, " | ", " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    s = stats or {}
    print(
        f"[Eval summary] running_time={s.get('running_time_sec', 'N/A')} | context_window_peak={s.get('context_window_peak', 'N/A')} | "
        + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        + f" | cost={s.get('estimated_cost', 'N/A')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
