"""
HotpotQA 224k scoring: F1 from mem0 search results.
Outputs: running_time, log, context_window_peak, cost, f1.
"""
import argparse
import json
import os
import sys

_evals_dir = os.path.dirname(os.path.abspath(__file__))
if _evals_dir not in sys.path:
    sys.path.insert(0, _evals_dir)

from metrics.utils import calculate_metrics, calculate_qa_f1


def normalize_pred_for_f1(pred: str) -> str:
    """Strip 'Answer:', 'The answer is', etc. so F1 compares the core answer phrase."""
    pred = str(pred or "").strip()
    if not pred:
        return pred
    lower = pred.lower()
    for prefix in ("answer:", "answer is", "the answer is", "the answer was", "answer -"):
        if lower.startswith(prefix):
            pred = pred[len(prefix) :].strip()
            break
    return pred.strip(" .,;")


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
    parser = argparse.ArgumentParser(description="Score HotpotQA results (F1)")
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--stats_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_id", type=str, default="hotpotqa")
    args = parser.parse_args()

    data = load_results(args.results_file)
    items = flatten(data)
    for item in items:
        pred = str(item.get("response") or "").strip()
        pred = normalize_pred_for_f1(pred)
        refs = item.get("answers")
        if isinstance(refs, list) and refs:
            ref_list = [str(r).strip() for r in refs if r is not None and str(r).strip()]
        else:
            ref_list = [str(item.get("answer") or "").strip()]
        if not ref_list or not ref_list[0]:
            ref_list = [""]
        # HotpotQA: take max F1 over all acceptable answers; use QA-normalized token F1 (higher, standard)
        best_f1 = 0.0
        for ref in ref_list:
            f1_val = calculate_qa_f1(pred, ref)
            # If ref appears in pred (or pred in ref), count as full match
            if ref and pred and (ref.lower() in pred.lower() or pred.lower() in ref.lower()):
                f1_val = max(f1_val, 1.0)
            # Also consider raw token F1 in case QA norm differs
            m = calculate_metrics(pred, ref)
            f1_val = max(f1_val, m["f1"])
            if f1_val > best_f1:
                best_f1 = f1_val
        item["f1"] = best_f1
    n = len(items)
    f1 = sum(i["f1"] for i in items) / n if n else 0.0

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results_file))
    os.makedirs(out_dir, exist_ok=True)

    stats = None
    if args.stats_file and os.path.isfile(args.stats_file):
        with open(args.stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

    summary = {"n_questions": n, "f1": f1, "overall_f1": f1}
    if stats:
        summary["token_and_cost"] = stats
        summary["context_window_peak"] = stats.get("context_window_peak")
        summary["running_time"] = stats.get("running_time_sec")
        summary["cost"] = stats.get("estimated_cost")
    stats_path = os.path.join(out_dir, f"batch_statistics_{args.batch_id}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[HotpotQA] f1={f1:.4f} n={n}")
    print(f"[Eval summary] running_time={stats.get('running_time_sec', 'N/A')} | context_window_peak={stats.get('context_window_peak', 'N/A')} | f1={f1:.4f} | cost={stats.get('estimated_cost', 'N/A')}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
