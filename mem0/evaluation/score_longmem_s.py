"""
LongMemEval-S scoring: accuracy from mem0 search results.
Outputs: running_time, log, context_window_peak, cost, acc.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

_evals_dir = os.path.dirname(os.path.abspath(__file__))
if _evals_dir not in sys.path:
    sys.path.insert(0, _evals_dir)


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
    parser = argparse.ArgumentParser(description="Score LongMemEval-S results (acc)")
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--stats_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_id", type=str, default="longmem_s")
    args = parser.parse_args()

    data = load_results(args.results_file)
    items = flatten(data)
    correct = 0
    for item in items:
        pred = str(item.get("response") or "").strip().lower()
        ref = str(item.get("answer") or "").strip().lower()
        item["correct"] = 1 if pred == ref else 0
        correct += item["correct"]
    n = len(items)
    acc = correct / n if n else 0.0

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results_file))
    os.makedirs(out_dir, exist_ok=True)

    stats = None
    if args.stats_file and os.path.isfile(args.stats_file):
        with open(args.stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

    summary = {
        "n_questions": n,
        "acc": acc,
        "overall_accuracy": acc,
    }
    if stats:
        summary["token_and_cost"] = stats
        summary["context_window_peak"] = stats.get("context_window_peak")
        summary["running_time"] = stats.get("running_time_sec")
        summary["cost"] = stats.get("estimated_cost")
        if stats.get("add_time_sec") is not None:
            summary["add_time_sec"] = stats["add_time_sec"]
        if stats.get("search_time_sec") is not None:
            summary["search_time_sec"] = stats["search_time_sec"]
    stats_path = os.path.join(out_dir, f"batch_statistics_{args.batch_id}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[LongMemEval-S] acc={acc:.4f} n={n}")
    rt = stats.get("running_time_sec") if stats else None
    peak = stats.get("context_window_peak") if stats else None
    cost = stats.get("estimated_cost") if stats else None
    rt_str = f"{rt:.4f}" if isinstance(rt, (int, float)) else "N/A"
    peak_str = str(peak) if peak is not None else "N/A"
    cost_str = f"{cost:.6f}" if isinstance(cost, (int, float)) else (str(cost) if cost is not None else "N/A")
    if stats and stats.get("add_time_sec") is not None and stats.get("search_time_sec") is not None:
        add_s = stats["add_time_sec"]
        search_s = stats["search_time_sec"]
        print(f"[Eval summary] running_time (total)={rt_str} | add_time_sec={add_s} | search_time_sec={search_s} | context_window_peak={peak_str} | acc={acc:.4f} | cost={cost_str}")
    else:
        print(f"[Eval summary] running_time={rt_str} | context_window_peak={peak_str} | acc={acc:.4f} | cost={cost_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
