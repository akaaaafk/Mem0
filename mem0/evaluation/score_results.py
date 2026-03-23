"""
LoCoMo 评估：从 mem0 search 结果算 F1/BLEU1，按类别汇总，控制台输出与保存路径与参考格式一致。
支持传入 token/费用统计，并支持 context window peak、各阶段 time、experiment log。
"""
import argparse
import json
import os
import sys
from collections import defaultdict

# 允许从 evaluation/ 或项目根运行
_evals_dir = os.path.dirname(os.path.abspath(__file__))
if _evals_dir not in sys.path:
    sys.path.insert(0, _evals_dir)

from metrics.utils import calculate_bleu_scores, calculate_metrics


def load_results(path):
    """加载 mem0 结果：{ "0": [ {question, answer, response, category, ...} ], ... }"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_and_filter(data):
    """展平为列表，跳过 category 5。"""
    items = []
    for conv_id, qa_list in data.items():
        for item in qa_list:
            cat = item.get("category")
            if cat is not None and int(cat) == 5:
                continue
            items.append({**item, "conv_id": conv_id})
    return items


def compute_scores(items):
    """对每条算 F1、BLEU1。"""
    for item in items:
        pred = str(item.get("response") or "").strip()
        ref = str(item.get("answer") or "").strip()
        m = calculate_metrics(pred, ref)
        b = calculate_bleu_scores(pred, ref)
        item["f1"] = m["f1"]
        item["bleu1"] = b.get("bleu1", 0.0)
    return items


def aggregate_by_category(items):
    """按 category 汇总 n, F1 avg, BLEU1 avg。"""
    by_cat = defaultdict(lambda: {"f1": [], "bleu1": []})
    for item in items:
        c = item.get("category", 0)
        by_cat[c]["f1"].append(item["f1"])
        by_cat[c]["bleu1"].append(item["bleu1"])
    return by_cat


def print_token_and_cost(stats):
    """打印 Token 与费用估算（与参考图一致）。"""
    if not stats:
        return
    inp = stats.get("input_tokens", 0)
    out = stats.get("output_tokens", 0)
    unit_in = stats.get("unit_price_input_per_1m", 3.0)
    unit_out = stats.get("unit_price_output_per_1m", 15.0)
    cost = stats.get("estimated_cost")
    if cost is None and (inp or out):
        cost = (inp / 1e6) * unit_in + (out / 1e6) * unit_out
    peak = stats.get("context_window_peak")
    search_time = stats.get("search_time_sec")
    running_time = stats.get("running_time_sec")
    print("Token 与费用估算 (Token and cost estimation)")
    print("  input tokens:  ", f"{inp:,}")
    print("  output tokens: ", f"{out:,}")
    print("  单价 (Unit price): 输入 (Input) ${}/1M  输出 (Output) ${}/1M".format(unit_in, unit_out))
    if cost is not None:
        print("  估算费用 (Estimated cost): ${:.4f}".format(cost))
    if peak is not None:
        print("  context window peak (单次最大 prompt tokens):", f"{peak:,}")
    if search_time is not None:
        print("  search 阶段 time (sec):", f"{search_time:.4f}")
    if running_time is not None:
        print("  total running time (sec):", f"{running_time:.4f}")
    print()


def print_score_summary(by_cat, items, run_name="LoCoMo"):
    """打印分数汇总：按类别 + 整体（含 accuracy）。"""
    print(f"{run_name} - 分数汇总 (Score summary)")
    print("按类别 (By Category):")
    for c in sorted(by_cat.keys(), key=lambda x: (int(x) if str(x).isdigit() else 0)):
        v = by_cat[c]
        n = len(v["f1"])
        f1_avg = sum(v["f1"]) / n if n else 0.0
        b1_avg = sum(v["bleu1"]) / n if n else 0.0
        acc = sum(v["correct"]) / n if n else 0.0
        print(f"  Category {c}:  n={n}  F1 avg={f1_avg:.4f}  BLEU1 avg={b1_avg:.4f}  accuracy={acc:.4f}")
    n_total = len(items)
    f1_overall = sum(i["f1"] for i in items) / n_total if n_total else 0.0
    b1_overall = sum(i["bleu1"] for i in items) / n_total if n_total else 0.0
    acc_overall = sum(i["correct"] for i in items) / n_total if n_total else 0.0
    print("整体 (Overall):")
    print(f"  问题数 (Number of questions): {n_total}")
    print(f"  平均 F1 (Average F1): {f1_overall:.4f}")
    print(f"  平均 BLEU1 (Average BLEU1): {b1_overall:.4f}")
    print(f"  准确率 (Accuracy, exact match): {acc_overall:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Score mem0 results (F1, BLEU1) and print summary like reference.")
    parser.add_argument("--results_file", type=str, required=True, help="mem0 search 结果 JSON 路径")
    parser.add_argument("--stats_file", type=str, default=None, help="可选：token/费用/时间统计 JSON")
    parser.add_argument("--output_dir", type=str, default=None, help="保存 batch_statistics / batch_results 的目录，默认与 results_file 同目录")
    parser.add_argument("--run_name", type=str, default="LoCoMo", help="打印标题用名称")
    parser.add_argument("--batch_id", type=str, default="0", help="batch 编号，用于文件名 batch_statistics_{id}.json")
    args = parser.parse_args()

    data = load_results(args.results_file)
    items = flatten_and_filter(data)
    items = compute_scores(items)
    by_cat = aggregate_by_category(items)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results_file))
    os.makedirs(out_dir, exist_ok=True)

    stats = None
    if args.stats_file and os.path.isfile(args.stats_file):
        with open(args.stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

    # 控制台输出（与参考图一致）
    print_token_and_cost(stats)
    print_score_summary(by_cat, items, run_name=args.run_name)

    # 保存文件
    stats_path = os.path.join(out_dir, f"batch_statistics_{args.batch_id}.json")
    results_path = os.path.join(out_dir, f"batch_results_{args.batch_id}.json")
    n_total = len(items)
    correct_total = sum(i["correct"] for i in items)
    summary = {
        "n_questions": n_total,
        "overall_f1": sum(i["f1"] for i in items) / n_total if items else 0,
        "overall_bleu1": sum(i["bleu1"] for i in items) / n_total if items else 0,
        "overall_accuracy": correct_total / n_total if n_total else 0,
        "by_category": {
            str(c): {
                "n": len(v["f1"]),
                "f1_avg": sum(v["f1"]) / len(v["f1"]) if v["f1"] else 0,
                "bleu1_avg": sum(v["bleu1"]) / len(v["bleu1"]) if v["bleu1"] else 0,
                "accuracy": sum(v["correct"]) / len(v["correct"]) if v["correct"] else 0,
            }
            for c, v in sorted(by_cat.items(), key=lambda x: (int(x[0]) if str(x[0]).isdigit() else 0))
        },
    }
    if stats:
        summary["token_and_cost"] = stats
        if "context_window_peak" in stats:
            summary["context_window_peak"] = stats["context_window_peak"]
        if "running_time_sec" in stats:
            summary["running_time_sec"] = stats["running_time_sec"]
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # batch_results: 带 f1/bleu1 的逐条结果（可精简字段）
    out_items = [
        {"question": i.get("question"), "answer": i.get("answer"), "response": i.get("response"), "category": i.get("category"), "f1": i["f1"], "bleu1": i["bleu1"], "correct": i["correct"]}
        for i in items
    ]
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out_items, f, indent=2, ensure_ascii=False)

    print("统计已保存 (Statistics saved):", os.path.abspath(stats_path))
    print("结果已保存 (Results saved):", os.path.abspath(results_path))
    log_path = os.path.join(out_dir, "experiment_log.jsonl")
    print("实验记录 (Experiment log):", os.path.abspath(log_path))
    # 每个 eval 统一输出一行：context_window_peak
    peak = (stats or {}).get("context_window_peak")
    cost = (stats or {}).get("estimated_cost")
    running_time = (stats or {}).get("running_time_sec")
    n_q = len(items)
    f1 = sum(i["f1"] for i in items) / n_q if n_q else 0
    bleu1 = sum(i["bleu1"] for i in items) / n_q if n_q else 0
    acc = sum(i["correct"] for i in items) / n_q if n_q else 0
    peak_str = f"{peak:,}" if peak is not None else "N/A"
    cost_str = f"{cost:.4f}" if cost is not None else "N/A"
    rt_str = f"{running_time:.4f}" if running_time is not None else "N/A"
    print(f"[Eval summary] total_running_time_sec={rt_str} | context_window_peak={peak_str} | n_questions={n_q} | overall_f1={f1:.4f} | overall_bleu1={bleu1:.4f} | accuracy={acc:.4f} | estimated_cost={cost_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
