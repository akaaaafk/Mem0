#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跑 100 个问题：用 Mem0 + LoCoMo 跑满 100 个 QA，保存结果并打印指标。

用法（在 e:\\baseline\\mem0 下）:
  python eval/run_100_questions.py
  python eval/run_100_questions.py --data ..\\data\\locomo\\locomo10.json --ollama-model llama3.2
"""

import sys
import os
import json
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 复用 locomo_test 的逻辑
from eval.locomo_test import (
    load_locomo,
    process_sample,
    compute_metrics_by_category,
    setup_logging,
    BGE_DIMS,
)

from tqdm import tqdm


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mem0 LoCoMo：跑满 100 个问题")
    parser.add_argument("--data", type=str, default=None, help="LoCoMo JSON 路径")
    parser.add_argument("--outdir", type=str, default="./results/locomo_100q", help="输出目录")
    parser.add_argument("--max-questions", type=int, default=100, help="最多跑多少题（默认 100）")
    parser.add_argument("--ollama-model", type=str, default="llama3.2")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--embedder-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--search-limit", type=int, default=10)
    args = parser.parse_args()

    if args.data is None:
        args.data = os.path.join(_REPO_ROOT, "..", "data", "locomo", "locomo10.json")
    args.data = os.path.abspath(args.data)
    if not os.path.exists(args.data):
        print("Data not found:", args.data)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    log_dir = os.path.join(args.outdir, "logs")
    logger = setup_logging(log_dir, "run_100q")
    logger.info("Data: %s", args.data)
    logger.info("Outdir: %s", args.outdir)
    logger.info("Max questions: %d", args.max_questions)
    logger.info("Ollama: %s @ %s", args.ollama_model, args.ollama_base_url)
    logger.info("Embedder: %s (dims=%d)", args.embedder_model, BGE_DIMS)

    samples = load_locomo(args.data)
    logger.info("Loaded %d samples", len(samples))

    all_results = []
    samples_used = 0
    t_total_start = datetime.now()

    for sample_idx in tqdm(range(len(samples)), desc="Samples"):
        if len(all_results) >= args.max_questions:
            break
        sample = samples[sample_idx]
        try:
            results = process_sample(
                sample,
                sample_idx,
                args.outdir,
                args.ollama_model,
                args.ollama_base_url,
                args.embedder_model,
                logger,
                search_limit=args.search_limit,
            )
            all_results.extend(results)
            samples_used += 1
            logger.info("Sample %d done, total QA so far: %d", sample_idx, len(all_results))
        except Exception as e:
            logger.exception("Sample %d failed: %s", sample_idx, e)

    # 只保留前 max_questions 条
    if len(all_results) > args.max_questions:
        all_results = all_results[: args.max_questions]
        logger.info("Trimmed to exactly %d questions", len(all_results))

    total_elapsed = (datetime.now() - t_total_start).total_seconds()
    logger.info("Total time: %.2fs, questions: %d, samples used: %d", total_elapsed, len(all_results), samples_used)

    if not all_results:
        logger.warning("No results to save.")
        return 0

    # 保存结果（与 GAM 格式一致）
    results_file = os.path.join(args.outdir, "batch_results_100_questions.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", results_file)

    # 指标
    summary_metrics, details = compute_metrics_by_category(
        all_results, pred_key="summary_answer", pred_field="answer"
    )
    all_f1 = [r["F1"] for r in details]
    all_b1 = [r["BLEU1"] for r in details]
    overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    overall_bleu1 = sum(all_b1) / len(all_b1) if all_b1 else 0.0

    print("\n" + "=" * 60)
    print("Mem0 LoCoMo — 100 问题结果")
    print("=" * 60)
    print("总题数: %d  总耗时: %.2fs  使用样本数: %d" % (len(all_results), total_elapsed, samples_used))
    print("Overall F1:   %.4f" % overall_f1)
    print("Overall BLEU1: %.4f" % overall_bleu1)
    print("\n按类别:")
    for r in summary_metrics:
        print("  Category %s: n=%d  F1=%.4f  BLEU1=%.4f" % (r["category"], r["count"], r["F1_avg"], r["BLEU1_avg"]))
    print("=" * 60)
    print("结果文件: %s" % results_file)

    statistics = {
        "total_questions": len(all_results),
        "samples_used": samples_used,
        "total_time_sec": round(total_elapsed, 2),
        "overall_f1_avg": overall_f1,
        "overall_bleu1_avg": overall_bleu1,
        "by_category": summary_metrics,
        "details": details,
    }
    stats_file = os.path.join(args.outdir, "batch_statistics_100_questions.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    logger.info("Saved statistics: %s", stats_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
