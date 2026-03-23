"""
RULER (128K) 数据集评估（follow GAM，Tinker + Qwen）。
输出目录：eval_qwen/results/ruler_128k/
接入数据与流程后在此实现；当前为占位。
用法（在 mem0_1 目录下）:
  python -m eval_qwen.run_ruler_128k
"""
import argparse
import os
import sys

_evals_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_evals_dir)


def _load_env():
    env_file = os.path.join(_evals_dir, ".env")
    if os.path.isfile(env_file):
        from dotenv import load_dotenv
        load_dotenv(env_file)
    output_folder = os.path.join(_evals_dir, "results", "ruler_128k")
    os.makedirs(output_folder, exist_ok=True)
    run_qdrant_base = os.path.join(output_folder, ".qdrant_run")
    os.makedirs(run_qdrant_base, exist_ok=True)
    os.environ["MEM0_DIR"] = run_qdrant_base


def main():
    parser = argparse.ArgumentParser(description="eval_qwen: RULER 128K 评估 (follow GAM)")
    parser.add_argument("--output_folder", type=str, default=None, help="结果目录，默认 eval_qwen/results/ruler_128k")
    args = parser.parse_args()

    _load_env()

    # TODO: 按 GAM 协议接入 RULER 128K 数据与 add_then_search + score 流程
    # 输出与 locomo 一致：time, log, f1, bleu, 花费 → results/ruler_128k/
    print("[run_ruler_128k] 未接入：请配置 RULER (128K) 数据与脚本（follow GAM）后在此实现")
    print("[Eval summary] context_window_peak=N/A | running_time=N/A | n_questions=0 | overall_f1=N/A | overall_bleu1=N/A | estimated_cost=N/A (RULER 128K not implemented)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
