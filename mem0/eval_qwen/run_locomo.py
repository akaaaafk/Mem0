"""
LoCoMo 数据集评估（Tinker + Qwen）。
输出：time, log, f1, bleu, accuracy, 花费, context_window_peak, running_time → eval_qwen/results/locomo/
用法（在 mem0_1 目录下）:
  python -m eval_qwen.run_locomo
  python -m eval_qwen.run_locomo --max_questions
  python -m eval_qwen.run_locomo --no-clear-lock
"""
import argparse
import json
import os
import subprocess
import sys
import time

_evals_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_evals_dir)
_evaluation_dir = os.path.join(_project_root, "evaluation")

def _load_env():
    env_file = os.path.join(_evals_dir, ".env")
    if os.path.isfile(env_file):
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"[eval_qwen] 已加载: {env_file}")
    else:
        print("[eval_qwen] 未找到 .env，请复制 .env.example 为 .env 并填写 TINKER_API_KEY")


def main():
    parser = argparse.ArgumentParser(description="eval_qwen: LoCoMo 评估（Tinker + Qwen）")
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="最多评估的问题数；不传则使用 LoCoMo 全部（10 个长对话的所有问题）",
    )
    parser.add_argument("--no-clear-lock", action="store_true", help="不主动删除 .lock")
    parser.add_argument("--output_folder", type=str, default=None, help="结果目录，默认 eval_qwen/results/locomo")
    args = parser.parse_args()

    _load_env()

    # 若 .env 未设置 Tinker，则用上面写的默认值
    if not os.getenv("TINKER_BASE_URL"):
        os.environ["TINKER_BASE_URL"] = TINKER_BASE_URL
    if not os.getenv("TINKER_API_KEY"):
        os.environ["TINKER_API_KEY"] = TINKER_API_KEY
    if not os.getenv("MODEL"):
        os.environ["MODEL"] = MODEL

    output_folder = args.output_folder or os.path.join(_evals_dir, "results", "locomo")
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    run_qdrant_base = os.path.join(output_folder, ".qdrant_run")
    os.makedirs(run_qdrant_base, exist_ok=True)
    os.environ["MEM0_DIR"] = run_qdrant_base

    if not args.no_clear_lock:
        if _evaluation_dir not in sys.path:
            sys.path.insert(0, _evaluation_dir)
        from src.memzero.local_memory import _qdrant_lock_path
        lock_path = _qdrant_lock_path()
        if os.path.isfile(lock_path):
            try:
                os.remove(lock_path)
                print(f"[run_locomo] 已删除锁文件: {lock_path}")
            except OSError as e:
                print(f"[run_locomo] 无法删除锁文件: {e}")

    cmd = [
        sys.executable,
        "-m",
        "evaluation.run_experiments",
        "--technique_type",
        "mem0",
        "--method",
        "add_then_search",
        "--score",
        "--output_folder",
        output_folder,
        "--batch_id",
        "locomo",
    ]

    # 不传 --max_questions 表示“用 LoCoMo 的十个长对话的全部问题”
    if args.max_questions is not None:
        cmd.extend(["--max_questions", str(args.max_questions)])
    print("[run_locomo] 执行:", " ".join(cmd))
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, cwd=_project_root, env=os.environ.copy()).returncode
    running_time_sec = time.perf_counter() - t0
    if ret == 0:
        stats_path = os.path.join(output_folder, "batch_statistics_locomo.json")
        if os.path.isfile(stats_path):
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                peak = stats.get("context_window_peak") or (stats.get("token_and_cost") or {}).get("context_window_peak")
                if peak is not None:
                    print(f"[eval_qwen] context_window_peak={peak:,}")
                else:
                    print("[eval_qwen] context_window_peak=N/A")
                stats["running_time_sec"] = round(running_time_sec, 4)
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
            except Exception:
                print("[eval_qwen] context_window_peak=N/A (could not read stats)")
        print(f"[eval_qwen] running_time_sec={running_time_sec:.4f}")
    else:
        print(f"[eval_qwen] running_time_sec={running_time_sec:.4f} (run failed)")
    return ret


if __name__ == "__main__":
    sys.exit(main())
