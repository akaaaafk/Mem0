# Usage (run from mem0/):
#   python -m eval_qwen.run_locomo
#   python -m eval_qwen.run_locomo --max_questions 50
#   python -m eval_qwen.run_locomo --no-clear-lock
# Results: eval_qwen/results/locomo/batch_statistics_locomo.json
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
        print(f"[eval_qwen] loaded: {env_file}")
    else:
        print("[eval_qwen] .env not found — copy .env.example to .env and set TINKER_API_KEY")


def main():
    parser = argparse.ArgumentParser(description="eval_qwen: LoCoMo eval (Tinker + Qwen)")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="max questions to evaluate; omit to run all LoCoMo questions")
    parser.add_argument("--no-clear-lock", action="store_true", help="skip deleting .lock file before run")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="results dir (default: eval_qwen/results/locomo)")
    args = parser.parse_args()

    _load_env()

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
                print(f"[run_locomo] removed lock: {lock_path}")
            except OSError as e:
                print(f"[run_locomo] could not remove lock: {e}")

    cmd = [
        sys.executable, "-m", "evaluation.run_experiments",
        "--technique_type", "mem0",
        "--method", "add_then_search",
        "--score",
        "--output_folder", output_folder,
        "--batch_id", "locomo",
    ]

    if args.max_questions is not None:
        cmd.extend(["--max_questions", str(args.max_questions)])
    print("[run_locomo] running:", " ".join(cmd))
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
