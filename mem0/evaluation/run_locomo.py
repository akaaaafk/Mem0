"""
LoCoMo evaluation with mem0. Backbone: tinker Qwen/Qwen3-30B-A3B-Instruct-2507, no graph.
Outputs: running_time, log, context_window_peak, cost, f1, bleu-1.
Usage (from mem0 repo root):
  python -m evaluation.run_locomo
  python -m evaluation.run_locomo --max_questions 50
"""
import argparse
import json
import os
import subprocess
import sys
import time

_evals_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_evals_dir)
if _evals_dir not in sys.path:
    sys.path.insert(0, _evals_dir)

from dotenv import load_dotenv

load_dotenv(os.path.join(_evals_dir, ".env"))

DEFAULT_LOCOMO_DATA = os.path.join(_evals_dir, "locomo", "locomo10.json")


def main():
    parser = argparse.ArgumentParser(description="mem0 LoCoMo evaluation (F1, BLEU-1)")
    parser.add_argument("--max_questions", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--output_folder", type=str, default=None, help="Output dir, default evaluation/results/locomo")
    parser.add_argument("--data_path", type=str, default=None, help="Override dataset path")
    args = parser.parse_args()

    output_folder = args.output_folder or os.path.join(_evals_dir, "results", "locomo")
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    run_qdrant_base = os.path.join(output_folder, ".qdrant_run")
    os.makedirs(run_qdrant_base, exist_ok=True)
    os.environ["MEM0_DIR"] = run_qdrant_base

    data_path = args.data_path or DEFAULT_LOCOMO_DATA
    if not os.path.isfile(data_path):
        print(f"[run_locomo] Warning: data not found at {data_path}. Create evaluation/locomo/ and add locomo10.json")

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
        "--data_path",
        data_path,
    ]
    if args.max_questions is not None:
        cmd.extend(["--max_questions", str(args.max_questions)])
    print("[run_locomo] Executing:", " ".join(cmd))
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, cwd=_project_root, env=os.environ.copy()).returncode
    running_time_sec = time.perf_counter() - t0
    stats_path = os.path.join(output_folder, "batch_statistics_locomo.json")
    if ret == 0 and os.path.isfile(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            stats["running_time_sec"] = round(running_time_sec, 4)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            peak = stats.get("context_window_peak") or (stats.get("token_and_cost") or {}).get("context_window_peak")
            print(f"[run_locomo] running_time_sec={running_time_sec:.4f} context_window_peak={peak}")
        except Exception:
            print(f"[run_locomo] running_time_sec={running_time_sec:.4f}")
    else:
        print(f"[run_locomo] running_time_sec={running_time_sec:.4f} (exit_code={ret})")
    return ret


if __name__ == "__main__":
    sys.exit(main())
