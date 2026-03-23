"""
LongMemEval-S evaluation with mem0. Backbone: tinker Qwen/Qwen3-30B-A3B-Instruct-2507, no graph.
Outputs: running_time, log, context_window_peak, cost, acc.

Usage (from mem0 repo root):
  python -m evaluation.run_longmem_s [--max_questions N]

Two-step run (avoids Qdrant lock / IndexError when add and search share one process):
  python -m evaluation.run_longmem_s --max_questions 10 --add-only   # step 1: add only, process exits
  python -m evaluation.run_longmem_s --max_questions 10 --search-only   # step 2: search + score (new process)
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

DEFAULT_LONGMEM_S_DATA = os.path.join(_evals_dir, "data", "longmemeval_s", "longmemeval_s_cleaned.json")


def main():
    parser = argparse.ArgumentParser(description="mem0 LongMemEval-S evaluation (acc)")
    parser.add_argument("--max_questions", type=int, default=None, help="Max questions")
    parser.add_argument("--output_folder", type=str, default=None, help="Output dir, default evaluation/results/longmem_s")
    parser.add_argument("--data_path", type=str, default=None, help="Override dataset path")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--add-only", action="store_true", help="Only run add (step 1 of two-step run); process exits, lock released")
    group.add_argument("--search-only", action="store_true", help="Only run search + score (step 2); use after --add-only in same output_folder")
    args = parser.parse_args()

    output_folder = args.output_folder or os.path.join(_evals_dir, "results", "longmem_s")
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    os.environ["MEM0_DIR"] = os.path.join(output_folder, ".qdrant_run")
    os.makedirs(os.environ["MEM0_DIR"], exist_ok=True)

    data_path = args.data_path or DEFAULT_LONGMEM_S_DATA
    if not os.path.isfile(data_path):
        print(f"[run_longmem_s] Warning: data not found at {data_path}. Add LongMemEval-S data in evaluation/longmem_s/")

    if args.add_only:
        method = "add"
    elif args.search_only:
        method = "search"
    else:
        method = "add_then_search"

    cmd = [
        sys.executable,
        "-m",
        "evaluation.run_experiments",
        "--technique_type",
        "mem0",
        "--method",
        method,
        "--output_folder",
        output_folder,
        "--batch_id",
        "longmem_s",
        "--data_path",
        data_path,
    ]
    if args.max_questions is not None:
        cmd.extend(["--max_questions", str(args.max_questions)])
    print("[run_longmem_s] Executing:", " ".join(cmd))
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, cwd=_project_root, env=os.environ.copy()).returncode
    running_time_sec = time.perf_counter() - t0

    if args.add_only:
        add_stats_path = os.path.join(output_folder, "add_phase_stats.json")
        add_stats = {
            "stage": "add",
            "time_sec": round(running_time_sec, 4),
            "batch_id": "longmem_s",
        }
        if args.max_questions is not None:
            add_stats["max_questions"] = args.max_questions
        try:
            with open(add_stats_path, "w", encoding="utf-8") as f:
                json.dump(add_stats, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        print(f"[run_longmem_s] add-only done. add_time_sec={running_time_sec:.4f} (saved to {os.path.basename(add_stats_path)}). Run --search-only next (same --output_folder and --max_questions).")
        return ret

    stats_path = os.path.join(output_folder, "batch_statistics_longmem_s.json")
    if ret == 0 and os.path.isfile(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            search_time_sec = round(running_time_sec, 4)
            stats["search_time_sec"] = search_time_sec
            add_time_sec = None
            if method == "search":
                log_path = os.path.join(output_folder, "experiment_log.jsonl")
                if os.path.isfile(log_path):
                    try:
                        with open(log_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                ent = json.loads(line)
                                if ent.get("stage") == "add" and ent.get("batch_id") == "longmem_s":
                                    add_time_sec = ent.get("time_sec")
                    except Exception:
                        pass
                if add_time_sec is not None:
                    stats["add_time_sec"] = round(float(add_time_sec), 4)
                    stats["running_time_sec"] = round(stats["add_time_sec"] + search_time_sec, 4)
                else:
                    stats["running_time_sec"] = search_time_sec
            else:
                stats["running_time_sec"] = search_time_sec
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            acc = stats.get("overall_accuracy")
            rt = stats.get("running_time_sec", running_time_sec)
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else (str(acc) if acc is not None else "N/A")
            if stats.get("add_time_sec") is not None:
                print(f"[run_longmem_s] add_time_sec={stats['add_time_sec']} search_time_sec={search_time_sec} total_running_time_sec={rt} acc={acc_str}")
            else:
                print(f"[run_longmem_s] running_time_sec={rt:.4f} acc={acc_str}")
        except Exception:
            print(f"[run_longmem_s] running_time_sec={running_time_sec:.4f} (exit_code={ret})")
    else:
        print(f"[run_longmem_s] running_time_sec={running_time_sec:.4f} (exit_code={ret})")

    results_file = os.path.join(output_folder, "mem0_results.json")
    if ret == 0 and os.path.isfile(results_file):
        subprocess.run(
            [
                sys.executable,
                os.path.join(_evals_dir, "score_longmem_s.py"),
                "--results_file", results_file,
                "--stats_file", stats_path,
                "--output_dir", output_folder,
                "--batch_id", "longmem_s",
            ],
            cwd=_evals_dir,
        )
    return ret


if __name__ == "__main__":
    sys.exit(main())
