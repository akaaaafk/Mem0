"""
LoCoMo evaluation with AWS Bedrock (Claude Haiku) as the answer model.
Uses existing mem0 evaluation: add phase unchanged (OpenRouter/local); search phase uses Bedrock
inference profile (model anthropic.claude-haiku-4-5-20251001-v1:0).

Usage (from mem0 repo root):
  python -m evaluation.haiku.run_locomo_haiku
  python -m evaluation.haiku.run_locomo_haiku --max_questions 50
  python -m evaluation.haiku.run_locomo_haiku --search_only   # add already done, re-run search + score

Environment:
  AWS credentials (profile or env) for Bedrock.
  Optional: AWS_REGION, BEDROCK_ACCOUNT_ID, BEDROCK_INFERENCE_PROFILE_ID.
  For add phase: OPENROUTER_* or MEM0_DIR; if MEM0_API_KEY unset, uses local mem0 (evaluation).
"""
import argparse
import json
import os
import subprocess
import sys
import time

_haiku_dir = os.path.dirname(os.path.abspath(__file__))
_evals_dir = os.path.dirname(_haiku_dir)
_mem0_root = os.path.dirname(_evals_dir)

if _mem0_root not in sys.path:
    sys.path.insert(0, _mem0_root)
if _evals_dir not in sys.path:
    sys.path.insert(0, _evals_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(_evals_dir, ".env"))

from evaluation.haiku.bedrock_client import (
    build_inference_profile_arn,
    DEFAULT_REGION,
    DEFAULT_ACCOUNT_ID,
    DEFAULT_INFERENCE_PROFILE_ID,
)
from evaluation.haiku.search_bedrock import run_search_loop


DEFAULT_LOCOMO_DATA = os.path.join(_evals_dir, "data", "locomo", "locomo10.json")
if not os.path.isfile(DEFAULT_LOCOMO_DATA):
    DEFAULT_LOCOMO_DATA = os.path.join(_evals_dir, "locomo", "locomo10.json")


def main():
    parser = argparse.ArgumentParser(description="LoCoMo with AWS Bedrock (Claude Haiku) for answer")
    parser.add_argument("--max_questions", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--output_folder", type=str, default=None, help="Output dir, default evaluation/haiku/results/locomo_haiku")
    parser.add_argument("--data_path", type=str, default=None, help="Override LoCoMo dataset path")
    parser.add_argument("--search_only", action="store_true", help="Skip add; run only search + score (add must be done)")
    parser.add_argument("--full_haiku", action="store_true", help="Use Bedrock Haiku for BOTH add (memory extraction) and search (answer). Default: add uses Qwen/OpenRouter.")
    # Match evaluation/run_experiments.py default (--top_k 30) for fair comparison to Qwen Mem0
    _default_top_k = int(os.environ.get("LOCOMO_TOP_K", "30"))
    parser.add_argument(
        "--top_k",
        type=int,
        default=_default_top_k,
        help="Top-k memories to retrieve (default 30, same as run_experiments; env LOCOMO_TOP_K)",
    )
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION", DEFAULT_REGION))
    parser.add_argument("--account_id", type=str, default=os.environ.get("BEDROCK_ACCOUNT_ID", DEFAULT_ACCOUNT_ID))
    parser.add_argument("--inference_profile_id", type=str, default=os.environ.get("BEDROCK_INFERENCE_PROFILE_ID", DEFAULT_INFERENCE_PROFILE_ID))
    args = parser.parse_args()

    output_folder = args.output_folder or os.path.join(_haiku_dir, "results", "locomo_haiku")
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    qdrant_base = os.path.join(output_folder, ".qdrant_run")
    os.environ["MEM0_DIR"] = qdrant_base
    os.makedirs(qdrant_base, exist_ok=True)

    data_path = args.data_path or DEFAULT_LOCOMO_DATA
    data_path = os.path.abspath(os.path.normpath(data_path))
    if not os.path.isfile(data_path):
        print(f"[run_locomo_haiku] Warning: data not found at {data_path}")
        return 1

    inference_profile_arn = build_inference_profile_arn(args.region, args.account_id, args.inference_profile_id)
    batch_id = "locomo"
    results_file = os.path.join(output_folder, "mem0_results.json")
    stats_path = os.path.join(output_folder, f"batch_statistics_{batch_id}.json")

    if not args.search_only:
        if args.full_haiku:
            # Add phase in-process: Bedrock Haiku extracts memories
            print("[run_locomo_haiku] Add phase: Bedrock Haiku extracts memories (full_haiku mode)")
            from evaluation.haiku.local_memory_haiku import create_local_memory_haiku_with_retry
            from src.memzero.add import MemoryADD
            t0 = time.perf_counter()
            add_client = create_local_memory_haiku_with_retry()
            memory_manager = MemoryADD(data_path=data_path, mem0_client=add_client)
            memory_manager.process_all_conversations(max_questions=args.max_questions)
            add_time = time.perf_counter() - t0
            print(f"[run_locomo_haiku] Add (Haiku) done in {add_time:.2f}s")
            # Release Qdrant lock before search phase opens the same path
            try:
                add_client.vector_store.client.close()
            except Exception:
                pass
            del memory_manager, add_client
        else:
            # Add phase: run mem0 evaluation add (Qwen / OpenRouter)
            cmd = [
                sys.executable,
                "-m",
                "evaluation.run_experiments",
                "--technique_type", "mem0",
                "--method", "add",
                "--output_folder", output_folder,
                "--batch_id", batch_id,
                "--data_path", data_path,
            ]
            if args.max_questions is not None:
                cmd.extend(["--max_questions", str(args.max_questions)])
            print("[run_locomo_haiku] Add phase (Qwen/OpenRouter):", " ".join(cmd))
            t0 = time.perf_counter()
            ret = subprocess.run(cmd, cwd=_mem0_root, env=os.environ.copy())
            add_time = time.perf_counter() - t0
            if ret.returncode != 0:
                print(f"[run_locomo_haiku] Add failed with exit code {ret.returncode}")
                return ret.returncode
            print(f"[run_locomo_haiku] Add done in {add_time:.2f}s")

    # Search phase: Bedrock
    if args.full_haiku:
        from evaluation.haiku.local_memory_haiku import create_local_memory_haiku_with_retry
        mem0_client = create_local_memory_haiku_with_retry()
    else:
        from src.memzero.local_memory import create_local_memory_with_retry
        mem0_client = create_local_memory_with_retry()
    run_stats = {}
    print("[run_locomo_haiku] Search phase (Bedrock Claude Haiku)...")
    t0 = time.perf_counter()
    run_search_loop(
        mem0_client=mem0_client,
        data_path=data_path,
        output_path=results_file,
        run_stats=run_stats,
        region=args.region,
        inference_profile_arn=inference_profile_arn,
        top_k=args.top_k,
        max_questions=args.max_questions,
    )
    running_time_sec = time.perf_counter() - t0
    run_stats["running_time_sec"] = round(running_time_sec, 4)
    unit_in = float(os.environ.get("UNIT_PRICE_INPUT_PER_1M", "0.25"))
    unit_out = float(os.environ.get("UNIT_PRICE_OUTPUT_PER_1M", "1.25"))
    run_stats["unit_price_input_per_1m"] = unit_in
    run_stats["unit_price_output_per_1m"] = unit_out
    run_stats["estimated_cost"] = (run_stats.get("input_tokens", 0) / 1e6) * unit_in + (run_stats.get("output_tokens", 0) / 1e6) * unit_out
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(run_stats, f, indent=2, ensure_ascii=False)

    log_path = os.path.join(output_folder, "experiment_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "stage": "search",
            "time_sec": run_stats.get("search_time_sec"),
            "context_window_peak": run_stats.get("context_window_peak"),
            "input_tokens": run_stats.get("input_tokens"),
            "output_tokens": run_stats.get("output_tokens"),
            "n_questions": run_stats.get("n_questions"),
            "batch_id": batch_id,
            "backend": "bedrock_haiku",
        }, ensure_ascii=False) + "\n")

    n_questions = run_stats.get("n_questions", 0)
    if n_questions == 0:
        print("[run_locomo_haiku] No questions processed; skipping score.")
        print(f"[run_locomo_haiku] running_time_sec={running_time_sec:.4f} context_window_peak={run_stats.get('context_window_peak')}")
        return 0

    # Score (use absolute paths so cwd=_evals_dir does not affect resolution)
    ret = subprocess.run(
        [
            sys.executable,
            os.path.join(_evals_dir, "score_results.py"),
            "--results_file", os.path.abspath(results_file),
            "--stats_file", os.path.abspath(stats_path),
            "--output_dir", os.path.abspath(output_folder),
            "--run_name", "LoCoMo-Haiku",
            "--batch_id", batch_id,
        ],
        cwd=_evals_dir,
    )
    print(f"[run_locomo_haiku] running_time_sec={running_time_sec:.4f} context_window_peak={run_stats.get('context_window_peak')}")
    return ret.returncode


if __name__ == "__main__":
    sys.exit(main())
