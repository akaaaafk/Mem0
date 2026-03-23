"""
HotpotQA 224k evaluation with mem0. Backbone: tinker Qwen/Qwen3-30B-A3B-Instruct-2507, no graph.
Outputs: running_time, log, context_window_peak, cost, f1.
Usage (from mem0 repo root):
  python -m evaluation.run_hotpotqa
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

DEFAULT_HOTPOTQA_DATA = os.path.join(_evals_dir, "data","hotpotqa", "eval_1600.json")


def main():
    parser = argparse.ArgumentParser(description="mem0 HotpotQA 224k evaluation (F1)")
    parser.add_argument("--max_questions", type=int, default=None, help="Max questions")
    parser.add_argument("--output_folder", type=str, default=None, help="Output dir, default evaluation/results/hotpotqa")
    parser.add_argument("--data_path", type=str, default=None, help="Override dataset path")
    parser.add_argument("--is_graph", action="store_true", help="Use graph-based memory")
    parser.add_argument("--max_prompt_chars", type=int, default=None, help="Max chars for retrieved context. Default 200000 (256k). Env RULER_MAX_PROMPT_CHARS overrides.")
    parser.add_argument("--search_only", action="store_true", help="Re-run only search + score (add must already be done; uses existing .qdrant_run)")
    parser.add_argument("--direct", action="store_true", help="Direct QA: feed full context to model (no add/search). Use with 128k-capable model + large max_prompt_chars for best F1.")
    args = parser.parse_args()

    output_folder = args.output_folder or os.path.join(_evals_dir, "results", "hotpotqa")
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    os.environ["MEM0_DIR"] = os.path.join(output_folder, ".qdrant_run")
    os.makedirs(os.environ["MEM0_DIR"], exist_ok=True)
    if args.max_prompt_chars is not None:
        os.environ["RULER_MAX_PROMPT_CHARS"] = str(args.max_prompt_chars)
    elif "RULER_MAX_PROMPT_CHARS" not in os.environ:
        os.environ["RULER_MAX_PROMPT_CHARS"] = "200000"  # 256k context; use 80000 for 32k

    data_path = args.data_path or DEFAULT_HOTPOTQA_DATA
    if not os.path.isfile(data_path):
        print(f"[run_hotpotqa] Error: data not found at {data_path}")
        print("Add HotpotQA data (e.g. eval_1600.json) to evaluation/hotpotqa/ or set --data_path to your file.")
        return 1

    if args.search_only:
        qdrant_dir = os.path.join(output_folder, ".qdrant_run")
        if not os.path.isdir(qdrant_dir):
            print(f"[run_hotpotqa] Error: --search_only requires add already done. Not found: {qdrant_dir}")
            return 1
        method = "search"
        print("[run_hotpotqa] search_only: running search then score (skip add)")
    elif args.direct:
        from tqdm import tqdm
        from src.memzero.search import answer_with_context

        max_context_chars = args.max_prompt_chars
        if max_context_chars is None:
            max_context_chars = int(os.environ.get("RULER_MAX_PROMPT_CHARS", "200000"))
        print(f"[run_hotpotqa] direct: full context -> model (max_context_chars={max_context_chars})")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("[run_hotpotqa] Error: direct mode expects a list of items (context, input, answers)")
            return 1
        if args.max_questions is not None:
            data = data[: args.max_questions]
        run_stats = {"input_tokens": 0, "output_tokens": 0, "context_window_peak": 0}
        results = {}
        t0 = time.perf_counter()
        for idx, item in enumerate(tqdm(data, desc="Direct HotpotQA")):
            ctx = item.get("context") or ""
            q = item.get("input") or ""
            answers = item.get("answers")
            if isinstance(answers, list) and answers:
                ref = str(answers[0]).strip()
            else:
                ref = str(item.get("answer") or "").strip()
            response_text, _, _ = answer_with_context(ctx, q, run_stats=run_stats, max_context_chars=max_context_chars)
            results[str(idx)] = [
                {"question": q, "answer": ref, "answers": answers if isinstance(answers, list) else [ref], "response": response_text}
            ]
        running_time_sec = time.perf_counter() - t0
        run_stats["running_time_sec"] = round(running_time_sec, 4)
        run_stats["n_questions"] = len(data)
        unit_in = float(os.getenv("UNIT_PRICE_INPUT_PER_1M", "3.0"))
        unit_out = float(os.getenv("UNIT_PRICE_OUTPUT_PER_1M", "15.0"))
        run_stats["estimated_cost"] = (run_stats["input_tokens"] / 1e6) * unit_in + (run_stats["output_tokens"] / 1e6) * unit_out
        results_file = os.path.join(output_folder, "mem0_results.json")
        stats_path = os.path.join(output_folder, "batch_statistics_hotpotqa.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(run_stats, f, indent=2, ensure_ascii=False)
        ret = subprocess.run(
            [
                sys.executable,
                os.path.join(_evals_dir, "score_hotpotqa.py"),
                "--results_file", results_file,
                "--stats_file", stats_path,
                "--output_dir", output_folder,
                "--batch_id", "hotpotqa",
            ],
            cwd=_evals_dir,
        ).returncode
        print(f"[run_hotpotqa] running_time_sec={running_time_sec:.4f}")
        return ret
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
        "hotpotqa",
        "--data_path",
        data_path,
        "--top_k",
        "80",
    ]
    if args.max_questions is not None:
        cmd.extend(["--max_questions", str(args.max_questions)])
    if args.is_graph:
        cmd.append("--is_graph")
    print("[run_hotpotqa] Executing:", " ".join(cmd))
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, cwd=_project_root, env=os.environ.copy()).returncode
    running_time_sec = time.perf_counter() - t0

    stats_path = os.path.join(output_folder, "batch_statistics_hotpotqa.json")
    if ret == 0 and os.path.isfile(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            stats["running_time_sec"] = round(running_time_sec, 4)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # Score HotpotQA: F1
    results_file = os.path.join(output_folder, "mem0_results.json")
    if ret == 0 and os.path.isfile(results_file):
        subprocess.run(
            [
                sys.executable,
                os.path.join(_evals_dir, "score_hotpotqa.py"),
                "--results_file", results_file,
                "--stats_file", stats_path,
                "--output_dir", output_folder,
                "--batch_id", "hotpotqa",
            ],
            cwd=_evals_dir,
        )
    print(f"[run_hotpotqa] running_time_sec={running_time_sec:.4f}")
    return ret


if __name__ == "__main__":
    sys.exit(main())
