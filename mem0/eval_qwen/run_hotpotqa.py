# Usage (run from mem0/):
#   python -m eval_qwen.run_hotpotqa
# Results: eval_qwen/results/hotpotqa/
# Status: stub — HotpotQA pipeline not yet implemented.
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
    output_folder = os.path.join(_evals_dir, "results", "hotpotqa")
    os.makedirs(output_folder, exist_ok=True)
    run_qdrant_base = os.path.join(output_folder, ".qdrant_run")
    os.makedirs(run_qdrant_base, exist_ok=True)
    os.environ["MEM0_DIR"] = run_qdrant_base


def main():
    parser = argparse.ArgumentParser(description="eval_qwen: HotpotQA (stub)")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="results dir (default: eval_qwen/results/hotpotqa)")
    args = parser.parse_args()

    _load_env()

    # TODO: implement HotpotQA add_then_search + score pipeline
    print("[run_hotpotqa] not implemented")
    print("[Eval summary] context_window_peak=N/A | running_time=N/A | n_questions=0 | overall_f1=N/A | overall_bleu1=N/A | estimated_cost=N/A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
