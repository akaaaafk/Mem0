# Usage (run from mem0/):
#   python -m eval_qwen.run_all_evals                              # LoCoMo only (default)
#   python -m eval_qwen.run_all_evals --max_questions 50
#   python -m eval_qwen.run_all_evals --no-locomo --longmem --hotpotqa --ruler
import argparse
import os
import subprocess
import sys

_evals_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_evals_dir)


def main():
    parser = argparse.ArgumentParser(description="eval_qwen: run all benchmarks")
    parser.add_argument("--no-locomo", action="store_true", help="skip LoCoMo (default: run)")
    parser.add_argument("--longmem", action="store_true", help="run LongMemEval-S (stub)")
    parser.add_argument("--hotpotqa", action="store_true", help="run HotpotQA (stub)")
    parser.add_argument("--ruler", action="store_true", help="run RULER 128K (stub)")
    parser.add_argument("--max_questions", type=int, default=50, help="LoCoMo max questions")
    args = parser.parse_args()

    ran = []

    if not args.no_locomo:
        cmd = [sys.executable, "-m", "eval_qwen.run_locomo", "--max_questions", str(args.max_questions)]
        print("[run_all_evals] LoCoMo:", " ".join(cmd))
        ret = subprocess.run(cmd, cwd=_project_root, env=os.environ.copy()).returncode
        if ret != 0:
            return ret
        ran.append("locomo")

    if args.longmem:
        cmd = [sys.executable, "-m", "eval_qwen.run_longmem_s"]
        print("[run_all_evals] LongMemEval-S:", " ".join(cmd))
        ret = subprocess.run(cmd, cwd=_project_root, env=os.environ.copy()).returncode
        if ret != 0:
            return ret
        ran.append("longmem_s")

    if args.hotpotqa:
        cmd = [sys.executable, "-m", "eval_qwen.run_hotpotqa"]
        print("[run_all_evals] HotpotQA:", " ".join(cmd))
        ret = subprocess.run(cmd, cwd=_project_root, env=os.environ.copy()).returncode
        if ret != 0:
            return ret
        ran.append("hotpotqa")

    if args.ruler:
        cmd = [sys.executable, "-m", "eval_qwen.run_ruler_128k"]
        print("[run_all_evals] RULER 128K:", " ".join(cmd))
        ret = subprocess.run(cmd, cwd=_project_root, env=os.environ.copy()).returncode
        if ret != 0:
            return ret
        ran.append("ruler_128k")

    if ran:
        print("[run_all_evals] done:", ", ".join(ran))
    return 0


if __name__ == "__main__":
    sys.exit(main())
