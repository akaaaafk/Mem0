"""
eval_qwen 全量评估入口：按数据集分别调用 run_locomo / run_longmem_s / run_hotpotqa / run_ruler_128k。
用法（在 mem0_1 目录下）:
  python -m eval_qwen.run_all_evals
  python -m eval_qwen.run_all_evals --no-locomo --longmem --hotpotqa --ruler
"""
import argparse
import os
import subprocess
import sys

_evals_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_evals_dir)


def main():
    parser = argparse.ArgumentParser(description="eval_qwen: 按数据集分别跑 LoCoMo / LongMemEval-S / HotpotQA / RULER")
    parser.add_argument("--no-locomo", action="store_true", help="不跑 LoCoMo（默认会跑 LoCoMo）")
    parser.add_argument("--longmem", action="store_true", help="跑 LongMemEval-S (follow LightMem)")
    parser.add_argument("--hotpotqa", action="store_true", help="跑 HotpotQA (follow GAM)")
    parser.add_argument("--ruler", action="store_true", help="跑 RULER 128K (follow GAM)")
    parser.add_argument("--max_questions", type=int, default=50, help="LoCoMo 最多问题数")
    args = parser.parse_args()

    run_locomo = not args.no_locomo
    ran = []
    if run_locomo:
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
        print("[run_all_evals] 已跑完:", ", ".join(ran))
    return 0


if __name__ == "__main__":
    sys.exit(main())
