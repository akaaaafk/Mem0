"""
一键跑 LoCoMo 评估（兼容旧用法，实际调用 run_locomo）。
用法（在 mem0_1 目录下）:
  python -m eval_qwen.run_full_eval
  python -m eval_qwen.run_full_eval --max_questions 50
"""
import subprocess
import sys

_evals_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_evals_dir)


def main():
    return subprocess.run(
        [sys.executable, "-m", "eval_qwen.run_locomo"] + sys.argv[1:],
        cwd=_project_root,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
