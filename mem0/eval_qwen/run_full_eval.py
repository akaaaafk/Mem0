# Usage (run from mem0/):
#   python -m eval_qwen.run_full_eval
#   python -m eval_qwen.run_full_eval --max_questions 50
# Alias for run_locomo — kept for backward compatibility.
import os
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
