"""
一键跑完 LoCoMo 评估：先尝试解除 Qdrant 锁，再执行 add_then_search + score。
用法（在 mem0_1 目录下）:
  python -m evaluation.run_full_eval
  python -m evaluation.run_full_eval --max_questions 50
  python -m evaluation.run_full_eval --no-clear-lock
"""
import argparse
import os
import subprocess
import sys

_evals_dir = os.path.dirname(os.path.abspath(__file__))
if _evals_dir not in sys.path:
    sys.path.insert(0, _evals_dir)


def main():
    parser = argparse.ArgumentParser(description="一键运行 add + search + score（可选先删 Qdrant 锁）")
    parser.add_argument("--max_questions", type=int, default=50, help="最多评估的问题数，默认 50")
    parser.add_argument("--no-clear-lock", action="store_true", help="不主动删除 .lock，完全依赖 create_local_memory_with_retry")
    parser.add_argument("--output_folder", type=str, default=None, help="结果输出目录，默认 mem0_1/results")
    args = parser.parse_args()

    # 使用本脚本所在 evaluation 的上级目录为项目根，确保子进程加载同一 evaluation
    project_root = os.path.dirname(_evals_dir)
    output_folder = args.output_folder or os.path.join(project_root, "results")
    output_folder = os.path.abspath(output_folder)

    if not args.no_clear_lock:
        from src.memzero.local_memory import _qdrant_lock_path
        lock_path = _qdrant_lock_path()
        if os.path.isfile(lock_path):
            try:
                os.remove(lock_path)
                print(f"[run_full_eval] 已删除锁文件: {lock_path}")
            except OSError as e:
                print(f"[run_full_eval] 无法删除锁文件（将依赖内部重试）: {e}")

    cmd = [
        sys.executable,
        "-m", "evaluation.run_experiments",
        "--technique_type", "mem0",
        "--method", "add_then_search",
        "--score",
        "--max_questions", str(args.max_questions),
        "--output_folder", output_folder,
    ]
    print("[run_full_eval] 执行:", " ".join(cmd))
    return subprocess.run(cmd, cwd=project_root).returncode


if __name__ == "__main__":
    sys.exit(main())
