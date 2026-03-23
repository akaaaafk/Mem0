import argparse
import json
import os
import sys
import time

# Allow "src" to resolve when run as: python -m evaluation.run_experiments (from mem0_1/) or python run_experiments.py (from evaluation/)
_evals_dir = os.path.dirname(os.path.abspath(__file__))
if _evals_dir not in sys.path:
    sys.path.insert(0, _evals_dir)

# Load .env from evaluation dir; fallback to mem0copy/evaluation/.env if present (for shared API config)
from dotenv import load_dotenv
load_dotenv(os.path.join(_evals_dir, ".env"))
_mem0copy_env = os.path.join(os.path.dirname(os.path.dirname(_evals_dir)), "mem0copy", "evaluation", ".env")
if os.path.isfile(_mem0copy_env):
    load_dotenv(_mem0copy_env)

# Dataset paths relative to evaluation/ so they work when cwd is mem0_1/ or evaluation/
DEFAULT_LOCOMO_DATA = os.path.join(_evals_dir, "locomo", "locomo10.json")
DEFAULT_LOCOMO_RAG_DATA = os.path.join(_evals_dir, "locomo", "locomo10_rag.json")

from src.utils import METHODS, TECHNIQUES


class Experiment:
    def __init__(self, technique_type, chunk_size):
        self.technique_type = technique_type
        self.chunk_size = chunk_size

    def run(self):
        print(f"Running experiment with technique: {self.technique_type}, chunk size: {self.chunk_size}")


def main():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument("--technique_type", choices=TECHNIQUES, default="mem0", help="Memory technique to use")
    parser.add_argument("--method", choices=METHODS, default="add", help="Method to use")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")
    parser.add_argument("--output_folder", type=str, default="results/", help="Output path for results")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top memories to retrieve")
    parser.add_argument("--filter_memories", action="store_true", default=False, help="Whether to filter memories")
    parser.add_argument("--is_graph", action="store_true", default=False, help="Whether to use graph-based search")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks to process")
    parser.add_argument("--max_samples", type=int, default=None, help="Max conversation samples (default: all). Overridden by --max_questions when set.")
    parser.add_argument("--max_questions", type=int, default=None, help="Limit by question count: add runs enough conversations to cover this many questions; search stops after this many questions. e.g. 50.")
    parser.add_argument("--score", action="store_true", help="After search, run score (F1/BLEU1) and print token/cost/summary like reference.")
    parser.add_argument("--batch_id", type=str, default="0", help="Batch id for batch_statistics_{batch_id}.json and experiment log.")

    args = parser.parse_args()

    # Ensure output folder exists (default results/ is relative to cwd)
    os.makedirs(args.output_folder, exist_ok=True)

    print(f"Running experiments with technique: {args.technique_type}, chunk size: {args.chunk_size}")

    if args.technique_type == "mem0":
        from src.memzero.add import MemoryADD
        from src.memzero.search import MemorySearch
        if args.method == "add":
            t0 = time.time()
            memory_manager = MemoryADD(data_path=DEFAULT_LOCOMO_DATA)
            memory_manager.process_all_conversations(max_samples=args.max_samples, max_questions=args.max_questions)
            add_time = time.time() - t0
            log_path = os.path.join(args.output_folder, "experiment_log.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"stage": "add", "time_sec": round(add_time, 4), "batch_id": args.batch_id}, ensure_ascii=False) + "\n")
            print(f"[add] time_sec={add_time:.4f} 实验记录已追加: {os.path.abspath(log_path)}")
        elif args.method == "add_then_search":
            # 同一进程内先 add 再 search，共用一个 Qdrant 连接；本次运行用独立目录，避免与其它进程抢锁
            total_run_t0 = time.time()
            shared_client = None
            if not os.getenv("MEM0_API_KEY"):
                run_qdrant_base = os.path.join(os.path.abspath(args.output_folder), ".qdrant_run")
                os.makedirs(args.output_folder, exist_ok=True)
                os.makedirs(run_qdrant_base, exist_ok=True)
                _prev_mem0 = os.environ.get("MEM0_DIR")
                os.environ["MEM0_DIR"] = run_qdrant_base
                try:
                    from src.memzero.local_memory import create_local_memory_with_retry
                    shared_client = create_local_memory_with_retry()
                finally:
                    if _prev_mem0 is not None:
                        os.environ["MEM0_DIR"] = _prev_mem0
                    else:
                        os.environ.pop("MEM0_DIR", None)
            t0 = time.time()
            memory_manager = MemoryADD(data_path=DEFAULT_LOCOMO_DATA, mem0_client=shared_client)
            memory_manager.process_all_conversations(max_samples=args.max_samples, max_questions=args.max_questions)
            add_time = time.time() - t0
            log_path = os.path.join(args.output_folder, "experiment_log.jsonl")
            os.makedirs(args.output_folder, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"stage": "add", "time_sec": round(add_time, 4), "batch_id": args.batch_id}, ensure_ascii=False) + "\n")
            print(f"[add] time_sec={add_time:.4f} 实验记录已追加: {os.path.abspath(log_path)}")
            output_file_path = os.path.join(args.output_folder, "mem0_results.json")
            run_stats = {}
            memory_searcher = MemorySearch(output_file_path, args.top_k, run_stats=run_stats, mem0_client=shared_client)
            memory_searcher.process_data_file(DEFAULT_LOCOMO_DATA, max_samples=args.max_samples, max_questions=args.max_questions)
            unit_in = float(os.getenv("UNIT_PRICE_INPUT_PER_1M", "3.0"))
            unit_out = float(os.getenv("UNIT_PRICE_OUTPUT_PER_1M", "15.0"))
            inp = run_stats.get("input_tokens", 0)
            out = run_stats.get("output_tokens", 0)
            run_stats["unit_price_input_per_1m"] = unit_in
            run_stats["unit_price_output_per_1m"] = unit_out
            run_stats["estimated_cost"] = (inp / 1e6) * unit_in + (out / 1e6) * unit_out
            run_stats["running_time_sec"] = round(time.time() - total_run_t0, 4)
            stats_path = os.path.join(args.output_folder, f"batch_statistics_{args.batch_id}.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(run_stats, f, indent=2, ensure_ascii=False)
            with open(log_path, "a", encoding="utf-8") as f:
                log_entry = {
                    "stage": "search",
                    "time_sec": run_stats.get("search_time_sec"),
                    "running_time_sec": run_stats.get("running_time_sec"),
                    "context_window_peak": run_stats.get("context_window_peak"),
                    "input_tokens": inp,
                    "output_tokens": out,
                    "n_questions": run_stats.get("n_questions"),
                    "batch_id": args.batch_id,
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            print(f"[search] time_sec={run_stats.get('search_time_sec', 0):.4f} running_time_sec={run_stats.get('running_time_sec')} context_window_peak={run_stats.get('context_window_peak')} 实验记录已追加: {os.path.abspath(log_path)}")
            if args.score:
                import subprocess
                subprocess.run([
                    sys.executable,
                    os.path.join(_evals_dir, "score_results.py"),
                    "--results_file", os.path.abspath(output_file_path),
                    "--stats_file", os.path.abspath(stats_path),
                    "--output_dir", os.path.abspath(args.output_folder),
                    "--run_name", "LoCoMo 10 样本",
                    "--batch_id", args.batch_id,
                ], cwd=_evals_dir)
        elif args.method == "search":
            t0_search = time.time()
            output_file_path = os.path.join(args.output_folder, "mem0_results.json")
            run_stats = {}
            memory_searcher = MemorySearch(output_file_path, args.top_k, run_stats=run_stats)
            memory_searcher.process_data_file(DEFAULT_LOCOMO_DATA, max_samples=args.max_samples, max_questions=args.max_questions)
            unit_in = float(os.getenv("UNIT_PRICE_INPUT_PER_1M", "3.0"))
            unit_out = float(os.getenv("UNIT_PRICE_OUTPUT_PER_1M", "15.0"))
            inp = run_stats.get("input_tokens", 0)
            out = run_stats.get("output_tokens", 0)
            run_stats["unit_price_input_per_1m"] = unit_in
            run_stats["unit_price_output_per_1m"] = unit_out
            run_stats["estimated_cost"] = (inp / 1e6) * unit_in + (out / 1e6) * unit_out
            run_stats["running_time_sec"] = round(time.time() - t0_search, 4)
            stats_path = os.path.join(args.output_folder, f"batch_statistics_{args.batch_id}.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(run_stats, f, indent=2, ensure_ascii=False)
            log_path = os.path.join(args.output_folder, "experiment_log.jsonl")
            log_entry = {
                "stage": "search",
                "time_sec": run_stats.get("search_time_sec"),
                "running_time_sec": run_stats.get("running_time_sec"),
                "context_window_peak": run_stats.get("context_window_peak"),
                "input_tokens": inp,
                "output_tokens": out,
                "n_questions": run_stats.get("n_questions"),
                "batch_id": args.batch_id,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            print(f"[search] time_sec={run_stats.get('search_time_sec', 0):.4f} running_time_sec={run_stats.get('running_time_sec')} context_window_peak={run_stats.get('context_window_peak')} 实验记录已追加: {os.path.abspath(log_path)}")
            if args.score:
                import subprocess
                subprocess.run([
                    sys.executable,
                    os.path.join(_evals_dir, "score_results.py"),
                    "--results_file", os.path.abspath(output_file_path),
                    "--stats_file", os.path.abspath(stats_path),
                    "--output_dir", os.path.abspath(args.output_folder),
                    "--run_name", "LoCoMo 10 样本",
                    "--batch_id", args.batch_id,
                ], cwd=_evals_dir)
    elif args.technique_type == "rag":
        from src.rag import RAGManager
        output_file_path = os.path.join(args.output_folder, f"rag_results_{args.chunk_size}_k{args.num_chunks}.json")
        rag_manager = RAGManager(data_path=DEFAULT_LOCOMO_RAG_DATA, chunk_size=args.chunk_size, k=args.num_chunks)
        rag_manager.process_all_conversations(output_file_path)
    elif args.technique_type == "langmem":
        from src.langmem import LangMemManager
        output_file_path = os.path.join(args.output_folder, "langmem_results.json")
        langmem_manager = LangMemManager(dataset_path=DEFAULT_LOCOMO_RAG_DATA)
        langmem_manager.process_all_conversations(output_file_path)
    elif args.technique_type == "zep":
        from src.zep.add import ZepAdd
        from src.zep.search import ZepSearch
        if args.method == "add":
            zep_manager = ZepAdd(data_path=DEFAULT_LOCOMO_DATA)
            zep_manager.process_all_conversations("1")
        elif args.method == "search":
            output_file_path = os.path.join(args.output_folder, "zep_search_results.json")
            zep_manager = ZepSearch()
            zep_manager.process_data_file(DEFAULT_LOCOMO_DATA, "1", output_file_path)
    elif args.technique_type == "openai":
        from src.openai.predict import OpenAIPredict
        output_file_path = os.path.join(args.output_folder, "openai_results.json")
        openai_manager = OpenAIPredict()
        openai_manager.process_data_file(DEFAULT_LOCOMO_DATA, output_file_path)
    else:
        raise ValueError(f"Invalid technique type: {args.technique_type}")


if __name__ == "__main__":
    main()
