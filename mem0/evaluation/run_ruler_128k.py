"""
RULER 128k evaluation with mem0. Backbone: OpenRouter Qwen/Qwen3-30B-A3B-Instruct-2507, no graph.
Runs all .jsonl datasets under evaluation/data/ruler except qa_1 and any qa_* (e.g. qa_2).
Outputs per dataset: running_time, log, context_window_peak, cost, retri.acc, MT acc, ACG acc, QA acc.
Default: RULER_MAX_PROMPT_CHARS=200000 (256k context); use 80000 for 32k.

Resume: If a run stopped after add (e.g. niah_* only has experiment_log "stage":"add", no mem0_results.json),
  use --fill_missing: for subdirs with add done but no mem0_results, we run search only (no re-add), then score.

Reaching benchmark-level (e.g. Retri ~93%%, MT ~90%%): Use --top_k 80 or 100 so more NIAH needles
  are in the retrieved set; default top_k=80 with 256k context.

Usage (from mem0 repo root):
  python -m evaluation.run_ruler_128k
  python -m evaluation.run_ruler_128k --max_datasets 1   # run one dataset then stop to check score
  python -m evaluation.run_ruler_128k --skip_done       # skip datasets that already have results; aggregate all at end
  python -m evaluation.run_ruler_128k --fill_missing   # fill missing batch_statistics: resume search-only where add done, else score-only or full run; then aggregate
  python -m evaluation.run_ruler_128k --top_k 100      # retrieve 100 memories (default 80); improves NIAH/retri.acc
  python -m evaluation.run_ruler_128k --rerun_search --top_k 80   # add already done: re-run only search with top_k=80, overwrite results (for higher retri.acc)
  python -m evaluation.run_ruler_128k --prefix niah               # run only datasets whose name starts with niah (niah_single_1, niah_multikey_1, etc.)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

_evals_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_evals_dir)
if _evals_dir not in sys.path:
    sys.path.insert(0, _evals_dir)

from dotenv import load_dotenv

load_dotenv(os.path.join(_evals_dir, ".env"))

# 256k context: default 200000 chars so prompt fits; use RULER_MAX_PROMPT_CHARS=80000 for 32k
if "RULER_MAX_PROMPT_CHARS" not in os.environ:
    os.environ["RULER_MAX_PROMPT_CHARS"] = "200000"

RULER_DATA_DIR = os.path.join(_evals_dir, "data", "ruler")
CATEGORY_TO_TASK = {"0": "retrieval", "1": "MT", "2": "ACG", "3": "QA"}
TASK_NAMES = {"retrieval": "retri.acc", "MT": "MT acc", "ACG": "AGG acc", "QA": "QA acc"}

# Optional: skip some datasets when using --rerun_search, e.g. "fwe,cwe,vt"
RULER_RERUN_SKIP = {
    stem.strip()
    for stem in os.getenv("RULER_RERUN_SKIP", "").split(",")
    if stem.strip()
}


def _collect_ruler_datasets(prefix=None):
    """All .jsonl in data/ruler except qa_1 and any qa_* (e.g. qa_2). If prefix is set, only stems starting with prefix (e.g. 'niah')."""
    if not os.path.isdir(RULER_DATA_DIR):
        return []
    out = []
    for name in sorted(os.listdir(RULER_DATA_DIR)):
        if not name.endswith(".jsonl"):
            continue
        stem = name[:-6]
        if stem == "qa_1" or stem.startswith("qa_"):
            continue
        if prefix is not None and not stem.startswith(prefix):
            continue
        out.append((stem, os.path.join(RULER_DATA_DIR, name)))
    return out


def _results_exist(output_folder, batch_id):
    """True if this dataset already has run results we can use for aggregation."""
    return os.path.isfile(os.path.join(output_folder, "mem0_results.json"))


def _batch_statistics_path(output_folder, batch_id):
    """Path to batch_statistics_{batch_id}.json for this run."""
    return os.path.join(output_folder, f"batch_statistics_{batch_id}.json")


def _add_done(subdir):
    """True if add stage has been run: .qdrant_run exists and experiment_log has 'stage': 'add'."""
    qdrant_dir = os.path.join(subdir, ".qdrant_run")
    if not os.path.isdir(qdrant_dir):
        return False
    log_path = os.path.join(subdir, "experiment_log.jsonl")
    if not os.path.isfile(log_path):
        return False
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("stage") == "add":
                    return True
    except (json.JSONDecodeError, OSError):
        pass
    return False


def _run_search_only(data_path, output_folder, batch_id, max_questions, top_k=30):
    """Resume: run only search (then score_ruler). Use when add is already done."""
    output_folder = os.path.abspath(output_folder)
    data_path = os.path.abspath(data_path)
    cmd = [
        sys.executable,
        "-m",
        "evaluation.run_experiments",
        "--technique_type", "mem0",
        "--method", "search",
        "--output_folder", output_folder,
        "--batch_id", batch_id,
        "--data_path", data_path,
        "--top_k", str(top_k),
    ]
    if max_questions is not None:
        cmd.extend(["--max_questions", str(max_questions)])
    print("[run_ruler_128k] Resuming (search only):", " ".join(cmd))
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, cwd=_project_root, env=os.environ.copy()).returncode
    running_time_sec = time.perf_counter() - t0
    stats_path = os.path.join(output_folder, f"batch_statistics_{batch_id}.json")
    if ret == 0 and os.path.isfile(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            stats["running_time_sec"] = round(running_time_sec, 4)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    results_file = os.path.join(output_folder, "mem0_results.json")
    if ret == 0 and os.path.isfile(results_file):
        subprocess.run(
            [
                sys.executable,
                os.path.join(_evals_dir, "score_ruler.py"),
                "--results_file", results_file,
                "--stats_file", stats_path,
                "--output_dir", output_folder,
                "--batch_id", batch_id,
            ],
            cwd=_evals_dir,
        )
    print(f"[run_ruler_128k] {batch_id} (search only) running_time_sec={running_time_sec:.4f}")
    return ret


def _fill_missing_batch_statistics(base_output, datasets_by_stem, max_questions, top_k=80):
    """Find subdirs missing batch_statistics; for each: re-run search (add_then_search or search-only resume) or score only."""
    stems_with_data = {s: p for s, p in datasets_by_stem}
    filled = []
    for stem in sorted(os.listdir(base_output)):
        subdir = os.path.join(base_output, stem)
        if not os.path.isdir(subdir):
            continue
        batch_id = f"ruler_128k_{stem}"
        stats_path = _batch_statistics_path(subdir, batch_id)
        if os.path.isfile(stats_path):
            continue
        results_file = os.path.join(subdir, "mem0_results.json")
        if os.path.isfile(results_file):
            # Search already done: run score_ruler only (no existing stats file to merge)
            print(f"[run_ruler_128k] Filling missing batch_statistics for {stem} (score only, mem0_results present)")
            ret = subprocess.run(
                [
                    sys.executable,
                    os.path.join(_evals_dir, "score_ruler.py"),
                    "--results_file", os.path.abspath(results_file),
                    "--output_dir", subdir,
                    "--batch_id", batch_id,
                ],
                cwd=_evals_dir,
            ).returncode
            if ret == 0:
                filled.append(stem)
            else:
                print(f"[run_ruler_128k] score_ruler failed for {stem}, exit code {ret}")
        else:
            # No mem0_results: resume with search-only if add already done, else full add_then_search
            if stem not in stems_with_data:
                print(f"[run_ruler_128k] Skip {stem}: no batch_statistics and no mem0_results, data path not in ruler dir")
                continue
            data_path = stems_with_data[stem]
            if _add_done(subdir):
                print(f"[run_ruler_128k] Filling missing for {stem} (resume: search only, add already done)")
                ret = _run_search_only(data_path, os.path.abspath(subdir), batch_id, max_questions, top_k)
            else:
                print(f"[run_ruler_128k] Filling missing batch_statistics for {stem} (full add_then_search + score)")
                ret = _run_one(data_path, subdir, batch_id, max_questions, top_k)
            if ret == 0:
                filled.append(stem)
            else:
                print(f"[run_ruler_128k] _run_one/_run_search_only failed for {stem}, exit code {ret}")
    return filled


def _rerun_search_only(base_output, datasets_by_stem, max_questions, top_k=80):
    """Re-run only search (no add) for every subdir where add is already done. Overwrites mem0_results and batch_statistics. Use with --top_k 80 to improve retri.acc."""
    stems_with_data = {s: p for s, p in datasets_by_stem}
    rerun = []
    for stem in sorted(os.listdir(base_output)):
        subdir = os.path.join(base_output, stem)
        if not os.path.isdir(subdir):
            continue
        if RULER_RERUN_SKIP and stem in RULER_RERUN_SKIP:
            print(f"[run_ruler_128k] Skip {stem}: in RULER_RERUN_SKIP")
            continue
        if stem not in stems_with_data:
            continue
        if not _add_done(subdir):
            print(f"[run_ruler_128k] Skip {stem}: add not done, cannot rerun search only")
            continue
        batch_id = f"ruler_128k_{stem}"
        data_path = stems_with_data[stem]
        print(f"[run_ruler_128k] Rerun search only: {stem} (top_k={top_k})")
        ret = _run_search_only(data_path, os.path.abspath(subdir), batch_id, max_questions, top_k)
        if ret == 0:
            rerun.append(stem)
        else:
            print(f"[run_ruler_128k] _run_search_only failed for {stem}, exit code {ret}")
    return rerun


def _aggregate_ruler_results(base_output, prefix=None):
    """Load all mem0_results.json under base_output, merge, compute four metrics, print and save.
    If prefix is set, only include subdirs whose name starts with prefix (e.g. niah).
    Uses same AGG matching as score_ruler: first_ref_only for CWE, FWE strict 3-in-order for FWE."""
    from score_ruler import (
        load_results,
        _response_matches_any,
        _response_matches_first_ref_only,
        _response_matches_fwe_strict,
    )

    all_items = []
    stems_done = []
    for stem in sorted(os.listdir(base_output)):
        if prefix is not None and not stem.startswith(prefix):
            continue
        subdir = os.path.join(base_output, stem)
        if not os.path.isdir(subdir):
            continue
        results_file = os.path.join(subdir, "mem0_results.json")
        if not os.path.isfile(results_file):
            continue
        try:
            data = load_results(results_file)
            for conv_id, qa_list in data.items():
                for item in qa_list:
                    all_items.append({**item, "conv_id": conv_id})
            stems_done.append(stem)
        except Exception as e:
            print(f"[run_ruler_128k] Warning: skip {stem}: {e}")
            continue

    if not all_items:
        print("[run_ruler_128k] No results to aggregate.")
        return

    by_task = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in all_items:
        pred = str(item.get("response") or "").strip()
        outputs = item.get("outputs")
        if isinstance(outputs, list) and len(outputs) > 0:
            refs = outputs
        else:
            ref = str(item.get("answer") or "").strip()
            refs = [ref] if ref else []
        task = item.get("task_type") or CATEGORY_TO_TASK.get(str(item.get("category", "")), "QA")
        if task == "ACG":
            if item.get("dataset") == "fwe" and len(refs) >= 3:
                correct = 1 if _response_matches_fwe_strict(pred, refs) else 0
            else:
                correct = 1 if refs and _response_matches_first_ref_only(pred, refs) else 0
        else:
            correct = 1 if refs and _response_matches_any(pred, refs) else 0
        by_task[task]["correct"] += correct
        by_task[task]["total"] += 1

    metrics = {}
    for task, name in TASK_NAMES.items():
        d = by_task.get(task, {"correct": 0, "total": 0})
        acc = d["correct"] / d["total"] if d["total"] else 0.0
        metrics[name] = acc
    total_correct = sum(by_task[t]["correct"] for t in TASK_NAMES)
    total_n = sum(by_task[t]["total"] for t in TASK_NAMES)
    overall = total_correct / total_n if total_n else 0.0

    summary = {
        "datasets_included": stems_done,
        "n_questions": len(all_items),
        "retri.acc": metrics.get("retri.acc", 0.0),
        "MT acc": metrics.get("MT acc", 0.0),
        "AGG acc": metrics.get("AGG acc", 0.0),
        "QA acc": metrics.get("QA acc", 0.0),
        "overall_accuracy": overall,
    }
    summary_path = os.path.join(base_output, "overall_ruler_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("[run_ruler_128k] Aggregated results (datasets: %s)" % ", ".join(stems_done))
    print("=" * 60)
    print("  retri.acc=%s | MT acc=%s | AGG acc=%s | QA acc=%s" % (
        metrics.get("retri.acc", 0),
        metrics.get("MT acc", 0),
        metrics.get("AGG acc", 0),
        metrics.get("QA acc", 0),
    ))
    print("  overall_accuracy=%s (%d/%d)" % (overall, total_correct, total_n))
    print("  Saved: %s" % summary_path)
    print("=" * 60)


def _run_one(data_path, output_folder, batch_id, max_questions, top_k=80):
    os.makedirs(output_folder, exist_ok=True)
    os.environ["MEM0_DIR"] = os.path.join(output_folder, ".qdrant_run")
    os.makedirs(os.environ["MEM0_DIR"], exist_ok=True)

    if not os.path.isfile(data_path):
        print(f"[run_ruler_128k] Warning: data not found at {data_path}")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "evaluation.run_experiments",
        "--technique_type",
        "mem0",
        "--method",
        "add_then_search",
        "--output_folder",
        output_folder,
        "--batch_id",
        batch_id,
        "--data_path",
        data_path,
        "--top_k",
        str(top_k),
    ]
    if max_questions is not None:
        cmd.extend(["--max_questions", str(max_questions)])
    print("[run_ruler_128k] Executing:", " ".join(cmd))
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, cwd=_project_root, env=os.environ.copy()).returncode
    running_time_sec = time.perf_counter() - t0

    stats_path = os.path.join(output_folder, f"batch_statistics_{batch_id}.json")
    if ret == 0 and os.path.isfile(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            stats["running_time_sec"] = round(running_time_sec, 4)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    results_file = os.path.join(output_folder, "mem0_results.json")
    if ret == 0 and os.path.isfile(results_file):
        subprocess.run(
            [
                sys.executable,
                os.path.join(_evals_dir, "score_ruler.py"),
                "--results_file", results_file,
                "--stats_file", stats_path,
                "--output_dir", output_folder,
                "--batch_id", batch_id,
            ],
            cwd=_evals_dir,
        )
    print(f"[run_ruler_128k] {batch_id} running_time_sec={running_time_sec:.4f}")
    return ret


def main():
    parser = argparse.ArgumentParser(description="mem0 RULER 128k evaluation (retri.acc, MT acc, ACG acc, QA acc)")
    parser.add_argument("--max_questions", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--max_datasets", type=int, default=None, help="Run only first N datasets then stop (e.g. 1 = run one, check score)")
    parser.add_argument("--skip_done", action="store_true", help="Skip datasets that already have mem0_results.json; at end aggregate all and print/save overall summary")
    parser.add_argument("--output_folder", type=str, default=None, help="Output dir, default evaluation/results/ruler_128k")
    parser.add_argument("--data_path", type=str, default=None, help="Run only this dataset path (single run, legacy)")
    parser.add_argument("--fill_missing", action="store_true", help="Only run datasets that are missing batch_statistics: re-run add_then_search or score_ruler as needed, then aggregate")
    parser.add_argument("--rerun_search", action="store_true", help="Re-run only search (add must be done). Overwrites mem0_results and batch_statistics. Use with --top_k 80 to improve retri.acc.")
    parser.add_argument("--top_k", type=int, default=80, help="Number of memories to retrieve (default 80 for 256k). Use 100 for higher NIAH/retri.acc (benchmark-level).")
    parser.add_argument("--prefix", type=str, default=None, help="Only run datasets whose name starts with this (e.g. niah to run only niah_single_1, niah_multikey_1, ...).")
    args = parser.parse_args()

    base_output = args.output_folder or os.path.join(_evals_dir, "results", "ruler_128k")
    base_output = os.path.abspath(base_output)

    if args.rerun_search:
        datasets = _collect_ruler_datasets(args.prefix)
        datasets_by_stem = list(datasets)
        rerun = _rerun_search_only(base_output, datasets_by_stem, args.max_questions, args.top_k)
        print(f"[run_ruler_128k] Reran search for: {rerun}")
        _aggregate_ruler_results(base_output, args.prefix)
        return 0

    if args.fill_missing:
        datasets = _collect_ruler_datasets(args.prefix)
        datasets_by_stem = list(datasets)
        filled = _fill_missing_batch_statistics(base_output, datasets_by_stem, args.max_questions, args.top_k)
        print(f"[run_ruler_128k] Filled batch_statistics for: {filled}")
        _aggregate_ruler_results(base_output, args.prefix)
        return 0

    if args.data_path is not None:
        # Single run with given data path (legacy)
        stem = os.path.splitext(os.path.basename(args.data_path))[0]
        batch_id = f"ruler_128k_{stem}"
        output_folder = os.path.join(base_output, stem)
        return _run_one(args.data_path, output_folder, batch_id, args.max_questions, args.top_k)

    datasets = _collect_ruler_datasets(args.prefix)
    if not datasets:
        print(f"[run_ruler_128k] No datasets found under {RULER_DATA_DIR} (excluding qa_1, qa_*)")
        return 1

    if args.max_datasets is not None:
        datasets = datasets[: args.max_datasets]
        print(f"[run_ruler_128k] Running {len(datasets)} dataset(s) then stop: {[s for s, _ in datasets]}")
    else:
        print(f"[run_ruler_128k] Running {len(datasets)} datasets: {[s for s, _ in datasets]}")
    last_ret = 0
    for stem, data_path in datasets:
        batch_id = f"ruler_128k_{stem}"
        output_folder = os.path.join(base_output, stem)
        if args.skip_done and _results_exist(output_folder, batch_id):
            print(f"[run_ruler_128k] Skipping {stem} (already has results)")
            continue
        last_ret = _run_one(data_path, output_folder, batch_id, args.max_questions, args.top_k)
        if last_ret != 0:
            print(f"[run_ruler_128k] Failed for {stem}, exit code {last_ret}")
    _aggregate_ruler_results(base_output, args.prefix)
    return last_ret


if __name__ == "__main__":
    sys.exit(main())
