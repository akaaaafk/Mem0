#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mem0 + LoCoMo 数据集测试

- Embedder: BAAI/bge-m3 (与 GAM 一致，避免维度问题)
- LLM: 本地 Ollama
- 输出格式与 GAM 一致: batch_results_*.json (question, gold_answer, category, summary_answer)
- 带 time、log 记录
"""

import sys
import os
import re
import json
import math
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

# 将 mem0 包加入 path（在 baseline/mem0 下运行）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tqdm import tqdm

# ========== 日志与时间 ==========

def setup_logging(log_dir: str, prefix: str = "locomo") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")
    logger = logging.getLogger("mem0_locomo")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Log file: %s", log_file)
    return logger


def log_time(logger: logging.Logger, msg: str, start: Optional[datetime] = None) -> datetime:
    now = datetime.now()
    if start is not None:
        elapsed = (now - start).total_seconds()
        logger.info("%s (elapsed: %.2fs)", msg, elapsed)
    else:
        logger.info("%s", msg)
    return now


# ========== 数据加载（与 GAM locomo_test 一致）==========

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_locomo(json_path: str) -> List[Dict[str, Any]]:
    data = load_json(json_path)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized LoCoMo JSON shape. Expect a list or {'samples': [...]}.")


def extract_sessions(conv_obj: Dict[str, Any]) -> List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]]:
    sessions: List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]] = []
    for k, v in conv_obj.items():
        m = re.match(r"^session_(\d+)$", k)
        if not (m and isinstance(v, list)):
            continue
        original_idx = int(m.group(1))
        idx = original_idx - 1
        ts = conv_obj.get(f"session_{original_idx}_date_time", "")
        ssum = conv_obj.get(f"session_{original_idx}_summary", None)
        sessions.append((idx, ts, v, ssum if isinstance(ssum, str) and ssum.strip() else None))
    sessions.sort(key=lambda x: x[0])
    return sessions


def session_to_text(
    idx: int, ts: str, turns: List[Dict[str, Any]], session_summary: Optional[str]
) -> str:
    lines = [f"=== SESSION {idx} - Dialogue Time(available to answer questions): {ts} ==="]
    lines.append("")
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        dia_id = turn.get("dia_id", "")
        text = turn.get("text", "")
        lines.append(f"{speaker} ({dia_id}): {text}")
    if session_summary:
        lines.append("")
        lines.append(f"Session {idx} summary: {session_summary}")
    return "\n".join(lines).strip()


def build_session_chunks_for_sample(sample: Dict[str, Any]) -> List[str]:
    conv = sample.get("conversation", {})
    sessions = extract_sessions(conv)
    return [session_to_text(idx, ts, turns, ssum) for idx, ts, turns, ssum in sessions]


def collect_qa_items_for_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    qas: List[Dict[str, Any]] = []
    sid = sample.get("sample_id", None)
    for q in sample.get("qa", []):
        qas.append({
            "sample_id": sid,
            "question": q.get("question"),
            "answer": q.get("answer"),
            "category": q.get("category"),
            "evidence": q.get("evidence"),
        })
    return qas


# ========== Prompt（与 GAM 一致）==========

def make_summary_prompt(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence. Answer with exact words from the context whenever possible.
For questions that require answering a date or time, strictly follow the format "15 July 2023" and provide a specific date whenever possible. For example, if you need to answer "last year," give the specific year of last year rather than just saying "last year." Only provide one year, date, or time, without any extra responses. If the question is about the duration, answer in the form of several years, months, or days.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""


def make_summary_prompt_category3(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence.
The question may need you to analyze and infer the answer from the summary.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""


# ========== 指标（与 GAM 一致）==========

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(s: str):
    s = normalize_text(s)
    return s.split() if s else []


def f1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if len(ptoks) == 0:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks) if ptoks else 0.0
    bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks) / len(ptoks)) if (ptoks and gtoks) else 0.0
    return bp * precision


def compute_metrics_by_category(items, pred_key: str = "summary_answer", pred_field: str = "answer"):
    agg = defaultdict(list)
    rows = []
    for idx, ex in enumerate(items, 1):
        cat = ex.get("category", "NA")
        gold = ex.get("gold_answer", "")
        pred = ""
        val = ex.get(pred_key, "")
        if isinstance(val, dict):
            pred = val.get(pred_field, "")
        else:
            pred = val
        f1 = f1_score(pred, gold)
        b1 = bleu1_score(pred, gold)
        agg[cat].append((f1, b1))
        rows.append({
            "q_idx": idx,
            "category": cat,
            "gold_answer": str(gold),
            "prediction": str(pred),
            "F1": f1,
            "BLEU1": b1,
        })
    summary = []
    for cat in sorted(agg.keys(), key=lambda x: str(x)):
        scores = agg[cat]
        if scores:
            summary.append({
                "category": cat,
                "count": len(scores),
                "F1_avg": sum(s[0] for s in scores) / len(scores),
                "BLEU1_avg": sum(s[1] for s in scores) / len(scores),
            })
    return summary, rows


# ========== Mem0 + Ollama 配置 ==========

# BAAI/bge-m3 维度
BGE_DIMS = 1024


def get_mem0_config(
    outdir: str,
    ollama_model: str = "llama3.2",
    ollama_base_url: str = "http://localhost:11434",
    embedder_model: str = "BAAI/bge-m3",
    collection_name: str = "mem0_locomo",
) -> Dict[str, Any]:
    path = os.path.join(outdir, "faiss_index")
    return {
        "version": "v1.1",
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": embedder_model,
                "embedding_dims": BGE_DIMS,
            },
        },
        "vector_store": {
            "provider": "faiss",
            "config": {
                "collection_name": collection_name,
                "path": path,
                "embedding_model_dims": BGE_DIMS,
                "distance_strategy": "cosine",
                "normalize_L2": True,
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": ollama_model,
                "ollama_base_url": ollama_base_url,
                "temperature": 0.3,
                "max_tokens": 2048,
            },
        },
    }


# ========== 使用 Ollama 生成短答案 ==========

def answer_with_ollama(
    category: Optional[int],
    summary: str,
    question: str,
    model: str,
    base_url: str,
    logger: logging.Logger,
) -> str:
    if category == 3:
        prompt = make_summary_prompt_category3(summary, question)
    else:
        prompt = make_summary_prompt(summary, question)
    try:
        from ollama import Client
        client = Client(host=base_url)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 128},
        )
        text = (response.get("message") or {}).get("content") or ""
        return text.strip()
    except Exception as e:
        logger.warning("Ollama generate error: %s", e)
        return ""


# ========== 单样本处理 ==========

def process_sample(
    sample: Dict[str, Any],
    sample_index: int,
    outdir: str,
    ollama_model: str,
    ollama_base_url: str,
    embedder_model: str,
    logger: logging.Logger,
    search_limit: int = 10,
) -> List[Dict[str, Any]]:
    sample_id = sample.get("sample_id", f"conv-{sample_index}")
    t_start = datetime.now()
    logger.info("Processing sample #%s: %s", sample_index, sample_id)

    try:
        from mem0 import Memory
    except ImportError:
        logger.error("mem0 not installed. Run from repo: pip install -e .[vector_stores,llms,extras]")
        raise

    session_chunks = build_session_chunks_for_sample(sample)
    logger.info("Sessions: %d", len(session_chunks))
    t_after_chunks = log_time(logger, "Built session chunks", t_start)

    sample_dir = os.path.join(outdir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)
    config = get_mem0_config(
        sample_dir,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        embedder_model=embedder_model,
        collection_name=f"mem0_{sample_id}",
    )
    memory = Memory.from_config(config)
    log_time(logger, "Memory (mem0) initialized with bge + ollama + faiss", t_after_chunks)

    # 将每个 session 作为一条 user 消息写入 mem0
    for i, chunk in enumerate(session_chunks):
        msg_start = datetime.now()
        memory.add(
            [{"role": "user", "content": chunk}],
            user_id=sample_id,
            infer=True,
        )
        logger.debug("Added session %d/%d in %.2fs", i + 1, len(session_chunks), (datetime.now() - msg_start).total_seconds())
    t_after_add = log_time(logger, "Added all sessions to mem0", t_after_chunks)

    qas = collect_qa_items_for_sample(sample)
    qa_results = []
    for i, qi in enumerate(qas):
        q = qi.get("question") or ""
        gold = qi.get("answer")
        cat = qi.get("category")
        if cat == 5:
            continue
        q_start = datetime.now()
        try:
            search_result = memory.search(q, user_id=sample_id, limit=search_limit, rerank=False)
            results_list = search_result.get("results") if isinstance(search_result, dict) else []
            summary = "\n".join(
                (r.get("memory") or r.get("data") or str(r))
                for r in results_list[:search_limit]
            ).strip() or "(No memories retrieved)"
            summary_answer = answer_with_ollama(cat, summary, q, ollama_model, ollama_base_url, logger)
            elapsed = (datetime.now() - q_start).total_seconds()
            logger.info("Q %d/%d answered in %.2fs: %s", i + 1, len(qas), elapsed, summary_answer[:80] if summary_answer else "")
            qa_results.append({
                "question": q,
                "gold_answer": gold,
                "category": cat,
                "summary_answer": summary_answer,
                "retrieved_count": len(results_list),
                "time_sec": round(elapsed, 2),
            })
        except Exception as e:
            logger.exception("Q %d failed: %s", i + 1, e)
            qa_results.append({
                "question": q,
                "gold_answer": gold,
                "category": cat,
                "error": str(e),
            })

    log_time(logger, "Finished sample %s (%d QA)" % (sample_id, len(qa_results)), t_start)
    return qa_results


# ========== 主函数 ==========

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mem0 + LoCoMo 测试 (BAAI/bge + Ollama)")
    parser.add_argument("--data", type=str, default=None, help="LoCoMo JSON 路径（默认使用 baseline/data/locomo/locomo10.json）")
    parser.add_argument("--outdir", type=str, default="./results/locomo", help="输出目录")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None, help="结束样本索引（不含）")
    parser.add_argument("--ollama-model", type=str, default="llama3.2", help="Ollama 模型名")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--embedder-model", type=str, default="BAAI/bge-m3", help="与 GAM 一致")
    parser.add_argument("--search-limit", type=int, default=10)
    args = parser.parse_args()

    if args.data is None:
        args.data = os.path.join(_REPO_ROOT, "..", "data", "locomo", "locomo10.json")
    if not os.path.isabs(args.data):
        args.data = os.path.abspath(args.data)
    if not os.path.exists(args.data):
        print("Data not found:", args.data)
        return 1

    log_dir = os.path.join(args.outdir, "logs")
    logger = setup_logging(log_dir, "mem0_locomo")
    logger.info("Data: %s", args.data)
    logger.info("Outdir: %s", args.outdir)
    logger.info("Ollama: %s @ %s", args.ollama_model, args.ollama_base_url)
    logger.info("Embedder: %s (dims=%d)", args.embedder_model, BGE_DIMS)

    samples = load_locomo(args.data)
    logger.info("Loaded %d samples", len(samples))
    if args.end_idx is None:
        args.end_idx = len(samples)
    args.end_idx = min(args.end_idx, len(samples))
    if args.start_idx >= args.end_idx:
        logger.error("Invalid range start=%s end=%s", args.start_idx, args.end_idx)
        return 1

    all_results = []
    for sample_idx in tqdm(range(args.start_idx, args.end_idx), desc="Samples"):
        sample = samples[sample_idx]
        t0 = datetime.now()
        try:
            results = process_sample(
                sample,
                sample_idx,
                args.outdir,
                args.ollama_model,
                args.ollama_base_url,
                args.embedder_model,
                logger,
                search_limit=args.search_limit,
            )
            all_results.extend(results)
            logger.info("Sample %d done in %.2fs, %d QA", sample_idx, (datetime.now() - t0).total_seconds(), len(results))
        except Exception as e:
            logger.exception("Sample %d failed: %s", sample_idx, e)

    if all_results:
        summary_file = os.path.join(args.outdir, f"batch_results_{args.start_idx}_{args.end_idx - 1}.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info("Saved batch results: %s", summary_file)

        summary_metrics, details = compute_metrics_by_category(all_results, pred_key="summary_answer", pred_field="answer")
        all_f1 = [r["F1"] for r in details]
        all_b1 = [r["BLEU1"] for r in details]
        overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
        overall_bleu1 = sum(all_b1) / len(all_b1) if all_b1 else 0.0
        logger.info("Overall F1: %.4f, BLEU1: %.4f", overall_f1, overall_bleu1)
        for r in summary_metrics:
            logger.info("Category %s: n=%d F1=%.4f BLEU1=%.4f", r["category"], r["count"], r["F1_avg"], r["BLEU1_avg"])

        statistics = {
            "total_samples": args.end_idx - args.start_idx,
            "total_questions": len(all_results),
            "overall_f1_avg": overall_f1,
            "overall_bleu1_avg": overall_bleu1,
            "by_category": summary_metrics,
            "details": details,
            "start_idx": args.start_idx,
            "end_idx": args.end_idx - 1,
        }
        stats_file = os.path.join(args.outdir, f"batch_statistics_{args.start_idx}_{args.end_idx - 1}.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        logger.info("Saved statistics: %s", stats_file)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
