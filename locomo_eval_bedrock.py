# Usage:
# TODO:
#   python eval_locomo.py --data mem0/evaluation/locomo/locomo10.json --memories mem0/evaluation/memories --max-samples 1

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

def _tokenize(text):
    text = str(text).lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ")
    return text.split()

def _f1(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    pred_tokens = set(_tokenize(prediction))
    ref_tokens = set(_tokenize(reference))
    common = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(ref_tokens)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def _bleu1(prediction: str, reference: str) -> float:
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        pred_tokens = nltk.word_tokenize(prediction.lower())
        ref_tokens = [nltk.word_tokenize(reference.lower())]
        return sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    except Exception:
        pt, rt = _tokenize(prediction), _tokenize(reference)
        if not rt:
            return 0.0
        return sum(1 for t in pt if t in rt) / len(rt) if rt else 0.0

def make_summary_prompt(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence. Answer with exact words from the context whenever possible.
For questions that require answering a date or time, strictly follow the format "15 July 2023" and provide a specific date whenever possible. For example, if you need to answer "last year," give the specific year of last year rather than just saying "last year." Only provide one year, date, or time, without any extra responses.
If the question is about the duration, answer in the form of several years, months, or days.

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

def make_prompt_for_category(context: str, question: str, category: int) -> str:
    if category == 3:
        return make_summary_prompt_category3(context, question)
    return make_summary_prompt(context, question)

def get_bedrock_client(region: str = "us-east-1"):
    import boto3
    return boto3.client("bedrock-runtime", region_name=region)

def invoke_bedrock_converse(client, model_id: str, system_content: str, max_tokens: int = 256, temperature: float = 0.0):
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": "Answer."}]}],
        system=[{"text": system_content}],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )
    out = response.get("output", {})
    msg = out.get("message", {})
    content = msg.get("content", [])
    text = content[0].get("text", "").strip() if content else ""
    usage = response.get("usage", {})
    inp = usage.get("inputTokens", 0)
    out_tok = usage.get("outputTokens", 0)
    return text, inp, out_tok

def load_memories(memories_dir: str, idx: int) -> str:
    path = os.path.join(memories_dir, f"{idx}.txt")
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def run_locomo_eval(
    dataset_path: str,
    memories_dir: str,
    results_dir: str,
    inference_profile_id: str = "5gzr4k820woo",
    region: str = "us-east-1",
    price_input_per_m: float = 3.0,
    price_output_per_m: float = 15.0,
):
    os.makedirs(results_dir, exist_ok=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = get_bedrock_client(region=region)

    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0
    context_window_peak = 0
    category_f1 = defaultdict(list)
    category_bleu1 = defaultdict(list)

    try:
        from tqdm import tqdm
        conv_iter = tqdm(enumerate(data), total=len(data), desc="Conversations")
    except ImportError:
        conv_iter = enumerate(data)

    start_time = time.perf_counter()
    for idx, item in conv_iter:
        qa_list = item.get("qa", [])
        memories = load_memories(memories_dir, idx)

        for q in qa_list:
            question = q.get("question", "")
            answer = q.get("answer", "")
            category = q.get("category", 0)
            try:
                cat_int = int(category) if category is not None else 0
            except (TypeError, ValueError):
                cat_int = 0

            if cat_int == 5:
                continue

            system_content = make_prompt_for_category(memories, question, cat_int)
            try:
                response_text, inp, out_tok = invoke_bedrock_converse(
                    client, inference_profile_id, system_content
                )
            except Exception as e:
                response_text = ""
                inp = out_tok = 0
                print(f" [Error] conv={idx} question: {e}", file=sys.stderr)

            total_input_tokens += inp
            total_output_tokens += out_tok
            if inp > context_window_peak:
                context_window_peak = inp

            f1 = _f1(response_text, answer)
            bleu1 = _bleu1(response_text, answer)
            category_f1[cat_int].append(f1)
            category_bleu1[cat_int].append(bleu1)

            all_results.append({
                "conversation_id": idx,
                "question": question,
                "answer": answer,
                "response": response_text,
                "category": cat_int,
                "f1": f1,
                "bleu1": bleu1,
                "input_tokens": inp,
                "output_tokens": out_tok,
            })

    cost = (total_input_tokens / 1_000_000 * price_input_per_m +
            total_output_tokens / 1_000_000 * price_output_per_m)
    end_time = time.perf_counter()
    experiment_time_seconds = end_time - start_time
    n_samples = len(data)
    n_total = len(all_results)
    cost_per_sample = cost / n_samples if n_samples else 0
    token_per_sample_input = total_input_tokens / n_samples if n_samples else 0
    token_per_sample_output = total_output_tokens / n_samples if n_samples else 0
    token_per_sample_total = (total_input_tokens + total_output_tokens) / n_samples if n_samples else 0

    batch_results_path = os.path.join(results_dir, "batch_results_0_9.json")
    batch_statistics_path = os.path.join(results_dir, "batch_statistics_0_9.json")

    stats = {"by_category": {}, "overall": {}}
    for cat in sorted(category_f1.keys()):
        f1_list = category_f1[cat]
        bleu_list = category_bleu1[cat]
        n = len(f1_list)
        if n == 0:
            continue
        f1_avg = sum(f1_list) / n
        bleu1_avg = sum(bleu_list) / n
        stats["by_category"][cat] = {"n": n, "F1_avg": f1_avg, "BLEU1_avg": bleu1_avg}

    n_total = len(all_results)
    overall_f1 = sum(r["f1"] for r in all_results) / n_total if n_total else 0.0
    overall_bleu1 = sum(r["bleu1"] for r in all_results) / n_total if n_total else 0.0
    stats["overall"] = {"问题数": n_total, "平均 F1": overall_f1, "平均 BLEU1": overall_bleu1}
    stats["token_usage"] = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "price_input_per_m": price_input_per_m,
        "price_output_per_m": price_output_per_m,
        "estimated_cost_usd": round(cost, 4),
    }

    with open(batch_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    with open(batch_statistics_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    sep = "=" * 60
    print(sep)
    print("Token 与费用估算")
    print(sep)
    print(f"  input_tokens:  {total_input_tokens}")
    print(f"  output_tokens: {total_output_tokens}")
    print(f"  context_window_peak (单次最大 input): {context_window_peak}")
    print(f"  Experiment Time: {experiment_time_seconds:.2f} s")
    print(f"  单价: 输入 ${price_input_per_m}/1M, 输出 ${price_output_per_m}/1M")
    print(f"  估算费用: ${cost:.4f}  |  Cost per Sample: ${cost_per_sample:.4f}")
    print(f"  Token per Sample: input={token_per_sample_input:.0f}, output={token_per_sample_output:.0f}, total={token_per_sample_total:.0f}")
    print(sep)
    print()
    print(sep)
    print("LoCoMo 10 样本 — 分数汇总")
    print(sep)
    print("按类别:")
    for cat in sorted(stats["by_category"].keys()):
        s = stats["by_category"][cat]
        print(f"  Category {cat}: n={s['n']}, F1_avg={s['F1_avg']:.4f}, BLEU1_avg={s['BLEU1_avg']:.4f}")
    print()
    print(f"整体: 问题数={n_total}, 平均 F1={overall_f1:.4f}, 平均 BLEU1={overall_bleu1:.4f}")
    print(sep)
    print(f"统计已保存: {batch_statistics_path}")
    print(f"结果已保存: {batch_results_path}")

    log_path = os.path.join(results_dir, "experiment_log.jsonl")
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "argv": sys.argv,
        "results_dir": os.path.abspath(results_dir),
        "batch_results": os.path.abspath(batch_results_path),
        "batch_statistics": os.path.abspath(batch_statistics_path),
        "dataset_path": os.path.abspath(dataset_path),
        "n_samples": n_samples,
        "total_questions": n_total,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "context_window_peak": context_window_peak,
        "experiment_time_seconds": round(experiment_time_seconds, 2),
        "cost_usd": round(cost, 4),
        "cost_per_sample": round(cost_per_sample, 4),
        "token_per_sample_input": round(token_per_sample_input, 2),
        "token_per_sample_output": round(token_per_sample_output, 2),
        "token_per_sample_total": round(token_per_sample_total, 2),
        "overall_f1": overall_f1,
        "overall_bleu1": overall_bleu1,
        "inference_profile_id": inference_profile_id,
        "region": region,
        "price_input_per_1m": price_input_per_m,
        "price_output_per_1m": price_output_per_m,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    print(f"实验记录已追加: {log_path}")

    return stats, all_results

def main():
    parser = argparse.ArgumentParser(description="LoCoMo 测评（Bedrock Inference Profile）")
    parser.add_argument("--dataset", type=str, default="dataset/locomo10.json", help="LoCoMo 数据集 JSON 路径")
    parser.add_argument("--memories-dir", type=str, default="memories", help="memories 目录（每轮对话对应 {idx}.txt）")
    parser.add_argument("--results-dir", type=str, default="./results/locomo_full", help="结果输出目录")
    parser.add_argument("--inference-profile-identifier", type=str, default="5gzr4k820woo", help="Bedrock Inference Profile ID")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS Region")
    parser.add_argument("--price-input", type=float, default=3.0, help="输入单价 ($/1M tokens)")
    parser.add_argument("--price-output", type=float, default=15.0, help="输出单价 ($/1M tokens)")
    args = parser.parse_args()

    run_locomo_eval(
        dataset_path=args.dataset,
        memories_dir=args.memories_dir,
        results_dir=args.results_dir,
        inference_profile_id=args.inference_profile_identifier,
        region=args.region,
        price_input_per_m=args.price_input,
        price_output_per_m=args.price_output,
    )

if __name__ == "__main__":
    main()
