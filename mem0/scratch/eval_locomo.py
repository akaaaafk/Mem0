# Usage:
# 运行不了时请检查：
#   1. AWS 凭证：在项目根目录执行 aws configure，填对 Access Key 和 Secret Key（本机时间要准，否则会报签名错误）。
#   2. 数据路径：--data 指向 locomo10.json（list，每项含 "qa"）；有 "conversation" 时可不用 --memories。
#   3. Inference Profile：--account-id 与 --inference-profile-id 必须是你在 Bedrock 里创建的 Application Inference Profile；IAM 需有 bedrock:InvokeModel。
#
# 用法:
#   python eval_locomo.py --data mem0/evaluation/locomo/locomo10.json --memories mem0/evaluation/memories
#   python eval_locomo.py --max-samples 1 --max-questions 5 --verbose

import argparse
import json
import os
import sys
import time
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.join(_SCRIPT_DIR, "mem0", "evaluation")
if os.path.isdir(_EVAL_DIR) and _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from prompts import ANSWER_PROMPT_ZEP
from metrics.utils import calculate_bleu_scores, calculate_metrics

DEFAULT_REGION = "us-east-1"
DEFAULT_ACCOUNT_ID = os.environ.get("BEDROCK_ACCOUNT_ID", "...")
DEFAULT_INFERENCE_PROFILE_ID = os.environ.get("BEDROCK_INFERENCE_PROFILE_ID", "...")

def build_inference_profile_arn(region: str, account_id: str, profile_id: str) -> str:
    return f"arn:aws:bedrock:{region}:{account_id}:application-inference-profile/{profile_id}"

def invoke_bedrock(
    region: str,
    model_id: str,
    prompt: str,
    max_tokens: int = 256,
):
    import boto3
    client = boto3.client("bedrock-runtime", region_name=region)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    resp = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    ret = json.loads(resp["body"].read())
    text = ""
    if ret.get("content") and isinstance(ret["content"], list) and len(ret["content"]) > 0:
        text = (ret["content"][0].get("text", "") or "").strip()
    usage = ret.get("usage", {})
    inp = usage.get("input_tokens", usage.get("inputTokens", 0))
    out_tok = usage.get("output_tokens", usage.get("outputTokens", 0))
    return text, inp, out_tok

def load_memories(memories_dir: str, idx: int) -> str:
    path = os.path.join(memories_dir, f"{idx}.txt")
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def conversation_to_memory(conv: dict) -> str:
    if not conv or not isinstance(conv, dict):
        return ""
    lines = []
    keys = sorted([k for k in conv if k.startswith("session_") and not k.endswith("_date_time")], key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)
    for k in keys:
        if not k[-1].isdigit():
            continue
        date_key = f"{k}_date_time"
        ts = conv.get(date_key, "")
        lines.append(f"=== {k.upper()} (Time: {ts}) ===")
        for turn in conv.get(k) or []:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "").strip()
            if text:
                lines.append(f"{speaker}: {text}")
        lines.append("")
    return "\n".join(lines).strip()

def main():
    parser = argparse.ArgumentParser(description="LoCoMo 评估（AWS Bedrock Inference）")
    parser.add_argument("--data", type=str, default=None, help="locomo10.json 路径")
    parser.add_argument("--memories", type=str, default=None, help="memories 目录（每 sample 对应 {idx}.txt）")
    parser.add_argument("--outdir", type=str, default="./results/eval_locomo_10", help="结果输出目录")
    parser.add_argument("--max-samples", type=int, default=1, help="跑几个 sample，默认 1")
    parser.add_argument("--max-questions", type=int, default=None, help="最多跑多少道题，默认不限制")
    parser.add_argument("--region", type=str, default=DEFAULT_REGION, help="AWS Region")
    parser.add_argument("--account-id", type=str, default=DEFAULT_ACCOUNT_ID, help="Bedrock 账号 ID")
    parser.add_argument("--inference-profile-id", type=str, default=DEFAULT_INFERENCE_PROFILE_ID, help="Application Inference Profile ID")
    parser.add_argument("--price-input", type=float, default=3.0, help="输入单价 $/1M tokens")
    parser.add_argument("--price-output", type=float, default=15.0, help="输出单价 $/1M tokens")
    parser.add_argument("--verbose", action="store_true", help="前几题打印 gold / model response")
    args = parser.parse_args()

    default_data = os.path.join(_SCRIPT_DIR, "mem0", "evaluation", "locomo", "locomo10.json")
    default_memories = os.path.join(_SCRIPT_DIR, "mem0", "evaluation", "memories")
    data_path = args.data or default_data
    memories_dir = args.memories or default_memories

    model_id = build_inference_profile_arn(args.region, args.account_id, args.inference_profile_id)
    try:
        import boto3
        _client = boto3.client("bedrock-runtime", region_name=args.region)
        _body = {"anthropic_version": "bedrock-2023-05-31", "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 10}
        _resp = _client.invoke_model(modelId=model_id, contentType="application/json", accept="application/json", body=json.dumps(_body))
        _ret = json.loads(_resp["body"].read())
        if not (_ret.get("content") and len(_ret["content"]) > 0):
            print("警告: Bedrock 返回无 content，请检查 Inference Profile 与模型。", file=sys.stderr)
    except Exception as e:
        print("Bedrock 连通性检查失败（首包即报错）：", file=sys.stderr)
        print(e, file=sys.stderr)
        if "InvalidSignatureException" in str(e) or "signature" in str(e).lower():
            print("→ 请重新 aws configure，确认 Secret Access Key 正确且本机时间准确。", file=sys.stderr)
        elif "ValidationException" in str(e) or "invalid" in str(e).lower() or "AccessDenied" in str(e):
            print("→ 请检查 --account-id / --inference-profile-id 与 IAM 权限（bedrock:InvokeModel）。", file=sys.stderr)
        return 1

    if not os.path.isfile(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        print("请指定 --data 指向 locomo10.json，例如:")
        print("  --data mem0/evaluation/locomo/locomo10.json")
        return 1

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("错误: 数据应为 list（每项含 qa）")
        return 1

    data = data[: args.max_samples]
    os.makedirs(args.outdir, exist_ok=True)

    def build_prompt(memories: str, question: str) -> str:
        return ANSWER_PROMPT_ZEP.replace("{{memories}}", memories).replace("{{question}}", question)

    print(f"样本数: {len(data)}")
    print(f"Inference: AWS Bedrock invoke_model ARN: {model_id[:72]}...")
    print("=" * 60)

    results = []
    total_in, total_out = 0, 0
    count = 0
    context_window_peak = 0
    start_time = time.perf_counter()

    for conv_idx, item in enumerate(data):
        qa_list = item.get("qa", [])
        memories = load_memories(memories_dir, conv_idx)
        if not memories.strip():
            memories = conversation_to_memory(item.get("conversation", {}))
            if memories and conv_idx == 0:
                print("  [无 memory 文件] 使用 data 中的 conversation 作为 context")

        for q in qa_list:
            if args.max_questions is not None and count >= args.max_questions:
                break
            if str(q.get("category", "")) == "5":
                continue

            question = str(q.get("question", "") or "")
            raw_answer = q.get("answer", "")
            answer = "" if raw_answer is None else str(raw_answer)
            prompt = build_prompt(memories, question)

            try:
                response_text, inp, out_tok = invoke_bedrock(args.region, model_id, prompt)
            except Exception as e:
                err_str = str(e)
                print(f"  [Error] conv={conv_idx} {e}", file=sys.stderr)
                if "InvalidSignatureException" in err_str or "signature" in err_str.lower():
                    print("  签名错误：请检查 AWS 凭证（Secret Access Key 是否正确、是否过期）。", file=sys.stderr)
                    print("  建议：重新运行 aws configure 填写正确的 Access Key 与 Secret Key；或检查系统时间是否准确。", file=sys.stderr)
                    return 1
                if "ValidationException" in err_str or "invalid" in err_str.lower():
                    print("  请检查 --region / --account-id / --inference-profile-id，以及 IAM 权限。", file=sys.stderr)
                    return 1
                response_text, inp, out_tok = "", 0, 0

            total_in += inp
            total_out += out_tok
            if inp > context_window_peak:
                context_window_peak = inp
            count += 1

            metrics = calculate_metrics(response_text, answer)
            bleu = calculate_bleu_scores(response_text, answer)
            f1 = metrics["f1"]
            bleu1 = bleu["bleu1"]

            results.append({
                "conv_idx": conv_idx,
                "q_idx": count,
                "category": q.get("category"),
                "question": question,
                "answer": answer,
                "response": response_text,
                "f1": f1,
                "bleu1": bleu1,
            })
            print(f"  [sample {conv_idx} q{count}] cat={q.get('category')} F1={f1:.4f} BLEU1={bleu1:.4f} | Q: {question[:50]}...")
            if args.verbose and count <= 3:
                print(f"      memories: {'（空）' if not memories.strip() else f'{len(memories)} 字符'}")
                print(f"      gold:    {repr(answer)}")
                print(f"      model:   {repr(response_text)}")

        if args.max_questions is not None and count >= args.max_questions:
            break

    if not results:
        print("没有有效问题结果")
        return 0
    if total_in == 0 and total_out == 0:
        print("\n错误: 所有题目 tokens 均为 0，Bedrock 调用可能全部失败。", file=sys.stderr)
        return 1

    end_time = time.perf_counter()
    experiment_time_seconds = end_time - start_time
    n_samples = len(data)
    cost_usd = (total_in / 1e6) * args.price_input + (total_out / 1e6) * args.price_output
    cost_per_sample = cost_usd / n_samples if n_samples else 0
    token_per_sample_input = total_in / n_samples if n_samples else 0
    token_per_sample_output = total_out / n_samples if n_samples else 0
    token_per_sample_total = (total_in + total_out) / n_samples if n_samples else 0

    n = len(results)
    avg_f1 = sum(r["f1"] for r in results) / n
    avg_bleu1 = sum(r["bleu1"] for r in results) / n

    print()
    print("=" * 60)
    print("LoCoMo 评估结果（AWS Bedrock Inference）")
    print("=" * 60)
    print(f"  问题数: {n}")
    print(f"  样本数: {n_samples}")
    print(f"  input_tokens:  {total_in}")
    print(f"  output_tokens: {total_out}")
    print(f"  context_window_peak (单次最大 input): {context_window_peak}")
    print(f"  Experiment Time: {experiment_time_seconds:.2f} s")
    print(f"  估算费用: ${cost_usd:.4f}  |  Cost per Sample: ${cost_per_sample:.4f}")
    print(f"  Token per Sample: input={token_per_sample_input:.0f}, output={token_per_sample_output:.0f}, total={token_per_sample_total:.0f}")
    print(f"  平均 F1:   {avg_f1:.4f}")
    print(f"  平均 BLEU1: {avg_bleu1:.4f}")
    print("=" * 60)

    out_path = os.path.join(args.outdir, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "total_questions": n, "avg_f1": avg_f1, "avg_bleu1": avg_bleu1}, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}")

    log_path = os.path.join(args.outdir, "experiment_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "argv": sys.argv,
            "outdir": os.path.abspath(args.outdir),
            "results_file": os.path.abspath(out_path),
            "max_samples": args.max_samples,
            "max_questions": args.max_questions,
            "n_samples": n_samples,
            "total_questions": n,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "context_window_peak": context_window_peak,
            "experiment_time_seconds": round(experiment_time_seconds, 2),
            "cost_usd": round(cost_usd, 4),
            "cost_per_sample": round(cost_per_sample, 4),
            "token_per_sample_input": round(token_per_sample_input, 2),
            "token_per_sample_output": round(token_per_sample_output, 2),
            "token_per_sample_total": round(token_per_sample_total, 2),
            "avg_f1": avg_f1,
            "avg_bleu1": avg_bleu1,
            "region": args.region,
            "account_id": args.account_id,
            "inference_profile_id": args.inference_profile_id,
            "price_input_per_1m": args.price_input,
            "price_output_per_1m": args.price_output,
        }, ensure_ascii=False) + "\n")
    print(f"实验记录已追加: {log_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
