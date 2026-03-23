"""
AWS Bedrock inference profile client for Claude Haiku.
Uses inference profile ARN with model anthropic.claude-haiku-4-5-20251001-v1:0.

Aligns with baseline general-agentic-memory-claude eval_haiku / gam.generator.claude_generator:
  InvokeModel, system in body["system"], user in body["messages"], no temperature.
"""
import json
import os


DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_ACCOUNT_ID = os.environ.get("BEDROCK_ACCOUNT_ID", "...")
DEFAULT_INFERENCE_PROFILE_ID = os.environ.get("BEDROCK_INFERENCE_PROFILE_ID", "...")


def build_inference_profile_arn(region: str, account_id: str, profile_id: str) -> str:
    """Inference profile ARN for InvokeModel modelId."""
    return f"arn:aws:bedrock:{region}:{account_id}:application-inference-profile/{profile_id}"


def invoke_bedrock(
    region: str,
    model_id: str,
    prompt: str,
    max_tokens: int = 1024,
    system: str | None = None,
) -> tuple[str, int, int]:
    """
    Call Bedrock with inference profile. Returns (text, input_tokens, output_tokens).
    model_id: full inference profile ARN from build_inference_profile_arn.
    Uses proper system + user roles. (Temperature not sent; some inference profiles do not allow it.)
    """
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    # Match Qwen pipeline: temperature=0 for deterministic answers (some inference profiles may not allow it)
    try:
        body["temperature"] = float(os.environ.get("BEDROCK_TEMPERATURE", "0"))
    except (TypeError, ValueError):
        body["temperature"] = 0
    if system and system.strip():
        body["system"] = system.strip()

    resp = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    ret = json.loads(resp["body"].read())
    text = ""
    content_list = ret.get("content")
    if content_list and isinstance(content_list, list):
        # Find first "text" block; skip "thinking" blocks (extended thinking)
        for block in content_list:
            if isinstance(block, dict) and block.get("type") == "text":
                text = (block.get("text", "") or "").strip()
                break
    usage = ret.get("usage", {}) if isinstance(ret.get("usage"), dict) else {}
    inp = usage.get("input_tokens", usage.get("inputTokens", 0))
    out_tok = usage.get("output_tokens", usage.get("outputTokens", 0))
    return text, inp, out_tok
