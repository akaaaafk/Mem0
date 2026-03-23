"""
mem0 LLM adapter for AWS Bedrock inference profile.
Wraps invoke_bedrock so mem0's Memory add phase uses Haiku (same profile as search).

Usage in local_memory_haiku.py:
    llm_config = LlmConfig(provider="bedrock_haiku", config={})
    (registered via monkey-patch in get_local_memory_haiku_config)
"""
import json
import os
from typing import Any, Dict, List, Optional

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase


DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_ACCOUNT_ID = os.environ.get("BEDROCK_ACCOUNT_ID", "...")
DEFAULT_INFERENCE_PROFILE_ID = os.environ.get("BEDROCK_INFERENCE_PROFILE_ID", "...")


def _build_arn(region: str, account_id: str, profile_id: str) -> str:
    return f"arn:aws:bedrock:{region}:{account_id}:application-inference-profile/{profile_id}"


class BedrockHaikuLLM(LLMBase):
    """
    mem0 LLM backend using AWS Bedrock inference profile (Claude Haiku).
    Implements generate_response() with system+user split (same as GAM claude_generator).
    No temperature passed (inference profile may not allow overrides).
    """

    def __init__(self, config: Optional[Any] = None):
        if config is None:
            config = BaseLlmConfig(model="bedrock-haiku-inference-profile")
        elif isinstance(config, dict):
            config = BaseLlmConfig(model=config.get("model", "bedrock-haiku-inference-profile"))
        super().__init__(config)
        self.region = os.environ.get("AWS_REGION", DEFAULT_REGION)
        self.account_id = os.environ.get("BEDROCK_ACCOUNT_ID", DEFAULT_ACCOUNT_ID)
        self.profile_id = os.environ.get("BEDROCK_INFERENCE_PROFILE_ID", DEFAULT_INFERENCE_PROFILE_ID)
        self.model_arn = _build_arn(self.region, self.account_id, self.profile_id)
        self.max_tokens = getattr(config, "max_tokens", None) or 1024

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Convert mem0 messages (list of {role, content}) to Bedrock system+user call.
        Ignores tools (not needed for memory extraction).
        Returns the text string mem0 expects.
        """
        import boto3

        system_parts = []
        user_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "") or ""
            if role == "system":
                system_parts.append(content)
            else:
                user_messages.append({"role": role, "content": content})

        if not user_messages:
            user_messages = [{"role": "user", "content": "\n".join(system_parts)}]
            system_parts = []

        body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "messages": user_messages,
        }
        # Match search_bedrock.invoke_bedrock: stable extraction for add phase
        try:
            body["temperature"] = float(os.environ.get("BEDROCK_TEMPERATURE", "0"))
        except (TypeError, ValueError):
            body["temperature"] = 0
        if system_parts:
            body["system"] = "\n".join(system_parts).strip()

        client = boto3.client("bedrock-runtime", region_name=self.region)
        resp = client.invoke_model(
            modelId=self.model_arn,
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
        return text
