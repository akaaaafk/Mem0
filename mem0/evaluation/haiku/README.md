# Haiku – LoCoMo with AWS Bedrock (Claude Haiku)

LoCoMo evaluation using **AWS Bedrock** as the answer model: inference profile `s4ovcks6u9cd` (memory-claude-sonnet) with model `anthropic.claude-haiku-4-5-20251001-v1:0`, region `us-east-1`.

- **Add (default)**: Uses OpenRouter/Qwen via `evaluation.run_experiments --method add` (same as `run_locomo`).
- **Add (`--full_haiku`)**: Uses **Bedrock Haiku** for memory extraction too (`bedrock_llm.BedrockHaikuLLM` injected into mem0's LlmFactory). Same inference profile as search.
- **Search**: Vector search with mem0 local client; answer step uses **Bedrock** `InvokeModel` with the inference profile ARN.

## Alignment with baseline (general-agentic-memory-claude eval_haiku)

Usage matches the baseline’s Haiku/Bedrock pattern:

- **Inference profile**: Same `account_id=920736616554`, `inference_profile_id=s4ovcks6u9cd`, `region=us-east-1` as in `longmemeval_s_run.py`, `longmemevals_run.py`, `locomo_test.py`, `hotpotqa_run_bedrock.py`, etc.
- **API**: `InvokeModel` (not Converse), ARN `arn:aws:bedrock:{region}:{account_id}:application-inference-profile/{profile_id}`.
- **Request body**: `anthropic_version: "bedrock-2023-05-31"`, `max_tokens`, `messages` (user content as `[{ "type": "text", "text": "..." }]`), and **`system`** for instructions + context (same as GAM `gam/generator/claude_generator.py`: system as top-level `body["system"]`).
- **Temperature**: Search + add (`bedrock_llm`) send `temperature` (default **0**, override with `BEDROCK_TEMPERATURE`). If your inference profile rejects it, set `BEDROCK_TEMPERATURE` empty or patch client — some profiles disallow overrides.

## Usage (from mem0 repo root)

```bash
# Default: add with Qwen/OpenRouter, search with Haiku/Bedrock
python -m evaluation.haiku.run_locomo_haiku

# Full Haiku: BOTH add (memory extraction) AND search use Bedrock Haiku
python -m evaluation.haiku.run_locomo_haiku --full_haiku

# Quick test (10 questions)
python -m evaluation.haiku.run_locomo_haiku --max_questions 10 --full_haiku

# Search only (re-use existing .qdrant_run)
python -m evaluation.haiku.run_locomo_haiku --search_only
```

## Environment

- **AWS**: Credentials for Bedrock (e.g. `AWS_PROFILE` or `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`).
- Optional: `AWS_REGION` (default `us-east-1`), `BEDROCK_ACCOUNT_ID`, `BEDROCK_INFERENCE_PROFILE_ID` (default `s4ovcks6u9cd`).
- Default add phase uses existing evaluation env (`OPENROUTER_*`, `MEM0_DIR`, etc.).
- `--full_haiku` add phase uses only AWS credentials (no OpenRouter needed).

### Tuning scores (F1 / BLEU)

- **`LOCOMO_TOP_K`**: default **30** (same as `run_experiments --top_k`). Raise (e.g. 45–80) if you want more retrieval context (may help or add noise). CLI `--top_k` overrides.
- **`LOCOMO_JSON_ANSWER`**: default **0** — same user message + `_extract_short_answer` style as Qwen `search.py`. Set to `1` for experimental JSON `{"answer":"..."}`.
- **`LOCOMO_BEDROCK_MAX_TOKENS`**: max output tokens (default **1024** if JSON off, **384** if JSON on).

## Output

- `evaluation/haiku/results/locomo_haiku/mem0_results.json` – per-question results.
- `evaluation/haiku/results/locomo_haiku/batch_statistics_locomo.json` – tokens, cost, time.
- Score summary (F1, BLEU-1, accuracy) via `evaluation/score_results.py`.
