# mem0 Evaluation

mem0 memory framework evaluation on the LoCoMo benchmark, supporting two inference backends:

- **Tinker / Qwen** (`eval_qwen/`) — OpenAI-compatible API, local Qdrant vector store
- **AWS Bedrock / Claude Haiku** (`evaluation/haiku/`) — Application Inference Profile

---

## Project Structure

```
mem0/
├── eval_qwen/              # Tinker + Qwen3 evaluation entry point
├── evaluation/
│   ├── haiku/              # Bedrock + Claude Haiku evaluation
│   ├── src/memzero/        # Core mem0 add/search logic
│   ├── metrics/            # F1 / BLEU scoring
│   ├── data/locomo/        # LoCoMo dataset
│   └── run_experiments.py  # Main experiment runner
├── mem0/                   # mem0 Python package (core library)
└── scratch/                # Standalone prototype scripts (archived)
```

---

## Setup

```bash
cd mem0
pip install -e .
pip install qdrant-client sentence-transformers python-dotenv
```

---

## Tinker / Qwen Backend (`eval_qwen/`)

Copy and fill in credentials:

```bash
cp eval_qwen/.env.example eval_qwen/.env
# Set TINKER_BASE_URL and TINKER_API_KEY in eval_qwen/.env
```

Run from the `mem0/` directory:

```bash
# LoCoMo full eval (add + search)
python -m eval_qwen.run_locomo

# Limit to N questions for quick test
python -m eval_qwen.run_locomo --max_questions 50

# All benchmarks (LoCoMo wired; others are stubs)
python -m eval_qwen.run_all_evals
```

Results are written to `eval_qwen/results/locomo/`:
- `batch_statistics_locomo.json` — tokens, cost, F1, BLEU
- `batch_results_locomo.json` — per-question results
- `experiment_log.jsonl` — run metadata

---

## AWS Bedrock / Claude Haiku Backend (`evaluation/haiku/`)

Requires AWS credentials with `bedrock:InvokeModel` permission.

Set environment variables (or use `aws configure`):

```bash
export AWS_REGION=us-east-1
export BEDROCK_ACCOUNT_ID=...
export BEDROCK_INFERENCE_PROFILE_ID=...
```

Run from the `mem0/` directory:

```bash
# Add with Qwen/Tinker, search/answer with Bedrock Haiku
python -m evaluation.haiku.run_locomo_haiku

# Both add and search use Bedrock Haiku
python -m evaluation.haiku.run_locomo_haiku --full_haiku

# Quick test — 10 questions
python -m evaluation.haiku.run_locomo_haiku --max_questions 10

# Re-run search only (reuse existing vector store)
python -m evaluation.haiku.run_locomo_haiku --search_only
```

Results are written to `evaluation/haiku/results/locomo_haiku/`.

---

## Standard Evaluation Runner (`evaluation/run_experiments.py`)

Used internally by `eval_qwen/` and `evaluation/haiku/`. Can also be called directly:

```bash
python -m evaluation.run_experiments \
  --technique_type mem0 \
  --method add_then_search \
  --score \
  --output_folder ./results/locomo \
  --batch_id locomo
```

| Flag | Description |
|---|---|
| `--technique_type` | `mem0`, `rag`, `langmem`, `zep` |
| `--method` | `add`, `search`, `add_then_search` |
| `--score` | Compute F1/BLEU after search |
| `--max_questions` | Limit number of questions |
| `--top_k` | Memories to retrieve per query (default 30) |

---

## Scoring

```bash
# Score an existing results folder
python -m evaluation.score_results --input_folder ./results/locomo

# LoCoMo full score breakdown (F1 by category)
python -m evaluation.run_experiments --method search --score --output_folder ./results/locomo
```

---

## Scratch Scripts (`scratch/`)

Standalone prototype scripts kept for reference. **Not part of the main pipeline.**

| Script | Description |
|---|---|
| `eval_locomo.py` | Bedrock InvokeModel direct eval from pre-built memory files |
| `locomo_eval_bedrock.py` | Bedrock Converse API eval from pre-built memory files |
| `eval_locomo_ollama_full.py` | Full local pipeline — Ollama LLM + embeddings + FAISS, no mem0 |
| `eval_locomo_add_local_eval_bedrock.py` | Hybrid — local Ollama add, Bedrock search/answer |
| `eval_locomo_mem0_full.py` | Full mem0 pipeline with Bedrock (add + search via mem0 library) |

Run from the repo root (`e:\mem0\`):

```bash
python mem0/scratch/locomo_eval_bedrock.py --dataset mem0/evaluation/data/locomo/locomo10.json
python mem0/scratch/eval_locomo_ollama_full.py --method both --max-samples 1 --max-questions 5
```
