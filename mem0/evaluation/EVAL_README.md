# mem0 Evaluation (LoCoMo, LongMemEval-S, HotpotQA 224k, RULER 128k)

Backbone: **tinker Qwen/Qwen3-30B-A3B-Instruct-2507**. Graph memory is disabled (default).

## Environment

- `OPENROUTER_BASE_URL`, `OPENROUTER_API_KEY` for the backbone (or `OPENAI_BASE_URL` / `OPENAI_API_KEY`, or legacy `TINKER_*`).
- `MODEL` defaults to `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- `EMBEDDER_MODEL` (default `BAAI/bge-m3`), `EMBEDDER_DIMS` (default `1024`).
- Optional: `UNIT_PRICE_INPUT_PER_1M`, `UNIT_PRICE_OUTPUT_PER_1M` for cost.

Copy `.env.example` to `.env` in this directory and fill in keys.

## Datasets and outputs

| Dataset          | Run script                 | Metrics                          | Common outputs                    |
|-----------------|----------------------------|----------------------------------|-----------------------------------|
| **LoCoMo**      | `python -m evaluation.run_locomo` | f1, bleu-1                       | running_time, log, context_window_peak, cost |
| **LongMemEval-S** | `python -m evaluation.run_longmem_s` | acc                              | same                              |
| **HotpotQA 224k** | `python -m evaluation.run_hotpotqa`  | f1                               | same                              |
| **RULER 128k** | `python -m evaluation.run_ruler_128k` | retri.acc, MT acc, ACG acc, QA acc | same                              |

All runs write:

- `results/<dataset>/mem0_results.json` – raw answers
- `results/<dataset>/batch_statistics_<id>.json` – metrics + token/cost/running_time/context_window_peak
- `results/<dataset>/experiment_log.jsonl` – log

## Data paths

- LoCoMo: `evaluation/locomo/locomo10.json`
- LongMemEval-S: `evaluation/longmem_s/longmem_s.json`
- HotpotQA 224k: `evaluation/hotpotqa_224k/hotpotqa_224k.json`
- RULER 128k: `evaluation/ruler_128k/ruler_128k.json`

Use `--data_path` to override. Data format for add_then_search: list of `{ "conversation": { "speaker_a", "speaker_b", ... }, "qa": [ { "question", "answer", "category"?, "task_type"? }, ... ] }`.

## Run from repo root (mem0)

```bash
cd mem0
python -m evaluation.run_locomo
python -m evaluation.run_locomo --max_questions 50
python -m evaluation.run_longmem_s
python -m evaluation.run_hotpotqa
python -m evaluation.run_ruler_128k
```
