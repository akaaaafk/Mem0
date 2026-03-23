# eval_qwen

使用 **Tinker** 作为 backbone、**Qwen/Qwen3-30B-A3B-Instruct-2507** 作为模型，跑 mem0 评估。  
输出与 reference（如 locomo_10_samples_with_stats）一致：**time、log、f1、bleu、花费**。

## 环境

1. 复制 `.env.example` 为 `.env`，填写 **Tinker** 配置（或保留示例中的 key 若可用）：
   - `TINKER_BASE_URL`：Tinker API 地址（OpenAI 兼容）
   - `TINKER_API_KEY`：Tinker API Key（如 `tml-5TT...`）
   - `MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507`
   - 若已用 `OPENAI_BASE_URL` / `OPENAI_API_KEY` 也会被识别（兼容旧配置）。

2. 在 **mem0_1** 目录下执行（不要进 eval_qwen 再跑）。每个数据集有独立脚本，结果写入各自子目录：

   | 数据集 | 脚本 | 结果目录 |
   |--------|------|----------|
   | LoCoMo | `python -m eval_qwen.run_locomo` | `eval_qwen/results/locomo/` |
   | LongMemEval-S | `python -m eval_qwen.run_longmem_s` | `eval_qwen/results/longmem_s/` |
   | HotpotQA | `python -m eval_qwen.run_hotpotqa` | `eval_qwen/results/hotpotqa/` |
   | RULER 128K | `python -m eval_qwen.run_ruler_128k` | `eval_qwen/results/ruler_128k/` |

   ```bash
   python -m eval_qwen.run_locomo
   python -m eval_qwen.run_locomo --max_questions 50
   # 兼容旧用法（内部调用 run_locomo）：
   python -m eval_qwen.run_full_eval
   ```

3. 各结果目录内：`experiment_log.jsonl`（time）、`batch_statistics_*.json`（token/花费）、`batch_results_*.json`（f1/bleu1），控制台打印汇总。

## 评估项

| 评估 | 状态 | 脚本 | 结果目录 |
|------|------|------|----------|
| **LoCoMo** | 已接入 | `run_locomo.py` | `results/locomo/` |
| **LongMemEval-S** (follow LightMem) | 待接入 | `run_longmem_s.py` | `results/longmem_s/` |
| **HotpotQA** (follow GAM) | 待接入 | `run_hotpotqa.py` | `results/hotpotqa/` |
| **RULER (128K)** (follow GAM) | 待接入 | `run_ruler_128k.py` | `results/ruler_128k/` |

全量入口（可勾选要跑的数据集）：

```bash
python -m eval_qwen.run_all_evals                    # 默认只跑 LoCoMo
python -m eval_qwen.run_all_evals --no-locomo --longmem --hotpotqa --ruler
```

## 需要改的地方（如需与 reference 完全一致）

- **evaluation** 已显式支持 **Tinker**：`evaluation` 内优先读 `TINKER_BASE_URL`、`TINKER_API_KEY`，未设时再读 `OPENAI_BASE_URL`、`OPENAI_API_KEY`。
- **单价**：`evaluation/run_experiments.py` 里 search 阶段费用估算使用 `unit_in=3.0`, `unit_out=15.0`（$/1M tokens）。若 Tinker/Qwen 有不同定价，可在 run_experiments 中改为从环境变量读取，或在 eval_qwen 的 .env 中增加并在调用前传入。
- **LongMemEval-S / HotpotQA / RULER**：在对应 `run_longmem_s.py`、`run_hotpotqa.py`、`run_ruler_128k.py` 中接入数据与流程，输出格式（time、log、f1、bleu、花费）与 LoCoMo 保持一致即可。
