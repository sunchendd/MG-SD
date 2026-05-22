# MG-SD / EARS 中文实验报告

日期：2026-05-22

## 1. 当前结论

这轮实验已经完成了 **GPU 版 vLLM 上 EARS / MG-SD 的实现、吞吐测试、药物段实体错误 pilot、300 条大样本实体错误实验**。

当前最重要的结论有 3 条：

1. **吞吐上**
   - `temperature=0.9` 时，EARS 和 MG-SD 都明显优于 baseline。
   - 在已完成的小规模吞吐测试里，EARS 是当前最稳的速度点。
   - MG-SD 仍保留相对 baseline 的吞吐收益，但相对 EARS 会有一定回退。

2. **实体错误上**
   - 12 条 pilot 曾出现正向信号：**MG-SD 优于 EARS**。
   - 但 300 条 medication-heavy 大样本实验没有复现这个优势。
   - 当前 300 条结果中，**EARS 略优于 MG-SD**。

3. **机制证据上**
   - 已加 `p1 / p2 / margin` 日志。
   - 但当前统计 **不支持** “low-margin entity token 更集中、MG-SD 因此修正实体错误” 这个假设。

所以目前更准确的结论是：

> 小样本 pilot 出现过正向信号，但 300 条大样本 proxy 实验没有复现，当前还不能直接宣称“MG-SD 比 EARS 实体错误更少”。

## 2. 实验环境

- 仓库目录：`/home/scd/ai-infra-tools`
- 实验结果目录：`/home/scd/MG-SD`
- 数据集：`/home/scd/mimic-iv-note/note/discharge.csv.gz`
- Target model：`/data/models/Qwen3-32B`
- Draft model：`/data/models/Qwen3-0.6B`
- 推理框架：本机已修改的 GPU 版 vLLM
- 可见 GPU：
  - L20：`0,1,3,4`
  - RTX 4090：`2,5`
- 本轮主要使用：**L20 GPU 0,1**
- 服务配置：`tensor_parallel_size=2`

## 3. 方法定义

- **Baseline**：标准 speculative decoding，不启用 EARS / MG-SD
- **EARS**：开启 `VLLM_EARS_BASE_TOLERANCE=0.1`
- **MG-SD**：在 EARS 基础上增加 margin gate
  - `VLLM_MGSD_ENABLED=1`
  - 主参数：`VLLM_MGSD_MARGIN_DELTA=0.10`
  - 对照参数：`0.05`、`0.20`

## 4. 服务启动方法

### 4.1 Baseline

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
unset VLLM_EARS_BASE_TOLERANCE VLLM_MGSD_ENABLED VLLM_MGSD_MARGIN_DELTA

vllm serve /data/models/Qwen3-32B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --speculative-config '{"model":"/data/models/Qwen3-0.6B","method":"draft_model","num_speculative_tokens":5,"parallel_drafting":false}'
```

### 4.2 EARS

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_EARS_BASE_TOLERANCE=0.1
unset VLLM_MGSD_ENABLED VLLM_MGSD_MARGIN_DELTA

vllm serve /data/models/Qwen3-32B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --speculative-config '{"model":"/data/models/Qwen3-0.6B","method":"draft_model","num_speculative_tokens":5,"parallel_drafting":false}'
```

### 4.3 MG-SD

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_EARS_BASE_TOLERANCE=0.1
export VLLM_MGSD_ENABLED=1
export VLLM_MGSD_MARGIN_DELTA=0.10

vllm serve /data/models/Qwen3-32B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --speculative-config '{"model":"/data/models/Qwen3-0.6B","method":"draft_model","num_speculative_tokens":5,"parallel_drafting":false}'
```

## 5. 测试方法

### 5.1 `temperature=0` smoke test

用途：确认实现正确、服务正常、不会破坏标准路径。

```bash
evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --tokenizer-path /data/models/Qwen3-32B \
  --dataset line_by_line \
  --dataset-path /home/scd/MG-SD/mimic_evalscope_messages.jsonl \
  --number 20 \
  --parallel 1 \
  --max-tokens 256 \
  --temperature 0.0 \
  --stream
```

### 5.2 `temperature=0.9` 吞吐对比

用途：在更高随机性下观察 speculative decoding 收益，并测试 `delta` 的合理性。

```bash
evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --tokenizer-path /data/models/Qwen3-32B \
  --dataset line_by_line \
  --dataset-path /home/scd/MG-SD/mimic_evalscope_messages.jsonl \
  --number 10 \
  --parallel 1 \
  --max-tokens 512 \
  --temperature 0.9 \
  --seed 123 \
  --stream
```

### 5.3 medication-section entity pilot

用途：比较 **Baseline / EARS / MG-SD** 在药物段续写中的实体错误 proxy。

```bash
cd /home/scd/MG-SD
python3 run_entity_eval_pilot.py --sample-count 12 --max-tokens 256 --temperature 0.9
```

### 5.4 300 条 medication-heavy 大样本实验

用途：扩大小样本 pilot，检查结论是否稳定，并记录 `p1 / p2 / margin`。

```bash
cd /home/scd/MG-SD
python3 run_entity_eval_pilot.py \
  --sample-count 300 \
  --max-tokens 256 \
  --temperature 0.9 \
  --logprobs 2 \
  --modes ears,mgsd-d0.10,mgsd-d0.05 \
  --output-dir /home/scd/MG-SD/entity_eval_large_300
```

## 6. 吞吐结果

### 6.1 `temperature=0` 结果

| 方法 | Output Throughput | Total Throughput | TPOT | Avg Latency | Accept Rate | Decoded Tok/Iter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 36.07 tok/s | 70.60 tok/s | 27.28 ms | 7.09 s | 0.59 | 2.44 |
| EARS | 36.30 tok/s | 71.04 tok/s | 27.15 ms | 7.05 s | 0.59 | 2.44 |
| MG-SD (`δ=0.10`) | 35.74 tok/s | 69.95 tok/s | 27.57 ms | 7.16 s | 0.59 | 2.44 |

结论：

- 这组更像 **sanity check**
- 三组结果非常接近
- `temperature=0` 不足以有效拉开 EARS 和 MG-SD

### 6.2 `temperature=0.9` 结果

| 方法 | Output Throughput | Total Throughput | TPOT | Avg Latency | Accept Rate | Decoded Tok/Iter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 30.53 tok/s | 45.65 tok/s | 32.53 ms | 16.76 s | 0.54 | 2.18 |
| EARS | 33.13 tok/s | 49.53 tok/s | 29.97 ms | 15.45 s | 0.56 | 2.29 |
| MG-SD (`δ=0.05`) | 32.78 tok/s | 49.01 tok/s | 30.28 ms | 15.61 s | 0.57 | 2.30 |
| MG-SD (`δ=0.10`) | 31.87 tok/s | 47.65 tok/s | 31.14 ms | 16.05 s | 0.56 | 2.28 |
| MG-SD (`δ=0.20`) | 32.24 tok/s | 48.20 tok/s | 30.80 ms | 15.88 s | 0.56 | 2.26 |

相对 Baseline：

- EARS：**+8.5% Output Throughput**
- MG-SD (`δ=0.05`)：**+7.4%**
- MG-SD (`δ=0.10`)：**+4.4%**
- MG-SD (`δ=0.20`)：**+5.6%**

结论：

- **速度优先默认**：`δ=0.05`
- **安全实验默认**：`δ=0.10`
- `δ=0.20` 暂时没有表现出明显优势

### 6.3 300 样本同批吞吐进展

目前 300 样本同批吞吐里：

- EARS / MG-SD 的 server log 已经有可提取结果
- baseline 同批补跑仍未完成

现有 log 末尾可读到的 `Avg generation throughput`：

| 方法 | 300 样本 log 中可提取的 Output Throughput |
| --- | ---: |
| EARS | 35.3 tok/s |
| MG-SD (`δ=0.10`) | 35.4 tok/s |
| MG-SD (`δ=0.05`) | 26.1 tok/s |

说明：

- 这组还不是最终的 baseline / EARS / MG-SD 完整对照表
- baseline 补跑失败的直接原因已定位为 **prompt truncation 没接上，触发 4096 context 上限**

## 7. 实体错误评估口径

当前“实体错误”不是人工标注金标准，而是 **regex + lexicon 的 proxy**。

### 7.1 数据来源与切分

- 数据源：`/home/scd/mimic-iv-note/note/discharge.csv.gz`
- 优先对齐：
  - `Discharge Medications:`
  - `Medications on discharge:`
- 切分方式：
  - note 前半段作为 `prompt`
  - 后半段药物段作为 `gold`
- 实际打分：
  - 使用 `gold_window = gold[:len(output)]`
  - 避免因输出更短而把未生成的 gold 尾部都算成 deletion

### 7.2 实体类别

当前统计 4 类：

1. **medications**
   - 命中手工药名字典 `MEDICATION_LEXICON`
2. **doses**
   - 正则抽取，例如 `5 mg`、`10 ml`、`2 units`
3. **frequencies**
   - 正则抽取，例如 `daily`、`bid`、`tid`、`q6h`、`prn`
4. **negations**
   - 正则抽取，例如 `no`、`not`、`denies`、`without`

### 7.3 样本筛选条件

只保留 **medication-heavy** 样本。

在 `gold[:800]` 里至少满足：

- 药名 `>= 2`
- 剂量 `>= 2`
- 频次 + 否定 `>= 1`

### 7.4 打分方式

对 `gold_window` 和 `output` 分别抽实体 Counter，然后按类别比较：

- **substitution**：gold 和输出都有，但词项不一致
- **deletion**：gold 有，输出缺失
- **insertion**：gold 没有，输出多出来

最后得到：

- `ceer = (substitutions + deletions + insertions) / gold_entities`
- `medication_error_rate = medication_errors / gold_medications`
- `dose_error_rate = dose_errors / gold_doses`
- `frequency_error_rate = frequency_errors / gold_frequencies`
- `negation_error_rate = negation_errors / gold_negations`

所以当前结果更准确的说法是：

> 基于药物段续写的 clinical entity fidelity proxy，不是最终人工标注 CEER 金标准。

## 8. 实体错误结果

### 8.1 12 条 pilot

| 方法 | Samples | Gold Entities | CEER | Med Error | Dose Error | Freq Error | Negation Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 12 | 300 | 0.773 | 0.943 | 0.859 | 0.639 | 1.125 |
| EARS | 12 | 285 | 0.698 | 0.755 | 0.852 | 0.559 | 1.000 |
| MG-SD (`δ=0.10`) | 12 | 275 | 0.647 | 0.654 | 0.795 | 0.523 | 1.125 |

MG-SD 相对 EARS：

- **CEER**：`0.698 -> 0.647`，**-7.3%**
- **Medication error**：`0.755 -> 0.654`，**-13.4%**
- **Dose error**：`0.852 -> 0.795`，**-6.7%**
- **Frequency error**：`0.559 -> 0.523`，**-6.5%**

结论：

- 小样本 pilot 有正向信号
- 说明 MG-SD 曾经表现出“压住药物/剂量漂移”的潜力

### 8.2 300 条 medication-heavy 大样本

| 方法 | Samples | CEER | Med Error | Dose Error | Freq Error | Negation Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EARS | 300 | 0.783 | 0.793 | 0.837 | 0.697 | 1.694 |
| MG-SD (`δ=0.10`) | 300 | 0.796 | 0.801 | 0.851 | 0.710 | 1.705 |
| MG-SD (`δ=0.05`) | 300 | 0.795 | 0.803 | 0.849 | 0.710 | 1.743 |

相对 EARS：

- MG-SD (`δ=0.10`)：CEER **+1.58%**
- MG-SD (`δ=0.05`)：CEER **+1.55%**

结论：

- 300 条大样本 **没有复现** 12 条 pilot 的优势
- 当前 aggregate proxy 上，**EARS 略优于 MG-SD**

## 9. `p1 / p2 / margin` 机制分析

这部分是**机制指标**，不是实体错误本身。

### 9.1 计算方式

1. 调用 completions API 时设置 `logprobs=2`
2. 对每个输出 token 读取 top-2 logprob
3. 转成概率：
   - `p1 = exp(top1_logprob)`
   - `p2 = exp(top2_logprob)`
4. 计算：
   - `margin = p1 - p2`
5. 再把 token 标成：
   - entity token
   - non-entity token

最终统计：

- `low_margin_entity_rate`
- `low_margin_non_entity_rate`
- `mean_entity_margin`
- `mean_non_entity_margin`

### 9.2 300 样本结果

| 方法 | Low-margin entity rate | Low-margin non-entity rate |
| --- | ---: | ---: |
| EARS | 0.0374 | 0.0933 |
| MG-SD (`δ=0.10`) | 0.0387 | 0.0923 |
| MG-SD (`δ=0.05`) | 0.0388 | 0.0923 |

结论：

- 当前 low-margin token 更多落在 **non-entity token**
- 没有看到“entity token 更 low-margin”的预期趋势
- 因此当前机制证据 **不支持** low-margin entity 假设

## 10. 当前进展与阻塞点

### 10.1 已完成

- GPU 版 MG-SD / EARS 代码实现
- `temperature=0` smoke test
- `temperature=0.9` baseline / EARS / MG-SD 吞吐 sweep
- 12 条 medication-section entity pilot
- 300 条大样本 entity experiment
- `p1 / p2 / margin` 日志链路
- 实体错误 proxy 说明文档

### 10.2 当前阻塞

1. **300 样本 baseline 同批吞吐还没补齐**
   - 失败原因已定位：
   - 某些样本 prompt 过长
   - 补跑脚本没有复用 prompt truncation 逻辑
   - 触发 `4096` context 上限

2. **小样本和大样本结论冲突**
   - 12 条 pilot 是正向
   - 300 条大样本是反向
   - 当前不能直接得出论文结论

3. **机制假设未被数据支持**
   - `low_margin_entity_rate` 没有高于 `low_margin_non_entity_rate`
   - 说明当前 margin-gate story 证据不足

## 11. 结果文件

- 主报告：`/home/scd/MG-SD/README.md`
- 中文总结：`/home/scd/MG-SD/experiment_summary.md`
- 温度 0：`/home/scd/MG-SD/evalscope/`
- 温度 0.9：`/home/scd/MG-SD/temp09/evalscope/`
- 12 条 entity pilot：`/home/scd/MG-SD/entity_eval/`
- 300 条大样本：`/home/scd/MG-SD/entity_eval_large_300/`

## 12. 当前建议

如果目标是继续把论文结论做实，建议按下面顺序推进：

1. 先补齐 **300 样本 baseline 同批吞吐**
2. 对 300 条结果做 **case-level 分析**
3. 复查当前 **proxy 是否足够反映真实 clinical entity risk**
4. 如果 proxy 仍不稳定，再考虑：
   - 更强的实体抽取器
   - 更严格的 medication/dose 子集
   - 更贴近临床风险的 gate 设计

当前最稳妥的表述是：

> MG-SD 在吞吐上仍保留相对 baseline 的收益，但在当前 300 条 proxy 实验中，没有证明其相对 EARS 的实体错误优势。
