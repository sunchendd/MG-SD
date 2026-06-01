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
   - baseline 补齐后，当前 300 条 aggregate proxy 排序是：**Baseline < EARS < MG-SD**。

3. **机制证据上**
   - 已加 `p1 / p2 / margin` 日志。
   - 但当前统计 **不支持** “low-margin entity token 更集中、MG-SD 因此修正实体错误” 这个假设。

所以目前更准确的结论是：

> 小样本 pilot 出现过正向信号，但 300 条大样本 proxy 实验没有复现该优势；补齐 baseline 后，aggregate proxy 上甚至是 baseline 最低，因此当前还不能宣称“MG-SD 比 EARS 或 baseline 实体错误更少”。

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

### 6.3 300 样本同批吞吐补跑（2xL20，256 output）

这轮补跑已经完成，统一使用同一批 cleaned 300 样本，`avg_output_tokens=256`。

| 方法 | Output Throughput | Total Throughput | Avg Latency | Avg Output Tokens |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 37.68 tok/s | 368.12 tok/s | 6.79 s | 256.0 |
| EARS | 38.60 tok/s | 377.07 tok/s | 6.63 s | 256.0 |
| MG-SD (`δ=0.10`) | 38.34 tok/s | 374.49 tok/s | 6.68 s | 256.0 |

相对 Baseline：

- EARS：**+2.43% Output Throughput**，**+2.43% Avg Latency 改善**
- MG-SD (`δ=0.10`)：**+1.73% Output Throughput**，**+1.70% Avg Latency 改善**

结论：

- 这组已经补齐了 baseline / EARS / MG-SD 的完整同批对照。
- **EARS 是当前最稳的吞吐点**。
- MG-SD 仍高于 baseline，但相对 EARS 有小幅回退。
- 由于这组是**长 prompt + 单并发 + 端到端口径**，decode 侧收益被 prefill 稀释，所以增幅明显小于前面的短样本 sweep。

### 6.4 4xL20 长输出吞吐（Qwen3-8B draft，10k / 4k）

这轮是用户后续追加的长输出配置，已经完整跑完。

- Target：`/data/models/Qwen3-32B`
- Draft：`/data/models/Qwen3-8B`
- GPU：`0,1,3,4`（4xL20）
- `tensor_parallel_size=4`
- `--max-model-len 10000`
- `--block-size 16`
- `--max-tokens 4000`
- `temperature=0.9`
- `concurrency=4`
- 数据集：`/home/scd/MG-SD/entity_eval_large_300_clean/pilot_dataset.jsonl`

| 方法 | Output Throughput | Total Throughput | Avg Latency | Avg Output Tokens | Avg Draft Accept Rate* |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 183.56 tok/s | 281.48 tok/s | 77.91 s | 3584.17 | 64.83% |
| EARS | 188.56 tok/s | 290.89 tok/s | 74.48 s | 3522.95 | 67.12% |
| MG-SD (`δ=0.10`) | 186.99 tok/s | 287.70 tok/s | 75.64 s | 3550.19 | 66.83% |

相对 Baseline：

- EARS：**+2.72% Output Throughput**，**+3.34% Total Throughput**，**-4.40% Avg Latency**
- MG-SD (`δ=0.10`)：**+1.87% Output Throughput**，**+2.21% Total Throughput**，**-2.91% Avg Latency**

补充解释：

- EARS / MG-SD 的平均输出长度略短于 baseline（`3522.95` / `3550.19` vs `3584.17`），所以看 Avg Latency 时不能只按绝对值解读。
- 若按 `wall_clock / output_tokens` 归一化，三组约为：
  - Baseline：`21.74 ms/token`
  - EARS：`21.14 ms/token`
  - MG-SD：`21.31 ms/token`
- 归一化后仍然是 **EARS 最优，MG-SD 次之，baseline 最慢**。

\* `Avg Draft Accept Rate` 来自对应 `*_server.log` 中 `SpecDecoding metrics` 的全程平均值，不是 `evalscope perf` 口径。

### 6.5 v5 吞吐回退实验补充（8 prompt / 4k output）

这部分补充 v5 之后为追回 acceptance / throughput 做的几组回退实验。  
注意：这组和 6.4 的 300 样本长输出口径不同，更接近固定负载下的 decode 吞吐对比。

| 方法 | Output Throughput | Total Throughput | TPOT | Avg Draft Accept Rate |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 160.71 tok/s | 200.89 tok/s | 24.73 ms | - |
| EARS | 168.89 tok/s | 211.12 tok/s | 20.36 ms | 66.98% |
| MG-SD v5 | 108.37 tok/s | 135.47 tok/s | 27.22 ms | 58.10% |
| MG-SD v5 `risk_only` | 159.77 tok/s | 199.71 tok/s | 21.94 ms | 77.11% |
| MG-SD v5 `ratio075` | 133.53 tok/s | 166.91 tok/s | 23.05 ms | 70.62% |
| MG-SD soft sweep `A` | 146.10 tok/s | 182.62 tok/s | 23.35 ms | 70.43% |

相对 Baseline：

- MG-SD v5 `risk_only`：**-0.59% Output Throughput**
- MG-SD v5 `ratio075`：**-16.92%**
- MG-SD soft sweep `A`：**-9.09%**

相对 MG-SD v5：

- MG-SD v5 `risk_only`：**+47.42% Output Throughput**
- MG-SD v5 `ratio075`：**+23.21%**
- MG-SD soft sweep `A`：**+34.81%**

结论：

- `risk_only` 基本把吞吐追回到 baseline 附近，但它后面的 300 条 CEER 没有保住 v5 的质量优势。
- `ratio075` 和 `A` 都能把 acceptance 拉回去一截，但速度仍明显低于 baseline / EARS。
- 这几组结果说明：**全局放松 gate 确实能追回吞吐，但会把质量又推回到 EARS 附近甚至更差。**

### 6.6 v6 条件化 gate 吞吐补充

v6 的思路是：**风险 token 继续严卡，非风险 token 单独放宽**。  
当前已经补跑完两组吞吐结果：

| 方法 | Output Throughput | Total Throughput | TPOT | Avg Draft Accept Rate |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 160.71 tok/s | 200.89 tok/s | 24.73 ms | - |
| EARS | 168.89 tok/s | 211.12 tok/s | 20.36 ms | 66.98% |
| v6_b | 174.18 tok/s | 217.73 tok/s | 22.35 ms | 76.35% |
| v6_c | 177.18 tok/s | 221.48 tok/s | 20.76 ms | 82.36% |

相对 Baseline：

- v6_b：**+8.38% Output Throughput**
- v6_c：**+10.25%**

相对 EARS：

- v6_b：**+3.13% Output Throughput**
- v6_c：**+4.91%**

结论：

- **v6 已经把吞吐追回并超过 EARS**。
- 当前吞吐最好的点是 `v6_c`，同时 acceptance 也最高。
- 但这两组目前还**没有补 300 条 entity eval**，所以暂时只能说“速度最好”，还不能说“整体最好”。

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
| Baseline | 300 | 0.780 | 0.766 | 0.840 | 0.701 | 1.702 |
| EARS | 300 | 0.782 | 0.793 | 0.834 | 0.697 | 1.694 |
| MG-SD (`δ=0.10`) | 300 | 0.808 | 0.814 | 0.858 | 0.727 | 1.682 |
| MG-SD (`δ=0.05`) | 300 | 0.815 | 0.826 | 0.869 | 0.729 | 1.631 |

相对 Baseline：

- EARS：CEER **+0.25%**
- MG-SD (`δ=0.10`)：CEER **+3.52%**
- MG-SD (`δ=0.05`)：CEER **+4.42%**

相对 EARS：

- MG-SD (`δ=0.10`)：CEER **+3.27%**
- MG-SD (`δ=0.05`)：CEER **+4.16%**

结论：

- 300 条大样本 **没有复现** 12 条 pilot 的优势
- baseline 补齐后，当前 aggregate proxy 上是 **Baseline 最低、EARS 次之、MG-SD 最差**
- EARS 只在 **dose / frequency / negation** 三类上略好于 baseline，但 **medication error** 反而高于 baseline
- 两组 MG-SD 重跑后都比刚才更差，其中 `δ=0.10` 相比 EARS 在 300 条上是 **82 better / 82 worse / 136 tie**，`δ=0.05` 是 **69 better / 73 worse / 158 tie**

### 8.3 `Qwen3-8B` + draft-conditioned 公式补充结果

这部分补充 v5 以及后续几组回退实验的实体结果。

#### 8.3.1 300 条 full eval

| 方法 | Samples | CEER | Med Error | Dose Error | Freq Error | Negation Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 300 | 0.780 | 0.766 | 0.840 | 0.701 | 1.702 |
| EARS 0.20 + 8B + TP4 | 300 | 0.800 | 0.781 | 0.871 | 0.708 | 1.942 |
| MG-SD v5 | 300 | 0.789 | 0.767 | 0.850 | 0.697 | 2.155 |
| MG-SD v5 `risk_only` | 300 | 0.801 | 0.782 | 0.864 | 0.715 | 1.905 |

结论：

- **v5 已经明显优于旧 MG-SD 和同配置 EARS**，但还没有压过 baseline。
- `risk_only` 把 negation 从 `2.155` 拉回到 `1.905`，但 medication / dose / frequency 都回退，最终 CEER 又回到 **EARS 附近**。
- 这说明 v5 的收益主要来自 **正确修正 gate 对象**，不是单纯把 gate 放松或只在风险 token 上开洞。

#### 8.3.2 pilot100 快速筛选结果

| 方法 | Samples | CEER | Med Error | Dose Error | Freq Error | Negation Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 100 | 0.756 | 0.749 | 0.818 | 0.682 | 1.412 |
| EARS | 100 | 0.776 | 0.726 | 0.878 | 0.678 | 1.848 |
| MG-SD v5 | 100 | 0.782 | 0.733 | 0.861 | 0.685 | 2.364 |
| Soft sweep `A` | 100 | 0.792 | 0.719 | 0.874 | 0.702 | 2.394 |
| `ratio075` | 100 | 0.814 | 0.753 | 0.890 | 0.726 | 2.314 |

结论：

- `A` 相比 pilot100 上的 v5 **没有继续改善 CEER**。
- `ratio075` 在 pilot100 上是这几组里最差的一档，不适合作为默认放松方向。
- 这组 quick filter 的意义主要是筛掉明显不好的回退策略，不应替代 300 条 full eval。

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
| Baseline | 0.0212 | 0.0827 |
| EARS | 0.0235 | 0.0823 |
| MG-SD (`δ=0.10`) | 0.0223 | 0.0836 |
| MG-SD (`δ=0.05`) | 0.0229 | 0.0834 |

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
- v5 draft-conditioned gate 公式验证
- `risk_only` / `ratio075` / soft sweep 回退实验
- v6 条件化 gate 吞吐补跑

当前阶段可以更准确地概括为：

1. **旧 MG-SD 的根因已经定位清楚**
   - 问题不只是 draft 模型太弱
   - 而是 gate 看的对象不对
2. **v5 已经证明“公式修正有效”**
   - 300 条 full eval 上优于旧 MG-SD，也优于同配置 EARS
   - 但仍略差于 baseline
3. **v6 已经把吞吐追回到当前最好**
   - 目前吞吐已经超过 EARS
   - 但质量侧还缺最终补跑

### 10.2 当前阻塞

1. **v6 还缺 300 条 entity eval**
   - 现在已经能说“v6 吞吐最好”
   - 但还不能说“v6 同时兼顾质量和速度”

2. **简单放松 gate 的路线已经基本证伪**
   - `risk_only`、`ratio075`、soft sweep 都能把 acceptance 拉回去
   - 但 CEER 会回退到 EARS 附近甚至更差

3. **当前剩余短板仍集中在 negation / dose**
   - v5 虽然整体 CEER 改善
   - 但 negation 仍明显高于 baseline
   - 一旦继续放松 gate，dose / unit / negation 很容易重新恶化

4. **机制证据仍需要更强支撑**
   - 当前 `low-margin entity` 假设证据仍偏弱
   - 后续更适合围绕 risk token / draft-conditioned gate 叙事，而不是继续强调全局 margin story

## 11. 结果文件

- 主报告：`/home/scd/MG-SD/README.md`
- 中文总结：`/home/scd/MG-SD/experiment_summary.md`
- 温度 0：`/home/scd/MG-SD/evalscope/`
- 温度 0.9：`/home/scd/MG-SD/temp09/evalscope/`
- 12 条 entity pilot：`/home/scd/MG-SD/entity_eval/`
- 300 条大样本：`/home/scd/MG-SD/entity_eval_large_300/`
- v5 300 条结果：`/home/scd/MG-SD/entity_eval_large_300_mgsd020_qwen8b_tp4_v5/`
- v5 `risk_only` 300 条结果：`/home/scd/MG-SD/entity_eval_large_300_mgsd020_qwen8b_tp4_v5_risk_only/`
- `ratio075` pilot100：`/home/scd/MG-SD/entity_eval_ratio075_pilot100_20260526_092232/`
- v5 长输出吞吐：`/home/scd/MG-SD/throughput_compare_20260525_124016/`
- `risk_only` 吞吐：`/home/scd/MG-SD/throughput_risk_only_20260526_020252/`
- `ratio075` 吞吐：`/home/scd/MG-SD/throughput_ratio075_20260526_090302/`
- soft sweep：`/home/scd/MG-SD/soft_sweep_20260526_103027/`
- v6 吞吐：`/home/scd/MG-SD/v6_sweep_20260529_062343/`

## 12. 当前建议

如果目标是继续把论文结论做实，建议按下面顺序推进：

1. 先补 **v6_b / v6_c 的 300 条 entity eval**
2. 如果 v6 质量仍回退，就继续只调 **risk path**
   - 优先收紧 `negation / numeric / unit`
   - 不要再做全局放松
3. 扩充 **多 token 风险词覆盖**
   - 尤其是单位、频次、否定短语
4. 对 v6 做 **case-level 分析**
   - 看它追回吞吐时，具体放回来了哪些 token
   - 确认是不是主要集中在 non-risk token
5. 如果 proxy 仍不稳定，再考虑：
   - 更强的实体抽取器
   - 更严格的 medication / dose 子集
   - 更贴近临床风险的 gate 设计

当前最稳妥的表述是：

> MG-SD 已经在 v5 证明“公式修正能改善实体错误”，最新 v6 也已经把吞吐追回并超过 EARS；但在补完 v6 的 300 条 entity eval 之前，还不能宣称它同时优于 EARS 和 baseline。
