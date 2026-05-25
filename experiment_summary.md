# MG-SD 实验总结

日期：2026-05-21

## 1. 实验硬件

- 当前机器可见 GPU：
  - L20: GPU `0,1,3,4`
  - RTX 4090: GPU `2,5`
- 本轮正式实验实际使用：**2 张 L20（GPU 0,1）**
- 服务配置：`tensor_parallel_size=2`

## 2. 模型与权重

- Target model: `/data/models/Qwen3-32B`
- Draft model: `/data/models/Qwen3-0.6B`
- 推理框架：已安装 vLLM（本轮使用本机已修改的 GPU rejection sampler）

## 3. 方法定义

- **Baseline**：标准 speculative decoding，不开启 EARS / MG-SD
- **EARS**：开启 `VLLM_EARS_BASE_TOLERANCE=0.1`
- **MG-SD**：在 EARS 基础上增加 margin gate
  - 主设置：`VLLM_MGSD_ENABLED=1`
  - 主实验默认：`VLLM_MGSD_MARGIN_DELTA=0.10`

## 4. 启动方法

### Baseline

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

### EARS

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

### MG-SD

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

### 5.1 temperature=0 smoke test

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

### 5.2 temperature=0.9 吞吐对比

用途：在更高随机性下观察 speculative decoding 收益，并测试 `delta` 合理性。

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

### 5.3 medication-section entity-error pilot

用途：直接比较 **Baseline / EARS / MG-SD** 在药物段续写中的实体错误趋势。

- 数据：12 条 MIMIC discharge notes
- 切分：优先对齐 `Discharge Medications:` 附近
- 生成：`temperature=0.9`, `max_tokens=256`
- 打分：regex-based CEER proxy
  - medications
  - doses
  - frequencies
  - negations

```bash
cd /home/scd/MG-SD
python3 run_entity_eval_pilot.py --sample-count 12 --max-tokens 256 --temperature 0.9
```

### 5.4 当前“实体错误”是怎么评估的

当前这套实体错误评估，**不是人工标注金标准**，而是一套 **regex + lexicon 的 proxy**。

#### 数据来源与切分

- 数据源：`/home/scd/mimic-iv-note/note/discharge.csv.gz`
- 优先对齐：
  - `Discharge Medications:`
  - `Medications on discharge:`
- 切分方式：
  - note 前半段作为 `prompt`
  - 后续药物段作为 `gold`
- 实际打分时，为避免输出比 gold 短导致大量伪 deletion：
  - 使用 `gold_window = gold[:len(output)]`

#### 实体类别

当前只统计 4 类：

1. **medications**
   - 来自手工药名字典 `MEDICATION_LEXICON`
2. **doses**
   - 由正则抽取，例如 `5 mg`、`10 ml`、`2 units`
3. **frequencies**
   - 由正则抽取，例如 `daily`、`bid`、`tid`、`q6h`、`prn`
4. **negations**
   - 由正则抽取，例如 `no`、`not`、`denies`、`without`

#### 样本筛选条件

不是随机抽样，而是先筛 **medication-heavy** continuation。

在 `gold[:800]` 内至少满足：

- 药名 `>= 2`
- 剂量 `>= 2`
- 频次 + 否定 `>= 1`

#### 打分方式

对 `gold_window` 和 `output` 分别抽实体 Counter，然后按类别比较：

- **substitution**：gold 和输出都有，但词项不一致
- **deletion**：gold 有，输出缺失
- **insertion**：gold 没有，输出多出来

最后汇总为：

- `ceer = (substitutions + deletions + insertions) / gold_entities`
- `medication_error_rate = medication_errors / gold_medications`
- `dose_error_rate = dose_errors / gold_doses`
- `frequency_error_rate = frequency_errors / gold_frequencies`
- `negation_error_rate = negation_errors / gold_negations`

所以当前“实体错误”更准确的说法是：

> 基于药物段续写的 clinical entity fidelity proxy，而不是最终人工标注 CEER 金标准。

### 5.5 当前 `p1 / p2 / margin` 是怎么来的

这部分是**机制分析指标**，不是实体错误本身。

来源：

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

它的作用是验证：

> MG-SD 拦下来的位置，是否真的集中在 low-margin entity tokens 上。

## 6. 测试结果

### 6.1 temperature=0 smoke test

| Method | Output Throughput | Total Throughput | TPOT | Avg Latency | Accept Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 36.07 tok/s | 70.60 tok/s | 27.28 ms | 7.09 s | 0.59 |
| EARS | 36.30 tok/s | 71.04 tok/s | 27.15 ms | 7.05 s | 0.59 |
| MG-SD (`δ=0.10`) | 35.74 tok/s | 69.95 tok/s | 27.57 ms | 7.16 s | 0.59 |

结论：这组更像 **sanity check**。三组非常接近，说明 `temperature=0` 不能有效拉开 EARS 和 MG-SD。

### 6.2 temperature=0.9 吞吐对比

| Method | Output Throughput | Total Throughput | TPOT | Avg Latency | Accept Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 30.53 tok/s | 45.65 tok/s | 32.53 ms | 16.76 s | 0.54 |
| EARS | 33.13 tok/s | 49.53 tok/s | 29.97 ms | 15.45 s | 0.56 |
| MG-SD (`δ=0.05`) | 32.78 tok/s | 49.01 tok/s | 30.28 ms | 15.61 s | 0.57 |
| MG-SD (`δ=0.10`) | 31.87 tok/s | 47.65 tok/s | 31.14 ms | 16.05 s | 0.56 |
| MG-SD (`δ=0.20`) | 32.24 tok/s | 48.20 tok/s | 30.80 ms | 15.88 s | 0.56 |

对比结论：

- EARS vs Baseline：**+8.5% Output Throughput**
- MG-SD (`δ=0.05`) vs Baseline：**+7.4%**
- MG-SD (`δ=0.10`) vs Baseline：**+4.4%**
- MG-SD (`δ=0.20`) vs Baseline：**+5.6%**

参数判断：

- **速度优先**：`δ=0.05`
- **安全实验默认**：`δ=0.10`
- `δ=0.20` 暂时没有明显优于 `δ=0.10`

### 6.3 300 样本同批吞吐补跑（2xL20，256 output）

这轮补跑已经完成，baseline / EARS / MG-SD 三组完整对照如下：

| Method | Output Throughput | Total Throughput | Avg Latency | Avg Output Tokens |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 37.68 tok/s | 368.12 tok/s | 6.79 s | 256.0 |
| EARS | 38.60 tok/s | 377.07 tok/s | 6.63 s | 256.0 |
| MG-SD (`δ=0.10`) | 38.34 tok/s | 374.49 tok/s | 6.68 s | 256.0 |

对比结论：

- EARS vs Baseline：**+2.43% Output Throughput**
- MG-SD (`δ=0.10`) vs Baseline：**+1.73%**
- 这组是**长 prompt + 单并发 + 端到端口径**，所以 speculative decode 收益被 prefill 稀释，EARS / MG-SD 的增益都比短样本 sweep 更小。

### 6.4 4xL20 长输出吞吐（Qwen3-8B draft，10k / 4k）

这轮长输出 benchmark 已跑完，启动参数已在 `*_server.log` 中确认：
- target：`/data/models/Qwen3-32B`
- draft：`/data/models/Qwen3-8B`
- `tensor_parallel_size=4`
- `--max-model-len 10000`
- `--block-size 16`
- `--max-tokens 4000`
- `temperature=0.9`
- `concurrency=4`

| Method | Output Throughput | Total Throughput | Avg Latency | Avg Output Tokens | Avg Draft Accept Rate* |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 183.56 tok/s | 281.48 tok/s | 77.91 s | 3584.17 | 64.83% |
| EARS | 188.56 tok/s | 290.89 tok/s | 74.48 s | 3522.95 | 67.12% |
| MG-SD (`δ=0.10`) | 186.99 tok/s | 287.70 tok/s | 75.64 s | 3550.19 | 66.83% |

对比结论：

- EARS vs Baseline：**+2.72% Output Throughput**，**+3.34% Total Throughput**，**-4.40% Avg Latency**
- MG-SD (`δ=0.10`) vs Baseline：**+1.87% Output Throughput**，**+2.21% Total Throughput**，**-2.91% Avg Latency**
- MG-SD 相比 EARS 仍有小幅回退（Output Throughput **-0.83%**）。
- 虽然 EARS / MG-SD 的平均输出长度略短于 baseline，但按 `wall_clock / output_tokens` 归一化后，EARS 仍然最佳，MG-SD 次之。

\* `Avg Draft Accept Rate` 来自对应 `*_server.log` 中 `SpecDecoding metrics` 的全程平均值，不是 `evalscope perf` 口径。

### 6.5 medication-section entity-error pilot

| Method | Samples | Gold Entities | CEER | Med Error | Dose Error | Freq Error | Negation Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 12 | 300 | 0.773 | 0.943 | 0.859 | 0.639 | 1.125 |
| EARS | 12 | 285 | 0.698 | 0.755 | 0.852 | 0.559 | 1.000 |
| MG-SD (`δ=0.10`) | 12 | 275 | 0.647 | 0.654 | 0.795 | 0.523 | 1.125 |

MG-SD vs EARS：

- **CEER proxy**：`0.698 -> 0.647`，**-7.3%**
- **Medication error**：`0.755 -> 0.654`，**-13.4%**
- **Dose error**：`0.852 -> 0.795`，**-6.7%**
- **Frequency error**：`0.559 -> 0.523`，**-6.5%**
- **Negation**：这轮没有改善

补充观察：

- 12 条样本里：MG-SD 优于 EARS `4` 条，差于 EARS `3` 条，持平 `5` 条
- 代表性 case：`10000032-DS-21`
  - baseline CEER: `0.476`
  - EARS CEER: `1.190`
  - MG-SD CEER: `0.476`
  - 该例中 EARS 出现明显额外药物/剂量插入，MG-SD 把这类漂移压住了

## 7. 测试结论

1. **工程层面**
   - GPU 版 MG-SD 已经跑通
   - `2xL20` 的 300 样本补跑和 `4xL20 + Qwen3-8B draft + 10k/4k` 长输出 benchmark 都已完成
   - 当前吞吐关系很稳定：**EARS > MG-SD > baseline**

2. **实验层面**
   - 测试链路本身是合理的：先吞吐，再 entity eval，再做 `δ` ablation 和 `margin` 分析
   - 但当前实体错误结论已经从“pilot 正向”变成了“300 条大样本未复现”
   - baseline 补齐后，aggregate proxy 排序进一步变成 **Baseline < EARS < MG-SD**
   - 因此现阶段不能把 MG-SD 写成“已证明优于 EARS”

3. **当前证据强度**
   - 吞吐证据已经够稳定
   - 但安全性证据不足，`margin gate` 的机制假设也没有被当前统计支持
   - 更准确的表述应是：**MG-SD 目前保留了相对 baseline 的加速收益，但尚未证明其相对 EARS 的实体错误优势；补齐 baseline 后，aggregate proxy 上甚至不是最优**

## 8. 测试方向是否正确

我的判断：**方向正确，但还需要把证据链补完整。**

原因：

1. **主问题对了**
   - 你要证明的不是“MG-SD 更快”，而是“MG-SD 相比 EARS 在保留大部分加速收益的同时，减少实体错误”
   - 现在的三层实验结构正好对应这个目标：
     - smoke test：确认实现和路径正确
     - throughput test：确认不会退化成没有收益
     - entity pilot：直接看安全收益

2. **数据切法对了**
   - 把 continuation 对齐到 `Discharge Medications:` 附近，比随机 note continuation 更容易测出药物/剂量/频次错误
   - 这个方向比单纯整段续写更接近“实体错误”核心问题

3. **下一步该怎么补强**
   - 扩到 **300-500 条** medication-heavy 样本
   - 保持 `temperature=0.9`
   - 主参数固定 `δ=0.10`，`δ=0.05` 作为 ablation
   - 增加 **`p1 / p2 / margin`** 日志
   - 证明：MG-SD 改善的那些错误位置，确实更集中在 **low-margin entity tokens**

如果做到这一步，测试方向就不仅是“正确”，而且会形成更完整的论文证据链。

## 9. 结果文件

- 总报告：`/home/scd/MG-SD/README.md`
- 当前正式总结：`/home/scd/MG-SD/experiment_summary.md`
- entity pilot 数据：`/home/scd/MG-SD/entity_eval/`

## 10. 300 条 medication-heavy 大样本实验

### 设置

- 样本数：`300`
- 温度：`0.9`
- 输出长度：`256`
- 方法：
  - `Baseline`
  - `EARS`
  - `MG-SD δ=0.10`
  - `MG-SD δ=0.05`
- 额外记录：
  - completion `logprobs`
  - `p1 / p2 / margin`
  - entity-token vs non-entity-token 的 low-margin 比例

### 大样本结果

| Method | Samples | CEER | Med Error | Dose Error | Freq Error | Negation Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 300 | 0.780 | 0.766 | 0.840 | 0.701 | 1.702 |
| EARS | 300 | 0.782 | 0.793 | 0.834 | 0.697 | 1.694 |
| MG-SD (`δ=0.10`) | 300 | 0.808 | 0.814 | 0.858 | 0.727 | 1.682 |
| MG-SD (`δ=0.05`) | 300 | 0.815 | 0.826 | 0.869 | 0.729 | 1.631 |

### 对比结论

#### EARS vs Baseline

- CEER：**+0.25%**（更差）
- Med Error：**+3.41%**
- Dose Error：**-0.74%**
- Freq Error：**-0.64%**
- Negation Error：**-0.44%**

#### MG-SD (`δ=0.10`) vs EARS

- CEER：**+3.27%**（更差）
- Med Error：**+2.73%**
- Dose Error：**+2.89%**
- Freq Error：**+4.27%**
- Negation Error：**-0.75%**

#### MG-SD (`δ=0.05`) vs EARS

- CEER：**+4.16%**（更差）
- Med Error：**+4.27%**
- Dose Error：**+4.17%**
- Freq Error：**+4.65%**
- Negation Error：**-3.77%**

### Margin 机制证据

当前结果**不支持**“low-margin entity tokens 更集中、MG-SD 因此修正实体错误”这个假设。

#### EARS

- entity low-margin rate: `0.0235`
- non-entity low-margin rate: `0.0823`

#### MG-SD (`δ=0.10`)

- entity low-margin rate: `0.0223`
- non-entity low-margin rate: `0.0836`

#### MG-SD (`δ=0.05`)

- entity low-margin rate: `0.0229`
- non-entity low-margin rate: `0.0834`

解释：

1. 在当前这套 proxy 标注和 continuation 设置下，**low-margin token 更多出现在 non-entity tokens，而不是 entity tokens**。
2. MG-SD 与 EARS 的 low-margin entity rate 差别也很小，没有形成强机制分离。
3. 因此，这轮 300 条实验没有给出支持性机制证据，反而更接近**反证**。

### Note-level 稳定性

#### EARS vs Baseline

- better: `102`
- worse: `114`
- same: `84`

#### MG-SD (`δ=0.10`) vs EARS

- better: `82`
- worse: `82`
- same: `136`

#### MG-SD (`δ=0.05`) vs EARS

- better: `69`
- worse: `73`
- same: `158`

这说明：

- baseline 补齐后可以看到：**EARS 并不是 aggregate proxy 最优，只是在部分类别上略优于 baseline**
- `δ=0.10` 并不是完全失效，它在部分样本上确实有改进
- 但总体均值上没有赢 EARS，而且这次重跑后差距比上一版更大
- 当前收益不稳定，方向还不够成立

## 11. 对“测试方向是否正确”的更新判断

**实验方法本身是合理的，但当前假设在这套设置下没有被数据支持。**

更具体地说：

1. **测试链路是对的**
   - 先做吞吐
   - 再做 medication-heavy entity eval
   - 再做 `δ` ablation
   - 再加 `p1 / p2 / margin`
   这条方法链路没有问题。

2. **但当前核心论点没有站住**
   - 12 条 pilot 给过正向信号
   - 300 条大样本结果反而显示 MG-SD 略差于 EARS
   - margin 统计也没有证明 entity tokens 更 low-margin

3. **所以当前结论应该调整**
   - 不能再说“当前实验已经证明 MG-SD 比 EARS 实体错误更少”
   - 更准确的说法是：
     > 在小样本 pilot 中观察到正向信号，但 300 条大样本 proxy 实验未复现该优势，当前假设需要重新审视。

4. **后续建议**
   - 优先检查当前 proxy 是否足够反映真实 clinical entity risk
   - 补真正的 token-level 对齐与人工 case 复核
   - 考虑把分析对象从“所有 entity token”缩到“药物/剂量密集错误位置”
   - 若仍无支持性证据，需要考虑：
     - `margin gate` 不是有效主因
     - 需要改成更强的 clinical backoff / token-type-aware gate
