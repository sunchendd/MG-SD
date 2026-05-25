# MG-SD v5 实体错误测试说明

## 1. 结论摘要

- 这轮有效改进的不是单纯把 draft 换成 `Qwen3-8B`，而是把 MG-SD gate 从 **`target top1-top2`** 改成了 **围绕当前 draft token 的风险判断**。
- 在 `4x L20 + Qwen3-32B target + Qwen3-8B draft + TP4`、300 条实体测试集上，MG-SD v5 的 CEER 从旧公式的 **0.812653** 降到 **0.788651**，绝对下降 **0.024002**。
- v5 也优于同配置 EARS `0.800384`，但仍略差于 baseline `0.780314`。
- 用共同窗口重算后排序不变：**baseline `0.789344` < v5 `0.800751` < EARS `0.813168` < 旧 MG-SD `0.820676`**。说明提升不是长度口径造成的，而是公式本身起作用了。
- 当前剩余主要短板是 **negation**，dose 也还没有压到 baseline 以下。

## 2. 新公式说明

### 2.1 旧公式为什么有问题

旧 MG-SD 的 gate 是：

```text
tol_ears = beta * (1 - p1)
tol_old  = tol_ears * 1[(p1 - p2) > delta]
```

其中：

- `p1` 是 target top-1 概率
- `p2` 是 target top-2 概率

这个 gate 控制的是 **target 分布前两名是否分得开**，而不是 **当前 draft token 本身是否值得放过**。  
因此会出现这种坏情况：

1. target 对正确 token 很自信，`p1 - p2` 很大
2. draft 给了错误 token
3. 旧公式仍然打开 tolerance，结果把错误 draft 放过去

### 2.2 v5 新公式

v5 先围绕 draft token 本身构造统计量：

```text
p_d   = p_target(d)
p_alt = max_{v != d} p_target(v)
m_d   = p_d - p_alt
r_d   = p_d / p1
```

然后把 tolerance 写成：

```text
tol_v5 =
  beta * (1 - p1)
  * sigmoid((m_d - delta) / tau_m)
  * sigmoid((r_d - rho) / tau_r)
  * risk_backoff
```

含义：

- `beta * (1 - p1)`：保留 EARS 的基础松弛项
- `sigmoid((m_d - delta) / tau_m)`：只有 draft token 自己接近可接受时才放松
- `sigmoid((r_d - rho) / tau_r)`：draft token 相对 top-1 的概率占比太低时继续压制
- `risk_backoff`：对高风险 token 类型做额外收紧

### 2.3 v5 实际使用参数

```text
base_tolerance = 0.20
margin_delta   = 0.00
soft_tau       = 0.03
draft_min_ratio= 0.85
ratio_tau      = 0.05
```

风险回退：

```text
negation  = 0.00
numeric   = 0.35
unit      = 0.35
frequency = 0.50
```

参考实现见：

- `/home/scd/ai-infra-tools/spec/ears/mgsd_formula.py`
- `/home/scd/ai-infra-tools/spec/ears/test_mgsd_formula.py`

## 3. 启动参数

### 3.1 服务配置

| 项目 | 参数 |
| --- | --- |
| 硬件 | 4 x L20 |
| CUDA_VISIBLE_DEVICES | `0,1,3,4` |
| target model | `/data/models/Qwen3-32B` |
| draft model | `/data/models/Qwen3-8B` |
| tensor parallel | `4` |
| speculative tokens | `5` |
| parallel drafting | `false` |
| max model len | `4096` |
| gpu memory util | `0.85` |
| eager | 未开启 `--enforce-eager` |
| port | `8000` |

启动命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 \
vllm serve /data/models/Qwen3-32B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --speculative-config '{"model":"/data/models/Qwen3-8B","method":"draft_model","num_speculative_tokens":5,"parallel_drafting":false}'
```

### 3.2 MG-SD v5 环境变量

```bash
export VLLM_EARS_BASE_TOLERANCE=0.20
export VLLM_MGSD_ENABLED=1
export VLLM_MGSD_MARGIN_DELTA=0.00
export VLLM_MGSD_SOFT_TAU=0.03
export VLLM_MGSD_DRAFT_MIN_RATIO=0.85
export VLLM_MGSD_RATIO_TAU=0.05
export VLLM_MGSD_NEGATION_BACKOFF=0.0
export VLLM_MGSD_NUMERIC_BACKOFF=0.35
export VLLM_MGSD_UNIT_BACKOFF=0.35
export VLLM_MGSD_FREQUENCY_BACKOFF=0.50
export VLLM_MGSD_NEGATION_TOKEN_IDS=537,902,2041,6857,14820,23101,46491
export VLLM_MGSD_NUMERIC_TOKEN_IDS=15,16,17,18,19,20,21,22,23,24
export VLLM_MGSD_UNIT_TOKEN_IDS=342,5651,8153,13742,15739,20697,47639
export VLLM_MGSD_FREQUENCY_TOKEN_IDS=7298,13112,14103,74760
```

### 3.3 评测生成参数

| 项目 | 参数 |
| --- | --- |
| dataset | `/home/scd/MG-SD/entity_eval_large_300/pilot_dataset.jsonl` |
| 样本数 | `300` |
| chunk size | `40` |
| server retries | `3` |
| sample retries | `3` |
| max tokens | `256` |
| temperature | `0.9` |
| seed | `123 + sample_idx` |
| logprobs | `2` |

## 4. 测试方法

1. 使用 `entity_eval_large_300/pilot_dataset.jsonl` 的 300 条 medication-heavy 样本。
2. 对比四组：
   - baseline
   - EARS 0.20 + `Qwen3-8B` + TP4
   - 旧 MG-SD 0.20 + `Qwen3-8B` + TP4
   - MG-SD v5 0.20 + `Qwen3-8B` + TP4
3. 每 40 条重启一次 vLLM 服务，单样本失败可自动重试，避免长跑中途污染后续结果。
4. 指标使用 `entity_eval.py`：
   - `CEER = (substitutions + deletions + insertions) / gold_entities`
   - 同时统计 medication / dose / frequency / negation 四类错误率
5. 由于当前聚合实现使用 `gold_window = gold[:len(output)]`，不同方法的 `gold_entities` 分母会有轻微差异，所以另外做了 **共同窗口重算**，验证排序是否稳定。

## 5. 测试结果

### 5.1 直接 summary 结果

| 方法 | CEER | Medication | Dose | Frequency | Negation |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.780314 | 0.766433 | 0.840254 | 0.701447 | 1.701923 |
| EARS 0.20 + 8B + TP4 | 0.800384 | 0.781190 | 0.870641 | 0.707531 | 1.941748 |
| 旧 MG-SD 0.20 + 8B + TP4 | 0.812653 | 0.781830 | 0.876774 | 0.728308 | 2.000000 |
| **MG-SD v5 0.20 + 8B + TP4** | **0.788651** | **0.767397** | **0.849925** | **0.697204** | **2.155340** |

### 5.2 v5 相对旧 MG-SD 的改善

| 指标 | 旧 MG-SD | v5 | 绝对改善 |
| --- | ---: | ---: | ---: |
| CEER | 0.812653 | 0.788651 | -0.024002 |
| Medication | 0.781830 | 0.767397 | -0.014433 |
| Dose | 0.876774 | 0.849925 | -0.026849 |
| Frequency | 0.728308 | 0.697204 | -0.031104 |
| Negation | 2.000000 | 2.155340 | +0.155340 |

误差计数也同步下降：

| 方法 | substitutions | deletions | insertions |
| --- | ---: | ---: | ---: |
| 旧 MG-SD | 2066 | 1904 | 1990 |
| v5 | 2048 | 1791 | 1915 |

### 5.3 共同窗口重算结果

| 方法 | CEER(common window) |
| --- | ---: |
| baseline | 0.789344 |
| **MG-SD v5** | **0.800751** |
| EARS 0.20 + 8B + TP4 | 0.813168 |
| 旧 MG-SD 0.20 + 8B + TP4 | 0.820676 |

这说明：

- v5 优于旧 MG-SD
- v5 也优于同配置 EARS
- 但 baseline 仍然最好

## 6. 结果解读

### 6.1 这次为什么有效

- 旧公式错在 gate 的对象不对：看的是 `p1 - p2`，不是 `p_target(draft)`
- v5 把 gate 绑定到 draft token 本身，再叠加 ratio gate 和 risk backoff
- 因此它主要减少了 **错误 draft 被高置信 top-1 误放行** 的情况

### 6.2 8B draft 有没有用

有用，但不是根因级修复。

- 单纯换成 `Qwen3-8B`，旧 MG-SD 仍然是 `0.812653`，说明 **8B 本身救不了旧公式**
- 把公式改成 v5 后才降到 `0.788651`，说明真正起作用的是 **gate 目标修正**
- 因此更准确的结论是：**8B 是必要的工程条件之一，但决定性因素是新公式**

### 6.3 还剩什么问题

- **negation** 仍然偏差最大，v5 比旧 MG-SD 还更差
- **dose** 虽然明显改善，但还没优于 baseline
- 当前 risk token 集仍是单 token 为主，像部分多 token 频次/单位表达还没完全覆盖

## 7. 关键结果文件

- baseline：
  - `/home/scd/MG-SD/entity_eval_large_300/baseline_summary.json`
- EARS 0.20 + 8B + TP4：
  - `/home/scd/MG-SD/entity_eval_large_300_ears020_qwen8b_tp4/ears-b0.20-qwen8b-tp4_summary.json`
- 旧 MG-SD 0.20 + 8B + TP4：
  - `/home/scd/MG-SD/entity_eval_large_300_mgsd020_qwen8b_tp4/mgsd-d0.20-qwen8b-tp4_summary.json`
- MG-SD v5：
  - `/home/scd/MG-SD/entity_eval_large_300_mgsd020_qwen8b_tp4_v5/mgsd-v5-d0.00-qwen8b-tp4_summary.json`

## 8. 当前结论

MG-SD v5 已经证明 **“公式优化有效”**：

1. 明显优于旧 MG-SD
2. 优于同配置 EARS
3. 公平口径下排序仍成立

但它还没有超过 baseline，所以如果继续优化，优先方向应放在：

1. negation / numeric / unit 的更强 backoff
2. 多 token 风险词覆盖
3. ratio gate 再收紧一档（例如更高的 `draft_min_ratio`）
