# MG-SD v5 新公式总结

## 1. 背景

MG-SD 原本想解决的问题，是在 speculative decoding 里避免把明显错误的 draft token 放过去，尤其是实体相关高风险 token。

旧版本的问题不在于 draft 模型太弱，而在于 **gate 的控制对象错了**。

## 2. 旧公式的问题

旧 MG-SD 的核心逻辑：

```text
tol_old = beta * (1 - p1) * 1[(p1 - p2) > delta]
```

其中：

- `p1`：target top-1 概率
- `p2`：target top-2 概率

这个公式看的是 **target top1 和 top2 的分离度**，但 rejection sampling 真正需要判断的是：

> 当前 draft token 自己，是否接近 target 可接受范围。

所以会出现错误放行：

1. target 对正确 token 很自信，`p1 - p2` 很大
2. draft 给的是错误 token
3. 旧公式仍然打开 tolerance
4. 最后把错误 draft 放过去

这就是旧公式的根因问题：**它没有绑定当前 draft token 本身。**

## 3. 新公式的核心思想

v5 不再看 `top1 - top2`，而是直接围绕 draft token `d` 本身建模。

先定义：

```text
p_d   = p_target(d)
p_alt = max_{v != d} p_target(v)
m_d   = p_d - p_alt
r_d   = p_d / p1
```

含义：

- `p_d`：target 对当前 draft token 的概率
- `p_alt`：除了 draft 之外，target 最强竞争项的概率
- `m_d`：draft token 相对最强替代项的 margin
- `r_d`：draft token 相对 top-1 的概率占比

## 4. v5 新公式

```text
tol_v5 =
  beta * (1 - p1)
  * sigmoid((m_d - delta) / tau_m)
  * sigmoid((r_d - rho) / tau_r)
  * risk_backoff
```

可以理解为四层控制：

1. **EARS 基础松弛项**
   - `beta * (1 - p1)`
   - target 越不自信，才越允许放松

2. **margin gate**
   - `sigmoid((m_d - delta) / tau_m)`
   - draft token 自己必须接近 target 可接受边界，才放开

3. **ratio gate**
   - `sigmoid((r_d - rho) / tau_r)`
   - 如果 draft token 相对 top-1 太弱，再压一次

4. **risk backoff**
   - 对高风险 token 类型额外收紧
   - 包括 negation、数字、单位、频次

## 5. 这次实际使用参数

```text
base_tolerance  = 0.20
margin_delta    = 0.00
soft_tau        = 0.03
draft_min_ratio = 0.85
ratio_tau       = 0.05
```

风险回退：

```text
negation  = 0.00
numeric   = 0.35
unit      = 0.35
frequency = 0.50
```

## 6. 新公式为什么有效

新公式本质上把 MG-SD 从：

> “看 target top1-top2 gap”

改成了：

> “看当前 draft token 自己是不是值得被放过”

这样做之后，可以明显减少下面这类问题：

- target 明明已经认定 draft 错了
- 但旧公式因为 `top1-top2` gap 大，仍然给 tolerance

v5 把这种错误放行压住了，所以实体错误显著改善。

## 7. 实测效果

在 **4xL20 / Qwen3-32B target / Qwen3-8B draft / 300 条 entity eval** 上：

| 方法 | CEER |
| --- | ---: |
| baseline | 0.780314 |
| EARS 0.20 + 8B | 0.800384 |
| 旧 MG-SD 0.20 + 8B | 0.812653 |
| **MG-SD v5** | **0.788651** |

结论：

- **明显优于旧 MG-SD**
- **优于同配置 EARS**
- **还没超过 baseline**

所以这个新公式已经证明：

> **方向是对的，公式修正是真有效的。**

## 8. 当前代价

在长输出吞吐测试里：

| 方法 | Output Throughput (tok/s) | TPOT (ms) | Spec Accept Rate |
| --- | ---: | ---: | ---: |
| baseline | 160.71 | 24.73 | - |
| EARS 0.20 + 8B | 168.89 | 20.36 | 66.98% |
| **MG-SD v5** | **108.37** | **27.22** | **58.10%** |

这说明：

- v5 虽然把实体错误修好了
- 但 gate 收紧后 acceptance 下降
- decode 阶段吞吐被明显拖慢

所以它现在的状态是：

> **精度更好，但性能退化。**

## 9. 一句话结论

**MG-SD v5 的核心改进，是把 gate 从“看 target top1-top2 gap”改成“看 draft token 本身是否接近可接受”。**

它已经证明能有效降低实体错误，但下一步还要继续把 acceptance rate 拉回去，才能兼顾吞吐和安全性。

## 10. 相关文件

- 实体测试说明：`/home/scd/MG-SD/MG-SD-v5-entity-eval-summary.md`
- 吞吐结果目录：`/home/scd/MG-SD/throughput_compare_20260525_124016`
- 参考公式实现：`/home/scd/ai-infra-tools/spec/ears/mgsd_formula.py`
- 参考测试：`/home/scd/ai-infra-tools/spec/ears/test_mgsd_formula.py`
