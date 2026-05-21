一、执行摘要
基于 EARS 投机解码工程管线，面向 AAAI 2027 投稿的压缩型实验方案。核心创新：在 EARS 的 $$1 - \max(P)$$ 置信度放松基础上，新增 Top-2 margin gate 作为安全门控，将加速任务从通用场景转向医疗安全感知场景。

---
二、从 EARS 到 MG-SD 的方法演进
EARS 基线（arXiv 2512.13194）
核心机制：$$Tol_i = \beta \cdot (1 - \max(P_{\text{target}}))$$
用目标模型的不置信度动态放宽接受阈值，解决 random rejection 问题。实验场景为通用任务（创意写作、OpenQA、GSM8K），吞吐提升最高 18.12%。
新方案：MG-SD Margin-Gated SD
核心公式：$$Tol_i = \beta \cdot (1 - p_{1,i}) \cdot \mathbf{1}[p_{1,i} - p_{2,i} > \delta]$$
- 保留 $$1 - p_1$$ 作为速度项（不确信时仍有加速空间）
- 新增 $$\mathbf{1}[p_1 - p_2 > \delta]$$ 作为安全门控项（低 margin 位置关闭 relaxation）
- 核心假设：低 $$p_1 - p_2$$ 的位置更可能是药物/剂量/否定词等临床实体碰撞
方法链
Standard SD → Confidence-Relaxed SD (EARS) → MG-SD hard gate → MG-SD soft gate (备选) → SafeSpec-Clinical (后续增强)

---
三、实验矩阵
维度
选择
说明
Target Model
Qwen3-32B
当前团队可跑通
Draft Model (主)
Qwen3-8B
同系列 tokenizer 一致
Draft Model (备)
Qwen3-4B
更小 draft 更易体现加速与实体风险
主数据集
MIMIC-IV-Note
Clinical note continuation 256-512 tokens
辅数据集
MedQA
USMLE 风格问答，验证医学推理保持
方法对比
方法
机制
用途
Target AR
纯自回归
质量参考和速度下界
Standard SD
标准接受规则
系统基础基线
Confidence-Relaxed SD
$$\beta \cdot (1-p_1)$$
证明纯 confidence 的风险
MG-SD
$$\beta \cdot (1-p_1) \cdot \mathbf{1}[p_1-p_2>\delta]$$
主方法

---
四、评估指标
效率指标
- Tokens/s：输出吞吐量
- Latency：端到端延迟
- Average Accepted Draft Length：每轮接受草稿长度
- Top-2 Overhead：额外 top-2 操作开销
医疗安全指标
- CEER：临床实体错误率（substitution + insertion + deletion）
- Med-CEER：药物实体错误率
- Dose/Number Error：剂量/数字/单位/频次错误
- Negation Error：否定词语义错误
- OCE：Overconfident Clinical Entity Error
医学推理指标
- Accuracy：MedQA 答案准确率
- Answer Flip Rate：相比 Standard SD 答案变化比例

---
五、6月初前执行排期
阶段
时间
任务
产出
第 0 阶段
立即-第 5 天
CITI 培训 + PhysioNet 申请；Qwen3-32B AR baseline 测试
数据审批提交；baseline 验证
第 1 阶段
第 1-3 天
EARS 代码加入 top-2 概率、margin、hard gate
MG-SD 可运行版本
第 2 阶段
第 4-7 天
MIMIC 300-500 条 pilot，三方法对比
CEER 初表；margin 分���初图
第 3 阶段
第 8-14 天
扩大 MIMIC 到 1000 条主实验
主结果表 + case studies
第 4 阶段
第 15-20 天
MedQA 实验，固定参数迁移
accuracy / answer flip 表
第 5 阶段
第 21-25 天
β/δ 消融 + overconfidence 诊断
Pareto 图；OCE 分析
第 6 阶段
第 26-30 天
整理图表，撰写论文素材
完整实验结果包
go/no-go 决策点
第 0 阶段数据审批和第 2 阶段 pilot 结果是两个关键决策点。pilot 如果 entity error 差异太小，直接调整方向，不上主实验。

---
六、风险与应对
风险
现象
应对
Margin 假设不成立
entity vs non-entity margin 分布差异不明显
降级为"候选碰撞风险控制"，补 clinical backoff
速度损失过大
tokens/s 接近 Standard SD
降低 δ 或切换到 soft gate
CEER 差异小
Confidence-Relaxed SD 未显著增加实体错误
换更小 draft / 更长生成长度 / 更高 temperature
创新性不足
审稿认为改动仅是 indicator gate
补充连续加权版本或 clinical knowledge 融合
Qwen3-32B 医学能力差
baseline 生成质量不行
换医学专用模型

---
七、论文图表规划
编号
图/表
作用
Table 1
MIMIC-IV-Note 主结果
核心效率+安全指标
Table 2
MedQA 结果
医学推理保持
Figure 1
Margin 分布图
entity vs non-entity 差异
Figure 2
Speed-safety Pareto
trade-off 最优性
Figure 3
Token-type acceptance rate
风险选择性保守
Figure 4
Case study
可解释性增强

---
八、关键行动项
[] 完成 CITI 培训并提交 PhysioNet 数据申请
[] Qwen3-32B 跑 MIMIC/MedQA AR baseline，验证生成长度
[] 在 rejection_sampler.py 中加入 top-2 概率提取和 margin gate
[] 跑 MIMIC 300-500 条 pilot，验证 EARS 实体安全问题是否成立
[] 确认 EEGPU 资源是否足够跑 32B + 多组实验
