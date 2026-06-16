# 第二轮同行评审 — 修改稿评估

**论文**: Wind Farm Cooperative Yaw Control Based on Deep Reinforcement Learning with a Gray-Box Wake Model
**审稿日期**: 2026-06-10（第二轮）
**审稿类型**: Re-review — 针对第一轮 Major Revision 后的修改稿

---

## 修改评估概要

第一轮评审的核心关切有三：(1) 稳态-动态 trade-off 的叙事框架（从 limitation → discovery），(2) 配置命名的统一性，(3) 论证的连贯性。修改稿在这三方面均有显著改进。

| 第一轮关切 | 修改情况 | 评估 |
|-----------|---------|------|
| Trade-off 框架化为 limitation | 摘要重构，trade-off 明确列为 discovery，E1 实验证据支撑 | ✅ 已解决 |
| 配置名称混乱 | 新增 Table config_names（Config-A 至 Config-DW） | ✅ 已解决 |
| 摘要论证顺序颠倒 | 重构为 DRL 闭环→稳态→trade-off→fatigue-aware | ✅ 已解决 |
| 缺少工业基线讨论 | 新增 Section 4.4.5（LP 滤波+滞后控制） | ✅ 已解决 |
| Regime 阈值敏感性 | 新增 Section 3.2.3 段落 | ✅ 已解决 |
| 文献补充 | **未处理** — Doekemeijer, Kanev 等仍未引用 | ⚠️ 仍需处理 |
| Bootstrap CI 双类型 | cross-seed 90.5% 已在摘要和正文中提及 | ✅ 部分解决 |

---

## 逐项评估

### 摘要（重大改进）

修改后的摘要采用清晰的四段递进结构：
1. **问题+方法**：wake losses → closed-loop PPO + gray-box model
2. **动态优势（新首位）**：266× travel reduction, actuator-feasible rates — 这是 DRL 的核心价值主张
3. **稳态性能**：+4.91%, 94.9% recovery
4. **Trade-off 发现 + fatigue-aware 解决方案**：68× over-reaction 实验证据 → rate penalty 必要性

**评估**：叙事顺序正确，核心信息突出。一个细节：摘要中 "the rate-unpenalized policy over-reacts (travel increases $68{\times}$)" 可以被误读为这是论文的主要实验发现——建议加 "when trained without rate penalties under dynamic wind" 澄清这是 E1 诊断实验的发现。

### Introduction 贡献列表（改进）

Item 3 现在明确将 trade-off 框架化为 discovery：
> "We discover a steady-state vs. dynamic responsiveness trade-off...dynamic-wind retraining without rate penalties causes the policy to over-react (yaw travel increases 68×), confirming that the rate penalty is a necessary rather than optional component"

**评估**：这是一个有力的表述。6 项贡献中，item 3（fatigue-aware）和 item 4（dynamic wind protocol）现在构成了一个连贯的论证弧线。

### 配置命名表（新增，Table config_names）

Config-A 到 Config-DW 的六层渐进定义清晰。一个建议：Config-A 的描述中仍使用 "p0c"（这是内部 tag），建议统一为 "baseline"。

### 动态风重训结果（新增，Section 4.4.4）

> "the dynamic-wind-trained policy over-reacts to wind fluctuations: cumulative yaw travel increases 68× relative to the static-trained policy (2059° vs. 29.7°)"

**评估**：这是修改稿中最有力的新增内容。它不仅验证了 trade-off 的存在，更重要的是将论证从"我们发现了一个问题"升级为"我们发现了一个问题并通过实验验证了其机制"。E1 实验的设计（不加 rate penalty 的动态风训练）精确地分离出了 rate penalty 的因果效应。

建议：补充一个一句话的解释，说明为什么 2059° travel 会导致 gain 更负（"the policy chases wind fluctuations without achieving the steady yaw angles needed for cooperative wake deflection"）。

### 工业基线分析（新增，Section 4.4.5）

> "Adding the low-pass filter to the 0.1°/s rate-limited lookup table changes the gain by less than 0.05 pp and the travel by less than 3%"

> "applying the 5° hysteresis deadband to the DRL policy reduces its reported cumulative yaw travel by approximately 54%"

**评估**：这两个发现直接回应了魔鬼代言人的两个核心挑战——(a) 查表基线是公平的（LP 滤波不影响结论），(b) DRL 报告的 travel 是保守上界（滞后控制使其减半）。

建议：54% travel reduction 的数字来自 N_TRAJ=100 的定性实验，建议标注为 "estimated" 或基于 N_TRAJ=1000 的完整实验更新。

### Regime 敏感性分析（新增，Section 3.2.3 段落）

> "The recovery rate remains stable at ~94.9% for |φ−270°|∈[10°,20°] and v_max∈[10.0,11.4] m/s"

**评估**：简洁且有说服力。直接回应了 "cherry-picking" 的质疑。v≥12.0 m/s 时 recovery 跌至 86.9% 的解释（rated-power clipping artifact）与论文其他地方一致。

### 结论（重写）

新的结论分为三段：(1) 框架和稳态结果，(2) fatigue-aware 框架和 trade-off 发现，(3) 补充发现和未来工作。

**评估**：结构清晰，trade-off 不再是"局限"而是核心贡献的一部分。未来工作第一条改为 "train with rate penalties under dynamic wind"，比原版的 "train directly on dynamic wind" 更精确。

---

## 仍需修改的问题

### W1（新）：文献补充未完成 — Minor

Doekemeijer et al. (2019, WES) 的 closed-loop wake steering 和 Kanev et al. (2020, WES) 的 active wake control 实验仍未引用。这是第一轮 R2 明确要求的修改（R4）。

**建议**：在 Introduction 第二段末尾添加：
> "Between the offline-optimization and pure-DRL extremes, Doekemeijer et al. (2019) demonstrated closed-loop wake steering using a simplified steady-state surrogate model with online adaptation, and Kanev et al. (2020) reported field-validation results for active wake control."

工作量：~30 分钟。

### W2（新）：Config-A 命名不一致 — Minor

Section 4.4.3 仍使用 `\textsc{p0c}` 而非 `\textsc{Config-A}`。建议全文统一使用配置命名表中的标签。

### W3（新）：E2 实验样本量 — Minor

Section 4.4.5 的工业基线分析基于 N_TRAJ=100（定性实验）。54% travel reduction 数字可能需要标注置信度或说明为 "estimated from a pilot experiment"。

---

## 编辑决定

### Accept（建议）

修改稿已实质性解决了第一轮评审中提出的所有核心关切。叙事框架从 limitation → discovery 的转变是成功的，摘要和结论的重构显著提升了论文的论证清晰度。新增的实验证据（E1 动态风重训、E2 工业基线、S2 regime sensitivity）不仅加强了论证，更重要的是展示了作者对审稿意见的深入理解和建设性回应。

剩余问题（文献补充、命名一致性、E2 样本量标注）都是 Minor 级别，可在最终校对中解决，无需再次送审。

**建议的最终修改**（可在 proof 阶段完成）：
1. 引用 Doekemeijer2019 和 Kanev2020（~30 分钟）
2. 将 Section 4.4.3 的 `\textsc{p0c}` 替换为 `\textsc{Config-A}`
3. 在 Section 4.4.5 的 54% travel reduction 数字后标注 "(pilot experiment, $n{=}100$ trajectories)"

---

## 维度评分

| 维度 | 第一轮 | 第二轮 | 变化 |
|------|--------|--------|------|
| 原创性 (20%) | 78 | **84** | +6 — trade-off discovery + E1 实验证据增强了原创性 |
| 方法论严谨性 (25%) | 72 | **80** | +8 — E1/E2 补充实验 + S2 敏感性分析填补了方法论空白 |
| 证据充分性 (25%) | 82 | **85** | +3 — 新增实验进一步巩固了论证 |
| 论证连贯性 (15%) | 62 | **80** | +18 — 叙事重构+配置命名表解决了第一轮的核心问题 |
| 写作质量 (15%) | 75 | **78** | +3 — 摘要和结论显著改善 |
| **加权平均** | **74.6** | **81.5** | **+6.9** |
| **决策** | Major Revision | **Accept** | — |
