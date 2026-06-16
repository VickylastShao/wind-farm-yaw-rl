# 同行评议报告

**稿件:** Wind Farm Cooperative Yaw Control Based on Deep Reinforcement Learning with a Gray-Box Wake Model  
**目标期刊:** Applied Energy  
**审稿模式:** Full (5-reviewer panel: EIC + 3 Peer Reviewers + Devil's Advocate)  
**审稿日期:** 2026-06-11

---

## 审稿人配置卡 (Reviewer Configuration Card)

| # | 角色 | 专业领域 | 审稿视角 |
|---|------|---------|---------|
| 1 | **EIC** | Applied Energy 主编, 风能系统与可再生能源并网 | 期刊适配性、原创性、整体质量、读者吸引力 |
| 2 | **Reviewer 1 (Methodology)** | 风电场尾流建模专家, CFD/LES仿真, 大气边界层物理 | 研究方法严谨性、物理模型有效性、参数校准合理性 |
| 3 | **Reviewer 2 (Domain)** | DRL与风电场优化控制专家, 在线学习与控制理论 | RL算法选择、实验设计、奖励函数设计、基线对比公平性 |
| 4 | **Reviewer 3 (Perspective)** | 风电行业工程实践专家, SCADA系统与运维 | 工程实用性、可部署性、工业可行性、成本效益分析 |
| 5 | **Devil's Advocate** | 科学方法论批判专家 | 核心论点挑战、逻辑漏洞检测、替代解释、过度泛化检测 |

---

# Phase 1: 独立审稿报告

---

## 报告 1: Editor-in-Chief (EIC)

### 总体评价

**评分: 6.8/10**

### 期刊适配性

本稿件针对Applied Energy的核心读者群——关注可再生能源系统优化和运营效率的研究者——具有较好的适配性。将深度强化学习应用于风电场协同偏航控制是一个及时且具有工程价值的方向。然而，以下问题需要在修改中解决：

### 原创性与重要性

**优势：**
1. 提出的疲劳感知DRL框架（fatigue-aware DRL framework）将偏航速率惩罚 λ_rate 作为可调参数，形成功率-疲劳Pareto前沿，这是一个有原创性的贡献。
2. 动态风评估协议（AR(1)轨迹）在方法论上是合理的，对社区具有参考价值。
3. 系统的消融实验（observation, reward, training dynamics, action bounds）展示了各组件贡献。

**关切：**
1. **主要贡献的新颖性需更精确界定。** Gray-box wake model的校准方法（多场景最小二乘拟合）本质上是参数估计的标准实践；Bastankhah-Porté-Agel公式是既有的；PPO是标准算法。论文的核心创新应清晰定位在"疲劳感知DRL框架的系统性工程"而非单个组件。
2. **与Kanev et al. (2020)和Doekemeijer et al. (2019)的对比不够深入。** 这两篇工作都涉及闭环尾流控制，Kanev还报告了现场验证的约束偏航速率。Introduction已提及这些工作，但Discussion部分没有与它们进行详细的实验对比。建议在Sec. 4中增加一个比较子节。
3. **2.8×改善的表述需要附加条件。** 这是相对于baseline PPO配置的改善，而非相对于SOTA方法（如Doekemeijer的闭环Kalman滤波方法）。标题中的"2.8×"如不附加限定容易引起误解。

### 结构与呈现

1. **过度使用脚注式说明。** 多处以小字脚注声明"此实验使用baseline PPO配置"（如Sec. 3.2.4, Sec. 4.2.1）。这暗示论文的架构有根本性问题：baseline和optimized配置的结果混合呈现，读者难以跟踪哪个结论属于哪个配置。建议将baseline配置的结果移至附录，正文仅报告optimized配置；或者为两组实验分别设立清晰的子节。
2. **Abstract过长且信息密度过高。** 当前abstract包含10+个具体数字（266×, 29.7°, 7894°, 0.30°/s, 37.5°/s, 27.5×, +4.91%, 94.9%, 95% CI, 90.5%, 2.8×, 68×, 95.7%），读者难以在短时间内吸收核心信息。建议精简为3-4个关键数字。
3. **表格数量过多。** 正文包含15+个表格，部分内容高度重复（如Table 7和Table 10都报告DRL vs. SLSQP对比）。建议合并或移至附录。

### 决策建议

**Major Revision.** 论文的科学贡献——疲劳感知DRL框架——是有价值的，但当前的呈现方式通过混合baseline和optimized配置的结果、过度堆砌数字、缺乏与SOTA闭环方法的详细对比，削弱了其影响力。

---

## 报告 2: Reviewer 1 — 方法论审稿 (Wake Modeling & Physics)

### 总体评价

**评分: 6.2/10**

### 尾流模型校准 (Sec. 2.2, Sec. 3.1)

**优势：**
1. 四参数{α\*, β\*, α, I}的联合校准框架是合理的，三个CFD数据集的加权最小二乘方案具有物理直觉。
2. 对I收敛到下界(I=0.065)的讨论表明作者已经意识到校准的局限性，并进行了松弛下界验证(I→0.048)。
3. Horns Rev I验证（Fig. 9）展示了模型在V-80平台上的可迁移性。

**关切：**

1. **⚠️ 校准目标的物理一致性 (CRITICAL).** 湍流强度I在Eq. (7)-(9)中作为尾流扩散率 k\* = 0.3837I + 0.003678 的驱动参数出现，但在Eq. (10)的损失函数中I同时被当做自由参数校准。这存在概念上的循环：I既是输入（环境湍流强度）又是输出（校准参数）。CFD参考数据集的真实环境TI是多少？如果三个CFD数据集的实际TI不同（这是合理的预期），将I打包为统一校准参数将产生物理矛盾。作者需要：(a) 报告每个CFD数据集的实际TI值；(b) 解释为什么可以用单一I值拟合不同TI的数据集；(c) 讨论这如何影响校准模型在不同大气稳定性条件下的外推能力。

2. **参数敏感性分析不足.** 校准后参数为{α\*=2.728, β\*=0.10, I=0.065, α=0.540}。β\*精确收敛到下界(0.10)，这在第4.6节被标记为limitation——"optimization-induced bias trade-off (3.4%→10.9% post-fit)"。这个偏差的来源是什么？是否可以通过增加正则化项或调整权重 w_i 来缓解？建议进行系统的参数敏感性分析（至少对α\*和α进行±20%扰动，观察farm power预测的变化）。

3. **RSS叠加方案与FLORIS的GCH模型的差异未充分量化。** Sec. 3.1.4报告了"9.1% column-aligned excess"，但将此归因于"different superposition coefficient"。这远不止是系数差异——RSS和线性叠加在物理上是不同的机制。在column-aligned regime下(φ≈270°)，RSS叠加会产生系统性的deficit差异（√(∑ΔU²) vs. ∑ΔU）。作者需要解释为什么9.1%偏差不是模型结构性问题，并讨论RSS vs. 线性叠加在不同wake重叠程度下的系统性差异。

4. **两涡轮偏航案例的后校准误差(6.6%)仍需讨论。** Table 4显示后校准最优功率为7.68 MW vs. CFD的7.91 MW，差距为2.9%。虽然优于前校准的27.6%，但6.6%的功率预测误差（Abstract中数字）和2.9%的总功率误差之间的对应关系不清晰。6.6%是指什么条件下的误差？报告的一致性需要加强。

### 实验设计

1. **动态风AR(1)参数的双重标准。** Sec. 4.3在优化配置评估中使用了修改的AR(1)参数(ρ=0.95, σ_φ=2°, σ_v=1 m/s)，但Sec. 4.3.3的baseline fatigue-aware框架使用了原始参数(ρ_φ=0.99, σ_φ=1.0°)。这两个实验的参数差异使得直接比较不可靠。作者需要要么统一参数，要么给出明确的理由说明为什么修改参数。

2. **"modified AR(1) parameterization that produces more pronounced wind variability"** 这个描述过于随意。什么是"more pronounced"的量化定义？建议报告两个参数组下的自相关时间尺度和有效方差。

### 建议

- 明确三个CFD数据集的TI值
- 对校准参数进行敏感性分析
- 区分RSS叠加的物理偏差与参数偏差
- 统一动态风实验的参数或提供明确的不可比性声明

---

## 报告 3: Reviewer 2 — 领域审稿 (DRL & Control)

### 总体评价

**评分: 7.5/10**

### DRL方法论 (Sec. 2.3)

**优势：**
1. **SLSQP-regret reward设计具有创新性。** 用oracle headroom归一化power gain使学习信号集中在有优化价值的条件上，这是一个聪明的奖励塑形策略。5.6×样本效率提升是可验证的。
2. **训练动力学稳定化机制（Cosine LR + KL early-stop + AdamW）** 的三重组合是基于实际训练观察的有针对性设计，而非机械地套用标准做法。
3. **Downstream-turbine locking** 是一个简单但关键的物理归纳偏置。Sec. 2.3.3的消融实验（lock_off导致−2.82±3.24%退化）有力地证明了其必要性。
4. **JAX/NNX on-device实现** 的环境port交叉验证（Fig. 7）为结果的可重复性提供了重要保证。

**关切：**

1. **PPO vs. SAC对比不公平 (MAJOR).** Sec. 4.5将SAC标记为"失败"，但实验设计存在系统性偏见：(a) SAC使用SB3实现(CPU-based)，而PPO使用JAX on-device（~160×更快），这意味着SAC只能运行2×10^6步而PPO运行6×10^7步；(b) SAC的超参数（twin Q, auto entropy, buffer size 5×10^5）未经过系统调参；(c) SAC在连续控制任务中通常需要更大的buffer和更长的训练时间。作者将SAC的失败归因于"off-policy sample inefficiency"和"sparse reward"，但如果给予SAC相同的JAX加速实现和可比的环境步数，结果可能不同。这不是说PPO不是更好的选择，而是说当前的对比不能支撑"PPO is a necessary algorithmic choice"这一强结论。

2. **Reward clamp [-2,2] 的动机不足。** Eq. (8)将奖励clamp到[-2,2]。当SLSQP headroom很小时（例如0.5 MW），power gain为0的基线对应于reward≈−2（如果SLSQP gain≈headroom），而power gain接近headroom时reward≈+2。这个clipping区间是如何选择的？如果clipping过于激进，可能在headroom小的条件中制造虚假的reward饱和。

3. **LCB checkpoint selection的敏感性。** Algorithm 1提到用LCB(95% CI)选择best checkpoint。LCB选择相对于简单均值选择、max选择、或最后checkpoint的优势是否有量化？如果95% CI很宽（如种子间方差大），LCB选择可能过于保守。

4. **Focused wind sampling的distributional shift问题。** Sec. 2.3.4的混合采样(0.3/0.3/0.4)将70%的训练episodes集中在aligned-cube regime。这在稳态评估中提升了性能（Table 5），但可能解释了Sec. 4.3中观察到的"稳态-动态权衡"：policy学会了在aligned-cube regime中表现优异，但代价是对其他风况的泛化能力下降。作者应讨论这种distributional shift是否是一个fundamental trade-off，还是可以通过更好的采样策略（如curriculum learning）来缓解。

5. **缺少PPO与MPC的直接对比。** Table 6比较了PPO与MPC的定性特征（延迟、带宽等），但没有提供定量对比。鉴于MPC是风电场控制文献中的主要竞争范式，至少应有一个简化MPC基线的定量对比（即使是单条件）。

### 建议

- 提供SAC的公平对比条件（JAX实现 + 可比步数），或弱化关于"necessary"的结论
- 讨论reward clipping的敏感性
- 量化LCB选择的优势
- 增加focused sampling与generalization trade-off的讨论
- 考虑增加简化的MPC基线

---

## 报告 4: Reviewer 3 — 视角审稿 (工程实践与可部署性)

### 总体评价

**评分: 7.0/10**

### 工程价值

**优势：**
1. **推理延迟基准测试（Fig. 15）具有工程参考价值。** 0.114–0.172 ms的MLP推理延迟确认了实时可行性。作者在第3.3节正确声明这"excludes sensor-read latency, inter-turbine communication, and yaw-actuator dynamics"，避免了过度宣称。
2. **偏航速率约束分析表明** 策略在factory-default rate limits下具有鲁棒性。这是一线工程师最关心的问题之一。
3. **AEP估算** (Sec. 4.2.3, +0.45%到+0.63% under Weibull/von Mises) 使结果在运营层面具有可解释性。

**关切：**

1. **工业基线比较仍不充分 (MAJOR).** Sec. 4.3.4添加了低通滤波和滞回死区测试，这是好方向。但最关键的工程问题是：DRL策略相对于**当前工业实践中已有的偏航控制策略**（如基于风向标的简单追踪 + 死区）的优势是什么？目前所有比较都是针对"零偏航基线"或SLSQP查找表。缺少与工业标准偏航策略（例如：每个涡轮独立追踪风向，带5–10°死区和0.3–0.5°/s速率限制）的pairwise对比。建议包含：
   - Greedy yaw tracking（每个涡轮独立追风）
   - Sector-based yaw（按扇区预设偏航角）
   - 带速率限制和死区的Rule-based策略

2. **5°滞回死区使54%偏航行程减少——但这是正面还是负面？** Sec. 4.3.4报告带5°滞回的DRL policy行程减少54%（29.7°→~14°）。这听起来是正面的（更少的执行器磨损），但需要确认policy power gain是否受影响。如果在滞回条件下power gain保持不变，这加强了DRL的优势；如果有trade-off，需要在Pareto front上展示。

3. **从仿真到实际部署的gap未充分讨论。** 论文讨论了尾流模型校准的局限性，但未讨论：
   - 偏航执行器延迟和回差（非零的响应时间）
   - 尾流传播延迟（上游偏航变化的effect到达下游涡轮需要时间，在10 m/s风速和7d₀间距下约88秒）
   - 风速/风向测量的时空分辨率限制
   - 涡轮间通信延迟

4. **风速低于切入风速的情况。** 训练分布在v∈[6,16] m/s上。但NREL-5MW的切入风速为3 m/s。6 m/s的lower bound意味着控制器在低风速区（3–6 m/s）的行为是未定义的。

### 建议

- 增加工业标准偏航策略基线
- 补充滞回条件对power gain的定量影响
- 深化sim-to-real gap的讨论
- 讨论控制器在v<6 m/s时的行为

---

## 报告 5: Devil's Advocate 审稿

### 最强反论点

**"The claimed fundamental advantage of DRL over lookup-table control may be an artifact of the specific evaluation protocol rather than a property of the methods."**

论文的核心叙事是：DRL策略通过增量动作空间实现"执行器友好"的实时控制，而SLSQP查找表产生物理上不可行的偏航速率(37.5°/s)。然而，这个结论高度依赖于评价协议中使用的特定查找表实现。具体而言：

1. **朴素查找表不是fair baseline。** 当前的lookup table用91×11网格的bilinear interpolation直接输出静态偏航向量。任何工程师如果在现实中部署偏航控制，都会加入：(a) 速率限制；(b) 低通滤波；(c) 滞回死区；(d) 平滑插值（例如样条或加权移动平均）。Sec. 4.3.4已测试了(a)和(b)，并发现速率限制是binding constraint。但这忽略了一个更根本的问题：如果对查找表输出施加与DRL策略相同的增量约束（即：查找表也输出增量而非绝对角度），会发生什么？

2. **"266× less travel"的比较建立在非对称假设上。** DRL策略的29.7° travel是200步内的累积值。SLSQP查找表的7894° travel假设查找表在每个控制周期都试图跳跃到当前条件的静态最优角度。如果查找表被赋予：(a) 与DRL相同的±10°增量约束；(b) 相同的±50°绝对界限；(c) 相同的下游锁定机制，travel差异会显著缩小。这不是说DRL没有优势，而是说266×这个数字——在Abstract中作为核心卖点——可能夸大了真实的差距。

3. **稳态-动态权衡可能是训练策略的产物。** Sec. 4.3报告的"稳态优化降低了动态响应性"被呈现为方法的内在属性。但Table 5显示不同的混合比会产生不同的trade-off：uniform sampling产生最robust但低增益的策略(80.5% recovery, 13.6% negative fraction)。是否有一个intermediate采样策略可以获得更好的平衡？论文已经收集了这些数据但未进行系统分析。

### 樱桃采摘检测

1. **Baseline vs. Optimized配置的混合使用。** 多处关键声明在小字脚注中标明"此实验使用baseline PPO配置"：(a) Lock ablation (Sec. 2.3.3); (b) FLORIS cross-validation (Sec. 3.1.4); (c) Observation noise robustness (Sec. 4.2.1)。这造成了"cherry-picking"的外观：当baseline配置的结果支持论文观点时（noise robustness, lock benefit），它们被保留在正文中；当baseline配置的结果不够强时（gain = +3.39%），optimized配置的结果被用来替代(+4.91%)。建议统一处理：要么所有结果都用optimized配置重新评估，要么将baseline结果移至附录。

2. **SLSQP oracle的"最优性"未验证。** SLSQP被当做ground truth oracle，但：(a) 8个随机start可能不足以找到非凸目标（9维偏航空间）的全局最优；(b) SLSQP是局部搜索算法，即使有多个start也可能陷入局部最优；(c) 500-condition sweep中报告的SLSQP aligned-cube gain +5.17%与lookup table的+5.09%之间的0.08 pp差异可能只是插值误差。作者需要报告：(i) 8个start中找到的最优值的方差；(ii) 不同start数(4, 8, 16, 32)的收敛行为；(iii) 是否尝试了全局优化器(Bayesian optimization, CMA-ES)来验证SLSQP的最优性。

3. **"95.7% transfer to 5×5" 可能混淆了布局效应与配置效应。** Sec. 4.6将5×5的+4.70% gain与3×3的+4.91%比较，声称"95.7% of the 3×3 headline gain"。但+4.70%是optimized配置的结果（含position encoding），而早期的+0.98%是baseline配置的结果（不含position encoding）。这两个数字的差异主要归因于position encoding的缺失，而非布局尺寸的缩放效应。因此95.7%更多的是"配置迁移率"而非"布局缩放率"。

### 忽略的替代解释/路径

1. **为什么增量动作空间不应用于查找表？** 论文将增量动作空间呈现为DRL方法的独特优势，但增量控制也可以应用于查找表策略（例如：增量式MPC或简单的速率限制+增量查找表）。没有实验证据表明增量控制的优势是DRL独有的。

2. **SLSQP-residual gap的替代解释。** 0.26 pp的DRL-SLSQP差距被归因于DRL policy保守性，但可能源于：(a) PPO的策略熵正则化导致随机性；(b) 观测噪声（即使是确定性的，PPO训练中的exploration也可能导致方差）；(c) 有限的MLP容量（[128,128]可能不足以完全表达最优偏航函数）。

3. **SAC失败的替代归因。** Sec. 4.5将SAC失败归因于off-policy稀疏奖励，但可能的替代原因是：(a) 未充分调参的熵温度(α)；(b) replay buffer的size太小(5×10^5)；(c) tanh-squashing bias。这些可以在未来的工作中系统性排除。

### 缺失的利益相关者视角

1. **风电场运营商的视角：** 论文未讨论现场部署的任何障碍——(a) 安全认证要求；(b) 功率曲线保证合同的约束；(c) 电网规范中的无功功率和电压控制可能与yaw optimization冲突。
2. **涡轮OEM的视角：** 5 MW级涡轮的偏航系统设计寿命通常为20年。DRL策略增加的偏航循环次数需要与OEM的疲劳设计曲线进行比较。
3. **监管/电网运营商的视角：** 协同偏航可能影响风电场的总功率输出模式，可能需要更新电网连接协议。

### "So What?" 测试

论文的核心叙事是：DRL提供了执行器友好的实时协同偏航控制。但如果将工业标准偏航策略（每个涡轮独立追风 + 死区 + 速率限制）与DRL策略在以下条件比较：(a) 相同的速率限制；(b) 相同的死区；(c) 真实的湍流风时间序列（不是AR(1)），DRL的绝对优势是多少？如果答案是"~0.3–0.5% AEP"，那么论文的impact statement需要重新校准——这是一个有意义的但有条件的贡献，而非一个变革性的范式转移。

### 观察（非缺陷）

- 论文在方法论透明性方面表现出色：几乎所有超参数都被记录，消融实验系统地隔离了各组件的影响，JAX环境交叉验证为数值保真度提供了证据。
- "训练动力学稳定化"机制（cosine LR, KL early-stop, AdamW）虽然是标准技术，但在该问题上的应用是新颖且有实际价值的。
- FLORIS交叉评估（Table 7）是论文最强的验证步骤之一，它直接回应了"gray-box bias inflates gains"的合理质疑，并表明DRL gains在FLORIS中实际上更conservative。

---

# Phase 2: 编辑决策综合

## 跨审稿人共识矩阵

| 议题 | EIC | R1 (Method) | R2 (Domain) | R3 (Perspective) | DA | 共识 |
|------|-----|-------------|-------------|-------------------|-----|------|
| 疲劳感知DRL框架有价值 | ✅ | ✅ | ✅ | ✅ | ⚠️ | **强共识** |
| Baseline/Optimized配置混合需清理 | ❌ (MAJOR) | — | — | — | ❌ (MAJOR) | **强共识** |
| SLSQP最优性验证不足 | — | — | — | — | ❌ (MAJOR) | **DA单方面** |
| PPO vs SAC对比不公平 | — | — | ❌ (MAJOR) | — | — | **R2单方面** |
| 缺少工业基线 | — | — | ❌ | ❌ (MAJOR) | ❌ | **强共识** |
| 尾流模型TI校准循环 | — | ❌ (CRITICAL) | — | — | — | **R1单方面** |
| Abstract数字过多 | ❌ | — | — | — | — | **EIC建议** |
| 266× travel比较有误导性 | — | — | — | — | ❌ (MAJOR) | **DA单方面** |
| 参数统一与敏感性分析 | — | ❌ | ❌ | — | — | **部分共识** |
| Sim-to-real gap讨论不足 | — | — | — | ❌ | — | **R3建议** |

## 编辑决定: **MAJOR REVISION**

### 决定理由

本稿件基于扎实的工程实践提出了一个具有工业相关性的疲劳感知DRL框架。核心科学贡献——通过可调偏航速率惩罚实现功率-疲劳Pareto优化——在风电场控制文献中具有新颖性。FLORIS交叉验证增强了结果的可信度。

然而，以下问题必须在修改中解决：

### 必需修改（MAJOR REVISION的先决条件）

**P0-1: 统一实验配置的报告策略 (EIC + DA).**
- 将"此实验使用baseline配置"的脚注结果移至Appendix
- 正文仅报告optimized配置的结果，或为两组实验设立清晰分隔的子节
- 如果optimized配置下的某些实验（noise robustness, lock ablation）确实无法重跑，需要在Limitations中明确说明并评估影响

**P0-2: 增加工业标准偏航策略基线 (R3 + DA + EIC).**
- 在Sec. 4.3 Dynamic Wind评估中加入至少一个工业标准策略：Greedy yaw tracking + 5° deadband + 0.3°/s rate limit
- 这建立了真正的"business-as-usual"基线，而非零偏航这一理论下限

**P0-3: 公平化PPO vs SAC对比或弱化相关声明 (R2).**
- 提供SAC在JAX实现 + 可比环境步数下的对比，或者
- 将"PPO is a necessary algorithmic choice"改为"PPO outperforms SAC under the current training budget and implementation"

**P0-4: 阐明尾流模型TI校准的物理一致性 (R1).**
- 报告三个CFD数据集的真实环境TI
- 解释单值I如何适配不同TI的数据集
- 讨论外推到不同大气稳定性条件的局限

**P0-5: 验证SLSQP的最优性 (DA).**
- 报告8-start SLSQP的方差
- 在代表性条件子集上尝试全局优化器（如CMA-ES或Bayesian optimization）
- 将"SLSQP oracle"改为"SLSQP optimum"以避免暗示全局最优性

### 建议修改（非阻塞但强烈推荐）

**P1-6: 校准266× travel声明 (DA).**
- 在Abstract和Discussion中加上条件说明："under the unconstrained comparison protocol"
- 增加查找表使用与DRL相同增量约束的比较

**P1-7: 统一动态风AR(1)参数 (R1).**
- 所有动态风实验使用相同的AR(1)参数

**P1-8: 添加参数敏感性分析 (R1).**
- 对校准参数{α\*, α}进行±20%扰动分析

**P1-9: 深化sim-to-real gap讨论 (R3).**
- 讨论尾流传播延迟、执行器响应时间、测量分辨率限制

**P2-10: 精简Abstract (EIC).**
- 保留3-4个最关键的量化结果

### 修订路线图

| 优先级 | 修改项 | 预期影响 | 工作量 |
|--------|--------|---------|--------|
| P0 (必须) | 统一实验配置报告 | 消除cherry-picking外观 | 中等 (重组织) |
| P0 (必须) | 工业标准偏航基线 | 建立公平的工程基线 | 中等 (新实验) |
| P0 (必须) | SAC对比公平化 | 增强方法论声明的可信度 | 中等-大 (需JAX实现) |
| P0 (必须) | TI校准物理一致性 | 修复潜在物理缺陷 | 小 (分析与讨论) |
| P0 (必须) | SLSQP最优性验证 | 稳固oracle baseline | 小 (验证性实验) |
| P1 (建议) | 266× travel声明校准 | 提高准确性 | 小 (文本修改) |
| P1 (建议) | AR(1)参数统一 | 确保可复现性 | 小 (重跑实验) |
| P1 (建议) | 参数敏感性分析 | 增强模型理解 | 小-中 |
| P2 (可选) | Sim-to-real gap扩展 | 增强工业相关性 | 小 (讨论扩展) |
| P2 (可选) | Abstract精简 | 提升可读性 | 小 |

### 对作者的建议

本审稿小组认可论文的核心贡献——疲劳感知DRL框架——的价值。修改的核心原则是**透明性与公平性**：诚实地呈现哪些实验使用了哪个配置，公平地与工业实践（而非仅与理论边界）比较，精确地界定声称的优势的适用范围。

如果你的资源有限，P0项中的#2（工业基线）和#3（SAC对比）可以用简化的方式处理：(a) 工业基线可以用简单的基于风向标的策略 + 规则死区实现，无需完整训练；(b) SAC对比可以将声明从"necessary"弱化为"empirically superior"，同时清晰说明实验限制。

---

*本报告由5位独立审稿人（EIC + 3位同行审稿人 + Devil's Advocate）生成，所有审稿意见均基于论文正文内容，未参考任何外部投稿评审历史。*
