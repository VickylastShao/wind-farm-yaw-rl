# Cover Letter

**Manuscript Title:** Cooperative Yaw Control for Wind Farm Power Maximization Using a Calibrated Gray-Box Wake Model and Reinforcement Learning

**Journal:** Wind Energy

Dear Editor,

We submit the manuscript "Cooperative Yaw Control for Wind Farm Power Maximization Using a Calibrated Gray-Box Wake Model and Reinforcement Learning" for consideration in *Wind Energy*.

**Why this fits Wind Energy.** The manuscript addresses a core wind-farm control problem—cooperative yaw steering for wake-induced power loss mitigation—using a control-oriented engineering methodology. The contribution lies at the intersection of wake modeling (calibrated Bastankhah–Porté-Agel Gaussian model, validated against FLORIS and Horns Rev LES), closed-loop control (real-time yaw policy via reinforcement learning), and actuator-aware dynamic operation. We believe this aligns with Wind Energy's readership of wind-farm operators, control engineers, and wake-modeling researchers.

**Problem and approach.** Wake-induced power losses motivate cooperative yaw control, yet conventional static optimization produces yaw vectors tied to single inflow snapshots and must be re-solved offline. We train a closed-loop reinforcement-learning policy on a CFD-calibrated gray-box wake model. The policy produces actuator-feasible yaw commands at sub-millisecond latency and is evaluated under steady-state, dynamic-wind, and noise-perturbed conditions.

**Key findings (all verified by new Config-E experiments):**
- The calibrated wake model reduces two-turbine yaw-case power-prediction error from 27.6% to 6.6%.
- The learned policy achieves +4.91% farm-power gain in wake-aligned regimes, recovering 94.9% of the SLSQP static optimum (bootstrap 95% CI [93.2%, 95.9%]).
- FLORIS cross-validation on 3,000 conditions with five independent seeds yields +5.02% aligned-cube gain (4.1% erosion relative to gray-box), confirming conservative gain estimates.
- Under dynamic AR(1) wind, a gated activation strategy reduces yaw travel by 43% (with 2° industrial deadband) while maintaining cooperative gains.
- Observation-noise tests (wind direction up to 10°, wind speed up to 1.0 m/s, yaw angle up to 2°) show negligible performance degradation.
- Downstream-turbine locking is validated as necessary under the optimized configuration (lock-off degrades gain from +2.42% to -0.46%).
- The framework transfers to 5×5 layouts with +4.70% gain.

**Scope and limitations.** This is a simulation and cross-validation study. The gray-box model is control-oriented; FLORIS serves as an independent cross-validation benchmark. No field SCADA validation or aeroelastic load assessment has been performed. We state these limitations explicitly in the manuscript.

**Originality.** This work is original, has not been published elsewhere, and is not under consideration by another journal. All authors have approved the manuscript.

We hope the manuscript is suitable for *Wind Energy* and look forward to the review process.

**Corresponding Author:**
Zhuang Shao
China Resources Power Technology Research Institute Co., Ltd.
Shenzhen 518000, Guangdong Province, China
E-mail: shaozhuang@crpower.com.cn
ORCID: 0000-0003-2496-0797

**Co-authors:**
Lijun Lei (leilijun6@crpower.com.cn), Rundian Energy Science and Technology Co., Ltd.
Peng Wang (wangpeng@ncwu.edu.cn), North China University of Water Resources and Electric Power
Liang Zheng (zhengliang35@crpower.com.cn), Rundian Energy Science and Technology Co., Ltd.
Jie Zhou (zhoujie365@crpower.com.cn), Rundian Energy Science and Technology Co., Ltd.

**Suggested Reviewers:**
- Jianxin Zhou (zjx@seu.edu.cn), Southeast University
- Zhenlong Wu (wuzhenlong2020@zzu.edu.cn), Zhengzhou University
- Cong Yu (congy@jhun.edu.cn), Jianghan University
- Hui Gu (guhuini@126.com)

Sincerely,
Zhuang Shao (on behalf of all authors)
