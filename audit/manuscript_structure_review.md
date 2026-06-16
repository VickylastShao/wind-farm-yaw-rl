# Manuscript Structure Review

## Basic Stats
- **Total words**: ~15740
- **Sections**: 66
- **References**: 0

## Current Structure
- **section**: Introduction
- **section**: Methodology
  - **subsection**: Overall framework
  - **subsection**: Gray-box wake model
    - **subsubsection**: Wind-speed--power model
    - **subsubsection**: Wake-deficit model
    - **subsubsection**: Wake superposition
    - **subsubsection**: Multi-scenario parameter optimization
  - **subsection**: DRL-based cooperative yaw controller
    - **subsubsection**: Algorithm choice: PPO
    - **subsubsection**: State, action and reward
    - **subsubsection**: Downstream-turbine locking
      - **paragraph**: Ablation: locking is necessary for stable cooperative yaw learning.
    - **subsubsection**: Training and evaluation protocol
    - **subsubsection**: On-device implementation
      - **paragraph**: Environment-port cross-validation.
- **section**: Experiments and Results
  - **subsection**: Stage I: wake-model calibration and validation
    - **subsubsection**: Calibration datasets
    - **subsubsection**: Optimization results
    - **subsubsection**: Horns Rev I validation
    - **subsubsection**: Cross-validation against FLORIS on the $3\!\times\!3$ NREL-5MW layout
      - **paragraph**: FLORIS cross-evaluation of trained policies.
  - **subsection**: Stage II: PPO control performance
    - **subsubsection**: Case A --- $1{\times
    - **subsubsection**: Case B --- $3{\times
      - **paragraph**: Online closed-loop behavior.
      - **paragraph**: Physical limits on closed-loop bandwidth.
      - **paragraph**: Comparison with offline numerical optimum.
    - **subsubsection**: Distribution-wise evaluation across the full training envelope
      - **paragraph**: Regime decomposition.
      - **paragraph**: A-priori justification of the regime cut.
      - **paragraph**: Regime-threshold sensitivity.
      - **paragraph**: Per-seed robustness in the aligned-cube regime.
      - **paragraph**: Operating envelope and headline figure.
    - **subsubsection**: Annual energy production impact
  - **subsection**: Online inference performance and comparison with existing control frameworks
- **section**: Discussion
  - **subsection**: Reward design: action penalties as a yaw-activity management tool
      - **paragraph**: Combined penalties hurt peak power performance.
      - **paragraph**: Rate-only penalties enable actuator-aware control.
  - **subsection**: Robustness and operational analysis
    - **subsubsection**: Observation noise robustness
    - **subsubsection**: Actuator rate constraints
  - **subsection**: Comparison with offline optimization
      - **paragraph**: Regime-wise DRL vs.\ SLSQP comparison.
      - **paragraph**: Lookup-table baseline.
      - **paragraph**: Toward dynamic evaluation.
      - **paragraph**: Caveat: rated-power clipping artifact.
      - **paragraph**: Conditions with negative DRL impact.
  - **subsection**: Dynamic wind performance: actuator feasibility and actuator-aware control
    - **subsubsection**: AR(1) turbulent wind model
    - **subsubsection**: Evaluation protocol
    - **subsubsection**: Original actuator-aware framework (baseline PPO configuration)
    - **subsubsection**: Dynamic-wind retraining confirms the necessity of rate penalties
    - **subsubsection**: Gated DRL: regime-conditional policy activation
    - **subsubsection**: Industrial baseline: low-pass filtering and hysteresis
  - **subsection**: Industrial baseline comparison
  - **subsection**: Algorithm comparison: PPO vs.\ SAC
  - **subsection**: Behavioral Cloning baseline
  - **subsection**: Parameter sensitivity of the calibrated wake model
  - **subsection**: Scalability limits
  - **subsection**: Limitations
- **section**: Conclusion
- **section**: Baseline configuration provenance
- **section**: SAC implementation details

## Wind Energy Fit Assessment

| Criterion | Status | Notes |
|-----------|--------|-------|
| Wind farm engineering focus | ⚠️ Partial | Currently reads as DRL+gray-box; needs more wind-farm-control framing |
| Wake steering context | ✅ | Bastankhah, FLORIS, Horns Rev all present |
| AEP / wind rose analysis | ⚠️ Buried | AEP section exists but is in robustness subsection |
| Actuator constraints | ✅ | Rate limits, yaw travel discussed |
| Dynamic inflow | ✅ | AR(1) wind modeling present |
| Field validation | ⚠️ Overstated | Horns Rev validation is of wake model only, not controller |
| Fatigue claims | ⚠️ Fixed | Changed to actuator-aware; load proxy still needed |
| FLORIS cross-validation | ❌ Config-A only | Must upgrade to Config-E |
| Industrial baseline | ✅ | Greedy tracking, rate-limited LUT |
| Scalability | ✅ | 5×5 layout shown |

## Config-A Content Requiring Upgrade
The following sections currently use results from Config-A (baseline) rather than Config-E (optimized):
1. Section 4.2 (lock ablation) — mentions Config-A explicitly
2. Section 4.3 (FLORIS cross-validation) — mentions Config-A explicitly  
3. Section 4.3 (observation noise) — mentions Config-A explicitly
4. Section 4.6 (original fatigue-aware framework) — baseline Config-A results

## Content Recommended for Supplementary
1. JAX profiling / implementation efficiency section (Section 3.2.5/4.5.4)
2. SAC algorithm comparison (Section 4.7) — keep 1-paragraph summary
3. BC baseline (Section 4.7.1) — keep as supporting evidence
4. Extended configuration tables (Config-A through Config-D details)
5. KL early-stop ablation details

## Missing Content
1. Wind-rose-weighted AEP analysis with public data
2. Secondary load proxy (yaw-misalignment aerodynamic load)
3. FLORIS SLSQP optimum comparison (for Config-E)
4. Observation-noise robustness for Config-E
5. Downstream-lock ablation for Config-E
6. Gated DRL full numerical evaluation
