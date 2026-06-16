# Reproducibility Status

## Per-Experiment Assessment

| Experiment | Script | Result File | Config-E? | CI? | Seeds | Verdict |
|-----------|--------|------------|-----------|-----|-------|---------|
| Config-E PPO Training | train_3x3_nnx_jaxenv.py | checkpoints_3x3_nnx_jaxenv/policy_seed*_sens_act10.pkl | Yes | No | 5 | reproducible |
| Marginal Reward Ablation | run_ablation_marginal_reward.sh → train_3x3_nnx_jaxenv.py | policy_seed*_marginal_reward.pkl | Yes | No | 3 | reproducible |
| KL ON Ablation | train_3x3_nnx_jaxenv.py (TARGET_KL=0.015) | policy_seed*_kl_on.pkl | Yes | No | 3 | reproducible |
| KL OFF Ablation | train_3x3_nnx_jaxenv.py (TARGET_KL=100) | policy_seed*_kl_off.pkl | Yes | No | 3 | reproducible |
| Parameter Perturbation | eval_param_perturbation_v2.py | param_perturbation_eval_v2.json | Yes | No | 1 | script exists, result missing |
| SAC Fair Training | train_sac_fair_v2.py | checkpoints_3x3_sac_fair/policy_seed*.pkl | Yes | No | 5 | reproducible |
| BC Baseline Eval | eval_sac_policy.py (adapted) | manual inline eval | **No (Config-A)** | No | 1 | partially reproducible |
| Gated DRL Eval | eval_gated_drl.py | gated_drl_results.json | Yes | No | 5 | script exists, full result missing |
| Dynamic Wind (Config-E) | eval_dynamic_optimized.py | dynamic_wind_optimized.json | Yes | No | 3 | result exists but script missing for Config-E |
| FLORIS Cross-Val (Config-A) | cross_validate_floris.py | floris_validation_stats.json | **No (Config-A)** | No | 5 | result exists (Config-A only) |
| Observation Noise (Config-A) | eval_observation_noise_robustness.py | obs_noise_robustness.json | **No (Config-A)** | No | 3 | result exists (Config-A only) |
| Downstream Lock (Config-A) | ablation_downstream_lock.py | threshold_sensitivity_locking.json | **No (Config-A)** | No | 3 | result exists (Config-A only) |
| SLSQP Optimum | compute_slsqp_optimum.py | lookup_table_baseline.json | Yes | No | 1 | reproducible |
| AEP Estimate | compute_aep_estimate.py | aep_estimate.json | Yes | No | 1 | result exists |
| 5x5 Training | train_5x5_nnx_jaxenv.py | checkpoints_5x5_nnx_jaxenv/ | Yes | No | 3 | partially reproducible |
| Bootstrap CI | compute_bootstrap_ci.py | bootstrap_ci_results.json | Yes | Yes | 5 | reproducible |

## Summary
- **Reproducible**: Experiments with complete scripts and results
- **Partially reproducible**: Results exist but scripts incomplete or vice versa  
- **Config-A only**: Must be upgraded to Config-E per revision requirements
- **Script exists, result missing**: Script written but evaluation not completed

## Critical Gaps Requiring Config-E Upgrade
1. FLORIS cross-validation (currently Config-A only)
2. Observation-noise robustness (currently Config-A only)
3. Downstream-lock ablation (currently Config-A only)
4. Dynamic-wind evaluation (result file exist for Config-E but needs verification)
