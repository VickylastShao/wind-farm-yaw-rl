# Experiment Inventory

## Training Experiments
| Tag | Checkpoints Found | Seeds | Steps | Status |
|-----|-------------------|-------|-------|--------|
| sens_act10 | 5 | 5 | 30015488 | ✅ Complete |
| p0c_ultimate | 5 | 5 | 60030976 | ✅ Complete |
| marginal_reward | 3 | 3 | 60030976 | ✅ Complete |
| kl_on | 3 | 3 | 30015488 | ✅ Complete |
| kl_off | 3 | 3 | 30015488 | ✅ Complete |
| bc | 18 | 5 | ? | ✅ Complete |
| full60m | 5 | 5 | 60030976 | ✅ Complete |
| dynamic_wind | 3 | 3 | 30015488 | ✅ Complete |

## Evaluation Scripts (22)
- **eval_drl_vs_slsqp_regime.py**: SLSQP
- **eval_dynamic_fast.py**: dynamic wind
- **eval_dynamic_industrial_baseline.py**: dynamic wind
- **eval_dynamic_optimized.py**: dynamic wind
- **eval_dynamic_trained.py**: dynamic wind
- **eval_dynamic_trained_fast.py**: dynamic wind
- **eval_dynamic_wind.py**: dynamic wind
- **eval_e2_compact.py**: general
- **eval_gated_drl.py**: gated DRL
- **eval_industrial_baselines.py**: industrial
- **eval_observation_noise_robustness.py**: noise
- **eval_p0c_policies.py**: general
- **eval_p0c_randomized.py**: general
- **eval_p1_5x5_randomized.py**: general
- **eval_param_perturbation.py**: general
- **eval_param_perturbation_v2.py**: general
- **eval_policy_in_floris.py**: FLORIS
- **eval_rate_limits.py**: general
- **eval_sac_policy.py**: SAC
- **eval_sac_randomized.py**: SAC
- **eval_slsqp_in_floris.py**: FLORIS
- **eval_unified_static_vs_slsqp.py**: SLSQP
