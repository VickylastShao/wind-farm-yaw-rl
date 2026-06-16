# Reproducibility Checklist

## Per-Experiment Status

| Experiment | Script | Config | Seeds | Result File | Reproducible? |
|-----------|--------|--------|-------|------------|---------------|
| Config-E PPO Training | train_3x3_nnx_jaxenv.py | env vars | 5 | policy_seed*_sens_act10.pkl | ✅ |
| FLORIS Cross-Val | eval_floris_configE.py | env vars | 5 | configE_floris_cross_validation.json | ✅ |
| Noise Robustness | eval_noise_batched.py | env vars | 5 | configE_noise_robustness.json | ✅ |
| Lock Ablation | train_3x3_nnx_jaxenv.py (NO_LOCK=1) | env vars | 3 | policy_seed*_lock_off_configE.pkl | ✅ |
| Gated DRL | eval_gated_fast.py | env vars | 5 | gated eval output | ✅ |
| λrate Sweep | eval_rate_policies.py | J=1, Config-A policies | 3 | eval output | ⚠️ Config-A only |
| Marginal Reward | run_ablation_marginal_reward.sh | env vars | 3 | policy_seed*_marginal_reward.pkl | ✅ |
| KL ON Ablation | run_ablation_kl_earlystop.sh | env vars | 3 | policy_seed*_kl_on.pkl | ✅ |
| KL OFF Ablation | run_ablation_kl_earlystop.sh | env vars | 3 | policy_seed*_kl_off.pkl | ✅ |
| SAC Fair Training | train_sac_fair_v2.py | env vars | 5 | policy_seed*.pkl | ✅ |
| BC Baseline | train_bc_nnx_jaxenv.py | env vars | 1 | policy_seed*_bc.pkl | ✅ |

## Environment
- Python 3.13, JAX 0.9.0.1, Flax NNX 0.12.4, Optax 0.2.6
- FLORIS 4.6.5
- GPU: NVIDIA RTX 4090 (24 GB)
- No requirements.txt available; dependencies installed ad-hoc

## Known Gaps
1. No requirements.txt or environment.yml — environment not fully reproducible
2. λrate sweep used Config-A policies — Config-E retraining recommended
3. Gated DRL and deadband are evaluation-only (no dedicated retraining)
4. Wind-rose AEP analysis not completed
5. Secondary load proxy not implemented
