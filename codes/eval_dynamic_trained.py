#!/usr/bin/env python3
"""Evaluate dynamic-wind-trained policies on 1000 AR(1) trajectories.

Uses the same protocol as eval_dynamic_optimized.py for fair comparison.
"""
import os, sys, json, time, pickle
import numpy as np
import jax, jax.numpy as jnp
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from windfarm_env import create_wind_farm_layout_3x3
from windfarm_env_jax import (env_reset, env_step, positions_to_jax,
                               inflow_speeds_jax, power_output_jax)
from train_3x3_nnx import ActorCritic

CKPT_DIR = "checkpoints_3x3_nnx_jaxenv"
FIG_DIR = "../latex_draft/figures"
os.makedirs(FIG_DIR, exist_ok=True)

DYN_TAG = "dynamic_wind"
N_TRAJ = 1000
TRAJ_LEN = 200
T_STEP = 10.0
SETTLE = 150
SIGMA_PHI, SIGMA_V = 2.0, 1.0
ALPHA_PHI, ALPHA_V = 0.95, 0.95

# Load layout
positions, _, _ = create_wind_farm_layout_3x3()
N = len(positions)
positions_j = positions_to_jax(positions)
obs_dim_per_step = 5 * N + 3
obs_dim = 3 * obs_dim_per_step
act_dim = N

# Load dynamic-wind policies
models = []
for s in range(3):
    ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_{DYN_TAG}.pkl")
    if not os.path.exists(ckpt):
        print(f"MISSING: {ckpt}")
        continue
    model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(0))
    graphdef, _ = nnx.split(model)
    with open(ckpt, "rb") as f:
        state = pickle.load(f)
    models.append(nnx.merge(graphdef, state))
    print(f"Loaded seed {s}: {ckpt}")

print(f"Loaded {len(models)} seeds for '{DYN_TAG}'")

# DRL rollout
@nnx.jit
def drl_rollout(m, init_state, init_obs):
    def body(carry, _):
        s, o = carry
        mean, _, _ = m(o.reshape(1, -1))
        a = jnp.clip(mean.reshape(N), -10.0, 10.0)
        s, o, _, _ = env_step(s, a, positions_j)
        return (s, o), s.gammas
    (final_s, _), gammas_traj = jax.lax.scan(body, (init_state, init_obs), None, length=TRAJ_LEN)
    return gammas_traj, final_s

# Generate AR(1) trajectories
rng = np.random.default_rng(20260609)
all_trajs = []
for traj in range(N_TRAJ):
    phi0 = rng.uniform(173, 353); v0 = rng.uniform(6, 16)
    phis = [phi0]; vs = [v0]
    for t in range(1, TRAJ_LEN):
        phis.append(ALPHA_PHI * phis[-1] + (1-ALPHA_PHI) * 270.0 + rng.normal(0, SIGMA_PHI))
        vs.append(ALPHA_V * vs[-1] + (1-ALPHA_V) * 11.4 + rng.normal(0, SIGMA_V))
    phis = np.clip(phis, 173, 353); vs = np.clip(vs, 6, 16)
    all_trajs.append((np.array(phis), np.array(vs)))

def compute_metrics(gammas_log, phis, vs):
    total_gain = 0.0; total_travel = 0.0; max_rate = 0.0
    prev_g = np.zeros(N)
    for t in range(TRAJ_LEN):
        g = gammas_log[t]
        inflow = inflow_speeds_jax(positions_j, jnp.float32(phis[t]),
                                    jnp.float32(vs[t]), jnp.array(g, dtype=jnp.float32))
        pwr = float(jnp.sum(power_output_jax(inflow, jnp.array(g, dtype=jnp.float32))) / 1e6)
        base_in = inflow_speeds_jax(positions_j, jnp.float32(phis[t]),
                                     jnp.float32(vs[t]), jnp.zeros(N, dtype=jnp.float32))
        base_pwr = float(jnp.sum(power_output_jax(base_in, jnp.zeros(N, dtype=jnp.float32))) / 1e6)
        if base_pwr > 0:
            total_gain += (pwr - base_pwr) / base_pwr * 100
        travel = np.abs(g - prev_g).sum()
        total_travel += travel
        max_rate = max(max_rate, travel / T_STEP)
        prev_g = g.copy()
    return total_gain / TRAJ_LEN, total_travel, max_rate

# Evaluate
print(f"\n{'='*60}")
print(f"Dynamic-wind-trained DRL evaluation ({len(models)} seeds)")
print(f"{'='*60}")

seed_gains, seed_travels, seed_rates = [], [], []
for seed_idx, model in enumerate(models):
    gains = np.zeros(N_TRAJ); travels = np.zeros(N_TRAJ); rates_arr = np.zeros(N_TRAJ)
    for traj in range(N_TRAJ):
        phis, vs = all_trajs[traj]
        key = jax.random.key(traj)
        sj, obs = env_reset(key, positions_j,
                             specific_wind_dir=jnp.float32(phis[0]),
                             specific_wind_speed=jnp.float32(vs[0]),
                             randomize_wind=False, j=3)
        gammas_traj_j, _ = drl_rollout(model, sj, obs)
        gammas_traj = np.asarray(gammas_traj_j)
        g, t, r = compute_metrics(gammas_traj, phis, vs)
        gains[traj] = g; travels[traj] = t; rates_arr[traj] = r

    eff = gains.mean() / travels.mean() * 100 if travels.mean() > 0 else 0
    print(f"  seed {seed_idx}: gain={gains.mean():+.4f}%, travel={travels.mean():.1f}°, "
          f"rate={rates_arr.mean():.4f}°/s, eff={eff:.4f}/100°")
    seed_gains.append(gains.mean()); seed_travels.append(travels.mean()); seed_rates.append(rates_arr.mean())

print(f"\n  MEAN: gain={np.mean(seed_gains):+.4f}% ± {np.std(seed_gains):.4f}, "
      f"travel={np.mean(seed_travels):.1f}° ± {np.std(seed_travels):.1f}, "
      f"rate={np.mean(seed_rates):.4f}°/s")

# Save
output = {
    "description": "Dynamic-wind-trained DRL policy evaluation (E1)",
    "tag": DYN_TAG, "n_seeds": len(models),
    "n_trajectories": N_TRAJ, "traj_length": TRAJ_LEN, "T_step_s": T_STEP,
    "ar1_params": {"sigma_phi": SIGMA_PHI, "sigma_v": SIGMA_V,
                   "alpha_phi": ALPHA_PHI, "alpha_v": ALPHA_V},
    "per_seed": [{"gain_mean": float(g), "travel_mean": float(t), "rate_mean": float(r)}
                 for g, t, r in zip(seed_gains, seed_travels, seed_rates)],
    "mean": {"gain_mean": float(np.mean(seed_gains)), "travel_mean": float(np.mean(seed_travels)),
             "rate_mean": float(np.mean(seed_rates))},
}
out_path = os.path.join(FIG_DIR, "dynamic_wind_trained_eval.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")

# Comparison with static-trained configs
print(f"\n{'='*60}")
print("COMPARISON: Dynamic-trained vs Static-trained under AR(1) wind")
print(f"{'='*60}")
print(f"{'Configuration':<30s} {'Gain':>8s} {'Travel':>8s} {'Rate':>8s}")
print("-" * 55)
# From paper / previous evals
comparisons = {
    "DRL dynamic-trained (E1)": (np.mean(seed_gains), np.mean(seed_travels), np.mean(seed_rates)),
    "DRL static-opt (sens_act10)": (-0.14, 29.7, 0.30),
    "DRL baseline (p0c)": (0.18, 123.0, 0.13),
    "Lookup unlimited": (4.82, 7894.0, 37.5),
    "Lookup 0.1°/s": (1.34, 817.0, 0.60),
}
for name, (g, t, r) in comparisons.items():
    print(f"{name:<30s} {g:+8.4f}% {t:8.1f}° {r:8.4f}°/s")
