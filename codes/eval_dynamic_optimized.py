#!/usr/bin/env python3
"""Dynamic wind evaluation: optimized DRL vs lookup table under AR(1) wind.

Re-evaluates the fatigue-aware DRL framework using the optimal configuration
(sens_act10: 0.3/0.3/0.4 mixture, ±10° bounds, γ=0.995, AdamW, etc.)
and compares against the lookup-table baseline from the original submission.
"""

import os, sys, json, time, pickle
import numpy as np
import jax, jax.numpy as jnp
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from windfarm_env import (create_wind_farm_layout_3x3, calculate_inflow_speeds,
                           power_output, C_T, I, d_0, alpha_star, beta_star, alpha)
from windfarm_env_jax import (env_reset, env_step, positions_to_jax,
                               inflow_speeds_jax, power_output_jax)
from train_3x3_nnx import ActorCritic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT_DIR = "checkpoints_3x3_nnx_jaxenv"
FIG_DIR = "../latex_draft/figures"
os.makedirs(FIG_DIR, exist_ok=True)

DYN_TAG = os.environ.get("DYN_TAG", "sens_act10")
N_TRAJ = int(os.environ.get("N_TRAJ", "1000"))
TRAJ_LEN = int(os.environ.get("TRAJ_LEN", "200"))
T_STEP = 10.0  # control period [s]
SETTLE = int(os.environ.get("SETTLE_STEPS", "150"))

# AR(1) parameters (identical to original eval)
SIGMA_PHI = 2.0
SIGMA_V = 1.0
ALPHA_PHI = 0.95
ALPHA_V = 0.95

# Lookup table settings
RATE_LIMITS = [None, 0.5, 0.3, 0.1]  # None = unlimited

# ---------------------------------------------------------------------------
# Load policies
# ---------------------------------------------------------------------------
positions, _, _ = create_wind_farm_layout_3x3()
N = len(positions)
positions_j = positions_to_jax(positions)

obs_dim_per_step = 5 * N + 3  # USE_POSITIONS=1
obs_dim = 3 * obs_dim_per_step  # J=3
act_dim = N
SETTLE_STATIC = SETTLE  # for JIT

models = []
for s in range(5):
    ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_{DYN_TAG}.pkl")
    if not os.path.exists(ckpt):
        continue
    model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(0))
    graphdef, _ = nnx.split(model)
    with open(ckpt, "rb") as f:
        state = pickle.load(f)
    models.append(nnx.merge(graphdef, state))
    print(f"Loaded seed {s}: {ckpt}")

if not models:
    print(f"ERROR: no checkpoints found for tag '{DYN_TAG}'")
    sys.exit(1)

print(f"Loaded {len(models)} seeds for tag '{DYN_TAG}'")

# ---------------------------------------------------------------------------
# DRL rollout (JIT-compiled)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Lookup table (numpy, from precomputed grid)
# ---------------------------------------------------------------------------
lt_path = os.path.join(FIG_DIR, "lookup_table_baseline.json")
yaw_path = os.path.join(FIG_DIR, "lookup_table_yaw.npy")
with open(lt_path) as f:
    lt = json.load(f)
phi_g = np.array(lt["phi_grid"], dtype=np.float32)
v_g = np.array(lt["v_grid"], dtype=np.float32)
yaw_table = np.load(yaw_path)

def lookup_yaw(phi, v):
    pi = np.clip(np.searchsorted(phi_g, phi) - 1, 0, len(phi_g) - 2)
    vi = np.clip(np.searchsorted(v_g, v) - 1, 0, len(v_g) - 2)
    wp = np.clip((phi - phi_g[pi]) / max(phi_g[pi+1] - phi_g[pi], 1e-6), 0, 1)
    wv = np.clip((v - v_g[vi]) / max(v_g[vi+1] - v_g[vi], 1e-6), 0, 1)
    return (yaw_table[pi,vi]*(1-wp)*(1-wv) + yaw_table[pi+1,vi]*wp*(1-wv)
            + yaw_table[pi,vi+1]*(1-wp)*wv + yaw_table[pi+1,vi+1]*wp*wv)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20260609)

drl_configs = {f"DRL {DYN_TAG}": models}

all_results = {}

for config_name, model_list in drl_configs.items():
    print(f"\n=== {config_name} ({len(model_list)} seeds) ===")
    seed_gains, seed_travels, seed_rates = [], [], []

    for seed_idx, model in enumerate(model_list):
        gains = np.zeros(N_TRAJ)
        travels = np.zeros(N_TRAJ)
        rates = np.zeros(N_TRAJ)

        for traj in range(N_TRAJ):
            phi0 = rng.uniform(173, 353)
            v0 = rng.uniform(6, 16)
            phis = [phi0]; vs = [v0]
            for t in range(1, TRAJ_LEN):
                phis.append(ALPHA_PHI * phis[-1] + (1-ALPHA_PHI) * 270.0 + rng.normal(0, SIGMA_PHI))
                vs.append(ALPHA_V * vs[-1] + (1-ALPHA_V) * 11.4 + rng.normal(0, SIGMA_V))
            phis = np.clip(phis, 173, 353)
            vs = np.clip(vs, 6, 16)

            key = jax.random.key(traj)
            sj, obs = env_reset(key, positions_j,
                                 specific_wind_dir=jnp.float32(phis[0]),
                                 specific_wind_speed=jnp.float32(vs[0]),
                                 randomize_wind=False, j=3)

            gammas_traj_j, final_s = drl_rollout(model, sj, obs)
            gammas_traj = np.asarray(gammas_traj_j)  # (TRAJ_LEN, N)

            total_gain = 0.0; total_travel = 0.0; max_rate = 0.0
            prev_g = np.zeros(N)
            for t in range(TRAJ_LEN):
                g = gammas_traj[t]
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

            gains[traj] = total_gain / TRAJ_LEN
            travels[traj] = total_travel
            rates[traj] = max_rate

        seed_gains.append(gains.mean())
        seed_travels.append(travels.mean())
        seed_rates.append(rates.mean())
        print(f"  seed {seed_idx}: gain={gains.mean():+.4f}%, travel={travels.mean():.1f}°, "
              f"rate={rates.mean():.4f}°/s, eff={gains.mean()/travels.mean()*100:.4f} per 100°")

    all_results[config_name] = {
        "gain_mean": float(np.mean(seed_gains)),
        "gain_std": float(np.std(seed_gains)),
        "travel_mean": float(np.mean(seed_travels)),
        "rate_mean": float(np.mean(seed_rates)),
        "efficiency": float(np.mean(seed_gains) / np.mean(seed_travels) * 100),
    }

# ---------------------------------------------------------------------------
# Lookup table evaluation
# ---------------------------------------------------------------------------
print(f"\n=== Lookup Table ===")
for rate_limit in RATE_LIMITS:
    rl_name = f"unlimited" if rate_limit is None else f"{rate_limit}°/s"
    max_dg = 999 if rate_limit is None else rate_limit * T_STEP

    gains = np.zeros(N_TRAJ); travels = np.zeros(N_TRAJ); rates_arr = np.zeros(N_TRAJ)
    for traj in range(N_TRAJ):
        phi0 = rng.uniform(173, 353); v0 = rng.uniform(6, 16)
        phis = [phi0]; vs = [v0]
        for t in range(1, TRAJ_LEN):
            phis.append(ALPHA_PHI * phis[-1] + (1-ALPHA_PHI) * 270.0 + rng.normal(0, SIGMA_PHI))
            vs.append(ALPHA_V * vs[-1] + (1-ALPHA_V) * 11.4 + rng.normal(0, SIGMA_V))
        phis = np.clip(phis, 173, 353); vs = np.clip(vs, 6, 16)

        total_gain = 0.0; total_travel = 0.0; max_rate = 0.0
        prev_g = np.zeros(N)
        for t in range(TRAJ_LEN):
            target = lookup_yaw(phis[t], vs[t])
            if rate_limit is not None:
                dg = np.clip(target - prev_g, -max_dg, max_dg)
                g = prev_g + dg
            else:
                g = target
            inflow = calculate_inflow_speeds(positions, phis[t], C_T, I, d_0, vs[t], g, alpha_star, beta_star, alpha)
            pwr = sum(power_output(inflow[i], g[i]) for i in range(N)) / 1e6
            base_in = calculate_inflow_speeds(positions, phis[t], C_T, I, d_0, vs[t], np.zeros(N), alpha_star, beta_star, alpha)
            base_pwr = sum(power_output(base_in[i], 0.0) for i in range(N)) / 1e6
            if base_pwr > 0:
                total_gain += (pwr - base_pwr) / base_pwr * 100
            travel = np.abs(g - prev_g).sum()
            total_travel += travel
            max_rate = max(max_rate, travel / T_STEP)
            prev_g = g.copy()
        gains[traj] = total_gain / TRAJ_LEN
        travels[traj] = total_travel
        rates_arr[traj] = max_rate

    eff = gains.mean() / travels.mean() * 100 if travels.mean() > 0 else 0
    print(f"  Lookup {rl_name:>10s}: gain={gains.mean():+.4f}%, travel={travels.mean():.1f}°, "
          f"rate={rates_arr.mean():.4f}°/s, eff={eff:.4f} per 100°")
    all_results[f"Lookup {rl_name}"] = {
        "gain_mean": float(gains.mean()), "travel_mean": float(travels.mean()),
        "rate_mean": float(rates_arr.mean()), "efficiency": float(eff),
    }

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = {
    "description": f"Dynamic wind evaluation: DRL {DYN_TAG} vs lookup table",
    "n_trajectories": N_TRAJ, "traj_length": TRAJ_LEN, "T_step_s": T_STEP,
    "drl_tag": DYN_TAG, "n_seeds": len(models),
    "ar1_params": {"sigma_phi": SIGMA_PHI, "sigma_v": SIGMA_V,
                   "alpha_phi": ALPHA_PHI, "alpha_v": ALPHA_V},
    "results": all_results,
}
out_path = os.path.join(FIG_DIR, "dynamic_wind_optimized.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")

# Comparison with old p0c
print("\n=== COMPARISON WITH OLD P0C ===")
print(f"{'Controller':<25s} {'Gain':>8s} {'Travel':>8s} {'Rate':>8s}")
print("-" * 52)
old_p0c = {"gain": 0.183, "travel": 123.0, "rate": 0.126}
old_lookup_unlimited = {"gain": 1.895, "travel": 2151.5, "rate": 3.320}
old_lookup_01 = {"gain": 1.042, "travel": 496.2, "rate": 0.098}
print(f"{'OLD DRL p0c':<25s} {old_p0c['gain']:+8.4f}% {old_p0c['travel']:8.1f}° {old_p0c['rate']:8.4f}°/s")
print(f"{'OLD Lookup unlimited':<25s} {old_lookup_unlimited['gain']:+8.4f}% {old_lookup_unlimited['travel']:8.1f}° {old_lookup_unlimited['rate']:8.4f}°/s")
print(f"{'OLD Lookup 0.1°/s':<25s} {old_lookup_01['gain']:+8.4f}% {old_lookup_01['travel']:8.1f}° {old_lookup_01['rate']:8.4f}°/s")
for name, res in all_results.items():
    print(f"{name:<25s} {res['gain_mean']:+8.4f}% {res['travel_mean']:8.1f}° {res['rate_mean']:8.4f}°/s")
