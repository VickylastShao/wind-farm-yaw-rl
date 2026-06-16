#!/usr/bin/env python3
"""Dynamic wind evaluation with industrial baselines: low-pass filter + hysteresis.

Extends eval_dynamic_optimized.py with:
  E2a: Lookup table + 1st-order low-pass filter (tau = 30, 60, 120 s)
  E2b: Hysteresis deadband applied to both lookup table and DRL policies
        (deadband = 2, 5, 8 degrees)

Key insight: hysteresis filters sub-threshold DRL micro-adjustments
(mean |Δγ| ≈ 0.03°/step << deadband), making reported travel an overestimate.
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

# AR(1) parameters (identical to eval_dynamic_optimized.py)
SIGMA_PHI = 2.0
SIGMA_V = 1.0
ALPHA_PHI = 0.95
ALPHA_V = 0.95

# Rate limits for lookup table
RATE_LIMITS = [None, 0.5, 0.3, 0.1]

# Low-pass filter time constants [s]
LP_TAUS = [30.0, 60.0, 120.0]

# Hysteresis deadbands [degrees]
DEADBANDS = [2.0, 5.0, 8.0]

# ---------------------------------------------------------------------------
# Load policies
# ---------------------------------------------------------------------------
positions, _, _ = create_wind_farm_layout_3x3()
N = len(positions)
positions_j = positions_to_jax(positions)

obs_dim_per_step = 5 * N + 3  # USE_POSITIONS=1
obs_dim = 3 * obs_dim_per_step  # J=3
act_dim = N

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

def apply_rate_limit(target, prev_g, max_dg):
    dg = np.clip(target - prev_g, -max_dg, max_dg)
    return prev_g + dg

def apply_lowpass(target, lp_prev, alpha):
    """1st-order low-pass filter: y_t = α·target + (1-α)·y_{t-1}"""
    return alpha * target + (1.0 - alpha) * lp_prev

def apply_hysteresis(g, prev_g, deadband):
    """Hysteresis deadband: don't move if |Δγ| < deadband for ALL turbines."""
    if np.all(np.abs(g - prev_g) < deadband):
        return prev_g.copy()
    return g

# ---------------------------------------------------------------------------
# Generate AR(1) trajectories (shared across all evaluations)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20260609)
all_trajs = []
for traj in range(N_TRAJ):
    phi0 = rng.uniform(173, 353)
    v0 = rng.uniform(6, 16)
    phis = [phi0]; vs = [v0]
    for t in range(1, TRAJ_LEN):
        phis.append(ALPHA_PHI * phis[-1] + (1-ALPHA_PHI) * 270.0 + rng.normal(0, SIGMA_PHI))
        vs.append(ALPHA_V * vs[-1] + (1-ALPHA_V) * 11.4 + rng.normal(0, SIGMA_V))
    phis = np.clip(phis, 173, 353)
    vs = np.clip(vs, 6, 16)
    all_trajs.append((np.array(phis), np.array(vs)))

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def compute_metrics(gammas_log, phis, vs):
    """Compute gain, travel, rate for a sequence of yaw vectors."""
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

def compute_metrics_hysteresis(gammas_log, phis, vs, deadband):
    """Compute metrics with hysteresis: sub-deadband moves are NOT executed."""
    total_gain = 0.0; total_travel = 0.0; max_rate = 0.0
    prev_g = np.zeros(N)
    actual_g = np.zeros(N)  # actual yaw after hysteresis
    for t in range(TRAJ_LEN):
        target_g = gammas_log[t]
        # Apply hysteresis to each turbine independently
        g = target_g.copy()
        for i in range(N):
            if abs(target_g[i] - actual_g[i]) < deadband:
                g[i] = actual_g[i]  # don't move
        actual_g = g.copy()
        # Compute power using ACTUAL (post-hysteresis) yaw
        inflow = inflow_speeds_jax(positions_j, jnp.float32(phis[t]),
                                    jnp.float32(vs[t]), jnp.array(g, dtype=jnp.float32))
        pwr = float(jnp.sum(power_output_jax(inflow, jnp.array(g, dtype=jnp.float32))) / 1e6)
        base_in = inflow_speeds_jax(positions_j, jnp.float32(phis[t]),
                                     jnp.float32(vs[t]), jnp.zeros(N, dtype=jnp.float32))
        base_pwr = float(jnp.sum(power_output_jax(base_in, jnp.zeros(N, dtype=jnp.float32))) / 1e6)
        if base_pwr > 0:
            total_gain += (pwr - base_pwr) / base_pwr * 100
        # Travel only counts MOVEMENTS that exceed deadband
        for i in range(N):
            if abs(g[i] - prev_g[i]) >= deadband:
                total_travel += abs(g[i] - prev_g[i])
        max_rate = max(max_rate, np.abs(g - prev_g).sum() / T_STEP)
        prev_g = g.copy()
    return total_gain / TRAJ_LEN, total_travel, max_rate

def compute_metrics_with_lp(gammas_log, phis, vs, alpha, rate_limit, max_dg):
    """Compute metrics where lookup table output goes through LP filter + rate limit."""
    total_gain = 0.0; total_travel = 0.0; max_rate = 0.0
    prev_g = np.zeros(N)
    lp_state = np.zeros(N)  # LP filter state
    for t in range(TRAJ_LEN):
        target = lookup_yaw(phis[t], vs[t])
        # Apply low-pass filter
        lp_state = apply_lowpass(target, lp_state, alpha)
        # Apply rate limit to filtered target
        if rate_limit is not None:
            g = apply_rate_limit(lp_state, prev_g, max_dg)
        else:
            g = lp_state
        # Compute power
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
    return total_gain / TRAJ_LEN, total_travel, max_rate

# ---------------------------------------------------------------------------
# E2a: Low-pass filter baseline
# ---------------------------------------------------------------------------
all_results = {}

print(f"\n{'='*60}")
print("E2a: Low-Pass Filter Baseline (Lookup Table)")
print(f"{'='*60}")

for tau in LP_TAUS:
    alpha = T_STEP / tau
    for rate_limit in RATE_LIMITS:
        rl_name = f"unlimited" if rate_limit is None else f"{rate_limit}°/s"
        max_dg = 999 if rate_limit is None else rate_limit * T_STEP
        label = f"Lookup LP τ={tau:.0f}s RL={rl_name}"

        gains = np.zeros(N_TRAJ); travels = np.zeros(N_TRAJ); rates_arr = np.zeros(N_TRAJ)
        for traj in range(N_TRAJ):
            phis, vs = all_trajs[traj]
            g, t, r = compute_metrics_with_lp(None, phis, vs, alpha, rate_limit, max_dg)
            gains[traj] = g; travels[traj] = t; rates_arr[traj] = r

        eff = gains.mean() / travels.mean() * 100 if travels.mean() > 0 else 0
        print(f"  {label:<35s}: gain={gains.mean():+.4f}%, travel={travels.mean():.1f}°, "
              f"rate={rates_arr.mean():.4f}°/s, eff={eff:.4f}/100°")
        all_results[label] = {
            "gain_mean": float(gains.mean()), "travel_mean": float(travels.mean()),
            "rate_mean": float(rates_arr.mean()), "efficiency": float(eff),
            "type": "lookup_lp", "tau": tau, "rate_limit": rl_name,
        }

# ---------------------------------------------------------------------------
# E2b: Hysteresis deadband (DRL + Lookup Table)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("E2b: Hysteresis Deadband (DRL + Lookup Table)")
print(f"{'='*60}")

# --- DRL with hysteresis ---
print("\n--- DRL policies with hysteresis ---")
for seed_idx, model in enumerate(models):
    for deadband in DEADBANDS:
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
            g, t, r = compute_metrics_hysteresis(gammas_traj, phis, vs, deadband)
            gains[traj] = g; travels[traj] = t; rates_arr[traj] = r

        eff = gains.mean() / travels.mean() * 100 if travels.mean() > 0 else 0
        label = f"DRL seed{seed_idx} db={deadband}°"
        print(f"  {label:<30s}: gain={gains.mean():+.4f}%, travel={travels.mean():.1f}°, "
              f"rate={rates_arr.mean():.4f}°/s, eff={eff:.4f}/100°")
        all_results[label] = {
            "gain_mean": float(gains.mean()), "travel_mean": float(travels.mean()),
            "rate_mean": float(rates_arr.mean()), "efficiency": float(eff),
            "type": "drl_hysteresis", "seed": seed_idx, "deadband": deadband,
        }

# --- Lookup table with hysteresis ---
print("\n--- Lookup table with hysteresis ---")
for deadband in DEADBANDS:
    for rate_limit in RATE_LIMITS:
        rl_name = f"unlimited" if rate_limit is None else f"{rate_limit}°/s"
        max_dg = 999 if rate_limit is None else rate_limit * T_STEP
        label = f"Lookup db={deadband}° RL={rl_name}"

        gains = np.zeros(N_TRAJ); travels = np.zeros(N_TRAJ); rates_arr = np.zeros(N_TRAJ)
        for traj in range(N_TRAJ):
            phis, vs = all_trajs[traj]
            total_gain = 0.0; total_travel = 0.0; max_rate = 0.0
            prev_g = np.zeros(N)
            for t in range(TRAJ_LEN):
                target = lookup_yaw(phis[t], vs[t])
                if rate_limit is not None:
                    g = apply_rate_limit(target, prev_g, max_dg)
                else:
                    g = target
                # Apply hysteresis
                g = apply_hysteresis(g, prev_g, deadband)
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
        print(f"  {label:<30s}: gain={gains.mean():+.4f}%, travel={travels.mean():.1f}°, "
              f"rate={rates_arr.mean():.4f}°/s, eff={eff:.4f}/100°")
        all_results[label] = {
            "gain_mean": float(gains.mean()), "travel_mean": float(travels.mean()),
            "rate_mean": float(rates_arr.mean()), "efficiency": float(eff),
            "type": "lookup_hysteresis", "deadband": deadband, "rate_limit": rl_name,
        }

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("SUMMARY: Key Comparisons")
print(f"{'='*60}")
print(f"{'Configuration':<40s} {'Gain':>8s} {'Travel':>8s} {'Rate':>8s} {'Eff/100°':>10s}")
print("-" * 75)

# Re-run baseline DRL (no hysteresis) for comparison
print("--- Baseline (no filter/hysteresis) ---")
drl_configs = {f"DRL {DYN_TAG}": models}
for config_name, model_list in drl_configs.items():
    seed_gains, seed_travels, seed_rates = [], [], []
    for seed_idx, model in enumerate(model_list):
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
        seed_gains.append(gains.mean()); seed_travels.append(travels.mean()); seed_rates.append(rates_arr.mean())
        eff = gains.mean() / travels.mean() * 100 if travels.mean() > 0 else 0
        print(f"  DRL seed{seed_idx:<30s} {gains.mean():+8.4f}% {travels.mean():8.1f}° {rates_arr.mean():8.4f}°/s {eff:10.4f}")

# Key comparison: DRL with db=5° vs DRL raw
print("\n--- KEY: DRL travel with hysteresis ---")
for deadband in DEADBANDS:
    for seed_idx in [0]:
        key_label = f"DRL seed{seed_idx} db={deadband}°"
        if key_label in all_results:
            r = all_results[key_label]
            print(f"  {key_label:<30s}: travel={r['travel_mean']:.1f}° (vs raw)")

# Key comparison: Lookup LP at 0.1°/s
print("\n--- KEY: Lookup LP filter at 0.1°/s rate limit ---")
for tau in LP_TAUS:
    key_label = f"Lookup LP τ={tau:.0f}s RL=0.1°/s"
    if key_label in all_results:
        r = all_results[key_label]
        print(f"  {key_label:<35s}: gain={r['gain_mean']:+.4f}%, travel={r['travel_mean']:.1f}°")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = {
    "description": "Industrial baseline experiments: low-pass filter + hysteresis",
    "n_trajectories": N_TRAJ, "traj_length": TRAJ_LEN, "T_step_s": T_STEP,
    "drl_tag": DYN_TAG, "n_seeds": len(models),
    "ar1_params": {"sigma_phi": SIGMA_PHI, "sigma_v": SIGMA_V,
                   "alpha_phi": ALPHA_PHI, "alpha_v": ALPHA_V},
    "lp_taus": LP_TAUS,
    "deadbands": DEADBANDS,
    "results": all_results,
}
out_path = os.path.join(FIG_DIR, "industrial_baseline_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
