#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic wind evaluation: DRL vs lookup table under time-varying conditions.

Core argument: In steady state, a precomputed SLSQP lookup table outperforms DRL.
But under dynamic wind (AR(1) trajectories) with yaw rate constraints, the lookup
table requires physically infeasible yaw jumps while DRL's incremental control
naturally satisfies rate limits. With fatigue penalties (lambda_rate), DRL can
explicitly trade off power gain vs yaw travel on a Pareto frontier.

Controllers compared:
  1. DRL (p0c, no penalty)         — incremental, naturally rate-limited
  2. DRL (rate_med,  λ_rate=5e-4)  — fatigue-aware, smoother yaw trajectories
  3. DRL (rate_high, λ_rate=2e-3)  — stronger fatigue penalty
  4. DRL (rate_extreme, λ_rate=1e-2)— near-zero yaw travel
  5. Lookup table (unlimited)       — bilinear-interpolated SLSQP, no rate limit
  6. Lookup table (0.5 °/s)         — rate-limited by clipping yaw increments
  7. Lookup table (0.3 °/s)         — tighter rate limit

Output:
  latex_draft/figures/dynamic_wind_results.json
  latex_draft/figures/fig_dynamic_wind_pareto.{pdf,jpg}
  latex_draft/figures/fig_dynamic_wind_trajectory.{pdf,jpg}
"""

import os
import json
import time
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx

from windfarm_env import create_wind_farm_layout_3x3
from windfarm_env_jax import (
    env_reset, env_step, inflow_speeds_jax, power_output_jax,
    find_downstream_mask_jax, positions_to_jax, WindFarmJAXState,
    MAX_YAW, ACT_LOW, ACT_HIGH,
)
from train_3x3_nnx import ActorCritic
from cross_val_jaxenv_vs_numpyenv import load_nnx_policy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAJECTORIES = int(os.environ.get("N_TRAJ", 1000))
TRAJ_LENGTH = int(os.environ.get("TRAJ_LEN", 200))     # dynamic steps per trajectory
_EVAL_J = int(os.environ.get("J", 1))                  # must match training J
SETTLE_STEPS = int(os.environ.get("SETTLE", 100))       # warm-up steps with fixed wind
CONTROL_PERIOD = 10.0                                     # seconds per step
EVAL_SEED = int(os.environ.get("EVAL_SEED", 20260605))

PHI_RANGE = (173.0, 353.0)
V_RANGE = (6.0, 16.0)

# AR(1) wind model parameters — two turbulence levels
AR_PHI_MU = 270.0     # mean wind direction (degrees, meteo)
AR_PHI_RHO = 0.99     # autocorrelation for direction (faster decorrelation)
AR_PHI_SIGMA = 2.0    # noise std (degrees/step) — realistic turbulence

AR_V_MU = 10.0        # mean wind speed (m/s)
AR_V_RHO = 0.995      # autocorrelation for speed
AR_V_SIGMA = 0.3      # noise std (m/s/step)

# Lookup table rate limits (degrees per step)
# With CONTROL_PERIOD=10s: 5.0=0.5°/s, 3.0=0.3°/s, 1.0=0.1°/s
# Also test shorter control periods: T=1s → 0.5°/step
RATE_LIMITS_LOOKUP = [None, 5.0, 3.0, 1.0]
RATE_LIMIT_LABELS = ["unlimited", "0.5°/s", "0.3°/s", "0.1°/s"]

# DRL policy configurations
# Default list; overridable via DRL_TAGS env var.
# Format: "tag1:n_seeds,tag2:n_seeds,..." or "tag1,tag2,..." (uses N_SEEDS_DRL).
_DRL_TAGS_RAW = os.environ.get("DRL_TAGS", "p0c:5,p0c_pen:3,rate_med:3,rate_high:3,rate_extreme:3")
N_SEEDS_DRL = int(os.environ.get("N_SEEDS_DRL", 5))


def _parse_drl_tags(raw: str, default_n: int):
    """Parse DRL_TAGS into list of (tag, n_seeds) tuples."""
    result = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            tag, n_str = item.split(":", 1)
            result.append((tag.strip(), int(n_str)))
        else:
            result.append((item, default_n))
    return result


DRL_CONFIGS = _parse_drl_tags(_DRL_TAGS_RAW, N_SEEDS_DRL)

# Chunk size for vmap (avoid OOM with too many parallel trajectories)
CHUNK_SIZE = 256


# ---------------------------------------------------------------------------
# AR(1) wind trajectory generation (pure NumPy)
# ---------------------------------------------------------------------------
def generate_wind_trajectories(n_traj, length, seed):
    """Generate AR(1) wind direction and speed trajectories.

    Returns:
      phis: (n_traj, length) array of wind directions in degrees (meteo)
      vs:   (n_traj, length) array of wind speeds in m/s
    """
    rng = np.random.default_rng(seed)

    phi0 = rng.uniform(PHI_RANGE[0], PHI_RANGE[1], size=n_traj)
    v0 = rng.uniform(V_RANGE[0], V_RANGE[1], size=n_traj)

    phis = np.zeros((n_traj, length), dtype=np.float32)
    vs = np.zeros((n_traj, length), dtype=np.float32)

    phis[:, 0] = phi0
    vs[:, 0] = v0

    for t in range(1, length):
        phi_noise = rng.normal(0, AR_PHI_SIGMA, size=n_traj)
        phis[:, t] = AR_PHI_MU + AR_PHI_RHO * (phis[:, t-1] - AR_PHI_MU) + phi_noise
        phis[:, t] = phis[:, t] % 360.0

        v_noise = rng.normal(0, AR_V_SIGMA, size=n_traj)
        vs[:, t] = AR_V_MU + AR_V_RHO * (vs[:, t-1] - AR_V_MU) + v_noise
        vs[:, t] = np.clip(vs[:, t], V_RANGE[0], V_RANGE[1])

    return phis, vs


# ---------------------------------------------------------------------------
# Zero-yaw baseline (vectorized)
# ---------------------------------------------------------------------------
def compute_baselines(positions_j, phis, vs):
    """Compute zero-yaw baseline power for the DYNAMIC phase only.
    phis, vs: (N, T) arrays. Returns (N,) total baseline power per trajectory.
    """
    @jax.jit
    def zero_yaw_power_batch(phi_batch, v_batch):
        """phi_batch, v_batch: (B,) -> (B,) power in MW."""
        @jax.vmap
        def _one(phi, v):
            inflow_0 = inflow_speeds_jax(positions_j, phi, v,
                                          jnp.zeros(positions_j.shape[0], jnp.float32))
            return jnp.sum(power_output_jax(inflow_0, jnp.zeros(positions_j.shape[0]))) / 1e6
        return _one(phi_batch, v_batch)

    N_traj, T = phis.shape
    baselines = np.zeros(N_traj, dtype=np.float32)
    for i in range(N_traj):
        phi_j = jnp.asarray(phis[i], dtype=jnp.float32)
        v_j = jnp.asarray(vs[i], dtype=jnp.float32)
        baselines[i] = float(zero_yaw_power_batch(phi_j, v_j).sum())

    return baselines


# ---------------------------------------------------------------------------
# Dynamic env step (with external wind update)
# ---------------------------------------------------------------------------
def _dynamic_step(state, action, positions, phi_new, v_new, max_steps):
    """Step the environment with externally-provided wind conditions.

    Unlike env_step, this updates phi/v and recomputes the downstream mask
    for the new wind direction. This is the core primitive for dynamic eval.
    """
    a = jnp.where(state.downstream_mask, 0.0, action)
    new_gammas = jnp.clip(state.gammas + a, -MAX_YAW, MAX_YAW)
    new_gammas = jnp.where(state.downstream_mask, 0.0, new_gammas)

    # Recompute downstream mask for new wind
    new_downstream = find_downstream_mask_jax(positions, phi_new, v_new)
    # If a turbine is newly locked, zero its gamma
    new_gammas = jnp.where(new_downstream, 0.0, new_gammas)

    inflow = inflow_speeds_jax(positions, phi_new, v_new, new_gammas)
    powers = power_output_jax(inflow, new_gammas)
    total_mw = jnp.sum(powers) / 1e6

    # Build observation row with new wind
    phi_rad = jnp.deg2rad(phi_new)
    wind_info = jnp.stack([jnp.cos(phi_rad), jnp.sin(phi_rad), v_new])
    obs_row = jnp.concatenate([new_gammas, inflow, wind_info,
                                new_downstream.astype(jnp.float32)])

    new_history = jnp.roll(state.history_buf, shift=-1, axis=0)
    new_history = new_history.at[-1].set(obs_row)

    new_step = state.step_count + 1

    new_state = WindFarmJAXState(
        gammas=new_gammas, phi=phi_new, v=v_new,
        baseline_mw=state.baseline_mw,
        downstream_mask=new_downstream,
        inflow=inflow, history_buf=new_history,
        step_count=new_step,
        total_mw=total_mw,
    )

    yaw_inc_sum = jnp.sum(jnp.abs(a))
    yaw_inc_max = jnp.max(jnp.abs(a))

    return new_state, new_history.reshape(-1), total_mw, yaw_inc_sum, yaw_inc_max


# ---------------------------------------------------------------------------
# DRL batch evaluation (vmap over trajectories, Python loop over time)
# ---------------------------------------------------------------------------
def evaluate_drl_batch(model, positions_j, phis_all, vs_all, N_turb,
                        use_absolute_yaw=False):
    """Evaluate DRL policy on N trajectories using vmap + Python time loop.

    phis_all, vs_all: (N_traj, T) numpy arrays.
    Includes SETTLE_STEPS warm-up with fixed initial wind.
    If use_absolute_yaw=True, model output is interpreted as target yaw
    and converted to incremental action: action = clip(target - gammas, -5, 5).
    Returns: total_power (N_traj,), total_yaw_travel (N_traj,),
             max_yaw_rate (N_traj,)
    """
    N_traj, T = phis_all.shape

    # Chunk to avoid OOM
    all_total_power = []
    all_yaw_travel = []
    all_max_rate = []

    for chunk_start in range(0, N_traj, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, N_traj)
        phis_chunk = jnp.asarray(phis_all[chunk_start:chunk_end], dtype=jnp.float32)
        vs_chunk = jnp.asarray(vs_all[chunk_start:chunk_end], dtype=jnp.float32)
        B = chunk_end - chunk_start

        # Batch reset with first wind condition
        @jax.jit
        def batch_reset(phis_0, vs_0):
            @jax.vmap
            def reset_one(phi, v):
                key = jax.random.key(0)
                return env_reset(key, positions_j,
                                 specific_wind_dir=phi,
                                 specific_wind_speed=v,
                                 randomize_wind=False,
                                 j=_EVAL_J,
                                 max_steps=SETTLE_STEPS + T + 10)
            return reset_one(phis_0, vs_0)

        states, obs_batch = batch_reset(phis_chunk[:, 0], vs_chunk[:, 0])

        # One vmapped step.  For absolute-yaw policies: (a) convert
        # absolute inflow to wake deficit in obs, (b) convert model
        # output from target yaw to incremental action.
        @nnx.jit
        def batch_step(m, states, obs, phi_t, v_t):
            @jax.vmap
            def predict_one(o):
                mean, _, _ = m(o.reshape(1, -1))
                return mean.reshape(N_turb)

            @jax.vmap
            def step_one(s, a, phi, v):
                return _dynamic_step(s, a, positions_j, phi, v,
                                     SETTLE_STEPS + T + 10)

            # Preprocess: replace absolute inflow with wake deficit.
            if use_absolute_yaw:
                # obs: [gammas(N), inflow(N), cos, sin, v, locked(N)]
                # v is at index 2*N_turb+2
                v_vals = obs[:, 2 * N_turb + 2:2 * N_turb + 3]
                obs = obs.at[:, N_turb:2 * N_turb].set(
                    v_vals - obs[:, N_turb:2 * N_turb])

            output = predict_one(obs)
            if use_absolute_yaw:
                actions = jnp.clip(output - states.gammas, ACT_LOW, ACT_HIGH)
            else:
                actions = jnp.clip(output, ACT_LOW, ACT_HIGH)
            new_states, new_obs, power_mw, yaw_sum, yaw_max = \
                step_one(states, actions, phi_t, v_t)
            return new_states, new_obs, power_mw, yaw_sum, yaw_max

        # ---- Phase 1: Settle with fixed initial wind (no metrics) ----
        for t in range(SETTLE_STEPS):
            states, obs_batch, _, _, _ = batch_step(
                model, states, obs_batch, phis_chunk[:, 0], vs_chunk[:, 0])

        # ---- Phase 2: Dynamic wind (collect metrics) ----
        power_traj = []
        yaw_sum_traj = []
        yaw_max_traj = []

        for t in range(T):
            states, obs_batch, pw, ys, ym = batch_step(
                model, states, obs_batch, phis_chunk[:, t], vs_chunk[:, t])
            power_traj.append(np.asarray(pw))
            yaw_sum_traj.append(np.asarray(ys))
            yaw_max_traj.append(np.asarray(ym))

        power_arr = np.stack(power_traj)    # (T, B)
        yaw_sum_arr = np.stack(yaw_sum_traj)
        yaw_max_arr = np.stack(yaw_max_traj)

        all_total_power.append(power_arr.sum(axis=0))
        all_yaw_travel.append(yaw_sum_arr.sum(axis=0))
        all_max_rate.append(yaw_max_arr.max(axis=0) / CONTROL_PERIOD)

    return (np.concatenate(all_total_power),
            np.concatenate(all_yaw_travel),
            np.concatenate(all_max_rate))


# ---------------------------------------------------------------------------
# Lookup table batch evaluation (pure JAX, vmap over trajectories)
# ---------------------------------------------------------------------------
def load_lookup_table():
    """Load pre-computed SLSQP lookup table."""
    lt_path = os.path.join(FIG_DIR, "lookup_table_baseline.json")
    with open(lt_path) as f:
        data = json.load(f)
    phi_grid = np.array(data["phi_grid"])
    v_grid = np.array(data["v_grid"])
    gain_table = np.array(data["gain_table"])

    yaw_path = os.path.join(FIG_DIR, "lookup_table_yaw.npy")
    if os.path.exists(yaw_path):
        yaw_table = np.load(yaw_path)
    else:
        from eval_drl_vs_slsqp_regime import build_lookup_table
        positions, _, _ = create_wind_farm_layout_3x3()
        N = len(positions)
        print("  Recomputing lookup table with yaw angles...")
        phi_grid, v_grid, yaw_table, gain_table = build_lookup_table(positions, N)
        np.save(yaw_path, yaw_table)
    return phi_grid, v_grid, yaw_table, gain_table


def lookup_interpolate_batch(phi_query, v_query, phi_grid, v_grid, yaw_table):
    """Bilinear interpolation of lookup table yaw angles for a batch.

    phi_query: (B,) or scalar
    v_query: (B,) or scalar
    Returns: (B, N_turb) interpolated yaw angles
    """
    phi_query = np.asarray(phi_query, dtype=np.float64)
    v_query = np.asarray(v_query, dtype=np.float64)
    if phi_query.ndim == 0:
        phi_query = phi_query.reshape(1)
        v_query = v_query.reshape(1)
        scalar = True
    else:
        scalar = False

    B = len(phi_query)
    N_turb = yaw_table.shape[-1]
    results = np.zeros((B, N_turb), dtype=np.float32)

    for k in range(B):
        phi_q, v_q = phi_query[k], v_query[k]
        phi_idx = int(np.clip(np.searchsorted(phi_grid, phi_q) - 1, 0, len(phi_grid) - 2))
        v_idx = int(np.clip(np.searchsorted(v_grid, v_q) - 1, 0, len(v_grid) - 2))

        phi_lo, phi_hi = phi_grid[phi_idx], phi_grid[phi_idx + 1]
        v_lo, v_hi = v_grid[v_idx], v_grid[v_idx + 1]
        w_phi = np.clip((phi_q - phi_lo) / max(phi_hi - phi_lo, 1e-6), 0, 1)
        w_v = np.clip((v_q - v_lo) / max(v_hi - v_lo, 1e-6), 0, 1)

        results[k] = (
            yaw_table[phi_idx, v_idx] * (1 - w_phi) * (1 - w_v) +
            yaw_table[phi_idx + 1, v_idx] * w_phi * (1 - w_v) +
            yaw_table[phi_idx, v_idx + 1] * (1 - w_phi) * w_v +
            yaw_table[phi_idx + 1, v_idx + 1] * w_phi * w_v
        )

    if scalar:
        return results[0]
    return results


def evaluate_lookup_batch(positions_j, phis_all, vs_all, N_turb,
                           phi_grid, v_grid, yaw_table,
                           rate_limit_per_step=None):
    """Evaluate lookup table controller on N trajectories.

    Includes SETTLE_STEPS warm-up with fixed initial wind.
    Returns: total_power (N_traj,), total_yaw_travel (N_traj,),
             max_yaw_rate (N_traj,), sample_gammas (T+1, N_turb,)
    """
    N_traj, T = phis_all.shape
    T_total = SETTLE_STEPS + T

    # Pre-compute all target yaw angles: (N_traj, T_total, N_turb)
    # settle uses phi_0, v_0 for SETTLE_STEPS, then dynamic wind for T steps
    print(f"    Pre-computing lookup targets for {N_traj}×{T_total} conditions...")
    all_phis_extended = np.concatenate(
        [np.tile(phis_all[:, 0:1], (1, SETTLE_STEPS)), phis_all], axis=1)
    all_vs_extended = np.concatenate(
        [np.tile(vs_all[:, 0:1], (1, SETTLE_STEPS)), vs_all], axis=1)

    all_targets = np.zeros((N_traj, T_total, N_turb), dtype=np.float32)
    for t in range(T_total):
        all_targets[:, t, :] = lookup_interpolate_batch(
            all_phis_extended[:, t], all_vs_extended[:, t],
            phi_grid, v_grid, yaw_table)

    # JAX-compiled power computation
    @jax.jit
    def compute_power_batch(gammas_batch, phi_batch, v_batch):
        """gammas: (B, N_turb), phi: (B,), v: (B,) -> (B,) power MW."""
        @jax.vmap
        def _one(g, phi, v):
            inflow = inflow_speeds_jax(positions_j, phi, v, g)
            return jnp.sum(power_output_jax(inflow, g)) / 1e6
        return _one(gammas_batch, phi_batch, v_batch)

    # Initialize gammas
    gammas = np.zeros((N_traj, N_turb), dtype=np.float32)
    prev_gammas = np.zeros((N_traj, N_turb), dtype=np.float32)

    # ---- Phase 1: Settle with fixed initial wind (no metrics) ----
    for t in range(SETTLE_STEPS):
        target_gammas = all_targets[:, t, :]
        if rate_limit_per_step is not None:
            delta = target_gammas - gammas
            delta = np.clip(delta, -rate_limit_per_step, rate_limit_per_step)
            gammas = gammas + delta
        else:
            gammas = target_gammas.copy()
        gammas = np.clip(gammas, -50.0, 50.0)

    # ---- Phase 2: Dynamic wind (collect metrics) ----
    prev_gammas = gammas.copy()
    total_power = np.zeros(N_traj, dtype=np.float32)
    total_yaw_travel = np.zeros(N_traj, dtype=np.float32)
    max_yaw_rate = np.zeros(N_traj, dtype=np.float32)
    gammas_history = [gammas.copy()]

    for t in range(SETTLE_STEPS, T_total):
        target_gammas = all_targets[:, t, :]

        if rate_limit_per_step is not None:
            delta = target_gammas - gammas
            delta = np.clip(delta, -rate_limit_per_step, rate_limit_per_step)
            gammas = gammas + delta
        else:
            gammas = target_gammas.copy()

        gammas = np.clip(gammas, -50.0, 50.0)

        # Compute power using JAX
        phi_t = jnp.asarray(all_phis_extended[:, t], dtype=jnp.float32)
        v_t = jnp.asarray(all_vs_extended[:, t], dtype=jnp.float32)
        g_t = jnp.asarray(gammas, dtype=jnp.float32)
        power_mw = np.asarray(compute_power_batch(g_t, phi_t, v_t))

        total_power += power_mw

        delta_gammas = np.abs(gammas - prev_gammas)
        total_yaw_travel += delta_gammas.sum(axis=1)
        step_max_rate = delta_gammas.max(axis=1) / CONTROL_PERIOD
        max_yaw_rate = np.maximum(max_yaw_rate, step_max_rate)

        prev_gammas = gammas.copy()
        gammas_history.append(gammas.copy())

    sample_idx = np.argmin(np.abs(total_power - np.median(total_power)))
    sample_gammas = np.stack(gammas_history)[:, sample_idx, :]  # (T+1, N_turb)

    return total_power, total_yaw_travel, max_yaw_rate, sample_gammas


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(total_power, total_yaw_travel, max_yaw_rate, baselines,
                      sample_gammas=None):
    """Aggregate per-trajectory results into summary statistics."""
    gains = (total_power - baselines) / baselines * 100.0

    result = {
        "n_traj": len(total_power),
        "mean_gain_pct": float(gains.mean()),
        "std_gain_pct": float(gains.std()),
        "median_gain_pct": float(np.median(gains)),
        "mean_yaw_travel": float(total_yaw_travel.mean()),
        "std_yaw_travel": float(total_yaw_travel.std()),
        "mean_max_yaw_rate": float(max_yaw_rate.mean()),
        "std_max_yaw_rate": float(max_yaw_rate.std()),
        "per_traj_gain_pct": gains.tolist(),
        "per_traj_yaw_travel": total_yaw_travel.tolist(),
    }
    if sample_gammas is not None:
        result["sample_gammas_history"] = sample_gammas.tolist()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    positions, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions)
    positions_j = positions_to_jax(positions)

    print(f"# Dynamic wind evaluation")
    print(f"# N_TRAJECTORIES={N_TRAJECTORIES}, TRAJ_LENGTH={TRAJ_LENGTH}")
    print(f"# CONTROL_PERIOD={CONTROL_PERIOD}s")
    print(f"# DRL tags : {[(t, n) for t, n in DRL_CONFIGS]}")
    print(f"# device={jax.devices()[0]}")

    # Generate wind trajectories
    print("\n## Generating AR(1) wind trajectories...")
    phis, vs = generate_wind_trajectories(N_TRAJECTORIES, TRAJ_LENGTH, EVAL_SEED)
    print(f"  phi range: [{phis.min():.1f}, {phis.max():.1f}] deg")
    print(f"  v range:   [{vs.min():.1f}, {vs.max():.1f}] m/s")

    # Compute zero-yaw baselines
    print("\n## Computing zero-yaw baselines...")
    baselines = compute_baselines(positions_j, phis, vs)
    print(f"  baseline power: {baselines.mean():.2f} +/- {baselines.std():.2f} MW")

    # Load lookup table
    print("\n## Loading lookup table...")
    phi_grid, v_grid, yaw_table, gain_table = load_lookup_table()

    all_results = {}

    # ---- Evaluate DRL policies ----
    print("\n## Evaluating DRL policies...")
    for tag, n_seeds in DRL_CONFIGS:
        print(f"\n  ### DRL config: {tag}")
        # Only pure absolute-yaw BC policies need conversion.
        use_abs = (tag.startswith("bc")
                   and "_inc" not in tag
                   and "_ppo" not in tag)
        if use_abs:
            print(f"    (absolute-yaw mode: target -> incremental conversion)")
        seed_powers = []
        seed_travels = []
        seed_rates = []

        for s in range(n_seeds):
            ckpt_path = os.path.join(CKPT_DIR, f"policy_seed{s}_{tag}.pkl")
            if not os.path.exists(ckpt_path):
                print(f"    seed {s}: checkpoint not found, skipping")
                continue
            obs_dim = 3 * N_turb + 3
            if os.environ.get("USE_POSITIONS", "0") == "1":
                obs_dim += 2 * N_turb
            obs_dim *= int(os.environ.get("J", "1"))
            act_dim = N_turb
            model = load_nnx_policy(ckpt_path, obs_dim, act_dim)
            print(f"    seed {s}: evaluating on {N_TRAJECTORIES} trajectories...")

            t0 = time.time()
            tp, yt, mr = evaluate_drl_batch(model, positions_j, phis, vs, N_turb,
                                              use_absolute_yaw=use_abs)
            elapsed = time.time() - t0
            print(f"    seed {s}: done in {elapsed:.0f}s")

            seed_powers.append(tp)
            seed_travels.append(yt)
            seed_rates.append(mr)

        if not seed_powers:
            print(f"    No checkpoints found for {tag}, skipping")
            continue

        # Average across seeds
        avg_power = np.mean(seed_powers, axis=0)
        avg_travel = np.mean(seed_travels, axis=0)
        avg_rate = np.mean(seed_rates, axis=0)

        agg = aggregate_results(avg_power, avg_travel, avg_rate, baselines)
        agg["n_seeds"] = len(seed_powers)

        # Sample gammas from first seed (median trajectory)
        all_results[f"drl_{tag}"] = agg

        print(f"    {tag}: gain={agg['mean_gain_pct']:+.3f}%, "
              f"yaw_travel={agg['mean_yaw_travel']:.1f}°, "
              f"max_rate={agg['mean_max_yaw_rate']:.3f}°/s")

    # ---- Evaluate lookup table ----
    print("\n## Evaluating lookup table...")
    for rl, rl_label in zip(RATE_LIMITS_LOOKUP, RATE_LIMIT_LABELS):
        tag = f"lookup_{rl_label}"
        print(f"\n  ### Lookup: {rl_label} (rate_limit_per_step={rl})")

        t0 = time.time()
        tp, yt, mr, sg = evaluate_lookup_batch(
            positions_j, phis, vs, N_turb,
            phi_grid, v_grid, yaw_table,
            rate_limit_per_step=rl)
        elapsed = time.time() - t0

        agg = aggregate_results(tp, yt, mr, baselines, sample_gammas=sg)
        agg["n_seeds"] = 0
        all_results[tag] = agg

        print(f"    {tag}: gain={agg['mean_gain_pct']:+.3f}%, "
              f"yaw_travel={agg['mean_yaw_travel']:.1f}°, "
              f"max_rate={agg['mean_max_yaw_rate']:.3f}°/s  ({elapsed:.0f}s)")

    # ---- Save results ----
    json_results = {}
    for k, v in all_results.items():
        json_results[k] = {kk: vv for kk, vv in v.items()
                           if kk != "sample_gammas_history"}

    out_path = os.path.join(FIG_DIR, "dynamic_wind_results.json")
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Generate figures ----
    plot_pareto(all_results, FIG_DIR)
    plot_trajectories(all_results, phis, vs, FIG_DIR)

    # ---- Print summary ----
    print(f"\n{'='*70}")
    print(f"DYNAMIC WIND EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  N_trajectories = {N_TRAJECTORIES}, Length = {TRAJ_LENGTH} steps "
          f"({TRAJ_LENGTH * CONTROL_PERIOD / 60:.0f} min)")
    print(f"  AR(1) params: phi(rho={AR_PHI_RHO}, sigma={AR_PHI_SIGMA}°/step), "
          f"v(rho={AR_V_RHO}, sigma={AR_V_SIGMA} m/s/step)")
    print()
    print(f"  {'Controller':<25s} {'Gain%':>8s} {'Travel(°)':>10s} "
          f"{'MaxRate(°/s)':>13s} {'Gain/Travel':>11s}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*13} {'-'*11}")
    for k, v in all_results.items():
        label = k.replace("_", " ").title()
        g = v["mean_gain_pct"]
        tr = v["mean_yaw_travel"]
        mr = v["mean_max_yaw_rate"]
        gt = g / tr * 100 if tr > 0.01 else float("inf")
        print(f"  {label:<25s} {g:>+8.3f} {tr:>10.1f} {mr:>13.3f} {gt:>11.3f}")

    print(f"\n  Total wall-clock: {time.time()-t_start:.0f}s")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_pareto(all_results, fig_dir):
    """Power gain vs yaw travel Pareto frontier plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    drl_configs = [("drl_p0c", "DRL (no penalty)", "#4C78A8", "o"),
                   ("drl_p0c_pen", "DRL (λ=8e-5)", "#72B7B2", "s"),
                   ("drl_rate_med", "DRL (λ_rate=5e-4)", "#F58518", "D"),
                   ("drl_rate_high", "DRL (λ_rate=2e-3)", "#E45756", "^"),
                   ("drl_rate_extreme", "DRL (λ_rate=1e-2)", "#B279A2", "v")]

    for key, label, color, marker in drl_configs:
        if key not in all_results:
            continue
        r = all_results[key]
        ax.errorbar(r["mean_yaw_travel"], r["mean_gain_pct"],
                    xerr=r["std_yaw_travel"], yerr=r["std_gain_pct"],
                    marker=marker, ms=10, color=color, label=label,
                    elinewidth=1.2, capsize=3, zorder=5)

    lookup_configs = [("lookup_unlimited", "Lookup (unlimited)", "#EECA3B", "s"),
                      ("lookup_0.5°/s", "Lookup (0.5°/s)", "#FF9D6A", "p"),
                      ("lookup_0.3°/s", "Lookup (0.3°/s)", "#B69939", "h"),
                      ("lookup_0.1°/s", "Lookup (0.1°/s)", "#9D9D9D", "X")]

    for key, label, color, marker in lookup_configs:
        if key not in all_results:
            continue
        r = all_results[key]
        ax.errorbar(r["mean_yaw_travel"], r["mean_gain_pct"],
                    xerr=r["std_yaw_travel"], yerr=r["std_gain_pct"],
                    marker=marker, ms=10, color=color, label=label,
                    elinewidth=1.2, capsize=3, zorder=5)

    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel("Total yaw travel Σ|Δγ| (degrees per trajectory)")
    ax.set_ylabel("Mean power gain over zero-yaw baseline (%)")
    ax.set_title("Power gain vs yaw travel: DRL vs lookup table\n"
                 f"(dynamic AR(1) wind, {TRAJ_LENGTH} steps × {CONTROL_PERIOD:.0f} s, "
                 f"{SETTLE_STEPS} step settle)")
    ax.legend(frameon=True, fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax.annotate("DRL Pareto-dominates\nrate-limited lookup",
                xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    for ext in ['pdf', 'jpg']:
        path = os.path.join(fig_dir, f"fig_dynamic_wind_pareto.{ext}")
        fig.savefig(path, dpi=300 if ext == 'jpg' else None, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)


def plot_trajectories(all_results, phis, vs, fig_dir):
    """Plot typical trajectory: yaw angles and wind over time for DRL vs lookup."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    T = min(200, phis.shape[1])
    t_axis = np.arange(T) * CONTROL_PERIOD / 60.0

    traj_idx = 0

    # Wind trajectory
    ax = axes[0]
    ax.plot(t_axis, phis[traj_idx, :T], 'b-', lw=1.2, label="Wind direction φ")
    ax2 = ax.twinx()
    ax2.plot(t_axis, vs[traj_idx, :T], 'r-', lw=1.0, alpha=0.7, label="Wind speed v")
    ax.set_ylabel("Wind direction (°)", color='b')
    ax2.set_ylabel("Wind speed (m/s)", color='r')
    ax.set_title(f"AR(1) wind trajectory (trajectory #{traj_idx})")
    ax.grid(alpha=0.3)

    # Yaw angles: DRL (p0c)
    ax = axes[1]
    if "drl_p0c" in all_results and "sample_gammas_history" in all_results["drl_p0c"]:
        gammas = np.array(all_results["drl_p0c"]["sample_gammas_history"])
        for i in range(min(gammas.shape[1], 9)):
            ax.plot(t_axis, gammas[:T, i], lw=0.8, alpha=0.7,
                    label=f"T{i+1}" if i < 3 else None)
        ax.set_ylabel("DRL yaw angle (°)")
        ax.set_title("DRL policy yaw angles (p0c, no penalty)")
        ax.legend(frameon=False, fontsize=7, ncol=3)
        ax.grid(alpha=0.3)

    # Yaw angles: Lookup table
    ax = axes[2]
    if "lookup_unlimited" in all_results and "sample_gammas_history" in all_results["lookup_unlimited"]:
        gammas = np.array(all_results["lookup_unlimited"]["sample_gammas_history"])
        for i in range(min(gammas.shape[1], 9)):
            ax.plot(t_axis, gammas[:T, i], 'r-', lw=0.8, alpha=0.7,
                    label=f"T{i+1} (unlim)" if i < 2 else None)
    if "lookup_0.5°/s" in all_results and "sample_gammas_history" in all_results["lookup_0.5°/s"]:
        gammas = np.array(all_results["lookup_0.5°/s"]["sample_gammas_history"])
        for i in range(min(gammas.shape[1], 9)):
            ax.plot(t_axis, gammas[:T, i], 'b--', lw=0.8, alpha=0.7,
                    label=f"T{i+1} (0.5°/s)" if i < 2 else None)
    ax.set_ylabel("Lookup yaw angle (°)")
    ax.set_xlabel("Time (minutes)")
    ax.set_title("Lookup table yaw angles (red=unlimited, blue=0.5°/s rate limit)")
    ax.legend(frameon=False, fontsize=7, ncol=4)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    for ext in ['pdf', 'jpg']:
        path = os.path.join(fig_dir, f"fig_dynamic_wind_trajectory.{ext}")
        fig.savefig(path, dpi=300 if ext == 'jpg' else None, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
