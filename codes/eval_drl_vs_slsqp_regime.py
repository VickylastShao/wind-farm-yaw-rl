#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRL vs SLSQP regime-wise comparison.

Evaluates both the DRL policy and SLSQP optimizer on the same 500
uniformly-sampled conditions, providing a per-condition head-to-head
comparison.  Directly addresses reviewer concerns about:
  C2: DRL vs offline optimization ceiling
  C4: Conditions where DRL actively reduces performance

Also computes a lookup-table baseline (SLSQP on a 91x11 grid, then
bilinear interpolation to arbitrary conditions) to quantify the
cost of "online adaptivity" vs a simple precomputed table.

Output:
  latex_draft/figures/drl_vs_slsqp_regime.json
  latex_draft/figures/fig_drl_vs_slsqp_scatter.{pdf,jpg}
  latex_draft/figures/lookup_table_baseline.json
"""

import os
import sys
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from flax import nnx

from windfarm_env import (
    create_wind_farm_layout_3x3,
    calculate_inflow_speeds, power_output,
    C_T, I, d_0, alpha_star, beta_star, alpha,
    u_rated, P_rated, rho, S, C_P, z_h,
)
from windfarm_env_jax import (
    env_reset, env_step, positions_to_jax,
    inflow_speeds_jax, power_output_jax,
)
from train_3x3_nnx import ActorCritic
from cross_val_jaxenv_vs_numpyenv import load_nnx_policy, SETTLE_STEPS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_COMPARE = int(os.environ.get("N_COMPARE", 500))
EVAL_SEED = int(os.environ.get("EVAL_SEED", 20260605))
PHI_RANGE = (173.0, 353.0)
V_RANGE = (6.0, 16.0)
N_SLSQP_STARTS = 8


def total_farm_power_np(gammas, positions, phi, v, N):
    """Compute total farm power (MW) using NumPy wake model."""
    gammas = np.asarray(gammas, dtype=np.float64)
    inflow = calculate_inflow_speeds(
        positions, phi, C_T, I, d_0, v, gammas, alpha_star, beta_star, alpha
    )
    total = sum(power_output(inflow[i], gammas[i]) for i in range(N))
    return total / 1e6


def optimize_slsqp(phi, v, positions, N, n_starts=N_SLSQP_STARTS, seed=42):
    """Multi-start SLSQP optimization."""
    rng = np.random.default_rng(seed)
    bounds = [(-50.0, 50.0)] * N

    best_power = -np.inf
    best_gammas = np.zeros(N)

    starts = [np.zeros(N)]
    for k in range(n_starts - 1):
        starts.append(rng.uniform(-30, 30, size=N))

    for x0 in starts:
        try:
            res = minimize(
                lambda g: -total_farm_power_np(g, positions, phi, v, N),
                x0, method='SLSQP', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-13},
            )
            pwr = -res.fun
            if pwr > best_power:
                best_power = pwr
                best_gammas = res.x.copy()
        except Exception:
            pass

    return best_power, best_gammas


def build_lookup_table(positions, N):
    """Pre-compute SLSQP optimal yaw angles on a grid.

    phi_grid: 91 values (173 to 353, step 2)
    v_grid: 11 values (6 to 16, step 1)
    """
    phi_grid = np.arange(173.0, 353.1, 2.0)
    v_grid = np.arange(6.0, 16.01, 1.0)

    n_phi = len(phi_grid)
    n_v = len(v_grid)

    yaw_table = np.zeros((n_phi, n_v, N))
    gain_table = np.zeros((n_phi, n_v))

    print(f"  Computing lookup table: {n_phi}×{n_v} = {n_phi*n_v} SLSQP solves...")

    for i, phi in enumerate(phi_grid):
        for j, v in enumerate(v_grid):
            opt_mw, opt_gammas = optimize_slsqp(phi, v, positions, N, seed=42+i*11+j)
            base_mw = total_farm_power_np(np.zeros(N), positions, phi, v, N)
            gain = (opt_mw - base_mw) / base_mw * 100.0 if base_mw > 0 else 0
            yaw_table[i, j] = opt_gammas
            gain_table[i, j] = gain
        if (i + 1) % 15 == 0:
            print(f"    phi row {i+1}/{n_phi} done")

    return phi_grid, v_grid, yaw_table, gain_table


def lookup_interpolate(phi_query, v_query, phi_grid, v_grid, yaw_table, gain_table):
    """Bilinear interpolation of lookup table yaw angles and gains."""
    # Find surrounding grid indices
    phi_idx = np.searchsorted(phi_grid, phi_query) - 1
    v_idx = np.searchsorted(v_grid, v_query) - 1
    phi_idx = np.clip(phi_idx, 0, len(phi_grid) - 2)
    v_idx = np.clip(v_idx, 0, len(v_grid) - 2)

    # Interpolation weights
    phi_lo, phi_hi = phi_grid[phi_idx], phi_grid[phi_idx + 1]
    v_lo, v_hi = v_grid[v_idx], v_grid[v_idx + 1]
    w_phi = (phi_query - phi_lo) / max(phi_hi - phi_lo, 1e-6)
    w_v = (v_query - v_lo) / max(v_hi - v_lo, 1e-6)
    w_phi = np.clip(w_phi, 0, 1)
    w_v = np.clip(w_v, 0, 1)

    # Bilinear interpolation for yaw angles
    yaw_interp = (
        yaw_table[phi_idx, v_idx] * (1 - w_phi) * (1 - w_v) +
        yaw_table[phi_idx + 1, v_idx] * w_phi * (1 - w_v) +
        yaw_table[phi_idx, v_idx + 1] * (1 - w_phi) * w_v +
        yaw_table[phi_idx + 1, v_idx + 1] * w_phi * w_v
    )

    # Bilinear interpolation for gain
    gain_interp = (
        gain_table[phi_idx, v_idx] * (1 - w_phi) * (1 - w_v) +
        gain_table[phi_idx + 1, v_idx] * w_phi * (1 - w_v) +
        gain_table[phi_idx, v_idx + 1] * (1 - w_phi) * w_v +
        gain_table[phi_idx + 1, v_idx + 1] * w_phi * w_v
    )

    return yaw_interp, float(gain_interp)


def main():
    t_start = time.time()
    positions, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions)
    positions_jax = positions_to_jax(positions)

    print(f"# DRL vs SLSQP regime-wise comparison")
    print(f"# N_COMPARE={N_COMPARE} conditions")
    print(f"# device={jax.devices()[0]}")

    # Sample conditions
    rng = np.random.default_rng(EVAL_SEED)
    phis = rng.uniform(PHI_RANGE[0], PHI_RANGE[1], size=N_COMPARE)
    vs = rng.uniform(V_RANGE[0], V_RANGE[1], size=N_COMPARE)

    # ---- Step 1: DRL evaluation ----
    print("\n## Step 1: DRL policy evaluation (5 seeds)")
    obs_dim = 3 * N_turb + 3
    act_dim = N_turb

    drl_gains_all = []
    for s in range(5):
        ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_p0c.pkl")
        model = load_nnx_policy(ckpt, obs_dim, act_dim)

        phis_j = jnp.asarray(phis, dtype=jnp.float32)
        vs_j = jnp.asarray(vs, dtype=jnp.float32)

        # Compute baselines
        @jax.jit
        def zero_yaw_baseline(phi, v):
            inflow_0 = inflow_speeds_jax(positions_jax, phi, v,
                                          jnp.zeros(N_turb))
            return jnp.sum(power_output_jax(inflow_0, jnp.zeros(N_turb))) / 1e6

        baselines = jax.vmap(zero_yaw_baseline)(phis_j, vs_j)
        baselines_np = np.asarray(baselines)

        # Run policy
        N_steps = SETTLE_STEPS

        @nnx.jit
        def run_policy(m, phis_j, vs_j):
            @jax.vmap
            def reset_one(phi, v):
                key = jax.random.key(0)
                state, obs = env_reset(key, positions_jax,
                                        specific_wind_dir=phi,
                                        specific_wind_speed=v,
                                        randomize_wind=False,
                                        max_steps=N_steps + 10)
                return state, obs

            states, obs_batch = reset_one(phis_j, vs_j)

            @jax.vmap
            def predict_one(o):
                mean, _, _ = m(o.reshape(1, -1))
                return mean.reshape(N_turb)

            @jax.vmap
            def step_one(s, a):
                return env_step(s, a, positions_jax, max_steps=N_steps + 10)

            def body(carry, _):
                states, obs = carry
                actions = predict_one(obs)
                actions = jnp.clip(actions, -5.0, 5.0)
                new_states, new_obs, _, _ = step_one(states, actions)
                return (new_states, new_obs), None

            (final_states, _), _ = jax.lax.scan(body, (states, obs_batch), None, length=N_steps)
            return final_states.total_mw, final_states.gammas

        total_mw, gammas = run_policy(model, phis_j, vs_j)
        gains = (total_mw - baselines) / baselines * 100.0
        drl_gains_all.append(np.asarray(gains))
        print(f"  seed {s}: mean gain = {float(np.asarray(gains).mean()):+.3f}%")

    drl_gains_stack = np.stack(drl_gains_all, axis=0)
    drl_mean = drl_gains_stack.mean(axis=0)

    # ---- Step 2: SLSQP per-condition optimization ----
    print(f"\n## Step 2: SLSQP optimization ({N_COMPARE} conditions, {N_SLSQP_STARTS} starts each)")
    slsqp_gains = np.empty(N_COMPARE)
    slsqp_gammas_list = []
    baselines_np2 = np.empty(N_COMPARE)

    for idx in range(N_COMPARE):
        phi, v = phis[idx], vs[idx]
        base = total_farm_power_np(np.zeros(N_turb), positions, phi, v, N_turb)
        opt_mw, opt_gammas = optimize_slsqp(phi, v, positions, N_turb, seed=42+idx)
        gain = (opt_mw - base) / base * 100.0 if base > 0 else 0
        slsqp_gains[idx] = gain
        slsqp_gammas_list.append(opt_gammas)
        baselines_np2[idx] = base
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  {idx+1}/{N_COMPARE} done ({elapsed:.0f}s)")

    # ---- Step 3: Lookup table baseline ----
    print(f"\n## Step 3: Lookup table baseline")
    phi_grid, v_grid, yaw_table, gain_table = build_lookup_table(positions, N_turb)

    # Interpolate to comparison conditions
    lookup_gains = np.empty(N_COMPARE)
    lookup_gammas_list = []
    for idx in range(N_COMPARE):
        yaw_interp, gain_interp = lookup_interpolate(
            phis[idx], vs[idx], phi_grid, v_grid, yaw_table, gain_table)
        lookup_gains[idx] = gain_interp
        lookup_gammas_list.append(yaw_interp)

    # Evaluate lookup-table yaw vectors in gray-box
    lookup_eval_gains = np.empty(N_COMPARE)
    for idx in range(N_COMPARE):
        pwr = total_farm_power_np(lookup_gammas_list[idx], positions,
                                   phis[idx], vs[idx], N_turb)
        base = baselines_np2[idx]
        lookup_eval_gains[idx] = (pwr - base) / base * 100.0 if base > 0 else 0

    # ---- Analysis ----
    dphi_arr = np.abs(((phis - 270.0 + 180.0) % 360.0) - 180.0)
    aligned_cube = (dphi_arr < 15.0) & (vs < 11.4)

    # 4-category breakdown
    cat_A = int(((slsqp_gains > 0.1) & (drl_mean > 0.01)).sum())
    cat_B = int(((slsqp_gains > 0.1) & (drl_mean <= 0.01)).sum())
    cat_C = int(((slsqp_gains <= 0.1) & (np.abs(drl_mean) <= 0.01)).sum())
    cat_D = int(((slsqp_gains <= 0.1) & (drl_mean < -0.01)).sum())

    # Recovery rate (where SLSQP > 0)
    valid = slsqp_gains > 0.1
    recovery = drl_mean[valid] / slsqp_gains[valid] * 100.0 if valid.sum() > 0 else np.array([0])
    mean_recovery = float(recovery.mean())

    # Save results
    per_condition = []
    for idx in range(N_COMPARE):
        per_condition.append({
            "phi": float(phis[idx]),
            "v": float(vs[idx]),
            "dphi": float(dphi_arr[idx]),
            "drl_gain_pct": float(drl_mean[idx]),
            "slsqp_gain_pct": float(slsqp_gains[idx]),
            "lookup_gain_pct": float(lookup_gains[idx]),
            "lookup_eval_gain_pct": float(lookup_eval_gains[idx]),
            "drl_slsqp_ratio_pct": float(drl_mean[idx] / slsqp_gains[idx] * 100)
                if slsqp_gains[idx] > 0.1 else None,
            "regime": "aligned_cube" if aligned_cube[idx] else "other",
        })

    result = {
        "description": "DRL vs SLSQP regime-wise comparison",
        "n_conditions": N_COMPARE,
        "eval_seed": EVAL_SEED,
        "slsqp_n_starts": N_SLSQP_STARTS,
        "lookup_table_grid": {"phi": f"{len(phi_grid)} pts", "v": f"{len(v_grid)} pts"},
        "summary": {
            "drl_marginal_pct": float(drl_mean.mean()),
            "slsqp_marginal_pct": float(slsqp_gains.mean()),
            "lookup_marginal_pct": float(lookup_eval_gains.mean()),
            "drl_aligned_cube_pct": float(drl_mean[aligned_cube].mean())
                if aligned_cube.sum() > 0 else None,
            "slsqp_aligned_cube_pct": float(slsqp_gains[aligned_cube].mean())
                if aligned_cube.sum() > 0 else None,
            "mean_drl_slsqp_recovery_pct": mean_recovery,
            "categories": {
                "A_slsqp_pos_drl_pos": cat_A,
                "B_slsqp_pos_drl_neg": cat_B,
                "C_both_near_zero": cat_C,
                "D_drl_actively_harmful": cat_D,
            },
        },
        "per_condition": per_condition,
    }

    out_path = os.path.join(FIG_DIR, "drl_vs_slsqp_regime.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {out_path}")

    # Save lookup table
    lt_result = {
        "description": "SLSQP lookup table baseline",
        "phi_grid": phi_grid.tolist(),
        "v_grid": v_grid.tolist(),
        "gain_table": gain_table.tolist(),
        "lookup_marginal_pct": float(lookup_eval_gains.mean()),
        "lookup_vs_slsqp_ratio": float(lookup_eval_gains.mean() / slsqp_gains.mean() * 100)
            if slsqp_gains.mean() > 0.01 else None,
    }
    lt_path = os.path.join(FIG_DIR, "lookup_table_baseline.json")
    with open(lt_path, "w") as f:
        json.dump(lt_result, f, indent=2)
    print(f"Saved {lt_path}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) DRL vs SLSQP scatter
    ax = axes[0]
    colors = np.where(aligned_cube, '#E45756', '#4C78A8')
    ax.scatter(slsqp_gains, drl_mean, s=12, alpha=0.6, c=colors)
    lims = [min(slsqp_gains.min(), drl_mean.min()), max(slsqp_gains.max(), drl_mean.max())]
    ax.plot(lims, lims, 'k--', lw=0.8, label='1:1')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.axvline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel("SLSQP optimal gain [%]")
    ax.set_ylabel("DRL policy gain [%]")
    ax.set_title("(a) DRL vs SLSQP per condition\n(red = aligned-cube)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Histogram: DRL gain, SLSQP gain, lookup gain
    ax = axes[1]
    bins = np.linspace(-3, 15, 60)
    ax.hist(drl_mean, bins=bins, alpha=0.5, color='#4C78A8', label='DRL', edgecolor='black', lw=0.3)
    ax.hist(slsqp_gains, bins=bins, alpha=0.4, color='#E45756', label='SLSQP', edgecolor='black', lw=0.3)
    ax.hist(lookup_eval_gains, bins=bins, alpha=0.3, color='#54A24B', label='Lookup table', edgecolor='black', lw=0.3)
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel("Farm-power gain [%]")
    ax.set_ylabel("Count")
    ax.set_title("(b) Gain distributions")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Recovery rate by regime
    ax = axes[2]
    dir_bins = [(0, 15), (15, 35), (35, 60), (60, 90.01)]
    v_bins = [(6, 8), (8, 11.4), (11.4, 14), (14, 16.01)]
    regime_labels = []
    recovery_vals = []
    for dlo, dhi in dir_bins:
        for vlo, vhi in v_bins:
            sel = (dphi_arr >= dlo) & (dphi_arr < dhi) & (vs >= vlo) & (vs < vhi)
            if sel.sum() < 5:
                continue
            valid_sel = sel & (slsqp_gains > 0.1)
            if valid_sel.sum() < 3:
                continue
            rec = drl_mean[valid_sel] / slsqp_gains[valid_sel] * 100
            regime_labels.append(f"|dphi|<{dhi:.0f}°\nv[{vlo:.0f},{vhi:.0f})")
            recovery_vals.append(float(rec.mean()))

    if regime_labels:
        y_pos = np.arange(len(regime_labels))
        colors_bar = ['#E45756' if r < 50 else '#54A24B' if r > 80 else '#F58518'
                       for r in recovery_vals]
        ax.barh(y_pos, recovery_vals, color=colors_bar, alpha=0.8, edgecolor='black', lw=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(regime_labels, fontsize=6)
        ax.axvline(50, color='red', lw=0.8, ls='--', label='50% recovery')
        ax.axvline(100, color='green', lw=0.8, ls='--', label='100% recovery')
        ax.set_xlabel("DRL/SLSQP recovery [%]")
        ax.set_title("(c) Recovery rate by regime")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("DRL vs SLSQP yaw optimization", fontsize=11)
    fig.tight_layout()
    for ext in ['pdf', 'jpg']:
        path = os.path.join(FIG_DIR, f"fig_drl_vs_slsqp_scatter.{ext}")
        fig.savefig(path, dpi=300 if ext == 'jpg' else None, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)

    # Print summary
    print(f"\n{'='*60}")
    print(f"DRL vs SLSQP COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  DRL marginal:     {drl_mean.mean():+.3f}%")
    print(f"  SLSQP marginal:   {slsqp_gains.mean():+.3f}%")
    print(f"  Lookup marginal:  {lookup_eval_gains.mean():+.3f}%")
    if aligned_cube.sum() > 0:
        print(f"  DRL aligned-cube:   {drl_mean[aligned_cube].mean():+.3f}%")
        print(f"  SLSQP aligned-cube: {slsqp_gains[aligned_cube].mean():+.3f}%")
    print(f"  Mean recovery:    {mean_recovery:.1f}%")
    print(f"  Categories: A={cat_A}  B={cat_B}  C={cat_C}  D={cat_D}")

    print(f"\n  Total wall-clock: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
