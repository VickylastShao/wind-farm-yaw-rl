#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLORIS cross-evaluation of trained DRL policies.

Runs the 5-seed trained policies in the gray-box JAX env to obtain per-condition
yaw vectors, then evaluates those same yaw vectors inside FLORIS (NREL 5MW,
GCH wake model) to quantify the real-world gain that the gray-box model
overestimates.

This is the GATEKEEPER experiment for the paper revision: the FLORIS-validated
gain determines whether the paper retains a positive DRL contribution (Scenario A)
or must be reframed as a methodological / negative-result contribution (Scenario B).

Output:
  latex_draft/figures/floris_cross_eval_results.json
  latex_draft/figures/fig_floris_cross_eval.{pdf,jpg}
"""

import os
import sys
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx

from windfarm_env import create_wind_farm_layout_3x3
from windfarm_env_jax import (
    env_reset, env_step, inflow_speeds_jax, power_output_jax,
    find_downstream_mask_jax, positions_to_jax,
)
from train_3x3_nnx import ActorCritic
from cross_val_jaxenv_vs_numpyenv import load_nnx_policy, SETTLE_STEPS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_SEEDS = int(os.environ.get("N_SEEDS", 5))
PHI_RANGE = (173.0, 353.0)
V_RANGE = (6.0, 16.0)
N_CONDITIONS = int(os.environ.get("N_CONDITIONS", 3000))
EVAL_SEED = int(os.environ.get("EVAL_SEED", 20260604))


# ---------------------------------------------------------------------------
# FLORIS helpers
# ---------------------------------------------------------------------------
def make_floris_model():
    """Create a FLORIS model with 3x3 NREL 5MW layout, GCH wake model."""
    from floris import FlorisModel

    fm = FlorisModel("defaults")

    # 3x3 layout matching windfarm_env.create_wind_farm_layout_3x3()
    d_0 = 126.0
    sx, sy = 7 * d_0, 7 * d_0
    tilt = np.radians(7.0)
    xs, ys = [], []
    for j in range(3):
        for i in range(3):
            xs.append(i * sx + j * sy * np.sin(tilt))
            ys.append(j * sy * np.cos(tilt))

    fm.set(layout_x=xs, layout_y=ys)
    return fm


def floris_farm_power(fm, phi, v, yaw_angles=None, ti=0.065):
    """Compute total farm power in FLORIS for a single condition.

    Args:
        fm: FlorisModel (already set with layout)
        phi: wind direction (meteorological, degrees)
        v: wind speed (m/s)
        yaw_angles: array of yaw angles (degrees), shape (N_turb,) or None
        ti: turbulence intensity

    Returns:
        total power in MW
    """
    n_turb = len(fm.layout_x)
    if yaw_angles is None:
        yaw_angles = np.zeros((1, n_turb))
    else:
        yaw_angles = np.asarray(yaw_angles).reshape(1, -1)

    fm.set(
        wind_directions=[float(phi)],
        wind_speeds=[float(v)],
        turbulence_intensities=[float(ti)],
        yaw_angles=yaw_angles,
    )
    fm.run()
    powers = fm.get_turbine_powers()  # (1, N_turb) in W
    return float(np.sum(powers)) / 1e6  # MW


# ---------------------------------------------------------------------------
# Gray-box batched evaluation (reuse eval_p0c_randomized pattern)
# ---------------------------------------------------------------------------
def build_batch(positions_jax, rng):
    """Sample N_CONDITIONS (phi, v) pairs — same as eval_p0c_randomized."""
    np_rng = np.random.default_rng(rng)
    phis = jnp.asarray(np_rng.uniform(PHI_RANGE[0], PHI_RANGE[1],
                                        size=N_CONDITIONS), dtype=jnp.float32)
    vs = jnp.asarray(np_rng.uniform(V_RANGE[0], V_RANGE[1],
                                      size=N_CONDITIONS), dtype=jnp.float32)

    @jax.jit
    def zero_yaw_baseline(phi, v):
        inflow_0 = inflow_speeds_jax(positions_jax, phi, v,
                                      jnp.zeros(positions_jax.shape[0]))
        baseline_mw = jnp.sum(power_output_jax(inflow_0,
                                                jnp.zeros(positions_jax.shape[0]))) / 1e6
        return baseline_mw

    baselines = jax.vmap(zero_yaw_baseline)(phis, vs)
    return phis, vs, baselines


def evaluate_batch_get_yaws(model, positions_jax, phis, vs, baselines, N_turb):
    """Evaluate policy and return both gains AND final yaw vectors."""
    N_steps = SETTLE_STEPS

    @nnx.jit
    def run(m, phis, vs):
        @jax.vmap
        def reset_one(phi, v):
            key = jax.random.key(0)
            state, obs = env_reset(key, positions_jax,
                                    specific_wind_dir=phi,
                                    specific_wind_speed=v,
                                    randomize_wind=False,
                                    max_steps=N_steps + 10)
            return state, obs

        states, obs_batch = reset_one(phis, vs)

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

    total_mw, gammas = run(model, phis, vs)
    gains = (total_mw - baselines) / baselines * 100.0
    return gains, gammas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    positions, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions)
    positions_jax = positions_to_jax(positions)

    print(f"# FLORIS cross-evaluation of trained DRL policies")
    print(f"# N_SEEDS={N_SEEDS}, N_CONDITIONS={N_CONDITIONS}")
    print(f"# device={jax.devices()[0]}")
    print(f"# FLORIS version={__import__('floris').__version__}")

    # Build same condition batch as eval_p0c_randomized
    phis, vs, gb_baselines = build_batch(positions_jax, EVAL_SEED)
    phis_np = np.asarray(phis)
    vs_np = np.asarray(vs)
    gb_baselines_np = np.asarray(gb_baselines)

    # Regime masks
    dphi_arr = np.abs(((phis_np - 270.0 + 180.0) % 360.0) - 180.0)
    aligned_cube = (dphi_arr < 15.0) & (vs_np < 11.4)

    # Initialize FLORIS
    print("\n## Initializing FLORIS model...")
    fm = make_floris_model()

    # Compute FLORIS zero-yaw baselines for all conditions
    print("## Computing FLORIS zero-yaw baselines...")
    floris_baselines = np.empty(N_CONDITIONS)
    for i in range(N_CONDITIONS):
        floris_baselines[i] = floris_farm_power(fm, phis_np[i], vs_np[i], yaw_angles=None)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N_CONDITIONS} baselines done ({time.time()-t_start:.0f}s)")
    print(f"  FLORIS baseline: mean={floris_baselines.mean():.2f} MW "
          f"(gray-box: {gb_baselines_np.mean():.2f} MW)")

    # Evaluate each seed
    per_seed_results = []
    all_gb_gains = []
    all_floris_gains = []

    for s in range(N_SEEDS):
        ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_p0c.pkl")
        if not os.path.exists(ckpt):
            print(f"  seed {s}: checkpoint not found, skipping")
            continue
        obs_dim = 3 * N_turb + 3
        act_dim = N_turb
        model = load_nnx_policy(ckpt, obs_dim, act_dim)
        print(f"\n## Seed {s}: JAX rollout + FLORIS evaluation...")

        # Step 1: JAX rollout to get gains and yaw vectors
        gb_gains, yaw_vectors = evaluate_batch_get_yaws(
            model, positions_jax, phis, vs, gb_baselines, N_turb)
        gb_gains_np = np.asarray(gb_gains)
        yaws_np = np.asarray(yaw_vectors)  # (N_CONDITIONS, N_turb)

        # Step 2: Evaluate each yaw vector in FLORIS
        floris_yawked = np.empty(N_CONDITIONS)
        for i in range(N_CONDITIONS):
            floris_yawked[i] = floris_farm_power(
                fm, phis_np[i], vs_np[i], yaw_angles=yaws_np[i])
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{N_CONDITIONS} FLORIS evals done ({time.time()-t_start:.0f}s)")

        # Step 3: Compute FLORIS-validated gains
        floris_gains = (floris_yawked - floris_baselines) / floris_baselines * 100.0

        # Statistics
        gb_marginal = float(gb_gains_np.mean())
        gb_aligned = float(gb_gains_np[aligned_cube].mean()) if aligned_cube.sum() > 0 else None
        floris_marginal = float(floris_gains.mean())
        floris_aligned = float(floris_gains[aligned_cube].mean()) if aligned_cube.sum() > 0 else None

        gb_aligned_s = f"{gb_aligned:+.3f}" if gb_aligned is not None else "N/A"
        floris_aligned_s = f"{floris_aligned:+.3f}" if floris_aligned is not None else "N/A"
        print(f"  Gray-box:    marginal={gb_marginal:+.3f}%  aligned-cube={gb_aligned_s}%")
        print(f"  FLORIS:      marginal={floris_marginal:+.3f}%  aligned-cube={floris_aligned_s}%")
        if floris_aligned is not None and gb_aligned is not None and gb_aligned > 0:
            print(f"  Erosion:     {(1 - floris_aligned/gb_aligned)*100:.1f}% "
                  f"(aligned-cube)")

        seed_result = {
            "seed": s,
            "gb_marginal_mean_pct": gb_marginal,
            "gb_aligned_cube_pct": gb_aligned,
            "floris_marginal_mean_pct": floris_marginal,
            "floris_aligned_cube_pct": floris_aligned,
            "per_condition_gb_gain": gb_gains_np.tolist(),
            "per_condition_floris_gain": floris_gains.tolist(),
            "per_condition_yaws": yaws_np.tolist(),
        }
        per_seed_results.append(seed_result)
        all_gb_gains.append(gb_gains_np)
        all_floris_gains.append(floris_gains)

    # Aggregate across seeds
    gb_stack = np.stack(all_gb_gains, axis=0)       # (n_seeds, N_CONDITIONS)
    floris_stack = np.stack(all_floris_gains, axis=0)
    gb_cond_mean = gb_stack.mean(axis=0)
    floris_cond_mean = floris_stack.mean(axis=0)

    # Bootstrap CIs
    rng = np.random.default_rng(42)
    B = 10000

    def bootstrap_ci(data, B=10000, ci=0.95):
        """Percentile bootstrap CI for the mean."""
        boot_means = np.empty(B)
        n = len(data)
        for b in range(B):
            sample = rng.choice(data, size=n, replace=True)
            boot_means[b] = sample.mean()
        lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
        hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
        return float(lo), float(hi)

    gb_marginal_ci = bootstrap_ci(gb_cond_mean)
    gb_aligned_ci = bootstrap_ci(gb_cond_mean[aligned_cube]) if aligned_cube.sum() > 0 else None
    floris_marginal_ci = bootstrap_ci(floris_cond_mean)
    floris_aligned_ci = bootstrap_ci(floris_cond_mean[aligned_cube]) if aligned_cube.sum() > 0 else None

    # Regime-binned analysis
    DIR_EDGES = [(0.0, 15.0), (15.0, 35.0), (35.0, 60.0), (60.0, 90.001)]
    V_EDGES = [(6.0, 8.0), (8.0, 11.4), (11.4, 14.0), (14.0, 16.001)]
    regime_bins = []
    for dlo, dhi in DIR_EDGES:
        sel_d = (dphi_arr >= dlo) & (dphi_arr < dhi)
        for vlo, vhi in V_EDGES:
            sel = sel_d & (vs_np >= vlo) & (vs_np < vhi)
            if sel.sum() < 5:
                continue
            regime_bins.append({
                "dir_range": f"|dphi|<{dhi:.0f}",
                "v_range": f"[{vlo:.0f},{vhi:.1f})",
                "n": int(sel.sum()),
                "gb_mean_pct": float(gb_cond_mean[sel].mean()),
                "floris_mean_pct": float(floris_cond_mean[sel].mean()),
                "erosion_pct": float((1 - floris_cond_mean[sel].mean() / max(gb_cond_mean[sel].mean(), 0.01)) * 100)
                    if gb_cond_mean[sel].mean() > 0.01 else None,
            })

    # Summary
    summary = {
        "description": "FLORIS cross-evaluation of trained DRL policies",
        "n_seeds": len(per_seed_results),
        "n_conditions": N_CONDITIONS,
        "eval_seed": EVAL_SEED,
        "floris_version": __import__('floris').__version__,
        "graybox_baseline_mw_mean": float(gb_baselines_np.mean()),
        "floris_baseline_mw_mean": float(floris_baselines.mean()),
        "baseline_discrepancy_pct": float((gb_baselines_np.mean() - floris_baselines.mean()) / floris_baselines.mean() * 100),
        "gb_marginal_mean_pct": float(gb_cond_mean.mean()),
        "gb_marginal_95ci": list(gb_marginal_ci),
        "gb_aligned_cube_pct": float(gb_cond_mean[aligned_cube].mean()) if aligned_cube.sum() > 0 else None,
        "gb_aligned_cube_95ci": list(gb_aligned_ci) if gb_aligned_ci else None,
        "floris_marginal_mean_pct": float(floris_cond_mean.mean()),
        "floris_marginal_95ci": list(floris_marginal_ci),
        "floris_aligned_cube_pct": float(floris_cond_mean[aligned_cube].mean()) if aligned_cube.sum() > 0 else None,
        "floris_aligned_cube_95ci": list(floris_aligned_ci) if floris_aligned_ci else None,
        "aligned_cube_erosion_pct": float((1 - floris_cond_mean[aligned_cube].mean() / gb_cond_mean[aligned_cube].mean()) * 100)
            if aligned_cube.sum() > 0 and gb_cond_mean[aligned_cube].mean() > 0 else None,
        "regime_bins": regime_bins,
        "per_seed": per_seed_results,
    }

    # Save JSON
    out_path = os.path.join(FIG_DIR, "floris_cross_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Scatter: gray-box gain vs FLORIS gain
    ax = axes[0, 0]
    ax.scatter(gb_cond_mean, floris_cond_mean, s=6, alpha=0.4, c=vs_np,
               cmap="viridis", vmin=V_RANGE[0], vmax=V_RANGE[1])
    lims = [min(gb_cond_mean.min(), floris_cond_mean.min()),
            max(gb_cond_mean.max(), floris_cond_mean.max())]
    ax.plot(lims, lims, 'k--', lw=0.8, label='1:1')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.axvline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel("Gray-box gain [%]")
    ax.set_ylabel("FLORIS-validated gain [%]")
    ax.set_title("(a) Per-condition gain: gray-box vs FLORIS")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # (b) FLORIS gain histogram
    ax = axes[0, 1]
    ax.hist(floris_cond_mean, bins=80, color="#4C78A8", alpha=0.8,
            edgecolor="black", linewidth=0.3, label="FLORIS")
    ax.hist(gb_cond_mean, bins=80, color="#E45756", alpha=0.4,
            edgecolor="black", linewidth=0.3, label="Gray-box")
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel("Farm-power gain [%]")
    ax.set_ylabel("Count")
    ax.set_title("(b) Gain distributions")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Gain erosion vs wind direction
    ax = axes[1, 0]
    # Bin by direction
    dir_bins = np.arange(173, 354, 5)
    dir_centers = 0.5 * (dir_bins[:-1] + dir_bins[1:])
    gb_by_dir = []
    floris_by_dir = []
    for k in range(len(dir_bins) - 1):
        sel = (phis_np >= dir_bins[k]) & (phis_np < dir_bins[k+1])
        if sel.sum() < 3:
            gb_by_dir.append(np.nan)
            floris_by_dir.append(np.nan)
        else:
            gb_by_dir.append(gb_cond_mean[sel].mean())
            floris_by_dir.append(floris_cond_mean[sel].mean())
    ax.plot(dir_centers, gb_by_dir, 'r-o', ms=3, lw=1.2, label="Gray-box")
    ax.plot(dir_centers, floris_by_dir, 'b-s', ms=3, lw=1.2, label="FLORIS")
    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.axvspan(255, 285, alpha=0.08, color='red')
    ax.set_xlabel("Wind direction φ [°]")
    ax.set_ylabel("Mean gain [%]")
    ax.set_title("(c) Gain by wind direction")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # (d) Gain erosion ratio by regime
    ax = axes[1, 1]
    regimes_labels = []
    floris_vals = []
    gb_vals = []
    for rb in regime_bins:
        if rb.get("erosion_pct") is not None and rb["gb_mean_pct"] > 0.1:
            label = f"{rb['dir_range']}, v{rb['v_range']}"
            regimes_labels.append(label)
            floris_vals.append(rb["floris_mean_pct"])
            gb_vals.append(rb["gb_mean_pct"])

    if regimes_labels:
        y_pos = np.arange(len(regimes_labels))
        ax.barh(y_pos - 0.15, gb_vals, 0.3, label="Gray-box", color="#E45756", alpha=0.8)
        ax.barh(y_pos + 0.15, floris_vals, 0.3, label="FLORIS", color="#4C78A8", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(regimes_labels, fontsize=7)
        ax.axvline(0, color='black', lw=0.5, ls='--')
        ax.set_xlabel("Mean gain [%]")
        ax.set_title("(d) Gray-box vs FLORIS gain by regime")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("FLORIS cross-evaluation of trained DRL yaw controller", fontsize=12, y=1.01)
    fig.tight_layout()

    for ext in ['pdf', 'jpg']:
        path = os.path.join(FIG_DIR, f"fig_floris_cross_eval.{ext}")
        fig.savefig(path, dpi=300 if ext == 'jpg' else None, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)

    # Print summary
    print(f"\n{'='*60}")
    print(f"FLORIS CROSS-EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Gray-box baseline:     {gb_baselines_np.mean():.2f} MW")
    print(f"  FLORIS baseline:       {floris_baselines.mean():.2f} MW  "
          f"(discrepancy: {(gb_baselines_np.mean()-floris_baselines.mean())/floris_baselines.mean()*100:.1f}%)")
    print(f"  Gray-box marginal:     {gb_cond_mean.mean():+.3f}%  "
          f"95% CI [{gb_marginal_ci[0]:+.3f}, {gb_marginal_ci[1]:+.3f}]")
    print(f"  FLORIS marginal:       {floris_cond_mean.mean():+.3f}%  "
          f"95% CI [{floris_marginal_ci[0]:+.3f}, {floris_marginal_ci[1]:+.3f}]")
    if aligned_cube.sum() > 0:
        print(f"  Gray-box aligned-cube: {gb_cond_mean[aligned_cube].mean():+.3f}%  "
              f"95% CI [{gb_aligned_ci[0]:+.3f}, {gb_aligned_ci[1]:+.3f}]")
        print(f"  FLORIS aligned-cube:   {floris_cond_mean[aligned_cube].mean():+.3f}%  "
              f"95% CI [{floris_aligned_ci[0]:+.3f}, {floris_aligned_ci[1]:+.3f}]")
        erosion = (1 - floris_cond_mean[aligned_cube].mean() / gb_cond_mean[aligned_cube].mean()) * 100
        print(f"  Gain erosion:          {erosion:.1f}% (aligned-cube regime)")

    print(f"\n  Total wall-clock: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
