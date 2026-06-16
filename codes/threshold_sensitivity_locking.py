#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threshold sensitivity analysis for the downstream-locking mechanism.

For each wind condition in the 3000-condition evaluation sweep, compute the
set of locked turbines under three deficit thresholds (0.5%, 1.0%, 2.0%).
Report:
  - fraction of conditions where the lock set changes vs. baseline (1%)
  - per-threshold aligned-cube gain (re-evaluate policies with the new mask)
  - per-threshold number of locked turbines (mean / max / distribution)

Output:
  latex_draft/figures/threshold_sensitivity_locking.json
"""

import os, json
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from windfarm_env import create_wind_farm_layout_3x3
from windfarm_env_jax import (
    find_downstream_mask_jax, inflow_speeds_jax, power_output_jax,
    env_reset, env_step, positions_to_jax,
)
from train_3x3_nnx import ActorCritic
from cross_val_jaxenv_vs_numpyenv import load_nnx_policy, SETTLE_STEPS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PHI_RANGE = (173.0, 353.0)
V_RANGE = (6.0, 16.0)
N_CONDITIONS = 3000
EVAL_SEED = 20260604
N_SEEDS = 5
THRESHOLDS = [0.005, 0.01, 0.02]  # 0.5%, 1.0%, 2.0%


def build_batch(positions_jax, rng):
    """Sample N_CONDITIONS (phi, v) pairs."""
    np_rng = np.random.default_rng(rng)
    phis = jnp.asarray(np_rng.uniform(PHI_RANGE[0], PHI_RANGE[1],
                                        size=N_CONDITIONS), dtype=jnp.float32)
    vs = jnp.asarray(np_rng.uniform(V_RANGE[0], V_RANGE[1],
                                      size=N_CONDITIONS), dtype=jnp.float32)

    @jax.jit
    def zero_yaw_baseline(phi, v):
        inflow_0 = inflow_speeds_jax(positions_jax, phi, v,
                                      jnp.zeros(positions_jax.shape[0]))
        baseline_mw = jnp.sum(power_output_jax(inflow_0, jnp.zeros(positions_jax.shape[0]))) / 1e6
        return baseline_mw

    baselines = jax.vmap(zero_yaw_baseline)(phis, vs)
    return phis, vs, baselines


def compute_masks_batch(positions_jax, phis, vs, threshold):
    """Compute locked masks for all conditions at a given threshold."""
    @jax.jit
    def mask_one(phi, v):
        return find_downstream_mask_jax(positions_jax, phi, v, threshold=threshold)
    return jax.vmap(mask_one)(phis, vs)  # (N_CONDITIONS, N_turb)


def evaluate_with_mask(model, positions_jax, phis, vs, baselines, N_turb,
                       masks_override):
    """
    Evaluate policy but override the downstream mask in the observation.

    We run the standard rollout but replace the locked-turbine portion
    of the observation with the mask computed under the alternative threshold.
    The *actual* env step still uses the default (1%) mask internally for
    action zeroing; we only change the observation to see how the policy
    responds to the *information* that different turbines are locked.

    Since the env's internal locking still zeros actions for the 1%-threshold
    turbines, this tests whether *misinforming* the policy about the lock set
    degrades performance, rather than testing a truly re-locked policy.

    For a full re-lock test, we also run a second mode where we modify
    the env state's downstream_mask at reset.
    """
    N_steps = SETTLE_STEPS

    @nnx.jit
    def run(m, phis, vs, masks):
        B = phis.shape[0]

        @jax.vmap
        def reset_one(phi, v, mask_override):
            key = jax.random.key(0)
            state, obs = env_reset(key, positions_jax,
                                    specific_wind_dir=phi,
                                    specific_wind_speed=v,
                                    randomize_wind=False,
                                    max_steps=N_steps + 10)
            # Replace the downstream mask in state and rebuild obs.
            state = state._replace(downstream_mask=mask_override)
            # Rebuild observation with the overridden mask.
            # Obs layout: [gammas(N), inflow(N), cos(phi), sin(phi), v, locked(N)]
            obs_dim = 3 * N_turb + 3
            # Extract parts from original obs (which used default mask)
            # and replace the last N_turb entries with the new mask.
            new_locked = mask_override.astype(jnp.float32)
            obs_new = obs.at[-N_turb:].set(new_locked)
            return state, obs_new

        states, obs_batch = reset_one(phis, vs, masks)

        @jax.vmap
        def predict_one(o):
            mean, _, _ = m(o.reshape(1, -1))
            return mean.reshape(N_turb)

        @jax.vmap
        def step_one(s, a):
            # Apply the overridden mask to zero actions for newly locked turbines
            a = a * (~s.downstream_mask).astype(jnp.float32)
            new_state, new_obs, reward, done = env_step(s, a, positions_jax,
                                                         max_steps=N_steps + 10)
            # Re-apply mask override on observation
            new_obs = new_obs.at[-N_turb:].set(s.downstream_mask.astype(jnp.float32))
            return new_state, new_obs, reward, done

        def body(carry, _):
            states, obs = carry
            actions = predict_one(obs)
            actions = jnp.clip(actions, -5.0, 5.0)
            # Zero actions for locked turbines (using overridden mask)
            actions = actions * (~states.downstream_mask).astype(jnp.float32)
            new_states, new_obs, _, _ = step_one(states, actions)
            # Force-zero yaw for locked turbines in state
            new_gammas = new_states.gammas * (~new_states.downstream_mask).astype(jnp.float32)
            new_states = new_states._replace(gammas=new_gammas)
            return (new_states, new_obs), None

        (final_states, _), _ = jax.lax.scan(body, (states, obs_batch), None, length=N_steps)
        return final_states.total_mw, final_states.gammas

    total_mw, gammas = run(model, phis, vs, masks_override)
    gains = (total_mw - baselines) / baselines * 100.0
    return gains


def main():
    positions, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions)
    positions_jax = positions_to_jax(positions)

    print(f"# Threshold sensitivity analysis for downstream locking")
    print(f"# N_SEEDS = {N_SEEDS}, N_CONDITIONS = {N_CONDITIONS}")
    print(f"# Thresholds: {THRESHOLDS}")
    print(f"# device = {jax.devices()[0]}")

    # Build wind condition batch (same as main eval)
    phis, vs, baselines = build_batch(positions_jax, EVAL_SEED)
    phis_np = np.asarray(phis)
    vs_np = np.asarray(vs)

    # Compute masks for all thresholds
    print("\n## Computing downstream masks for all thresholds...")
    all_masks = {}
    for thr in THRESHOLDS:
        masks = compute_masks_batch(positions_jax, phis, vs, thr)
        all_masks[thr] = np.asarray(masks)
        n_locked = np.asarray(masks).sum(axis=1)
        print(f"  threshold={thr:.3f} (={thr*100:.1f}%): "
              f"mean locked = {n_locked.mean():.2f}, "
              f"max = {n_locked.max():.0f}, "
              f"min = {n_locked.min():.0f}")

    # Compare masks vs. baseline (1%)
    baseline_thr = 0.01
    baseline_masks = all_masks[baseline_thr]
    for thr in THRESHOLDS:
        if thr == baseline_thr:
            continue
        diff = np.any(all_masks[thr] != baseline_masks, axis=1)
        n_diff = diff.sum()
        print(f"\n  threshold={thr:.3f} vs baseline (0.01): "
              f"{n_diff}/{N_CONDITIONS} conditions ({100*n_diff/N_CONDITIONS:.1f}%) "
              f"have different lock sets")

    # Direction/speed binning for alignment analysis
    dphi_arr = np.abs(((phis_np - 270.0 + 180.0) % 360.0) - 180.0)
    aligned_cube = (dphi_arr < 15.0) & (vs_np < 11.4)

    results = {
        "n_conditions": N_CONDITIONS,
        "n_seeds": N_SEEDS,
        "thresholds": THRESHOLDS,
        "per_threshold": {},
    }

    # For each threshold, evaluate all seeds
    for thr in THRESHOLDS:
        print(f"\n{'='*60}")
        print(f"## Evaluating with threshold = {thr:.3f} (={thr*100:.1f}%)")
        masks_jax = jnp.asarray(all_masks[thr])

        per_seed_gains = []
        for s in range(N_SEEDS):
            ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_p0c.pkl")
            if not os.path.exists(ckpt):
                print(f"  seed {s}: checkpoint not found, skipping")
                continue
            obs_dim = 3 * N_turb + 3
            act_dim = N_turb
            model = load_nnx_policy(ckpt, obs_dim, act_dim)
            print(f"  seed {s}: evaluating...", end=" ", flush=True)

            gains = evaluate_with_mask(model, positions_jax, phis, vs,
                                        baselines, N_turb, masks_jax)
            gains_np = np.asarray(gains)
            per_seed_gains.append(gains_np)
            print(f"mean = {gains_np.mean():+.3f}%")

        if not per_seed_gains:
            continue

        g_stack = np.stack(per_seed_gains, axis=0)
        cond_mean = g_stack.mean(axis=0)

        # Overall stats
        thr_result = {
            "threshold_pct": thr * 100,
            "mean_gain_pct": float(cond_mean.mean()),
            "marginal_mean_pct": float(cond_mean.mean()),
            "aligned_cube_mean_pct": float(cond_mean[aligned_cube].mean()) if aligned_cube.sum() > 0 else None,
            "aligned_cube_n": int(aligned_cube.sum()),
            "n_locked_mean": float(all_masks[thr].sum(axis=1).mean()),
            "n_locked_max": int(all_masks[thr].sum(axis=1).max()),
            "lock_set_diff_vs_baseline_pct": None,
            "per_seed_mean_pct": [float(g.mean()) for g in per_seed_gains],
        }

        # Lock set difference vs baseline
        if thr != baseline_thr:
            diff = np.any(all_masks[thr] != baseline_masks, axis=1)
            thr_result["lock_set_diff_vs_baseline_pct"] = float(100 * diff.sum() / N_CONDITIONS)

        results["per_threshold"][f"thr_{thr*100:.1f}pct"] = thr_result

        print(f"  Marginal mean: {thr_result['marginal_mean_pct']:+.3f}%")
        if thr_result["aligned_cube_mean_pct"] is not None:
            print(f"  Aligned-cube:  {thr_result['aligned_cube_mean_pct']:+.3f}% "
                  f"(n={thr_result['aligned_cube_n']})")

    # Save results
    out_path = os.path.join(FIG_DIR, "threshold_sensitivity_locking.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")

    # Summary comparison table
    print(f"\n{'='*60}")
    print(f"{'Threshold':>12s}  {'Marginal':>10s}  {'Aligned-cube':>13s}  "
          f"{'Mean #locked':>13s}  {'Diff vs 1%':>11s}")
    for thr in THRESHOLDS:
        key = f"thr_{thr*100:.1f}pct"
        r = results["per_threshold"][key]
        diff_str = "---" if r["lock_set_diff_vs_baseline_pct"] is None else f"{r['lock_set_diff_vs_baseline_pct']:.1f}%"
        ac = f"{r['aligned_cube_mean_pct']:+.3f}%" if r['aligned_cube_mean_pct'] is not None else "N/A"
        print(f"  {thr*100:.1f}%       {r['marginal_mean_pct']:+.3f}%    {ac:>13s}  "
              f"{r['n_locked_mean']:.2f}         {diff_str:>11s}")


if __name__ == "__main__":
    main()
