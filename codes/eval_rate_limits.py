#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actuator rate-limit evaluation.

Quantifies how yaw rate constraints erode DRL policy gains.
We do NOT retrain; we evaluate the already-trained policy under
constrained execution, where the yaw increment per step is limited
by rate_limit_per_step = yaw_rate * control_period.

Typical yaw rate: 0.5 deg/s (industry standard for large turbines)
Control period sweep: 1, 5, 10, 30, 60 s
  → rate_limit_per_step: 0.5, 2.5, 5, 15, 30 deg/step

The original training uses ±5 deg/step (no rate limit). When the
rate limit is tighter, the policy's commanded increments are clipped.

Output:
  latex_draft/figures/rate_limit_results.json
  latex_draft/figures/fig_rate_limit_sensitivity.{pdf,jpg}
"""

import os
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx

from windfarm_env import create_wind_farm_layout_3x3
from windfarm_env_jax import (
    env_reset, env_step, positions_to_jax,
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

YAW_RATE = 0.5  # deg/s (typical)
CONTROL_PERIODS = [1, 5, 10, 30, 60]  # seconds
# rate_limit_per_step = YAW_RATE * CONTROL_PERIOD


def evaluate_rate_limited(model, positions_jax, phis, vs, baselines, N_turb,
                          rate_limit_per_step):
    """Evaluate policy with yaw rate limit.

    The policy outputs an action (yaw increment), but it is clipped to
    [-rate_limit_per_step, +rate_limit_per_step] before being applied
    to the environment.
    """
    N_steps = SETTLE_STEPS
    B = phis.shape[0]
    rl = jnp.float32(rate_limit_per_step)

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
            # Apply rate limit (clip yaw increment)
            actions = jnp.clip(actions, -rl, rl)
            new_states, new_obs, _, _ = step_one(states, actions)
            return (new_states, new_obs), None

        (final_states, _), _ = jax.lax.scan(body, (states, obs_batch), None, length=N_steps)
        return final_states.total_mw, final_states.gammas

    total_mw, gammas = run(model, phis, vs)
    gains = (total_mw - baselines) / baselines * 100.0
    yaws = jnp.abs(gammas)
    return gains, yaws.max(axis=1), yaws.mean(axis=1)


def main():
    t_start = time.time()
    positions, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions)
    positions_jax = positions_to_jax(positions)

    print(f"# Actuator rate-limit evaluation")
    print(f"# YAW_RATE = {YAW_RATE} deg/s")
    print(f"# Control periods: {CONTROL_PERIODS} s")
    print(f"# N_SEEDS={N_SEEDS}, N_CONDITIONS={N_CONDITIONS}")
    print(f"# device={jax.devices()[0]}")

    # Build condition batch
    from eval_p0c_randomized import build_batch
    phis, vs, baselines = build_batch(positions_jax, EVAL_SEED)
    baselines_np = np.asarray(baselines)
    phis_np = np.asarray(phis)
    vs_np = np.asarray(vs)

    # Regime masks
    dphi_arr = np.abs(((phis_np - 270.0 + 180.0) % 360.0) - 180.0)
    aligned_cube = (dphi_arr < 15.0) & (vs_np < 11.4)

    # Also include "unlimited" baseline (5 deg/step = original training)
    rate_limits = [5.0]  # original (no effective limit)
    for cp in CONTROL_PERIODS:
        rate_limits.append(YAW_RATE * cp)

    results = {}

    for rl in rate_limits:
        key_str = f"rl_{rl:.1f}"
        if rl >= 5.0:
            desc = f"unlimited (±5°/step)"
            cp_s = "unlimited"
        else:
            cp = rl / YAW_RATE
            desc = f"rate=±{rl:.1f}°/step (T={cp:.0f}s)"
            cp_s = f"{cp:.0f}s"

        print(f"\n## {desc}")

        seed_gains = []
        for s in range(N_SEEDS):
            ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_p0c.pkl")
            obs_dim = 3 * N_turb + 3
            act_dim = N_turb
            model = load_nnx_policy(ckpt, obs_dim, act_dim)

            gains, max_yaws, mean_yaws = evaluate_rate_limited(
                model, positions_jax, phis, vs, baselines, N_turb, rl)
            seed_gains.append(np.asarray(gains))

        gains_stack = np.stack(seed_gains, axis=0)
        cond_mean = gains_stack.mean(axis=0)

        result = {
            "rate_limit_per_step_deg": rl,
            "control_period_s": cp_s,
            "description": desc,
            "marginal_mean_pct": float(cond_mean.mean()),
            "aligned_cube_pct": float(cond_mean[aligned_cube].mean())
                if aligned_cube.sum() > 0 else None,
            "per_seed_means": [float(sg.mean()) for sg in seed_gains],
        }
        results[key_str] = result
        ac_s = f"  aligned-cube={result['aligned_cube_pct']:+.3f}%" \
            if result['aligned_cube_pct'] is not None else ""
        print(f"  marginal={result['marginal_mean_pct']:+.3f}%{ac_s}")

    # Save
    out_path = os.path.join(FIG_DIR, "rate_limit_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Gain vs control period
    ax = axes[0]
    cp_vals = [0] + CONTROL_PERIODS  # 0 = unlimited
    rl_vals_plot = [5.0] + [YAW_RATE * cp for cp in CONTROL_PERIODS]
    marginal_y = [results[f"rl_{rl:.1f}"]["marginal_mean_pct"] for rl in rl_vals_plot]
    aligned_y = [results[f"rl_{rl:.1f}"]["aligned_cube_pct"] for rl in rl_vals_plot]

    ax.plot(cp_vals, marginal_y, 'b-o', ms=7, lw=1.5, label="Marginal mean")
    ax.plot(cp_vals, aligned_y, 'r-s', ms=7, lw=1.5, label="Aligned-cube")
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("Control period T [s]")
    ax.set_ylabel("Mean gain [%]")
    ax.set_title("(a) Gain vs control period (yaw rate = 0.5 °/s)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Gain vs rate limit per step
    ax = axes[1]
    rl_sorted = sorted(rl_vals_plot)
    marginal_y2 = [results[f"rl_{rl:.1f}"]["marginal_mean_pct"] for rl in rl_sorted]
    aligned_y2 = [results[f"rl_{rl:.1f}"]["aligned_cube_pct"] for rl in rl_sorted]

    ax.plot(rl_sorted, marginal_y2, 'b-o', ms=7, lw=1.5, label="Marginal mean")
    ax.plot(rl_sorted, aligned_y2, 'r-s', ms=7, lw=1.5, label="Aligned-cube")
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel(r"Rate limit per step [°]")
    ax.set_ylabel("Mean gain [%]")
    ax.set_title("(b) Gain vs yaw increment limit")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Actuator rate-limit sensitivity of DRL yaw controller", fontsize=11)
    fig.tight_layout()
    for ext in ['pdf', 'jpg']:
        path = os.path.join(FIG_DIR, f"fig_rate_limit_sensitivity.{ext}")
        fig.savefig(path, dpi=300 if ext == 'jpg' else None, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)

    print(f"\nTotal wall-clock: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
