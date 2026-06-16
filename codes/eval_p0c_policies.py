# -*- coding: utf-8 -*-
"""
Evaluate the P0-c convergence policies on the same 19-point deterministic
(phi, v) grid used by cross_val_jaxenv_vs_numpyenv.py, and aggregate across
the 3 seeds.

This is the paper-evidence "trained policy beats baseline" table + figure.

Reads:
  checkpoints_3x3_nnx_jaxenv/policy_seed{0,1,2}_p0c.pkl

Writes:
  latex_draft/figures/fig_p0c_eval.{pdf,jpg}
  latex_draft/figures/p0c_eval.json
"""

import os
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax import nnx

from windfarm_env import WindFarmYawEnv, create_wind_farm_layout_3x3
from train_3x3_nnx import ActorCritic
from cross_val_jaxenv_vs_numpyenv import (
    build_eval_grid, load_nnx_policy, make_nnx_predict_fn,
    rollout_policy, rollout_zero_yaw, SETTLE_STEPS,
)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR),
                       "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_SEEDS = 3


def evaluate_seed(seed: int, env, grid):
    ckpt = os.path.join(CKPT_DIR, f"policy_seed{seed}_p0c.pkl")
    assert os.path.exists(ckpt), f"missing {ckpt}"
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    model = load_nnx_policy(ckpt, obs_dim, act_dim)
    predict_fn = make_nnx_predict_fn(model)

    rows = []
    for phi, v in grid:
        g_pol, mw_pol, mw_base, max_yaw, mean_yaw = rollout_policy(
            predict_fn, env, phi, v, SETTLE_STEPS)
        rows.append(dict(phi=phi, v=v,
                         policy_gain_pct=g_pol,
                         policy_total_mw=mw_pol,
                         baseline_mw=mw_base,
                         max_abs_yaw_deg=max_yaw,
                         mean_abs_yaw_deg=mean_yaw))
    return rows


def main():
    positions, R, C = create_wind_farm_layout_3x3()
    env = WindFarmYawEnv(positions, R, C, j=1, randomize_wind=False,
                         max_steps=SETTLE_STEPS + 10)
    grid = build_eval_grid()

    # Zero-yaw baseline (env-only; same across seeds).
    zero_gains = [rollout_zero_yaw(env, phi, v, SETTLE_STEPS)
                  for phi, v in grid]

    # Per-seed policy evaluation.
    all_seed_gains = np.zeros((N_SEEDS, len(grid)), dtype=np.float32)
    per_seed_rows = []
    for s in range(N_SEEDS):
        print(f"\n# seed {s}")
        rows = evaluate_seed(s, env, grid)
        per_seed_rows.append(rows)
        for i, r in enumerate(rows):
            all_seed_gains[s, i] = r["policy_gain_pct"]
            print(f"  phi={r['phi']:5.1f} v={r['v']:4.1f}  "
                  f"policy gain {r['policy_gain_pct']:+7.3f}%  "
                  f"max|yaw|={r['max_abs_yaw_deg']:5.1f}deg")

    seed_mean_per_cond = all_seed_gains.mean(axis=0)
    seed_std_per_cond = all_seed_gains.std(axis=0)

    print(f"\n=== aggregate ===")
    for i, (phi, v) in enumerate(grid):
        print(f"  phi={phi:5.1f} v={v:4.1f}  "
              f"3-seed gain = {seed_mean_per_cond[i]:+6.3f}% "
              f"+/- {seed_std_per_cond[i]:.3f}%  "
              f"(zero-yaw = {zero_gains[i]:+.3f}%)")

    overall_mean = float(seed_mean_per_cond.mean())
    overall_std = float(seed_mean_per_cond.std())
    n_pos = int((seed_mean_per_cond > 0).sum())
    n_beats_zero = int(((seed_mean_per_cond > np.array(zero_gains))).sum())
    print(f"\n  overall mean gain = {overall_mean:+.3f}%  "
          f"std over conditions = {overall_std:.3f}%")
    print(f"  positive on {n_pos}/{len(grid)} conditions")
    print(f"  beats zero-yaw on {n_beats_zero}/{len(grid)}")

    # Figure: per-condition seed-aggregated gain + per-seed final ep_rew curves.
    n = len(grid)
    x = np.arange(n)
    labels = [f"({phi:.0f},{v:.0f})" for phi, v in grid]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 4.2))

    w = 0.36
    axA.bar(x - w / 2, zero_gains, w, color="#888",
            label=f"Zero-yaw baseline  mean={np.mean(zero_gains):+.3f}%",
            edgecolor="black", linewidth=0.4)
    axA.bar(x + w / 2, seed_mean_per_cond, w, yerr=seed_std_per_cond, capsize=2,
            color="#E45756",
            label=f"NNX jax-env P0-c (3 seeds x 3e7)  mean={overall_mean:+.3f}%",
            edgecolor="black", linewidth=0.4)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    axA.set_ylabel("Farm-power gain over baseline [%]")
    axA.set_title("P0-c deterministic eval, 3-seed mean +/- std "
                  "(settle = 150 steps)", fontsize=10)
    axA.axhline(0, color="black", lw=0.5)
    axA.grid(alpha=0.3, axis="y")
    axA.legend(frameon=False, fontsize=8, loc="best")

    # Per-seed scatter of overall gain per condition to show consistency.
    for s in range(N_SEEDS):
        axB.plot(x, all_seed_gains[s], "o-", lw=1.0, ms=4,
                 label=f"seed {s}  mean={all_seed_gains[s].mean():+.2f}%")
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    axB.set_ylabel("Farm-power gain [%]")
    axB.set_title("Per-seed agreement across conditions\n"
                  "(tight band = stable policy, no seed lottery)",
                  fontsize=10)
    axB.axhline(0, color="black", lw=0.5)
    axB.grid(alpha=0.3)
    axB.legend(frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    out_fig = os.path.join(FIG_DIR, "fig_p0c_eval")
    fig.savefig(out_fig + ".pdf", bbox_inches="tight")
    fig.savefig(out_fig + ".jpg", dpi=180, bbox_inches="tight")
    print(f"\n  -> {out_fig}.pdf  {out_fig}.jpg")

    out = dict(
        n_seeds=N_SEEDS,
        total_steps_per_seed=30_000_000,
        n_envs=256,
        settle_steps=SETTLE_STEPS,
        eval_grid=[list(pt) for pt in grid],
        zero_yaw_gain_pct=[float(g) for g in zero_gains],
        per_seed=[[dict(phi=r["phi"], v=r["v"],
                        policy_gain_pct=r["policy_gain_pct"],
                        max_abs_yaw_deg=r["max_abs_yaw_deg"],
                        mean_abs_yaw_deg=r["mean_abs_yaw_deg"])
                   for r in per_seed_rows[s]] for s in range(N_SEEDS)],
        seed_aggregate=dict(
            mean_gain_per_condition=[float(g) for g in seed_mean_per_cond],
            std_gain_per_condition=[float(g) for g in seed_std_per_cond],
            overall_mean_gain_pct=overall_mean,
            std_over_conditions=overall_std,
            n_positive_of=n_pos,
            n_beats_zero_of=n_beats_zero,
            n_total=len(grid),
        ),
    )
    out_path = os.path.join(FIG_DIR, "p0c_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
