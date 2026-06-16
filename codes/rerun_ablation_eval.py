# -*- coding: utf-8 -*-
"""Rerun the evaluation phase of the S3 ablation from existing checkpoints.

The original ablation_downstream_lock.py crashed in evaluate() because it
passed a batched action from venv.step() into the underlying single-env
step(), causing IndexError. This script bypasses venv.step entirely:
- VecNormalize is used only to normalize the observation before predict
- env.step is called directly with the unbatched action
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sbx import PPO
    backend = "sbx"
except ImportError:
    from stable_baselines3 import PPO
    backend = "sb3"

import windfarm_env as wfe
from windfarm_env import WindFarmYawEnv, create_wind_farm_layout_3x3


OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "latex_draft", "figures",
)
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "checkpoints_ablation")

N_SEEDS = 3
SETTLE_STEPS = 150

_ORIGINAL_FIND_DOWNSTREAM = wfe.find_downstream_turbines


def enable_lock():
    wfe.find_downstream_turbines = _ORIGINAL_FIND_DOWNSTREAM


def disable_lock():
    wfe.find_downstream_turbines = lambda positions, wind_dir, U_inf: []


def evaluate(condition, seed, eval_grid):
    if condition == "lock_on":
        enable_lock()
    else:
        disable_lock()

    ckpt = os.path.join(CKPT_DIR, f"ppo_{condition}_seed{seed}.zip")
    vn_path = os.path.join(CKPT_DIR, f"vecnormalize_{condition}_seed{seed}.pkl")

    positions, R, C = create_wind_farm_layout_3x3()
    env = WindFarmYawEnv(positions, R, C, j=1, randomize_wind=False, max_steps=SETTLE_STEPS + 10)

    venv = DummyVecEnv([lambda: env])
    vn = VecNormalize.load(vn_path, venv)
    vn.training = False
    vn.norm_reward = False

    model = PPO.load(ckpt)

    gains = []
    for phi, v in eval_grid:
        obs, _ = env.reset(options=dict(specific_wind_dir=float(phi),
                                        specific_wind_speed=float(v)))
        for _ in range(SETTLE_STEPS):
            obs_norm = vn.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=True)
            action = np.asarray(action).reshape(-1)
            obs, _, _, _, _ = env.step(action)
        gain = (env.current_total_mw - env.baseline_mw) / env.baseline_mw * 100
        gains.append(float(gain))
    return float(np.mean(gains)), float(np.std(gains)), gains


def main():
    eval_grid = [(phi, 11.4) for phi in np.linspace(200, 340, 8)]
    eval_grid += [(270.0, v) for v in (8.0, 11.4, 14.0)]

    eval_results = []
    for seed in range(N_SEEDS):
        for cond in ("lock_on", "lock_off"):
            mean_gain, std_gain, gains = evaluate(cond, seed, eval_grid)
            eval_results.append(dict(condition=cond, seed=seed,
                                     eval_mean_pct=mean_gain,
                                     eval_std_pct=std_gain,
                                     per_point_pct=gains))
            print(f"  [{cond} seed={seed}] eval mean gain = "
                  f"{mean_gain:+.2f} +/- {std_gain:.2f} %")

    by_cond = {c: [e["eval_mean_pct"] for e in eval_results if e["condition"] == c]
               for c in ("lock_on", "lock_off")}
    summary = {c: dict(mean=float(np.mean(by_cond[c])),
                       std=float(np.std(by_cond[c])),
                       n_seeds=len(by_cond[c])) for c in by_cond}

    print(f"\nSummary:")
    for c, s in summary.items():
        print(f"  {c}: {s['mean']:+.2f} +/- {s['std']:.2f} %  (n={s['n_seeds']})")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    labels = list(summary.keys())
    means = [summary[c]["mean"] for c in labels]
    stds = [summary[c]["std"] for c in labels]
    ax.bar(labels, means, yerr=stds, capsize=6,
           color=["#4C78A8", "#E45756"], edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Mean farm-power gain over baseline [%]")
    ax.set_title(f"Downstream-lock ablation (3x3, n={N_SEEDS} seeds)")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_lock_ablation_eval.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig_lock_ablation_eval.jpg"),
                dpi=200, bbox_inches="tight")

    out = dict(backend=backend, n_seeds=N_SEEDS,
               settle_steps=SETTLE_STEPS,
               eval_grid=[list(pt) for pt in eval_grid],
               per_run=eval_results, summary=summary)
    with open(os.path.join(OUT_DIR, "lock_ablation_stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved bar chart + JSON to {OUT_DIR}")


if __name__ == "__main__":
    main()
