# -*- coding: utf-8 -*-
"""
Downstream-locking ablation (S3): does forcing the most-downstream turbines
to gamma = 0 actually help, or is it just a hand-engineered prior that the
policy could discover on its own?

Trains two PPO controllers on the 3x3 farm with otherwise identical
hyperparameters:
  (A) lock-enabled    : default WindFarmYawEnv behavior
  (B) lock-disabled   : monkey-patch find_downstream_turbines to return [],
                        so the env never locks any turbine.

Both runs are evaluated on the same fixed grid of (phi, v) test conditions
and the mean farm-power gain over baseline is reported. Per the paper we
report:
  - mean reward curve (training)
  - mean evaluation power gain (post-training)
  - sample efficiency: env-steps to reach 80 % of (A)'s final gain.

Outputs:
  latex_draft/figures/
    fig_lock_ablation_curve.{pdf,jpg}
    fig_lock_ablation_eval.{pdf,jpg}
    lock_ablation_stats.json
  codes/checkpoints_ablation/
    ppo_lock_on_seedN.zip / ppo_lock_off_seedN.zip

Usage:
  cd codes
  python ablation_downstream_lock.py                       # 3 seeds, 1e7 steps each
  N_SEEDS=2 TOTAL_STEPS=5e6 python ablation_downstream_lock.py    # quick smoke test
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

try:
    from sbx import PPO
    backend = "sbx"
except ImportError:
    from stable_baselines3 import PPO
    backend = "sb3"
    print("[warn] sbx not installed; falling back to torch PPO.")

import windfarm_env as wfe
from windfarm_env import (
    WindFarmYawEnv,
    calculate_inflow_speeds, power_output,
    create_wind_farm_layout_3x3,
    C_T, I, d_0, alpha_star, beta_star, alpha,
)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_SCRIPT_DIR)
OUT_DIR = os.path.join(_PROJ_ROOT, "latex_draft", "figures")
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_ablation")
TB_DIR = os.path.join(_SCRIPT_DIR, "tb_ablation")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)


N_SEEDS = int(os.environ.get("N_SEEDS", 3))
N_ENVS = int(os.environ.get("N_ENVS", 16))
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 1e7)))
N_STEPS = int(os.environ.get("N_STEPS", 2048))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))


_ORIGINAL_FIND_DOWNSTREAM = wfe.find_downstream_turbines


def enable_lock():
    wfe.find_downstream_turbines = _ORIGINAL_FIND_DOWNSTREAM


def disable_lock():
    """Replace the downstream finder with one that returns []."""
    wfe.find_downstream_turbines = lambda positions, wind_dir, U_inf: []


def make_env_factory(seed_offset):
    """Returns a thunk that builds a 3x3 env. The lock state is captured at
    call time (when SubprocVecEnv forks), so callers must call
    enable_lock() / disable_lock() *before* constructing the SubprocVecEnv."""
    positions, R, C = create_wind_farm_layout_3x3()
    def _make():
        return WindFarmYawEnv(positions, R, C, j=1, randomize_wind=True, max_steps=200)
    return _make


def train_one_condition(condition, seed):
    if condition == "lock_on":
        enable_lock()
    elif condition == "lock_off":
        disable_lock()
    else:
        raise ValueError(condition)

    print(f"\n{'='*60}\n# Training [{condition}] seed = {seed}\n{'='*60}")
    np.random.seed(seed)

    venv = SubprocVecEnv([make_env_factory(seed + 1000 * k) for k in range(N_ENVS)])
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    model = PPO(
        "MlpPolicy", venv,
        n_steps=N_STEPS, batch_size=BATCH_SIZE,
        learning_rate=3e-4, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=TB_DIR, seed=seed, verbose=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS,
                tb_log_name=f"{condition}_seed{seed}")
    elapsed = time.time() - t0

    final_path = os.path.join(CKPT_DIR, f"ppo_{condition}_seed{seed}.zip")
    vn_path = os.path.join(CKPT_DIR, f"vecnormalize_{condition}_seed{seed}.pkl")
    model.save(final_path)
    venv.save(vn_path)
    venv.close()

    return dict(condition=condition, seed=seed,
                final_path=final_path, vn_path=vn_path,
                wall_clock_s=elapsed)


def evaluate(record, eval_grid):
    """Roll out the trained policy on a fixed grid of (phi, v) and return
    mean farm-power gain over the no-yaw baseline.

    Implementation note: we deliberately bypass venv.step(). The original
    code wrapped a single env in DummyVecEnv and called venv.step(action)
    inside the loop -- the resulting (1, N) batched action was forwarded
    into the underlying env.step, causing IndexError on the downstream
    mask. Instead we keep VecNormalize ONLY for observation normalization
    via vn.normalize_obs(), and step the underlying env directly with a
    flattened action. This mirrors rerun_ablation_eval.py.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    if record["condition"] == "lock_on":
        enable_lock()
    else:
        disable_lock()

    positions, R, C = create_wind_farm_layout_3x3()
    env = WindFarmYawEnv(positions, R, C, j=1, randomize_wind=False, max_steps=200)

    venv_for_vn = DummyVecEnv([lambda: env])
    vn = VecNormalize.load(record["vn_path"], venv_for_vn)
    vn.training = False
    vn.norm_reward = False

    model = PPO.load(record["final_path"])

    gains = []
    for phi, v in eval_grid:
        obs, _ = env.reset(options=dict(specific_wind_dir=float(phi),
                                        specific_wind_speed=float(v)))
        for _ in range(150):  # let the controller settle
            obs_norm = vn.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=True)
            action = np.asarray(action).reshape(-1)
            obs, _, _, _, _ = env.step(action)
        gain = (env.current_total_mw - env.baseline_mw) / env.baseline_mw * 100
        gains.append(float(gain))
    return float(np.mean(gains)), float(np.std(gains))


def main():
    print(f"# Downstream-lock ablation (S3)")
    print(f"# backend     : {backend}")
    print(f"# seeds/cond  : {N_SEEDS}")
    print(f"# parallel env: {N_ENVS}")
    print(f"# total steps : {TOTAL_STEPS:.2e} per run")

    records = []
    for seed in range(N_SEEDS):
        for cond in ("lock_on", "lock_off"):
            records.append(train_one_condition(cond, seed))

    eval_grid = [(phi, 11.4) for phi in np.linspace(200, 340, 8)]
    eval_grid += [(270.0, v) for v in (8.0, 11.4, 14.0)]

    eval_results = []
    for r in records:
        mean_gain, std_gain = evaluate(r, eval_grid)
        eval_results.append(dict(condition=r["condition"], seed=r["seed"],
                                 eval_mean_pct=mean_gain, eval_std_pct=std_gain))
        print(f"  [{r['condition']} seed={r['seed']}] eval mean gain = "
              f"{mean_gain:.2f} +/- {std_gain:.2f} %")

    by_cond = {c: [e["eval_mean_pct"] for e in eval_results if e["condition"] == c]
               for c in ("lock_on", "lock_off")}
    summary = {c: dict(mean=float(np.mean(by_cond[c])),
                       std=float(np.std(by_cond[c])),
                       n_seeds=len(by_cond[c])) for c in by_cond}

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    labels = list(summary.keys())
    means = [summary[c]["mean"] for c in labels]
    stds = [summary[c]["std"] for c in labels]
    ax.bar(labels, means, yerr=stds, capsize=6, color=["#4C78A8", "#E45756"],
           edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Mean farm-power gain over baseline [%]")
    ax.set_title(f"Downstream-lock ablation (3x3, n={N_SEEDS} seeds)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_lock_ablation_eval.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig_lock_ablation_eval.jpg"),
                dpi=200, bbox_inches="tight")

    out = dict(backend=backend, n_seeds=N_SEEDS, n_envs=N_ENVS,
               total_steps=TOTAL_STEPS, eval_grid=eval_grid,
               per_run=eval_results, summary=summary)
    with open(os.path.join(OUT_DIR, "lock_ablation_stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved bar chart + JSON to {OUT_DIR}")


if __name__ == "__main__":
    main()
