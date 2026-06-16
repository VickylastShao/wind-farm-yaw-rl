# -*- coding: utf-8 -*-
"""
3x3 PPO training (local, CPU-only).

Trains a PPO controller on the 9-turbine farm with SubprocVecEnv parallel
rollouts. Backend auto-selects: SBX (JAX-CPU, XLA-jit) if `sbx-rl` is
installed, else SB3 (torch-CPU). Both run fine on the T400-class local
host because at this scale CPU-bound rollout dominates and the policy
network is tiny.

Outputs:
  codes/checkpoints_3x3/ppo_3x3_seedN_final.zip + VecNormalize pkl
  codes/tb_3x3/PPO_seedN/                          tensorboard logs
  latex_draft/figures/
    fig_3x3_training_curve.{pdf,jpg}
    3x3_training_stats.json

Usage:
    cd codes
    python train_3x3_local.py                                   # 2 seeds x 2e6 steps
    N_SEEDS=1 TOTAL_STEPS=5e5 python train_3x3_local.py         # smoke test
    N_ENVS=4 python train_3x3_local.py                          # lower CPU pressure
    BACKEND=sb3 python train_3x3_local.py                       # force torch SB3
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

_force = os.environ.get("BACKEND", "").lower()
if _force == "sb3":
    from stable_baselines3 import PPO
    backend = "sb3-torch-cpu"
else:
    try:
        from sbx import PPO
        backend = "sbx-jax-cpu"
    except ImportError:
        from stable_baselines3 import PPO
        backend = "sb3-torch-cpu"

from windfarm_env import WindFarmYawEnv, create_wind_farm_layout_3x3


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "latex_draft", "figures")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_3x3")
TB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_3x3")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)


N_SEEDS = int(os.environ.get("N_SEEDS", 2))
N_ENVS = int(os.environ.get("N_ENVS", 8))
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 2e6)))
N_STEPS = int(os.environ.get("N_STEPS", 2048))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))


def make_env_factory(seed_offset):
    positions, R, C = create_wind_farm_layout_3x3()
    def _make():
        return WindFarmYawEnv(positions, R, C, j=1, randomize_wind=True, max_steps=200)
    return _make


def train_one_seed(seed):
    print(f"\n{'='*60}\n# Training 3x3 controller ({backend}), seed = {seed}\n{'='*60}")
    np.random.seed(seed)

    venv = SubprocVecEnv([make_env_factory(seed + 1000 * k) for k in range(N_ENVS)])
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    ppo_kwargs = dict(
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=3e-4,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=TB_DIR,
        seed=seed,
        verbose=1,
    )
    if backend == "sb3-torch-cpu":
        ppo_kwargs["device"] = "cpu"
    model = PPO("MlpPolicy", venv, **ppo_kwargs)

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 250_000 // N_ENVS),
        save_path=CKPT_DIR,
        name_prefix=f"ppo_3x3_seed{seed}",
        save_vecnormalize=True,
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt_cb,
                tb_log_name=f"PPO_seed{seed}")
    elapsed = time.time() - t0

    final_path = os.path.join(CKPT_DIR, f"ppo_3x3_seed{seed}_final.zip")
    vn_path = os.path.join(CKPT_DIR, f"vecnormalize_seed{seed}.pkl")
    model.save(final_path)
    venv.save(vn_path)
    venv.close()

    return dict(seed=seed, final_path=final_path, vn_path=vn_path,
                wall_clock_s=elapsed, total_steps=TOTAL_STEPS, n_envs=N_ENVS)


def load_tb_curves():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[warn] tensorboard not importable; skipping curve plot.")
        return None
    curves = {}
    for entry in sorted(os.listdir(TB_DIR)):
        run_dir = os.path.join(TB_DIR, entry)
        if not os.path.isdir(run_dir):
            continue
        try:
            ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
            ea.Reload()
            if "rollout/ep_rew_mean" in ea.Tags().get("scalars", []):
                events = ea.Scalars("rollout/ep_rew_mean")
                steps = np.array([e.step for e in events])
                vals = np.array([e.value for e in events])
                curves[entry] = (steps, vals)
        except Exception as exc:
            print(f"[tb] skip {entry}: {exc}")
    return curves


def plot_training_curves(curves, fname):
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label, (steps, vals) in curves.items():
        ax.plot(steps, vals, lw=1.2, label=label)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode mean reward")
    ax.set_title(f"3x3 closed-loop training (SB3 PPO torch-CPU, {N_ENVS} envs)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fname + ".pdf", bbox_inches="tight")
    fig.savefig(fname + ".jpg", dpi=200, bbox_inches="tight")


def main():
    print(f"# 3x3 local training")
    print(f"# backend     : {backend}")
    print(f"# seeds       : {N_SEEDS}")
    print(f"# parallel env: {N_ENVS}")
    print(f"# total steps : {TOTAL_STEPS:.2e} per seed")
    print(f"# n_steps     : {N_STEPS}")
    print(f"# batch_size  : {BATCH_SIZE}")

    run_records = []
    for s in range(N_SEEDS):
        run_records.append(train_one_seed(seed=s))

    curves = load_tb_curves()
    plot_training_curves(curves, os.path.join(OUT_DIR, "fig_3x3_training_curve"))

    stats = dict(
        backend=backend,
        n_seeds=N_SEEDS,
        n_envs=N_ENVS,
        total_steps=TOTAL_STEPS,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        runs=run_records,
    )
    with open(os.path.join(OUT_DIR, "3x3_training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved checkpoints to {CKPT_DIR}")
    print(f"Saved figure + stats to {OUT_DIR}")


if __name__ == "__main__":
    main()
