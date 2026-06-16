# -*- coding: utf-8 -*-
"""
Stage A 5x5 closed-loop training script (SBX + Numba physics path).

Trains a PPO controller on the 25-turbine farm using SBX (JAX-accelerated SB3)
with 16 parallel SubprocVecEnv workers. Designed to be run on the RTX 4090
host after the project has been migrated per MIGRATION_4090.md.

Outputs:
  ppo_5x5_seedN.zip          one model per seed
  tb_5x5/PPO_<seed>/         TensorBoard logs
  latex_draft/figures/
    fig_5x5_training_curve.{pdf,jpg}
    5x5_training_stats.json

Prerequisites (see MIGRATION_4090.md §1):
  pip install sbx-rl "jax[cuda12]" numba stable-baselines3 gymnasium

Usage:
  cd codes
  python train_5x5_sbx.py                       # default: 3 seeds, 3e7 steps each
  TOTAL_STEPS=1e7 N_SEEDS=1 python train_5x5_sbx.py    # quick smoke test
  N_ENVS=8 python train_5x5_sbx.py              # half the workers (lower-RAM host)
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
    print("[warn] sbx not installed; falling back to torch SB3 PPO (much slower on 4090).")

from windfarm_env import WindFarmYawEnv, d_0, z_h


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "latex_draft", "figures")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_5x5")
TB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_5x5")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)


N_SEEDS = int(os.environ.get("N_SEEDS", 3))
N_ENVS = int(os.environ.get("N_ENVS", 16))
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 3e7)))
N_STEPS = int(os.environ.get("N_STEPS", 4096))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 512))


def create_wind_farm_layout_5x5():
    """5x5 grid, 7-deg tilt, 7*d_0 spacing. Identical to benchmark_inference_latency.py."""
    N_rows, N_cols = 5, 5
    spacing = 7 * d_0
    ang = np.radians(7.0)
    pos = []
    for i in range(N_rows):
        for j in range(N_cols):
            x = -i * spacing * np.sin(ang) + j * spacing
            y = i * spacing * np.cos(ang)
            pos.append((x, y, z_h))
    return pos, N_rows, N_cols


def make_env_factory(seed_offset):
    positions, R, C = create_wind_farm_layout_5x5()
    def _make():
        env = WindFarmYawEnv(positions, R, C, j=1, randomize_wind=True, max_steps=200)
        return env
    return _make


def train_one_seed(seed):
    print(f"\n{'='*60}\n# Training 5x5 controller, seed = {seed}, backend = {backend}\n{'='*60}")
    np.random.seed(seed)

    venv = SubprocVecEnv([make_env_factory(seed + 1000 * k) for k in range(N_ENVS)])
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    model = PPO(
        "MlpPolicy", venv,
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
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=TB_DIR,
        seed=seed,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 500_000 // N_ENVS),
        save_path=CKPT_DIR,
        name_prefix=f"ppo_5x5_seed{seed}",
        save_vecnormalize=True,
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt_cb,
                tb_log_name=f"PPO_seed{seed}")
    elapsed = time.time() - t0

    final_path = os.path.join(CKPT_DIR, f"ppo_5x5_seed{seed}_final.zip")
    model.save(final_path)
    venv.save(os.path.join(CKPT_DIR, f"vecnormalize_seed{seed}.pkl"))
    venv.close()

    return dict(seed=seed, final_path=final_path, wall_clock_s=elapsed,
                total_steps=TOTAL_STEPS, n_envs=N_ENVS)


def load_tb_curves():
    """Best-effort: read scalar 'rollout/ep_rew_mean' from each PPO_seedN run."""
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
    ax.set_title("5x5 closed-loop training (SBX PPO, 16 envs)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fname + ".pdf", bbox_inches="tight")
    fig.savefig(fname + ".jpg", dpi=200, bbox_inches="tight")


def main():
    print(f"# Stage A 5x5 training")
    print(f"# backend     : {backend}")
    print(f"# seeds       : {N_SEEDS}")
    print(f"# parallel env: {N_ENVS}")
    print(f"# total steps : {TOTAL_STEPS:.2e} per seed")

    run_records = []
    for s in range(N_SEEDS):
        run_records.append(train_one_seed(seed=s))

    curves = load_tb_curves()
    plot_training_curves(curves, os.path.join(OUT_DIR, "fig_5x5_training_curve"))

    stats = dict(
        backend=backend,
        n_seeds=N_SEEDS,
        n_envs=N_ENVS,
        total_steps=TOTAL_STEPS,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        runs=run_records,
    )
    with open(os.path.join(OUT_DIR, "5x5_training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved checkpoints to {CKPT_DIR}")
    print(f"Saved figure + stats to {OUT_DIR}")


if __name__ == "__main__":
    main()
