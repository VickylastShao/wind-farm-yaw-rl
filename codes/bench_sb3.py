# -*- coding: utf-8 -*-
"""
SB3 PPO baseline for the NNX-vs-SB3 wall-clock benchmark.

This is *not* the paper-evidence training run -- it is a small, fast,
side-by-side run that uses the exact same hyperparameters as
train_3x3_nnx.py so wall-clock and FPS can be compared head-to-head.

Differences from train_3x3_local.py:
  * Writes to a benchmark-specific output dir; never touches the
    paper's checkpoints_3x3 / fig_3x3_training_curve.* files.
  * VecNormalize is configured with norm_reward=False so the only
    delta vs the NNX run is the framework (SB3+torch vs NNX+JAX-GPU).
  * SyncVectorEnv (DummyVecEnv) instead of SubprocVecEnv to remove the
    multiprocessing-overhead variable; train_3x3_nnx.py also uses sync.
  * Records wall-clock breakdown into bench_3x3/metrics_sb3_seedN.json.

Env vars mirror train_3x3_nnx.py:
  N_SEEDS / N_ENVS / TOTAL_STEPS / N_STEPS / BATCH_SIZE / N_EPOCHS
"""

import os
import json
import time

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from windfarm_env import WindFarmYawEnv, create_wind_farm_layout_3x3


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(_SCRIPT_DIR, "bench_3x3")
os.makedirs(BENCH_DIR, exist_ok=True)


N_SEEDS = int(os.environ.get("N_SEEDS", 1))
N_ENVS = int(os.environ.get("N_ENVS", 8))
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 50_000)))
N_STEPS = int(os.environ.get("N_STEPS", 2048))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
N_EPOCHS = int(os.environ.get("N_EPOCHS", 10))

# Same constants as train_3x3_nnx.py
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.005
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
NET_ARCH = [128, 128]

DEVICE = os.environ.get("SB3_DEVICE", "cpu")  # cpu|cuda|auto


def make_env(seed_offset):
    positions, R, C = create_wind_farm_layout_3x3()

    def _thunk():
        return WindFarmYawEnv(positions, R, C, j=1, randomize_wind=True,
                              max_steps=200)
    return _thunk


def train_one_seed(seed: int) -> dict:
    print(f"\n{'='*60}\n# SB3 PPO seed={seed}  device={DEVICE}\n{'='*60}")
    np.random.seed(seed)

    # DummyVecEnv == sync; mirrors SyncVectorEnv in train_3x3_nnx.py.
    venv = DummyVecEnv([make_env(seed + 1000 * k) for k in range(N_ENVS)])
    venv = VecMonitor(venv)
    # norm_reward=False to mirror the NNX run, which does not normalize rewards.
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False,
                        clip_obs=10.0, gamma=GAMMA)

    model = PPO(
        "MlpPolicy", venv,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=dict(net_arch=NET_ARCH),
        seed=seed,
        verbose=0,
        device=DEVICE,
    )

    # SB3 logger -> in-memory list (no tensorboard, to keep it minimal).
    import collections
    log_records = collections.defaultdict(list)
    from stable_baselines3.common.callbacks import BaseCallback

    class CapturingCallback(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_end(self):
            # Pull the most recent ep_info from the model's ep buffer.
            buf = self.model.ep_info_buffer
            if buf and len(buf):
                recent = list(buf)[-20:]
                log_records["ep_rew_mean"].append(
                    float(np.mean([e["r"] for e in recent])))
                log_records["ep_len_mean"].append(
                    float(np.mean([e["l"] for e in recent])))
                log_records["total_steps"].append(self.num_timesteps)

    cb = CapturingCallback()

    t0 = time.time()
    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=cb)
    finally:
        elapsed = time.time() - t0

    fps = TOTAL_STEPS / max(1e-9, elapsed)
    iters = []
    for i, (r, l, s) in enumerate(zip(log_records["ep_rew_mean"],
                                       log_records["ep_len_mean"],
                                       log_records["total_steps"])):
        iters.append(dict(iteration=i, total_env_steps=int(s),
                          ep_rew_mean=r, ep_len_mean=l))

    metrics = dict(
        seed=seed,
        backend=f"sb3-torch-{DEVICE}",
        n_seeds=N_SEEDS,
        n_envs=N_ENVS,
        total_steps_target=TOTAL_STEPS,
        total_env_steps=TOTAL_STEPS,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        wall_clock_s=elapsed,
        fps=fps,
        final_ep_rew_mean=(log_records["ep_rew_mean"][-1]
                            if log_records["ep_rew_mean"] else None),
        iterations=iters,
    )
    with open(os.path.join(BENCH_DIR, f"metrics_sb3_seed{seed}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    venv.close()
    print(f"  wall-clock: {elapsed:.1f}s   fps: {fps:.0f}   "
          f"final ep_rew_mean: {metrics['final_ep_rew_mean']}")
    return metrics


def main():
    print(f"# SB3 PPO baseline (for NNX A/B)")
    print(f"# device      : {DEVICE}")
    print(f"# seeds       : {N_SEEDS}")
    print(f"# parallel env: {N_ENVS}")
    print(f"# total steps : {TOTAL_STEPS}")
    print(f"# n_steps     : {N_STEPS}")
    print(f"# batch_size  : {BATCH_SIZE}")
    print(f"# n_epochs    : {N_EPOCHS}")

    all_metrics = []
    for s in range(N_SEEDS):
        all_metrics.append(train_one_seed(s))

    summary = dict(
        backend=f"sb3-torch-{DEVICE}",
        n_seeds=N_SEEDS,
        per_seed=all_metrics,
        wall_clock_mean_s=float(np.mean(
            [m["wall_clock_s"] for m in all_metrics])),
        fps_mean=float(np.mean([m["fps"] for m in all_metrics])),
    )
    with open(os.path.join(BENCH_DIR, "summary_sb3.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote per-seed metrics + summary to {BENCH_DIR}")


if __name__ == "__main__":
    main()
