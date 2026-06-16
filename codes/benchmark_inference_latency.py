# -*- coding: utf-8 -*-
"""
Inference-latency benchmark for the PPO controller.

Reports mean / std / p50 / p95 / p99 (ms) of a single policy forward pass
for N=2 (1x2), N=9 (3x3), and N=25 (5x5) layouts on the host CPU,
and saves both raw samples and a publication-ready histogram figure.

Usage:
    cd codes
    python benchmark_inference_latency.py
"""

import os
import time
import json
import platform
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from windfarm_env import (
    WindFarmYawEnv,
    create_wind_farm_layout,
    create_wind_farm_layout_3x3,
    d_0, z_h,
)


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "latex_draft", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def create_wind_farm_layout_5x5():
    """5x5 grid with the same 7-degree tilt and 7*d_0 spacing as the other helpers."""
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


def bench(layout_fn, label, n_warmup=300, n_iter=3000):
    positions, N_rows, N_cols = layout_fn()
    env = WindFarmYawEnv(positions, N_rows, N_cols, j=1, randomize_wind=True)
    venv = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", venv, verbose=0, device="cpu",
                policy_kwargs=dict(net_arch=[128, 128]))
    obs = venv.reset()

    for _ in range(n_warmup):
        model.predict(obs, deterministic=True)

    s_us = np.empty(n_iter)
    for i in range(n_iter):
        t0 = time.perf_counter_ns()
        model.predict(obs, deterministic=True)
        s_us[i] = (time.perf_counter_ns() - t0) / 1000.0

    s_ms = s_us / 1000.0
    stats = dict(
        label=label,
        N=len(positions),
        obs_dim=int(env.observation_space.shape[0]),
        act_dim=int(env.action_space.shape[0]),
        n_iter=n_iter,
        mean_ms=float(s_ms.mean()),
        std_ms=float(s_ms.std()),
        p50_ms=float(np.percentile(s_ms, 50)),
        p95_ms=float(np.percentile(s_ms, 95)),
        p99_ms=float(np.percentile(s_ms, 99)),
        min_ms=float(s_ms.min()),
        max_ms=float(s_ms.max()),
    )
    print(f"\n[{label}] N={stats['N']}  obs={stats['obs_dim']}  act={stats['act_dim']}  "
          f"iter={n_iter}")
    for k in ("mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
        print(f"  {k:8s} = {stats[k]:.4f}")
    return s_ms, stats


def main():
    print(f"# Inference latency benchmark")
    print(f"# host    : {platform.node()}")
    print(f"# os      : {platform.platform()}")
    print(f"# python  : {platform.python_version()}")
    print(f"# torch   : {torch.__version__}")
    print(f"# device  : CPU (forced)")

    cases = [
        (create_wind_farm_layout, "1x2"),
        (create_wind_farm_layout_3x3, "3x3"),
        (create_wind_farm_layout_5x5, "5x5"),
    ]
    samples = {}
    all_stats = []
    for fn, label in cases:
        s_ms, stats = bench(fn, label)
        samples[label] = s_ms
        all_stats.append(stats)

    np.savez(os.path.join(OUT_DIR, "inference_latency_samples.npz"), **samples)
    with open(os.path.join(OUT_DIR, "inference_latency_stats.json"), "w") as f:
        json.dump(dict(host=platform.node(),
                       os=platform.platform(),
                       python=platform.python_version(),
                       torch=torch.__version__,
                       cases=all_stats), f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, (label, s_ms), stats in zip(axes, samples.items(), all_stats):
        ax.hist(s_ms, bins=60, color="#4C78A8", edgecolor="white", linewidth=0.3)
        ax.axvline(stats["p50_ms"], color="k", lw=1.0, label=f"p50 = {stats['p50_ms']:.2f} ms")
        ax.axvline(stats["p95_ms"], color="r", lw=1.0, ls="--",
                   label=f"p95 = {stats['p95_ms']:.2f} ms")
        ax.set_title(f"{label}  ($N={stats['N']}$)")
        ax.set_xlabel("Inference latency (ms)")
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("Count")
    fig.suptitle(
        f"PPO policy forward-pass latency on CPU "
        f"({platform.python_version()}, torch {torch.__version__})",
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_inference_latency.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig_inference_latency.jpg"), dpi=200, bbox_inches="tight")

    print("\n# LaTeX-ready row (drop into Table 4):")
    for s in all_stats:
        print(f"  N={s['N']:2d}  {s['mean_ms']:.3f} ± {s['std_ms']:.3f} ms  "
              f"(p95 = {s['p95_ms']:.3f} ms)")
    print(f"\nSaved:")
    print(f"  {OUT_DIR}/inference_latency_samples.npz")
    print(f"  {OUT_DIR}/inference_latency_stats.json")
    print(f"  {OUT_DIR}/fig_inference_latency.pdf  (+ .jpg)")


if __name__ == "__main__":
    main()
