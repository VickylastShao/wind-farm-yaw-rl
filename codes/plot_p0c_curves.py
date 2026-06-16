# -*- coding: utf-8 -*-
"""
Paper-evidence learning curve for the P0-c convergence run (3 seeds x 3e7
steps on the NNX-JAX-env-GPU stack).

Reads per-seed metrics_seed{0,1,2}_p0c.json (written by
train_3x3_nnx_jaxenv.py with OUT_TAG=p0c) and plots ep_rew_mean vs
total_env_steps with a shaded inter-seed std band.

Output:
  latex_draft/figures/fig_3x3_training_curve_p0c.{pdf,jpg}
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR),
                       "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_SEEDS = int(os.environ.get("N_SEEDS", 5))
SMOOTH_WIN = 5   # rolling-mean window for plotting (in iterations)


def load_seed(seed: int):
    path = os.path.join(CKPT_DIR, f"metrics_seed{seed}_p0c.json")
    with open(path) as f:
        d = json.load(f)
    iters = d["iterations"]
    steps = np.array([it["total_env_steps"] for it in iters], dtype=np.float64)
    ep_rew = np.array([it["ep_rew_mean"] for it in iters], dtype=np.float64)
    kl = np.array([it["approx_kl"] for it in iters], dtype=np.float64)
    clip = np.array([it["clip_frac"] for it in iters], dtype=np.float64)
    return dict(steps=steps, ep_rew=ep_rew, kl=kl, clip=clip,
                wall_clock_s=d["wall_clock_s"],
                fps=d["fps"],
                final_ep_rew_mean=d.get("final_ep_rew_mean"))


def smooth(x, w):
    if w <= 1 or len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def main():
    seeds = [load_seed(s) for s in range(N_SEEDS)]

    # All seeds use identical iteration schedule, so steps align across seeds.
    steps = seeds[0]["steps"]
    rew_stack = np.stack([s["ep_rew"] for s in seeds], axis=0)
    rew_mean = rew_stack.mean(axis=0)
    rew_std = rew_stack.std(axis=0)
    rew_mean_s = smooth(rew_mean, SMOOTH_WIN)
    rew_std_s = smooth(rew_std, SMOOTH_WIN)

    kl_mean = np.stack([s["kl"] for s in seeds], axis=0).mean(axis=0)
    clip_mean = np.stack([s["clip"] for s in seeds], axis=0).mean(axis=0)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.0))

    # ep_rew_mean plot.
    for i, s in enumerate(seeds):
        axA.plot(s["steps"] / 1e6, smooth(s["ep_rew"], SMOOTH_WIN),
                 lw=0.8, alpha=0.45, color=f"C{i}",
                 label=f"seed {i}  final={s['final_ep_rew_mean']:+.2f}")
    axA.plot(steps / 1e6, rew_mean_s, lw=1.8, color="#000",
             label="5-seed mean")
    axA.fill_between(steps / 1e6, rew_mean_s - rew_std_s, rew_mean_s + rew_std_s,
                     alpha=0.15, color="#000")
    axA.set_xlabel("Environment steps  (millions)")
    axA.set_ylabel("Episode mean reward  (per-turbine avg power gain x10)")
    axA.set_title("3x3 PPO learning curve, NNX + windfarm_env_jax\n"
                  f"(5 seeds x 3e7 steps, 256 envs, GPU)",
                  fontsize=10)
    axA.grid(alpha=0.3)
    axA.legend(frameon=False, fontsize=8, loc="best")
    axA.axhline(0, color="black", lw=0.4)

    # KL + clip-frac diagnostics on the same axis (twin-y).
    axB.plot(steps / 1e6, smooth(kl_mean, SMOOTH_WIN),
             color="#4C78A8", lw=1.4, label="approx KL  (left)")
    axB.set_xlabel("Environment steps  (millions)")
    axB.set_ylabel("Approx KL  (target band ~ 0.01-0.05)")
    axB.tick_params(axis="y", labelcolor="#4C78A8")
    axB.grid(alpha=0.3)

    axB2 = axB.twinx()
    axB2.plot(steps / 1e6, smooth(clip_mean, SMOOTH_WIN),
              color="#E45756", lw=1.4, label="clip fraction  (right)")
    axB2.set_ylabel("Clip fraction  (PPO clipping rate)")
    axB2.tick_params(axis="y", labelcolor="#E45756")

    lines, labels = axB.get_legend_handles_labels()
    lines2, labels2 = axB2.get_legend_handles_labels()
    axB.legend(lines + lines2, labels + labels2,
               frameon=False, fontsize=8, loc="upper right")
    axB.set_title("PPO update diagnostics (5-seed mean)\n"
                  "stable KL + small clip-frac = healthy learning",
                  fontsize=10)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_3x3_training_curve_p0c")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".jpg", dpi=180, bbox_inches="tight")
    print(f"  -> {out}.pdf  {out}.jpg")

    wallclocks = [s["wall_clock_s"] for s in seeds]
    fpss = [s["fps"] for s in seeds]
    finals = [s["final_ep_rew_mean"] for s in seeds]
    summary = dict(
        n_seeds=N_SEEDS,
        total_steps_per_seed=30_000_000,
        n_envs=256,
        wall_clock_mean_s=float(np.mean(wallclocks)),
        wall_clock_total_s=float(np.sum(wallclocks)),
        fps_mean=float(np.mean(fpss)),
        per_seed_final_ep_rew=[float(x) for x in finals],
        per_seed_final_ep_rew_mean=float(np.mean(finals)),
        per_seed_final_ep_rew_std=float(np.std(finals)),
    )
    out_json = os.path.join(FIG_DIR, "3x3_training_stats_p0c.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {out_json}")

    print(f"\n=== P0-c summary ===")
    print(f"  wall-clock per seed: {np.mean(wallclocks):.1f}s "
          f"(total: {np.sum(wallclocks):.1f}s = {np.sum(wallclocks)/60:.1f} min)")
    print(f"  fps mean           : {np.mean(fpss):.0f}")
    print(f"  final ep_rew_mean  : "
          f"{np.mean(finals):+.3f} +/- {np.std(finals):.3f}")


if __name__ == "__main__":
    main()
