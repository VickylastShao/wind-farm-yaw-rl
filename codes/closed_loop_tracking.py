# -*- coding: utf-8 -*-
"""
Closed-loop tracking experiment under time-varying inflow.

Tests whether the trained PPO controller can track shifting wind direction
within a single episode -- the property that justifies the "closed-loop
real-time controller" claim in the paper. Three protocols are evaluated on
the 3x3 grid:

  (A) Step changes:   phi: 270 -> 280 -> 260 -> 275, each segment 80 steps.
  (B) Slow drift:     phi: 260 -> 290 linearly over 240 steps (0.125 deg/step).
  (C) Fast drift:     phi: 260 -> 290 linearly over  60 steps (0.5  deg/step).

For each protocol we record total farm power, per-step yaw vector, and the
no-yaw baseline power under the *current* phi, then compute tracking
error and settling time after each step change.

Outputs:
  figures/fig_tracking_step.{pdf,jpg}
  figures/fig_tracking_drift.{pdf,jpg}
  figures/tracking_stats.json

Requires:
  - trained PPO model checkpoint; configurable via PPO_MODEL_PATH below
    (a freshly-instantiated random policy is used as a fallback so the
    script runs even without a checkpoint, but tracking will be poor).
"""

import os
import json
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sbx import PPO as _SBX_PPO
except ImportError:
    _SBX_PPO = None
from stable_baselines3 import PPO as _SB3_PPO

from windfarm_env import (
    WindFarmYawEnv,
    calculate_inflow_speeds,
    power_output,
    create_wind_farm_layout_3x3,
    C_T, I, d_0, alpha_star, beta_star, alpha,
)


OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "latex_draft", "figures",
)
os.makedirs(OUT_DIR, exist_ok=True)

PPO_MODEL_PATH = os.environ.get("PPO_MODEL_PATH", "")
VECNORM_PATH = os.environ.get("VECNORM_PATH", "")
BACKEND = os.environ.get("PPO_BACKEND", "auto").lower()  # sbx | sb3 | auto
N_SEEDS = 3
WIND_SPEED = 11.4


def _resolve_ppo():
    """Return the PPO class to use for load/init based on BACKEND."""
    if BACKEND == "sbx":
        if _SBX_PPO is None:
            raise RuntimeError("PPO_BACKEND=sbx requested but sbx is not installed.")
        return _SBX_PPO
    if BACKEND == "sb3":
        return _SB3_PPO
    # auto: try SBX first (newer trained models), fall back to SB3
    return _SBX_PPO if _SBX_PPO is not None else _SB3_PPO


def load_or_init_policy(env):
    venv = DummyVecEnv([lambda: env])
    vn = None
    if PPO_MODEL_PATH and os.path.exists(PPO_MODEL_PATH):
        if VECNORM_PATH and os.path.exists(VECNORM_PATH):
            print(f"[policy] loading VecNormalize {VECNORM_PATH}")
            vn = VecNormalize.load(VECNORM_PATH, venv)
            vn.training = False
            vn.norm_reward = False
        target_env = vn if vn is not None else venv

        last_err = None
        for cls_name, cls in (("sbx", _SBX_PPO), ("sb3", _SB3_PPO)):
            if cls is None:
                continue
            if BACKEND in ("sbx", "sb3") and BACKEND != cls_name:
                continue
            try:
                print(f"[policy] loading {PPO_MODEL_PATH} via {cls_name} PPO")
                model = cls.load(PPO_MODEL_PATH, env=target_env)
                return model, vn
            except Exception as exc:
                last_err = exc
                print(f"[policy] {cls_name} load failed: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"all PPO backends failed; last error: {last_err}")

    # Refusing a silent fallback to a randomly-initialized policy: such a
    # run would still produce fig_tracking_*.{pdf,jpg} and overwrite the
    # checked-in evidence chain. Caller must set PPO_MODEL_PATH to a real
    # checkpoint or skip this script.
    raise FileNotFoundError(
        "PPO_MODEL_PATH is not set or does not exist; refusing to fall back "
        "to a randomly-initialized policy because that would overwrite the "
        "tracking figures with meaningless data. Set PPO_MODEL_PATH to a "
        "trained checkpoint (and optionally VECNORM_PATH to its VecNormalize "
        "pickle), e.g.:\n"
        "  PPO_MODEL_PATH=checkpoints_3x3/ppo_3x3_seed0_final.zip \\\n"
        "  VECNORM_PATH=checkpoints_3x3/vecnormalize_seed0.pkl \\\n"
        "  python closed_loop_tracking.py"
    )


def baseline_power_at(positions, phi, v):
    u0 = calculate_inflow_speeds(
        positions, phi, C_T, I, d_0, v,
        np.zeros(len(positions)), alpha_star, beta_star, alpha,
    )
    return sum(power_output(ui, 0.0) for ui in u0) / 1e6


def run_protocol(model, vn, env, positions, phi_schedule, label, seed):
    obs, _ = env.reset(seed=seed,
                       options=dict(specific_wind_dir=float(phi_schedule[0]),
                                    specific_wind_speed=WIND_SPEED))
    powers_mw, yaws, baselines = [], [], []
    for t, phi in enumerate(phi_schedule):
        env.current_phi = float(phi)
        obs_in = vn.normalize_obs(obs) if vn is not None else obs
        action, _ = model.predict(obs_in, deterministic=True)
        obs, _, _, _, _ = env.step(action)
        p_mw = sum(power_output(u, g) for u, g in
                   zip(env.current_inflow_speeds, env.current_gammas)) / 1e6
        powers_mw.append(p_mw)
        yaws.append(env.current_gammas.copy())
        baselines.append(baseline_power_at(positions, float(phi), WIND_SPEED))
    return (np.asarray(powers_mw), np.asarray(yaws),
            np.asarray(baselines), np.asarray(phi_schedule))


def step_schedule():
    seg = 80
    return np.concatenate([
        np.full(seg, 270.0),
        np.full(seg, 280.0),
        np.full(seg, 260.0),
        np.full(seg, 275.0),
    ])


STEP_SEGMENTS = [
    ("phi=270", 0, 80),
    ("phi=280", 80, 160),
    ("phi=260", 160, 240),
    ("phi=275", 240, 320),
]
STEP_TRANSIENT = 30  # skip first 30 steps of each segment when computing steady gain


def drift_schedule(n_steps, phi_start=260.0, phi_end=290.0):
    return np.linspace(phi_start, phi_end, n_steps)


def settling_time(power, baseline, change_idx, tol=0.01):
    """Return the number of steps after change_idx for (power - baseline)
    to enter a tube of width tol*baseline around its post-change steady value."""
    if change_idx + 30 >= len(power):
        return None
    steady = np.mean(power[change_idx + 30: change_idx + 60])
    band = tol * steady
    for k in range(change_idx, len(power)):
        if abs(power[k] - steady) <= band and \
           np.all(np.abs(power[k:min(k + 10, len(power))] - steady) <= band):
            return k - change_idx
    return None


def plot_step_protocol(results, fname, per_segment_summary=None):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]))
    ax_p, ax_phi = axes
    for r in results:
        power, _, base, phi = r["data"]
        steps = np.arange(len(power))
        ax_p.plot(steps, power, lw=1.0, alpha=0.85,
                  label=f"seed {r['seed']}")
        if r["seed"] == results[0]["seed"]:
            ax_p.plot(steps, base, "k--", lw=1.0, label="no-yaw baseline")
            ax_phi.plot(steps, phi, "k-", lw=1.2)
    if per_segment_summary is not None:
        ymin, ymax = ax_p.get_ylim()
        for lbl, lo, hi in STEP_SEGMENTS:
            ax_p.axvline(lo, color="gray", lw=0.5, alpha=0.5)
            g = per_segment_summary[lbl]["mean"]
            color = "tab:green" if g > 0 else "tab:red"
            ax_p.text((lo + hi) / 2, ymin + 0.92 * (ymax - ymin),
                      f"{lbl}\n{g:+.2f}%", ha="center", va="top",
                      fontsize=7.5, color=color,
                      bbox=dict(facecolor="white", edgecolor="none",
                                alpha=0.75, pad=1))
    ax_p.set_ylabel("Farm power [MW]")
    ax_p.legend(frameon=False, fontsize=8, ncol=4, loc="lower right")
    ax_p.grid(alpha=0.3)
    ax_phi.set_xlabel("Control step")
    ax_phi.set_ylabel(r"$\phi$ [deg]")
    ax_phi.grid(alpha=0.3)
    fig.suptitle("Closed-loop tracking: step changes in wind direction "
                 r"($U_\infty = 11.4$ m/s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fname + ".pdf", bbox_inches="tight")
    fig.savefig(fname + ".jpg", dpi=200, bbox_inches="tight")


def plot_drift_protocol(results_slow, results_fast, fname):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=False)
    for ax, results, ttl in [
        (axes[0], results_slow, "Slow drift (0.125 deg/step)"),
        (axes[1], results_fast, "Fast drift (0.50 deg/step)"),
    ]:
        for r in results:
            power, _, base, phi = r["data"]
            ax.plot(phi, power, lw=1.0, alpha=0.85, label=f"seed {r['seed']}")
            if r["seed"] == results[0]["seed"]:
                ax.plot(phi, base, "k--", lw=1.0, label="no-yaw baseline")
        ax.set_xlabel(r"Wind direction $\phi$ [deg]")
        ax.set_ylabel("Farm power [MW]")
        ax.set_title(ttl, fontsize=10)
        ax.legend(frameon=False, fontsize=8, ncol=4)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname + ".pdf", bbox_inches="tight")
    fig.savefig(fname + ".jpg", dpi=200, bbox_inches="tight")


def main():
    positions, N_rows, N_cols = create_wind_farm_layout_3x3()
    base_env = WindFarmYawEnv(positions, N_rows, N_cols, j=1, randomize_wind=False)
    model, vn = load_or_init_policy(base_env)

    stats: dict[str, Any] = dict(model_checkpoint=PPO_MODEL_PATH or "(random init)",
                                 vecnormalize=VECNORM_PATH or "(none)")

    # --- Protocol A: step changes ---
    print("\n# Protocol A: step changes")
    sched_A = step_schedule()
    results_A = []
    settle = []
    per_segment: dict[str, list[float]] = {lbl: [] for lbl, _, _ in STEP_SEGMENTS}
    for seed in range(N_SEEDS):
        data = run_protocol(model, vn, base_env, positions, sched_A, "step", seed)
        results_A.append(dict(seed=seed, data=data))
        power, _, base, _ = data
        for idx in (80, 160, 240):
            s = settling_time(power, base, idx)
            if s is not None:
                settle.append(s)
        for lbl, lo, hi in STEP_SEGMENTS:
            ss = slice(lo + STEP_TRANSIENT, hi)
            p_ss = float(power[ss].mean())
            b_ss = float(base[ss].mean())
            per_segment[lbl].append((p_ss - b_ss) / b_ss * 100)
    stats["step_mean_gain_pct"] = float(np.mean([
        (r["data"][0] - r["data"][2]).mean() / r["data"][2].mean() * 100
        for r in results_A
    ]))
    stats["step_per_segment_pct"] = {
        lbl: dict(mean=float(np.mean(vals)),
                  std=float(np.std(vals)),
                  values=vals)
        for lbl, vals in per_segment.items()
    }
    stats["step_best_segment_gain_pct"] = float(max(
        v["mean"] for v in stats["step_per_segment_pct"].values()
    ))
    stats["step_settling_steps_mean"] = (float(np.mean(settle)) if settle else None)
    stats["step_settling_steps_std"] = (float(np.std(settle)) if settle else None)
    print(f"  mean gain over baseline (whole window): "
          f"{stats['step_mean_gain_pct']:.2f} %")
    for lbl, summary in stats["step_per_segment_pct"].items():
        print(f"    {lbl} steady gain: "
              f"{summary['mean']:+.2f} +/- {summary['std']:.2f} %")
    print(f"  best segment: {stats['step_best_segment_gain_pct']:+.2f} %")
    if settle:
        print(f"  settling time: {stats['step_settling_steps_mean']:.1f} "
              f"+/- {stats['step_settling_steps_std']:.1f} steps")
    plot_step_protocol(results_A, os.path.join(OUT_DIR, "fig_tracking_step"),
                       per_segment_summary=stats["step_per_segment_pct"])

    # --- Protocols B/C: slow and fast drift ---
    print("\n# Protocols B/C: drift")
    sched_B = drift_schedule(240)
    sched_C = drift_schedule(60)
    results_B, results_C = [], []
    for seed in range(N_SEEDS):
        results_B.append(dict(seed=seed,
                              data=run_protocol(model, vn, base_env, positions,
                                                sched_B, "slow", seed)))
        results_C.append(dict(seed=seed,
                              data=run_protocol(model, vn, base_env, positions,
                                                sched_C, "fast", seed)))
    stats["slow_mean_gain_pct"] = float(np.mean([
        (r["data"][0] - r["data"][2]).mean() / r["data"][2].mean() * 100
        for r in results_B
    ]))
    stats["fast_mean_gain_pct"] = float(np.mean([
        (r["data"][0] - r["data"][2]).mean() / r["data"][2].mean() * 100
        for r in results_C
    ]))
    print(f"  slow drift gain: {stats['slow_mean_gain_pct']:.2f} %")
    print(f"  fast drift gain: {stats['fast_mean_gain_pct']:.2f} %")
    plot_drift_protocol(results_B, results_C,
                        os.path.join(OUT_DIR, "fig_tracking_drift"))

    with open(os.path.join(OUT_DIR, "tracking_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved figures and stats to {OUT_DIR}")


if __name__ == "__main__":
    main()
