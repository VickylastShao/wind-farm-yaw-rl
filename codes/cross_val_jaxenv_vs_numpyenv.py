# -*- coding: utf-8 -*-
"""
Cross-validate the JAX-env-trained NNX policy by deterministically rolling it
out on the SAME numpy env (windfarm_env.WindFarmYawEnv) and checking that the
farm-power gain over the all-zero-yaw baseline is physically sensible.

Why no head-to-head with the old sbx checkpoint?
  * The sbx-rl wheel pins jax<0.7 / flax<0.12, which is incompatible with the
    nnx-0.12 stack we run NNX training on. Loading the sbx checkpoint would
    require either downgrading jax (breaks the new training stack) or writing
    a hand-rolled state-dict loader (more variables, not fewer).
  * Physics cross-check already proved windfarm_env_jax.py and windfarm_env.py
    are numerically identical (max inflow err 1.08e-6 m/s, max power err 0.36 W,
    downstream-mask 21/21 agreement). So if the JAX-env-trained policy
    produces a sane positive gain when *evaluated* in the numpy env, the env
    port has not introduced a systematic offset in the learning signal.

The verdict gate:
  * The trained policy must beat the zero-yaw baseline by a positive mean
    farm-power gain across a 19-point (phi, v) eval grid.
  * Negative or near-zero mean gain would indicate the policy is broken (the
    env port shifted the reward distribution, or the 5M-step budget was too
    small to learn anything useful on the new stack).
  * Any positive mean gain green-lights P0-c (3 seeds x 3e7 steps), where the
    larger budget will deliver paper-grade policy quality.

Output:
  latex_draft/figures/fig_xval_jaxenv_policy_gain.{pdf,jpg}
  latex_draft/figures/xval_jaxenv_policy_gain.json
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


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NNX_CKPT = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv",
                        "policy_seed0_xval.pkl")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR),
                       "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SETTLE_STEPS = 150


def build_eval_grid():
    """19-point (phi, v) grid: 8 wind dirs at rated speed + 3 speeds on-axis
    + 8 off-axis condition points covering the trained wind window."""
    grid = [(float(phi), 11.4) for phi in np.linspace(200, 340, 8)]
    grid += [(270.0, v) for v in (8.0, 11.4, 14.0)]
    grid += [(float(phi), 10.0) for phi in (220.0, 260.0, 290.0, 320.0)]
    grid += [(float(phi), 12.0) for phi in (230.0, 280.0, 310.0, 340.0)]
    return grid


def load_nnx_policy(path: str, obs_dim: int, act_dim: int) -> ActorCritic:
    """Restore an ActorCritic from the state pickle saved by
    train_3x3_nnx_jaxenv.py."""
    model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(0))
    graphdef, _ = nnx.split(model)
    with open(path, "rb") as f:
        state = pickle.load(f)
    return nnx.merge(graphdef, state)


def make_nnx_predict_fn(model: ActorCritic):
    """Return obs[np.ndarray] -> action[np.ndarray] using the mean (deterministic)
    action of the NNX policy."""

    @nnx.jit
    def _mean(m, x):
        mean, _, _ = m(x)
        return mean

    def predict(obs_np):
        obs_j = jnp.asarray(obs_np, dtype=jnp.float32)
        mean = _mean(model, obs_j)
        return np.asarray(mean)

    return predict


def rollout_policy(predict_fn, env, phi, v, settle):
    """Roll out `predict_fn` deterministically on `env` starting from (phi, v).
    Returns (gain_pct, final_total_mw, baseline_mw, max_yaw, mean_abs_yaw)."""
    obs, _ = env.reset(options=dict(specific_wind_dir=phi,
                                    specific_wind_speed=v))
    for _ in range(settle):
        action = predict_fn(obs)
        action = np.asarray(action).reshape(-1)
        obs, _, _, _, _ = env.step(action)
    gain = (env.current_total_mw - env.baseline_mw) / env.baseline_mw * 100.0
    return (float(gain), float(env.current_total_mw),
            float(env.baseline_mw), float(np.max(np.abs(env.current_gammas))),
            float(np.mean(np.abs(env.current_gammas))))


def rollout_zero_yaw(env, phi, v, settle):
    """Sanity-check baseline: zero-action policy. Should produce gain ~ 0%."""
    obs, _ = env.reset(options=dict(specific_wind_dir=phi,
                                    specific_wind_speed=v))
    n_act = env.action_space.shape[0]
    zero = np.zeros(n_act, dtype=np.float32)
    for _ in range(settle):
        obs, _, _, _, _ = env.step(zero)
    gain = (env.current_total_mw - env.baseline_mw) / env.baseline_mw * 100.0
    return float(gain)


def evaluate():
    positions, R, C = create_wind_farm_layout_3x3()
    env = WindFarmYawEnv(positions, R, C, j=1, randomize_wind=False,
                         max_steps=SETTLE_STEPS + 10)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    model = load_nnx_policy(NNX_CKPT, obs_dim, act_dim)
    predict_fn = make_nnx_predict_fn(model)

    grid = build_eval_grid()
    rows = []
    for phi, v in grid:
        g_pol, mw_pol, mw_base, max_yaw, mean_yaw = rollout_policy(
            predict_fn, env, phi, v, SETTLE_STEPS)
        g_zero = rollout_zero_yaw(env, phi, v, SETTLE_STEPS)
        rows.append(dict(phi=phi, v=v,
                         policy_gain_pct=g_pol,
                         zero_gain_pct=g_zero,
                         policy_total_mw=mw_pol,
                         baseline_mw=mw_base,
                         max_abs_yaw_deg=max_yaw,
                         mean_abs_yaw_deg=mean_yaw))
        print(f"  phi={phi:5.1f} v={v:4.1f}  "
              f"policy gain {g_pol:+7.3f}%  zero gain {g_zero:+7.3f}%  "
              f"max|yaw|={max_yaw:5.1f}deg")
    return rows


def make_fig(rows, out_path):
    n = len(rows)
    x = np.arange(n)
    w = 0.4
    labels = [f"({r['phi']:.0f},{r['v']:.0f})" for r in rows]
    policy = [r["policy_gain_pct"] for r in rows]
    zero = [r["zero_gain_pct"] for r in rows]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 4.0))

    axA.bar(x - w / 2, zero, w, color="#888",
            label=f"Zero-yaw baseline  mean={np.mean(zero):+.3f}%",
            edgecolor="black", linewidth=0.4)
    axA.bar(x + w / 2, policy, w, color="#E45756",
            label=f"NNX jax-env (5M)   mean={np.mean(policy):+.3f}%",
            edgecolor="black", linewidth=0.4)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    axA.set_ylabel("Farm-power gain over baseline [%]")
    axA.set_title("Per-condition deterministic eval  (settle = 150 steps)",
                  fontsize=10)
    axA.axhline(0, color="black", lw=0.5)
    axA.grid(alpha=0.3, axis="y")
    axA.legend(frameon=False, fontsize=8, loc="best")

    max_yaw = [r["max_abs_yaw_deg"] for r in rows]
    mean_yaw = [r["mean_abs_yaw_deg"] for r in rows]
    axB.bar(x - w / 2, mean_yaw, w, color="#4C78A8",
            label=f"mean |yaw|  ({np.mean(mean_yaw):.1f} deg avg)",
            edgecolor="black", linewidth=0.4)
    axB.bar(x + w / 2, max_yaw, w, color="#F58518",
            label=f"max  |yaw|  ({np.mean(max_yaw):.1f} deg avg)",
            edgecolor="black", linewidth=0.4)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    axB.set_ylabel("Yaw magnitude [deg]")
    axB.set_title("Learned controller yaw envelope per condition\n"
                  "(non-trivial yaw => policy is doing something)",
                  fontsize=10)
    axB.grid(alpha=0.3, axis="y")
    axB.axhline(50, color="red", lw=0.5, ls="--",
                label="env yaw limit (+/-50 deg)")
    axB.legend(frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    fig.savefig(out_path + ".jpg", dpi=180, bbox_inches="tight")
    print(f"  -> {out_path}.pdf  {out_path}.jpg")


def main():
    assert os.path.exists(NNX_CKPT), f"missing policy: {NNX_CKPT}"

    print("# Cross-validation: JAX-env-trained NNX policy, evaluated in numpy env")
    print(f"# settle steps    : {SETTLE_STEPS}")
    print(f"# eval grid size  : {len(build_eval_grid())}")
    print(f"# policy          : {os.path.relpath(NNX_CKPT)}\n")

    rows = evaluate()

    policy_gains = np.array([r["policy_gain_pct"] for r in rows])
    zero_gains = np.array([r["zero_gain_pct"] for r in rows])
    advantage = policy_gains - zero_gains  # extra %-points over doing nothing

    print(f"\n=== summary ===")
    print(f"  policy gain   mean = {policy_gains.mean():+.3f}%  "
          f"std = {policy_gains.std():.3f}%  "
          f"min/max = {policy_gains.min():+.2f}/{policy_gains.max():+.2f}%")
    print(f"  zero-yaw gain mean = {zero_gains.mean():+.3f}%  "
          f"std = {zero_gains.std():.3f}%")
    print(f"  advantage     mean = {advantage.mean():+.3f}pp  "
          f"(>0 means controller beats doing nothing)")
    n_positive = int((policy_gains > 0).sum())
    n_beats_zero = int((policy_gains > zero_gains).sum())
    print(f"  positive on {n_positive}/{len(rows)} conditions")
    print(f"  beats zero-yaw on {n_beats_zero}/{len(rows)} conditions")

    # Verdict: jax-env-trained policy must learn SOMETHING positive on average.
    # 5M steps is small; we expect a modest but non-zero gain. The full P0-c
    # 3 seeds x 3e7 will deliver paper-grade numbers.
    pass_mean = float(policy_gains.mean()) > 0.0
    pass_advantage = float(advantage.mean()) > 0.0
    pass_consistency = n_beats_zero >= len(rows) // 2
    verdict = "PASS" if (pass_mean and pass_advantage and pass_consistency) else "REVIEW"
    print(f"\n=== verdict: {verdict} ===")
    if verdict == "PASS":
        print("  jax-env policy yields physically-sensible positive gain")
        print("  -> GREEN-LIGHT P0-c (3 seeds x 3e7 steps, jax-env stack)")
    else:
        print("  policy did not consistently beat the zero-yaw baseline.")
        print("  investigation candidates:")
        print("    * 5M steps too short for new stack to find local optimum")
        print("    * obs scale shift (NNX trained on raw obs, vs SB3 used VecNormalize)")
        print("    * reward shaping or downstream-mask diff between envs")

    make_fig(rows, os.path.join(FIG_DIR, "fig_xval_jaxenv_policy_gain"))

    out = dict(
        candidate_label="NNX jax-env (5M steps, N_ENVS=256)",
        candidate_ckpt=os.path.relpath(NNX_CKPT),
        settle_steps=SETTLE_STEPS,
        eval_grid=[[r["phi"], r["v"]] for r in rows],
        per_condition=rows,
        policy_summary=dict(mean_gain_pct=float(policy_gains.mean()),
                            std_gain_pct=float(policy_gains.std()),
                            min_gain_pct=float(policy_gains.min()),
                            max_gain_pct=float(policy_gains.max())),
        zero_yaw_summary=dict(mean_gain_pct=float(zero_gains.mean()),
                              std_gain_pct=float(zero_gains.std())),
        advantage_summary=dict(mean_pp=float(advantage.mean()),
                               n_positive_of=int(n_positive),
                               n_beats_zero_of=int(n_beats_zero),
                               n_total=len(rows)),
        verdict=verdict,
    )
    out_path = os.path.join(FIG_DIR, "xval_jaxenv_policy_gain.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
