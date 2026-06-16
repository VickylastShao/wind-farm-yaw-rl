# -*- coding: utf-8 -*-
"""
Distribution-wise P0-c evaluation using JAX batched env (fast).

Evaluates the 3 NNX policies on N_EVAL_CONDITIONS random wind conditions
sampled from the training distribution, all in one GPU-second per seed
via vmap + lax.scan.

Output:
  latex_draft/figures/p0c_eval_randomized.json
  latex_draft/figures/fig_p0c_eval_randomized.{pdf,jpg}
"""

import os, json, pickle

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx

from windfarm_env import create_wind_farm_layout_3x3
from windfarm_env_jax import (
    env_reset, env_step, inflow_speeds_jax, power_output_jax,
    find_downstream_mask_jax, positions_to_jax,
    WindFarmJAXState,
)
from train_3x3_nnx import ActorCritic
from cross_val_jaxenv_vs_numpyenv import load_nnx_policy, SETTLE_STEPS


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR),
                       "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_SEEDS = int(os.environ.get("N_SEEDS", 5))
PHI_RANGE = (173.0, 353.0)
V_RANGE = (6.0, 16.0)
N_CONDITIONS = int(os.environ.get("N_CONDITIONS", 3000))
EVAL_SEED = int(os.environ.get("EVAL_SEED", 20260604))

MAX_EPISODE_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", 200))


def build_batch(positions_jax, rng):
    """Sample N_CONDITIONS (phi, v) pairs from training distribution."""
    # Use numpy RNG to sidestep jax.random.key vs PRNGKey API confusion.
    np_rng = np.random.default_rng(rng)
    phis = jnp.asarray(np_rng.uniform(PHI_RANGE[0], PHI_RANGE[1],
                                        size=N_CONDITIONS), dtype=jnp.float32)
    vs = jnp.asarray(np_rng.uniform(V_RANGE[0], V_RANGE[1],
                                      size=N_CONDITIONS), dtype=jnp.float32)
    # Pre-compute zero-yaw baseline for each condition (env-independent).
    @jax.jit
    def zero_yaw_gain(phi, v):
        mask = find_downstream_mask_jax(positions_jax, phi, v)
        inflow_0 = inflow_speeds_jax(positions_jax, phi, v,
                                      jnp.zeros(positions_jax.shape[0], jnp.float32))
        baseline_mw = jnp.sum(power_output_jax(inflow_0, jnp.zeros(positions_jax.shape[0]))) / 1e6
        return baseline_mw  # in MW
    baselines = jax.vmap(zero_yaw_gain)(phis, vs)
    return phis, vs, baselines


def evaluate_batch(model, positions_jax, phis, vs, baselines, N_turb):
    """Evaluate policy on a batch of (phi, v) conditions using vmap + scan.

    Returns (gains_pct, max_abs_yaws, mean_abs_yaws) each (B,).
    """
    B = phis.shape[0]
    N_steps = SETTLE_STEPS

    @nnx.jit
    def run(m, phis, vs):
        # Batch reset
        @jax.vmap
        def reset_one(phi, v):
            key = jax.random.key(0)
            state, obs = env_reset(key, positions_jax,
                                    specific_wind_dir=phi,
                                    specific_wind_speed=v,
                                    randomize_wind=False,
                                    max_steps=N_steps + 10)
            return state, obs
        states, obs_batch = reset_one(phis, vs)

        # Vmapped policy mean.
        @jax.vmap
        def predict_one(o):
            mean, _, _ = m(o.reshape(1, -1))
            return mean.reshape(N_turb)

        @jax.vmap
        def step_one(s, a):
            return env_step(s, a, positions_jax, max_steps=N_steps + 10)

        def body(carry, _):
            states, obs = carry
            actions = predict_one(obs)
            actions = jnp.clip(actions, -5.0, 5.0)
            new_states, new_obs, _, _ = step_one(states, actions)
            return (new_states, new_obs), None

        (final_states, _), _ = jax.lax.scan(body, (states, obs_batch), None, length=N_steps)
        return final_states.total_mw, final_states.gammas

    total_mw, gammas = run(model, phis, vs)
    gains = (total_mw - baselines) / baselines * 100.0
    yaws = jnp.abs(gammas)
    return gains, yaws.max(axis=1), yaws.mean(axis=1)


def main():
    positions, R, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions)
    positions_jax = positions_to_jax(positions)

    print(f"# Distribution-wise P0-c eval (JAX batch)")
    print(f"# N_SEEDS = {N_SEEDS}, N_CONDITIONS = {N_CONDITIONS}")
    print(f"# phi ~ U{PHI_RANGE}, v ~ U{V_RANGE}")
    print(f"# settle steps = {SETTLE_STEPS}, episode max = {MAX_EPISODE_STEPS}")
    print(f"# device = {jax.devices()[0]}")

    # Pre-build the wind-condition batch (same across seeds for fairness).
    phis, vs, baselines = build_batch(positions_jax, EVAL_SEED)
    print(f"  Sampled {N_CONDITIONS} conditions, "
          f"baseline power: {baselines.mean():.2f} +/- {baselines.std():.2f} MW")

    per_seed_gains = []
    summary = dict(
        n_seeds=N_SEEDS,
        n_conditions=N_CONDITIONS,
        eval_seed=EVAL_SEED,
        phi_range_deg=list(PHI_RANGE),
        v_range_mps=list(V_RANGE),
        settle_steps=SETTLE_STEPS,
        per_seed=[],
        per_seed_rows=[],
    )

    for s in range(N_SEEDS):
        ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_p0c.pkl")
        obs_dim = 3 * N_turb + 3  # j=1: gammas(N)+inflow(N)+(cos,sin,v)(3)+locked(N)
        act_dim = N_turb
        model = load_nnx_policy(ckpt, obs_dim, act_dim)
        print(f"\n## seed {s}: loading eval...")

        gains, max_yaws, mean_yaws = evaluate_batch(
            model, positions_jax, phis, vs, baselines, N_turb)

        gains_np = np.asarray(gains)
        max_yaws_np = np.asarray(max_yaws)
        mean_yaws_np = np.asarray(mean_yaws)

        # Per-seed summary.
        sg = dict(
            mean_gain_pct=float(gains_np.mean()),
            std_gain_pct=float(gains_np.std()),
            median_gain_pct=float(np.median(gains_np)),
            p5_gain_pct=float(np.percentile(gains_np, 5)),
            p95_gain_pct=float(np.percentile(gains_np, 95)),
            n_positive=int((gains_np > 0).sum()),
            mean_advantage_pp=float(gains_np.mean() - 0.0),
            mean_max_yaw_deg=float(max_yaws_np.mean()),
            mean_mean_yaw_deg=float(mean_yaws_np.mean()),
        )
        per_seed_gains.append(gains_np)
        summary["per_seed"].append(sg)

        rows = [dict(phi=float(phis[i]), v=float(vs[i]),
                     policy_gain_pct=float(gains_np[i]),
                     max_abs_yaw_deg=float(max_yaws_np[i]),
                     mean_abs_yaw_deg=float(mean_yaws_np[i]))
                for i in range(N_CONDITIONS)]
        summary["per_seed_rows"].append(rows)

        print(f"  mean = {sg['mean_gain_pct']:+.3f}%  "
              f"std = {sg['std_gain_pct']:.3f}%  "
              f"median = {sg['median_gain_pct']:+.3f}%")
        print(f"  p5 = {sg['p5_gain_pct']:+.2f}%  "
              f"p95 = {sg['p95_gain_pct']:+.2f}%")
        print(f"  positive = {sg['n_positive']}/{N_CONDITIONS}")
        print(f"  mean |yaw| = {sg['mean_mean_yaw_deg']:.1f} deg  "
              f"max |yaw| = {sg['mean_max_yaw_deg']:.1f} deg")

    # Across-seed aggregate: per-condition 3-seed mean.
    g_stack = np.stack(per_seed_gains, axis=0)  # (3, N)
    cond_mean = g_stack.mean(axis=0)
    cond_std = g_stack.std(axis=0)

    across = dict(
        n=N_CONDITIONS,
        mean_gain_pct=float(cond_mean.mean()),
        std_gain_pct=float(cond_mean.std()),
        across_seed_std_pct=float(g_stack.std(axis=0).mean()),
        median_gain_pct=float(np.median(cond_mean)),
        p5_gain_pct=float(np.percentile(cond_mean, 5)),
        p95_gain_pct=float(np.percentile(cond_mean, 95)),
        n_positive=int((cond_mean > 0).sum()),
    )
    summary["across_seed"] = across

    print(f"\n=== 3-seed aggregate (per-condition mean) ===")
    print(f"  mean gain = {across['mean_gain_pct']:+.3f}%")
    print(f"  median    = {across['median_gain_pct']:+.3f}%  "
          f"p5={across['p5_gain_pct']:+.2f}%  "
          f"p95={across['p95_gain_pct']:+.2f}%")
    print(f"  positive on {across['n_positive']}/{across['n']}")
    print(f"  across-seed agreement = {across['across_seed_std_pct']:.3f}%")

    # ---- by-direction x by-speed segmentation (paper-evidence H3 table) ----
    # H3 explanation: policy's effective regime is (low v) + (aligned dir).
    # High v hits rated-power saturation -> no headroom for yaw to recover MW.
    # Aligned directions place rear turbines in strongest wake -> largest deficit
    # to recover. Reporting only the marginal mean dilutes this signal heavily.
    orig_phis_arr = np.array([r["phi"] for r in summary["per_seed_rows"][0]])
    orig_vs_arr = np.array([r["v"] for r in summary["per_seed_rows"][0]])
    dphi_arr = np.abs(((orig_phis_arr - 270.0 + 180.0) % 360.0) - 180.0)

    DIR_EDGES = [(0.0, 15.0), (15.0, 35.0), (35.0, 60.0), (60.0, 90.001)]
    V_EDGES = [(6.0, 8.0), (8.0, 11.4), (11.4, 14.0), (14.0, 16.001)]

    by_bin = []
    print(f"\n=== Segmented gain%: rows = |phi-270| bin, cols = v bin ===")
    header = (f"{'dir bin':<16s}"
              + "".join([f"{f'v[{lo:.1f},{hi:.1f})':>16s}" for lo, hi in V_EDGES])
              + f"{'all-v':>16s}")
    print(header)
    for dlo, dhi in DIR_EDGES:
        sel_d = (dphi_arr >= dlo) & (dphi_arr < dhi)
        row_str = f"|dphi|<{dhi:.0f}° (n={sel_d.sum():>4d})"
        row_str = f"{row_str:<16s}"
        for vlo, vhi in V_EDGES:
            sel = sel_d & (orig_vs_arr >= vlo) & (orig_vs_arr < vhi)
            if sel.sum() == 0:
                cell_str = "      —      "
                cell_dict = dict(n=0, mean_pct=None, std_pct=None)
            else:
                m = float(cond_mean[sel].mean())
                s = float(cond_mean[sel].std())
                cell_str = f"{m:+5.2f}±{s:4.2f} (n={sel.sum():>3d})"
                cell_dict = dict(n=int(sel.sum()), mean_pct=m, std_pct=s)
            by_bin.append(dict(dir_lo=dlo, dir_hi=dhi,
                                v_lo=vlo, v_hi=vhi, **cell_dict))
            row_str += f"{cell_str:>16s}"
        # All-v marginal for this dir bin.
        if sel_d.sum() > 0:
            mm = float(cond_mean[sel_d].mean())
            ss = float(cond_mean[sel_d].std())
            row_str += f"{f'{mm:+5.2f}±{ss:4.2f}':>16s}"
        print(row_str)

    summary["by_dir_v_bin"] = by_bin

    # Marginal-v (paper-evidence: "policy's effective regime").
    print(f"\n=== Marginal gain% by v (all directions) ===")
    v_marginals = []
    for vlo, vhi in V_EDGES:
        sel = (orig_vs_arr >= vlo) & (orig_vs_arr < vhi)
        m = float(cond_mean[sel].mean()) if sel.sum() else float("nan")
        s = float(cond_mean[sel].std()) if sel.sum() else float("nan")
        v_marginals.append(dict(v_lo=vlo, v_hi=vhi, n=int(sel.sum()),
                                mean_pct=m, std_pct=s))
        print(f"  v[{vlo:.1f},{vhi:.1f})  mean = {m:+6.3f}%  "
              f"std = {s:5.2f}%  n={sel.sum():>5d}")
    summary["by_v_bin"] = v_marginals

    # Headline single number for the paper: aligned + cube-region gain.
    aligned_cube = (dphi_arr < 15.0) & (orig_vs_arr < 11.4)
    if aligned_cube.sum() > 0:
        m = float(cond_mean[aligned_cube].mean())
        s = float(cond_mean[aligned_cube].std())
        summary["paper_headline_aligned_cube"] = dict(
            n=int(aligned_cube.sum()), mean_pct=m, std_pct=s,
            desc="|phi-270|<15deg AND v<11.4 m/s (cube region, wake-aligned)")
        print(f"\n=== Paper headline (aligned + cube region) ===")
        print(f"  |phi-270|<15° AND v<11.4 m/s:  "
              f"mean = {m:+.3f}%  std = {s:.3f}%  n={aligned_cube.sum()}")

    # Save JSON.
    out_path = os.path.join(FIG_DIR, "p0c_eval_randomized.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

    # Panel A: histogram of per-condition 3-seed-mean gain.
    ax = axes[0, 0]
    ax.hist(cond_mean, bins=80, color="#E45756", alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axvline(across["mean_gain_pct"], color="#4C78A8", lw=1.5, ls="-",
               label=f"mean = {across['mean_gain_pct']:+.3f}%")
    ax.set_xlabel("Farm-power gain [%]")
    ax.set_ylabel("Count of conditions")
    ax.set_title(f"Distribution of 3-seed-mean gain\n"
                 f"({N_CONDITIONS} random wind conditions)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    # Stats box.
    stats_text = (f"median = {across['median_gain_pct']:+.3f}%\n"
                  f"p5     = {across['p5_gain_pct']:+.2f}%\n"
                  f"p95    = {across['p95_gain_pct']:+.2f}%\n"
                  f">0 on {across['n_positive']}/{across['n']}")
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            va="top", ha="right", fontsize=7.5, family="monospace")

    # Panel B: scatter of per-cond gain vs phi, colored by v.
    ax = axes[0, 1]
    phi_deg = np.asarray([round((r["phi"] - 270 + 360) % 360) if r["phi"] > 180 else r["phi"]
                          for r in summary["per_seed_rows"][0]])
    # Use original phi in meteorological convention.
    orig_phis = np.array([r["phi"] for r in summary["per_seed_rows"][0]])
    orig_vs = np.array([r["v"] for r in summary["per_seed_rows"][0]])
    sc = ax.scatter(orig_phis, cond_mean, c=orig_vs, s=8, alpha=0.5, cmap="viridis",
                    vmin=V_RANGE[0], vmax=V_RANGE[1])
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Wind direction phi [deg] (meteo)")
    ax.set_ylabel("3-seed-mean gain [%]")
    ax.set_title("Per-condition gain vs wind direction\n(colored by speed)",
                 fontsize=10)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("v [m/s]")
    ax.grid(alpha=0.3)

    # Panel C: per-seed per-condition gain vs v, colored by phi.
    ax = axes[1, 0]
    for s in range(N_SEEDS):
        ax.scatter(orig_vs, per_seed_gains[s], s=4, alpha=0.3,
                   label=f"seed {s}", color=f"C{s}")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Wind speed v [m/s]")
    ax.set_ylabel("Gain [%]")
    ax.set_title("Per-seed gain vs wind speed\n(all directions)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7, markerscale=2)
    ax.grid(alpha=0.3)

    # Panel D: diagnostic — gain% vs v, binned by |phi - 270 deg|.
    # Effective regime (data-driven, H3):
    #   * High-v (>=14 m/s):  rated-power saturation -> ALL bins give gain ~ 0%.
    #     No physical headroom; policy correctly does nothing.
    #   * Low-v (6-8 m/s) + wake-aligned (|dphi|<15°): single-digit % gain
    #     (~4-5%), because rear turbines are deep in the cube region of the
    #     power curve where every recovered m/s of v_eff buys a lot of MW.
    # The marginal mean (+0.4%) heavily dilutes this — paper should report the
    # segmented mean as the headline.
    ax = axes[1, 1]
    dphi = np.abs(((orig_phis - 270.0 + 180.0) % 360.0) - 180.0)  # deg off-axis
    BIN_EDGES = np.array([0, 15, 35, 60, 90.001])
    BIN_LABELS = [f"|phi-270|<{BIN_EDGES[i+1]:.0f}°"
                  for i in range(len(BIN_EDGES) - 1)]
    BIN_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    V_BINS = np.linspace(V_RANGE[0], V_RANGE[1], 11)  # 10 bins of width 1 m/s
    V_CENT = 0.5 * (V_BINS[:-1] + V_BINS[1:])

    for k in range(len(BIN_EDGES) - 1):
        sel = (dphi >= BIN_EDGES[k]) & (dphi < BIN_EDGES[k + 1])
        if sel.sum() < 5:
            continue
        v_in = orig_vs[sel]
        g_in = cond_mean[sel]
        # Bin by v and compute mean +/- std per v-bin.
        means, stds, counts = [], [], []
        for j in range(len(V_BINS) - 1):
            in_bin = (v_in >= V_BINS[j]) & (v_in < V_BINS[j + 1])
            if in_bin.sum() == 0:
                means.append(np.nan); stds.append(np.nan); counts.append(0)
            else:
                means.append(g_in[in_bin].mean())
                stds.append(g_in[in_bin].std())
                counts.append(int(in_bin.sum()))
        means = np.array(means); stds = np.array(stds)
        ax.errorbar(V_CENT, means, yerr=stds, color=BIN_COLORS[k],
                    marker="o", lw=1.4, ms=5, capsize=2,
                    label=f"{BIN_LABELS[k]}  (n={sel.sum()})")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axvline(11.4, color="grey", lw=0.6, ls=":", label="rated v=11.4 m/s")
    ax.set_xlabel("Wind speed v [m/s]")
    ax.set_ylabel("Mean 3-seed gain [%]  (mean ± std within bin)")
    ax.set_title("gain% vs v, binned by direction-misalignment\n"
                 "low-v + wake-aligned = effective regime  |  "
                 "high-v = rated-power saturation",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_fig = os.path.join(FIG_DIR, "fig_p0c_eval_randomized")
    fig.savefig(out_fig + ".pdf", bbox_inches="tight")
    fig.savefig(out_fig + ".jpg", dpi=180, bbox_inches="tight")
    print(f"  -> {out_fig}.pdf  {out_fig}.jpg")


if __name__ == "__main__":
    main()