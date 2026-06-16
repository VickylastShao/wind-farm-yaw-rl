# -*- coding: utf-8 -*-
"""
Reward-penalty ablation figure: A/B learning curves for 3x3 PPO with and
without yaw-magnitude + yaw-rate penalties.

Loads:
  checkpoints_3x3_nnx_jaxenv/summary_p0c.json     (no-penalty baseline)
  checkpoints_3x3_nnx_jaxenv/summary_p0c_pen.json (with-penalty variant)

Output:
  latex_draft/figures/fig_reward_penalty_ablation.{pdf,jpg}
  latex_draft/figures/reward_penalty_ablation.json
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR),
                       "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _load(tag):
    p = os.path.join(CKPT_DIR, f"summary_{tag}.json")
    with open(p) as f:
        return json.load(f)


def _curve(per_seed, key="ep_rew_mean"):
    """Stack per-seed iteration logs into (n_seeds, n_iters) for `key`."""
    rows = []
    n_iters = min(len(s["iterations"]) for s in per_seed)
    for s in per_seed:
        rows.append([s["iterations"][i].get(key, np.nan)
                     for i in range(n_iters)])
    return np.array(rows, dtype=np.float32), n_iters


def main():
    nopen = _load("p0c")
    pen = _load("p0c_pen")

    print(f"# no-penalty  : seeds={nopen['n_seeds']} "
          f"final_ep_rew={[s['final_ep_rew_mean'] for s in nopen['per_seed']]}")
    print(f"# with-penalty: seeds={pen['n_seeds']} "
          f"lambda_mag={pen['lambda_mag']} lambda_rate={pen['lambda_rate']} "
          f"final_ep_rew={[s['final_ep_rew_mean'] for s in pen['per_seed']]}")

    rew_n, n_iters_n = _curve(nopen["per_seed"], "ep_rew_mean")
    rew_p, n_iters_p = _curve(pen["per_seed"], "ep_rew_mean")
    steps_n, _ = _curve(nopen["per_seed"], "total_env_steps")
    steps_p, _ = _curve(pen["per_seed"], "total_env_steps")
    # Use seed-0 step axis (all seeds match up to identical iteration counts).
    x_n = steps_n[0] / 1e6
    x_p = steps_p[0] / 1e6

    # ----- summary numbers -----
    last_k = 10
    nopen_last = rew_n[:, -last_k:].mean(axis=1)
    pen_last = rew_p[:, -last_k:].mean(axis=1)
    summary = dict(
        nopen=dict(
            tag="p0c",
            lambda_mag=0.0, lambda_rate=0.0,
            per_seed_final20=nopen_last.tolist(),
            mean=float(nopen_last.mean()), std=float(nopen_last.std()),
            best=float(nopen_last.max()),
        ),
        pen=dict(
            tag="p0c_pen",
            lambda_mag=pen["lambda_mag"], lambda_rate=pen["lambda_rate"],
            per_seed_final20=pen_last.tolist(),
            mean=float(pen_last.mean()), std=float(pen_last.std()),
            best=float(pen_last.max()),
        ),
    )

    # ----- entropy + clip frac (diagnostic) -----
    ent_n, _ = _curve(nopen["per_seed"], "entropy")
    ent_p, _ = _curve(pen["per_seed"], "entropy")

    summary["nopen"]["entropy_final20_mean"] = float(ent_n[:, -last_k:].mean())
    summary["pen"]["entropy_final20_mean"] = float(ent_p[:, -last_k:].mean())

    print(f"\n=== Final-{last_k}-iter ep_rew_mean ===")
    print(f"  no-penalty   : {summary['nopen']['mean']:+.2f} ± "
          f"{summary['nopen']['std']:.2f}  (best={summary['nopen']['best']:+.2f})")
    print(f"  with-penalty : {summary['pen']['mean']:+.2f} ± "
          f"{summary['pen']['std']:.2f}  (best={summary['pen']['best']:+.2f})")

    out_json = os.path.join(FIG_DIR, "reward_penalty_ablation.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")

    # ----- figure (single-panel: entropy panel removed per peer review C2,
    # since both runs retain ~16-nat entropy and the panel was visually flat) -----
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.4))

    # learning curves: mean ± min/max envelope across seeds.
    for arr, x, color, label in [
            (rew_n, x_n, "#4C78A8", "no penalty (baseline)"),
            (rew_p, x_p, "#E45756",
             f"with penalty (λ_mag=λ_rate={pen['lambda_mag']:.0e})"),
    ]:
        mean = np.nanmean(arr, axis=0)
        lo, hi = np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)
        ax.plot(x, mean, color=color, lw=1.6, label=label)
        for s in range(arr.shape[0]):
            ax.plot(x, arr[s], color=color, lw=0.5, alpha=0.4)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Environment steps (×10$^6$)")
    ax.set_ylabel("Episode return (last-20 mean)")
    ax.set_title("3×3 reward-design ablation: "
                 "λ=8e-5 action penalties reduce attained return",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # NOTE: per-seed final-iter entropy (no-pen vs with-pen) stays in
    # reward_penalty_ablation.json (`entropy_final20_mean` field) — see caption.

    fig.tight_layout()
    out_fig = os.path.join(FIG_DIR, "fig_reward_penalty_ablation")
    fig.savefig(out_fig + ".pdf", bbox_inches="tight")
    fig.savefig(out_fig + ".jpg", dpi=180, bbox_inches="tight")
    print(f"  -> {out_fig}.pdf  {out_fig}.jpg")


if __name__ == "__main__":
    main()
