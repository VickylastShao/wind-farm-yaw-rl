#!/usr/bin/env python3
"""
Compute bootstrap 95% confidence intervals on all headline gains.

Input files:
  - p0c_eval_randomized.json      (3x3 eval)
  - p1_5x5_eval_randomized.json   (5x5 eval)
  - 3x3_training_stats_p0c.json   (training return)

B=10000 bootstrap resamples, seed=42.
"""

import json
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
B = 10_000
RNG_SEED = 42
FIG_DIR = Path("/home/gpu/sz_workspace/JAX-WFCOYAW-RL/latex_draft/figures")

EVAL_FILES = {
    "3x3": FIG_DIR / "p0c_eval_randomized.json",
    "5x5": FIG_DIR / "p1_5x5_eval_randomized.json",
}
TRAIN_FILE = FIG_DIR / "3x3_training_stats_p0c.json"

# Aligned-cube filter: |phi - 270| < 15 AND v < 11.4
ALIGNED_PHI_CENTER = 270.0
ALIGNED_PHI_HALF = 15.0
ALIGNED_V_MAX = 11.4

# 4x4 regime bins
PHI_EDGES = [173, 218, 263, 308, 353]
V_EDGES = [6.0, 8.5, 11.0, 13.5, 16.0]

OUT_FILE = FIG_DIR / "bootstrap_ci_results.json"


# ── Helpers ────────────────────────────────────────────────────────────

def load_eval(path):
    with open(path) as f:
        return json.load(f)


def condition_matrix(data):
    """Return (n_seeds, n_conditions) array of policy_gain_pct and
    (n_conditions,) arrays of phi and v."""
    n_seeds = data["n_seeds"]
    rows = data["per_seed_rows"]  # list of n_seeds lists of condition dicts
    n_cond = len(rows[0])
    gains = np.zeros((n_seeds, n_cond))
    phis = np.zeros(n_cond)
    vs = np.zeros(n_cond)
    for s in range(n_seeds):
        for i, cond in enumerate(rows[s]):
            gains[s, i] = cond["policy_gain_pct"]
            if s == 0:
                phis[i] = cond["phi"]
                vs[i] = cond["v"]
    return gains, phis, vs


def seed_averaged_gains(gains):
    """Average across seeds → (n_conditions,) vector."""
    return gains.mean(axis=0)


def aligned_cube_mask(phis, vs):
    return (np.abs(phis - ALIGNED_PHI_CENTER) < ALIGNED_PHI_HALF) & (vs < ALIGNED_V_MAX)


def bootstrap_ci_over_conditions(gain_vec, mask=None, rng=None):
    """Bootstrap CI by resampling conditions. Returns (point, [lo, hi])."""
    if mask is not None:
        gain_vec = gain_vec[mask]
    point = float(gain_vec.mean())
    n = len(gain_vec)
    indices = rng.integers(0, n, size=(B, n))
    boot_means = gain_vec[indices].mean(axis=1)
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return point, [lo, hi]


def bootstrap_ci_inter_seed_on_mean(gains, mask=None, rng=None):
    """Inter-seed bootstrap CI on the aligned-cube mean gain.

    Resample seeds with replacement. For each bootstrap sample, compute
    the seed-averaged aligned-cube mean gain.
    Returns (point, [lo, hi]).
    """
    n_seeds = gains.shape[0]
    n_cond = gains.shape[1]

    # Per-seed mean gain over the masked conditions
    if mask is not None:
        per_seed_means = np.array([gains[s, mask].mean() for s in range(n_seeds)])
    else:
        per_seed_means = gains.mean(axis=1)

    point = float(per_seed_means.mean())

    seed_indices = rng.integers(0, n_seeds, size=(B, n_seeds))
    boot_means = per_seed_means[seed_indices].mean(axis=1)
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return point, [lo, hi]


def regime_bins(phis, vs):
    """Assign each condition to a (phi_bin, v_bin) index. Returns (n_cond, 2) int array."""
    phi_bin = np.digitize(phis, PHI_EDGES[1:-1])  # 0..3
    v_bin = np.digitize(vs, V_EDGES[1:-1])         # 0..3
    return phi_bin, v_bin


def bootstrap_regime_table(gain_vec, phi_bin, v_bin, rng):
    """Per-bin bootstrap CIs for 4x4 regime table.

    Returns dict: "{phi_bin}_{v_bin}": {"point": ..., "ci_95": [...], "n_conditions": ...}
    """
    result = {}
    for pb in range(4):
        for vb in range(4):
            mask = (phi_bin == pb) & (v_bin == vb)
            n_cond = int(mask.sum())
            if n_cond == 0:
                result[f"{pb}_{vb}"] = {"point": None, "ci_95": [None, None], "n_conditions": 0}
                continue
            sub = gain_vec[mask]
            point = float(sub.mean())
            indices = rng.integers(0, n_cond, size=(B, n_cond))
            boot_means = sub[indices].mean(axis=1)
            lo = float(np.percentile(boot_means, 2.5))
            hi = float(np.percentile(boot_means, 97.5))
            result[f"{pb}_{vb}"] = {"point": round(point, 4), "ci_95": [round(lo, 4), round(hi, 4)], "n_conditions": n_cond}
    return result


def bootstrap_training_return(seed_returns, rng):
    """Inter-seed CI on final episode return."""
    n = len(seed_returns)
    point = float(np.mean(seed_returns))
    indices = rng.integers(0, n, size=(B, n))
    boot_means = seed_returns[indices].mean(axis=1)
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return point, [lo, hi]


# ── Main ───────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RNG_SEED)

    # Load training stats
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)
    train_returns = np.array(train_data["per_seed_final_ep_rew"])

    all_results = {}

    for tag, fpath in EVAL_FILES.items():
        print(f"\n{'='*60}")
        print(f"  {tag} Farm")
        print(f"{'='*60}")

        data = load_eval(fpath)
        gains, phis, vs = condition_matrix(data)
        avg_gains = seed_averaged_gains(gains)
        n_seeds = gains.shape[0]
        n_cond = gains.shape[1]

        ac_mask = aligned_cube_mask(phis, vs)
        n_ac = int(ac_mask.sum())

        print(f"  Seeds: {n_seeds}, Conditions: {n_cond}, Aligned-cube conditions: {n_ac}")

        # 1. Marginal mean gain (inter-condition bootstrap)
        marg_point, marg_ci = bootstrap_ci_over_conditions(avg_gains, mask=None, rng=rng)
        print(f"\n  Marginal mean gain:  {marg_point:.4f}%  CI95: [{marg_ci[0]:.4f}, {marg_ci[1]:.4f}]")

        # 2. Aligned-cube mean gain (inter-condition bootstrap)
        ac_point, ac_ci = bootstrap_ci_over_conditions(avg_gains, mask=ac_mask, rng=rng)
        print(f"  Aligned-cube gain:   {ac_point:.4f}%  CI95: [{ac_ci[0]:.4f}, {ac_ci[1]:.4f}]")

        # 3. Inter-seed CI on aligned-cube mean
        ac_seed_point, ac_seed_ci = bootstrap_ci_inter_seed_on_mean(gains, mask=ac_mask, rng=rng)
        print(f"  Aligned-cube (inter-seed): {ac_seed_point:.4f}%  CI95: [{ac_seed_ci[0]:.4f}, {ac_seed_ci[1]:.4f}]")

        # 4. Regime table
        phi_bin, v_bin = regime_bins(phis, vs)
        regime = bootstrap_regime_table(avg_gains, phi_bin, v_bin, rng)

        print(f"\n  Regime table (4 phi bins x 4 speed bins):")
        print(f"  Phi bins: [{PHI_EDGES[0]},{PHI_EDGES[1]}), [{PHI_EDGES[1]},{PHI_EDGES[2]}), [{PHI_EDGES[2]},{PHI_EDGES[3]}), [{PHI_EDGES[3]},{PHI_EDGES[4]}]")
        print(f"  V bins:   [{V_EDGES[0]},{V_EDGES[1]}), [{V_EDGES[1]},{V_EDGES[2]}), [{V_EDGES[2]},{V_EDGES[3]}), [{V_EDGES[3]},{V_EDGES[4]}]")
        print(f"  {'Phi\\V':<8}", end="")
        for vb in range(4):
            print(f" [{V_EDGES[vb]},{V_EDGES[vb+1]})       ", end="")
        print()
        for pb in range(4):
            label = f"[{PHI_EDGES[pb]},{PHI_EDGES[pb+1]})"
            print(f"  {label:<12}", end="")
            for vb in range(4):
                key = f"{pb}_{vb}"
                r = regime[key]
                if r["point"] is None:
                    print(f"{'N/A':<20}", end="")
                else:
                    print(f" {r['point']:>6.3f} [{r['ci_95'][0]:>6.3f},{r['ci_95'][1]:>6.3f}]", end="")
            print()

        # 5. Training return (only for 3x3)
        train_result = None
        if tag == "3x3":
            tr_point, tr_ci = bootstrap_training_return(train_returns, rng)
            print(f"\n  Training final return (inter-seed): {tr_point:.4f}  CI95: [{tr_ci[0]:.4f}, {tr_ci[1]:.4f}]")
            train_result = {
                "point": round(tr_point, 4),
                "ci_95": [round(tr_ci[0], 4), round(tr_ci[1], 4)],
                "std_type": "inter-seed"
            }

        # Assemble result dict
        all_results[tag] = {
            "marginal_mean_gain_pct": {
                "point": round(marg_point, 4),
                "ci_95": [round(marg_ci[0], 4), round(marg_ci[1], 4)],
                "std_type": "inter-condition"
            },
            "aligned_cube_gain_pct": {
                "point": round(ac_point, 4),
                "ci_95": [round(ac_ci[0], 4), round(ac_ci[1], 4)],
                "std_type": "inter-condition"
            },
            "aligned_cube_inter_seed_ci": {
                "point": round(ac_seed_point, 4),
                "ci_95": [round(ac_seed_ci[0], 4), round(ac_seed_ci[1], 4)],
                "std_type": "inter-seed"
            },
            "regime_table_ci": regime,
        }
        if train_result is not None:
            all_results[tag]["training_return"] = train_result

    # Write output JSON
    with open(OUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults written to {OUT_FILE}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for tag in all_results:
        r = all_results[tag]
        print(f"\n  {tag}:")
        print(f"    Marginal mean gain:  {r['marginal_mean_gain_pct']['point']:.4f}%  "
              f"CI95: [{r['marginal_mean_gain_pct']['ci_95'][0]:.4f}, {r['marginal_mean_gain_pct']['ci_95'][1]:.4f}]")
        print(f"    Aligned-cube gain:   {r['aligned_cube_gain_pct']['point']:.4f}%  "
              f"CI95: [{r['aligned_cube_gain_pct']['ci_95'][0]:.4f}, {r['aligned_cube_gain_pct']['ci_95'][1]:.4f}]")
        print(f"    Aligned-cube (seed): {r['aligned_cube_inter_seed_ci']['point']:.4f}%  "
              f"CI95: [{r['aligned_cube_inter_seed_ci']['ci_95'][0]:.4f}, {r['aligned_cube_inter_seed_ci']['ci_95'][1]:.4f}]")
        if "training_return" in r:
            print(f"    Training return:     {r['training_return']['point']:.4f}  "
                  f"CI95: [{r['training_return']['ci_95'][0]:.4f}, {r['training_return']['ci_95'][1]:.4f}]")


if __name__ == "__main__":
    main()
