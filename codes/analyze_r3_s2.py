#!/usr/bin/env python3
"""
R3: Bootstrap CI dual-type report (condition-sampling + seed-sampling)
S2: Regime threshold sensitivity analysis

Uses unified_static_vs_slsqp.json for the optimized config (sens_act10).
"""
import json
import numpy as np
from pathlib import Path

FIG_DIR = Path("/home/gpu/sz_workspace/JAX-WFCOYAW-RL/latex_draft/figures")
B = 10_000
RNG_SEED = 42

def main():
    rng = np.random.default_rng(RNG_SEED)

    # Load optimized config data
    with open(FIG_DIR / "unified_static_vs_slsqp.json") as f:
        data = json.load(f)

    conditions = data["per_condition"]  # list of 500 condition dicts
    n_cond = len(conditions)

    # Extract arrays
    phis = np.array([c["phi"] for c in conditions])
    vs = np.array([c["v"] for c in conditions])
    dphis = np.array([c["dphi"] for c in conditions])
    slsqp_gains = np.array([c["slsqp_gain_pct"] for c in conditions])
    drl_gains = np.array([c["policies"]["sens_act10"]["gain_pct"] for c in conditions])

    print(f"Loaded {n_cond} conditions")
    print(f"SLSQP aligned-cube gain: {slsqp_gains[(dphis < 15) & (vs < 11.4)].mean():.4f}%")
    print(f"DRL aligned-cube gain:   {drl_gains[(dphis < 15) & (vs < 11.4)].mean():.4f}%")

    # ══════════════════════════════════════════════════════════════════
    # R3: Bootstrap CI on recovery rate
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("R3: Bootstrap CI on Recovery Rate (sens_act10)")
    print(f"{'='*60}")

    # Aligned-cube mask
    ac_mask = (dphis < 15.0) & (vs < 11.4)
    n_ac = int(ac_mask.sum())
    print(f"Aligned-cube conditions: {n_ac}")

    # Recovery rate = ratio of MEANS (not mean of per-condition ratios).
    # Per-condition ratios are unstable when SLSQP gain is near zero.
    # We bootstrap the ratio: resample conditions, compute mean_drl/mean_slsqp.
    def bootstrap_recovery_ratio(drl_vals, slsqp_vals, mask, rng, B=10000):
        """Bootstrap CI on recovery = mean(drl)/mean(slsqp) * 100 over masked conditions."""
        d = drl_vals[mask]
        s = slsqp_vals[mask]
        n = len(d)
        point = float(d.mean() / max(s.mean(), 1e-9) * 100)
        indices = rng.integers(0, n, size=(B, n))
        boot_ratios = d[indices].mean(axis=1) / np.maximum(s[indices].mean(axis=1), 1e-9) * 100
        lo = float(np.percentile(boot_ratios, 2.5))
        hi = float(np.percentile(boot_ratios, 97.5))
        return point, lo, hi

    # Recovery rate over aligned-cube conditions
    rec_point, rec_lo, rec_hi = bootstrap_recovery_ratio(drl_gains, slsqp_gains, ac_mask, rng)
    print(f"\nRecovery rate (aligned-cube, condition-sampling, ratio-of-means):")
    print(f"  Point: {rec_point:.2f}%")
    print(f"  CI95:  [{rec_lo:.2f}%, {rec_hi:.2f}%]")

    # Recovery over ALL conditions (for comparison)
    all_mask = np.ones(n_cond, dtype=bool)
    all_rec_point, all_rec_lo, all_rec_hi = bootstrap_recovery_ratio(drl_gains, slsqp_gains, all_mask, rng)
    print(f"\nRecovery rate (ALL conditions, ratio-of-means):")
    print(f"  Point: {all_rec_point:.2f}%")
    print(f"  CI95:  [{all_rec_lo:.2f}%, {all_rec_hi:.2f}%]")

    # Also compute recovery over off-axis conditions (|dphi| >= 15)
    off_axis_mask = (dphis >= 15.0) & (vs < 11.4)
    if off_axis_mask.sum() > 0:
        off_point, off_lo, off_hi = bootstrap_recovery_ratio(drl_gains, slsqp_gains, off_axis_mask, rng)
        print(f"\nRecovery rate (off-axis, |dphi|>=15°, v<11.4, ratio-of-means):")
        print(f"  Point: {off_point:.2f}%")
        print(f"  CI95:  [{off_lo:.2f}%, {off_hi:.2f}%]")
        print(f"  n={off_axis_mask.sum()} (SLSQP mean={slsqp_gains[off_axis_mask].mean():.4f}% — near zero, recovery ill-defined)")

    # ══════════════════════════════════════════════════════════════════
    # S2: Regime threshold sensitivity
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("S2: Regime Threshold Sensitivity (sens_act10)")
    print(f"{'='*60}")

    phi_half_widths = [5, 10, 15, 20, 25, 30]
    v_maxes = [10.0, 10.5, 11.0, 11.4, 12.0]

    print(f"\n{'|dphi| threshold':>16s} | {'v<11.4 gain':>10s} | {'n_cond':>6s} | {'recovery':>8s}")
    print("-" * 58)

    for hw in phi_half_widths:
        mask = (dphis < hw) & (vs < 11.4)
        n = int(mask.sum())
        drl_g = drl_gains[mask].mean()
        slsqp_g = slsqp_gains[mask].mean()
        rec = drl_g / slsqp_g * 100 if slsqp_g > 0.01 else 0
        print(f"  |dphi| < {hw:>2d}°         | {drl_g:>+10.4f}% | {n:>6d} | {rec:>7.2f}%")

    print(f"\n{'v threshold':>16s} | {'|dphi|<15 gain':>12s} | {'n_cond':>6s} | {'recovery':>8s}")
    print("-" * 58)

    for vmax in v_maxes:
        mask = (dphis < 15.0) & (vs < vmax)
        n = int(mask.sum())
        drl_g = drl_gains[mask].mean()
        slsqp_g = slsqp_gains[mask].mean()
        rec = drl_g / slsqp_g * 100 if slsqp_g > 0.01 else 0
        print(f"  v < {vmax:>4.1f} m/s       | {drl_g:>+12.4f}% | {n:>6d} | {rec:>7.2f}%")

    # 2D sensitivity heatmap
    print(f"\n{'='*60}")
    print("2D Sensitivity: DRL gain for each (|dphi| threshold, v threshold) pair")
    print(f"{'='*60}")
    print(f"{'':>12s}", end="")
    for vmax in v_maxes:
        print(f" v<{vmax:4.1f}", end="")
    print()
    for hw in phi_half_widths:
        print(f"  |dphi|<{hw:>2d}° ", end="")
        for vmax in v_maxes:
            mask = (dphis < hw) & (vs < vmax)
            drl_g = drl_gains[mask].mean()
            print(f" {drl_g:+7.3f}", end="")
        print()

    # Save results
    output = {
        "description": "R3 & S2: Bootstrap CI + regime sensitivity for sens_act10",
        "recovery_rate": {
            "method": "ratio-of-means bootstrap (avoids per-condition ratio instability)",
            "aligned_cube": {
                "point_pct": round(rec_point, 2),
                "ci95": [round(rec_lo, 2), round(rec_hi, 2)],
                "n_conditions": int(ac_mask.sum()),
            },
            "all_conditions": {
                "point_pct": round(all_rec_point, 2),
                "ci95": [round(all_rec_lo, 2), round(all_rec_hi, 2)],
                "n_conditions": n_cond,
            },
            "note": "Seed-sampling CI requires per-seed per-condition data. "
                    "The cross-seed evaluation (seed 20260609, recovery=90.5%) provides an independent estimate. "
                    "For the paper, recommend reporting both: condition-sampling CI (uncertainty from condition sampling) "
                    "and cross-seed evaluation (uncertainty from random seed)."
        },
        "regime_sensitivity": {
            "phi_half_widths": phi_half_widths,
            "v_maxes": v_maxes,
            "by_phi_threshold": [],
            "by_v_threshold": [],
        }
    }

    for hw in phi_half_widths:
        mask = (dphis < hw) & (vs < 11.4)
        output["regime_sensitivity"]["by_phi_threshold"].append({
            "|dphi|_threshold": hw,
            "n_conditions": int(mask.sum()),
            "drl_gain_pct": round(float(drl_gains[mask].mean()), 4),
            "slsqp_gain_pct": round(float(slsqp_gains[mask].mean()), 4),
            "recovery_pct": round(float(drl_gains[mask].mean() / max(slsqp_gains[mask].mean(), 0.01) * 100), 2),
        })

    for vmax in v_maxes:
        mask = (dphis < 15.0) & (vs < vmax)
        output["regime_sensitivity"]["by_v_threshold"].append({
            "v_threshold": vmax,
            "n_conditions": int(mask.sum()),
            "drl_gain_pct": round(float(drl_gains[mask].mean()), 4),
            "slsqp_gain_pct": round(float(slsqp_gains[mask].mean()), 4),
            "recovery_pct": round(float(drl_gains[mask].mean() / max(slsqp_gains[mask].mean(), 0.01) * 100), 2),
        })

    out_path = FIG_DIR / "review_r3_s2_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
