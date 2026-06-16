#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gain decomposition: quantify the structure of the marginal +0.42% mean gain.

Reads p0c_eval_randomized.json and slsqp_optimum_results.json to compute:
  - How many conditions are in each regime (aligned-cube, marginal, cross-wind)
  - What fraction of the marginal mean comes from the aligned-cube regime
  - How many conditions have DRL actively reducing performance (gain < 0)
  - 4-category breakdown: (SLSQP>0 & DRL>0), (SLSQP>0 & DRL<0), etc.

Output:
  latex_draft/figures/gain_decomposition.json
"""

import os
import json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures")
DRL_JSON = os.path.join(FIG_DIR, "p0c_eval_randomized.json")
SLSQP_JSON = os.path.join(FIG_DIR, "slsqp_optimum_results.json")


def main():
    # Load DRL eval data
    with open(DRL_JSON) as f:
        drl_data = json.load(f)

    # Load SLSQP data
    with open(SLSQP_JSON) as f:
        slsqp_data = json.load(f)

    # ---- DRL per-condition gains (5-seed average) ----
    # per_seed_rows[i] has 3000 entries for seed i
    n_seeds = drl_data["n_seeds"]
    n_conds = drl_data["n_conditions"]

    # Stack gains: (n_seeds, n_conds)
    all_gains = []
    all_phis = []
    all_vs = []
    for seed_idx in range(n_seeds):
        rows = drl_data["per_seed_rows"][seed_idx]
        gains = [r["policy_gain_pct"] for r in rows]
        all_gains.append(gains)
        if seed_idx == 0:
            all_phis = [r["phi"] for r in rows]
            all_vs = [r["v"] for r in rows]

    gains_stack = np.array(all_gains)  # (n_seeds, n_conds)
    gains_mean = gains_stack.mean(axis=0)  # (n_conds,)
    phis = np.array(all_phis)
    vs = np.array(all_vs)

    # ---- Regime masks ----
    dphi = np.abs(((phis - 270.0 + 180.0) % 360.0) - 180.0)
    aligned_cube = (dphi < 15.0) & (vs < 11.4)
    aligned_above_rated = (dphi < 15.0) & (vs >= 11.4)
    near_aligned = (dphi >= 15.0) & (dphi < 35.0)
    cross_wind = dphi >= 35.0

    regimes = {
        "aligned_cube": aligned_cube,
        "aligned_above_rated": aligned_above_rated,
        "near_aligned": near_aligned,
        "cross_wind": cross_wind,
    }

    # ---- Regime statistics ----
    regime_stats = {}
    marginal_total = gains_mean.mean()

    for name, mask in regimes.items():
        n = int(mask.sum())
        if n == 0:
            continue
        regime_gains = gains_mean[mask]
        mean_g = float(regime_gains.mean())
        n_positive = int((regime_gains > 0).sum())
        n_negative = int((regime_gains < -0.001).sum())
        n_zero = int((np.abs(regime_gains) <= 0.001).sum())
        contribution_to_marginal = float(mean_g * n / n_conds)

        regime_stats[name] = {
            "n_conditions": n,
            "fraction": n / n_conds,
            "mean_gain_pct": mean_g,
            "median_gain_pct": float(np.median(regime_gains)),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_zero": n_zero,
            "contribution_to_marginal_pp": contribution_to_marginal,
            "fraction_of_marginal": contribution_to_marginal / marginal_total if abs(marginal_total) > 1e-6 else None,
        }

    # ---- DRL negative-gain analysis (C4) ----
    n_negative = int((gains_mean < -0.001).sum())
    n_zero = int((np.abs(gains_mean) <= 0.001).sum())
    n_positive = int((gains_mean > 0.001).sum())

    negative_gain_mean = float(gains_mean[gains_mean < -0.001].mean()) if n_negative > 0 else None
    positive_gain_mean = float(gains_mean[gains_mean > 0.001].mean()) if n_positive > 0 else None

    # ---- SLSQP vs DRL 4-category breakdown ----
    # Match DRL conditions to nearest SLSQP condition
    slsqp_by_cond = {}
    for r in slsqp_data["results"]:
        slsqp_by_cond[(round(r["phi"], 1), round(r["v"], 1))] = r

    # For the 75 SLSQP conditions, get DRL gains
    slsqp_drl_comparison = []
    for r in slsqp_data["results"]:
        phi_s, v_s = r["phi"], r["v"]
        # Find nearby DRL conditions
        dist = np.sqrt(((phis - phi_s) / 180.0) ** 2 + ((vs - v_s) / 10.0) ** 2)
        nearest = np.argmin(dist)
        drl_g = float(gains_mean[nearest])

        slsqp_g = r["slsqp_gain_pct"]
        slsqp_drl_comparison.append({
            "phi": phi_s,
            "v": v_s,
            "label": r["label"],
            "slsqp_gain_pct": slsqp_g,
            "drl_gain_pct": drl_g,
        })

    # 4-category counts
    SLSQP_POS = 0.1  # threshold for "meaningful positive"
    DRL_POS = 0.01

    cat_A = sum(1 for c in slsqp_drl_comparison
                if c["slsqp_gain_pct"] > SLSQP_POS and c["drl_gain_pct"] > DRL_POS)
    cat_B = sum(1 for c in slsqp_drl_comparison
                if c["slsqp_gain_pct"] > SLSQP_POS and c["drl_gain_pct"] <= DRL_POS)
    cat_C = sum(1 for c in slsqp_drl_comparison
                if c["slsqp_gain_pct"] <= SLSQP_POS and abs(c["drl_gain_pct"]) <= DRL_POS)
    cat_D = sum(1 for c in slsqp_drl_comparison
                if c["slsqp_gain_pct"] <= SLSQP_POS and c["drl_gain_pct"] < -DRL_POS)

    # ---- Output ----
    result = {
        "description": "Gain decomposition: structure of the marginal mean DRL gain",
        "n_conditions": n_conds,
        "n_seeds": n_seeds,
        "marginal_mean_pct": float(marginal_total),
        "regime_breakdown": regime_stats,
        "drl_negative_analysis": {
            "n_negative": n_negative,
            "fraction_negative": n_negative / n_conds,
            "n_zero": n_zero,
            "fraction_zero": n_zero / n_conds,
            "n_positive": n_positive,
            "fraction_positive": n_positive / n_conds,
            "negative_gain_mean_pct": negative_gain_mean,
            "positive_gain_mean_pct": positive_gain_mean,
        },
        "slsqp_vs_drl_categories": {
            "A_slsqp_pos_drl_pos": {"count": cat_A, "description": "SLSQP>0 & DRL>0"},
            "B_slsqp_pos_drl_neg": {"count": cat_B, "description": "SLSQP>0 & DRL≤0 (DRL fails to recover)"},
            "C_slsqp_neg_drl_zero": {"count": cat_C, "description": "SLSQP≈0 & DRL≈0 (correctly passive)"},
            "D_slsqp_neg_drl_neg": {"count": cat_D, "description": "SLSQP≈0 & DRL<0 (DRL actively harmful)"},
        },
        "slsqp_drl_comparison": slsqp_drl_comparison,
    }

    out_path = os.path.join(FIG_DIR, "gain_decomposition.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"GAIN DECOMPOSITION SUMMARY")
    print(f"{'='*60}")
    print(f"  Marginal mean gain:  {marginal_total:+.3f}%")
    print(f"\n  Regime breakdown:")
    for name, s in regime_stats.items():
        frac_contrib = f"{s['fraction_of_marginal']:.1%}" if s.get('fraction_of_marginal') else "N/A"
        print(f"    {name:25s}: n={s['n_conditions']:4d} ({s['fraction']:.1%})  "
              f"mean={s['mean_gain_pct']:+.3f}%  "
              f"contrib={s['contribution_to_marginal_pp']:+.3f}pp ({frac_contrib})")

    print(f"\n  DRL negative-gain conditions (C4):")
    print(f"    Negative: {n_negative} ({n_negative/n_conds:.1%})")
    print(f"    Zero:     {n_zero} ({n_zero/n_conds:.1%})")
    print(f"    Positive: {n_positive} ({n_positive/n_conds:.1%})")
    if negative_gain_mean:
        print(f"    Mean negative gain: {negative_gain_mean:+.4f}%")

    print(f"\n  SLSQP vs DRL 4-category (on {len(slsqp_drl_comparison)} SLSQP conditions):")
    print(f"    A (SLSQP>0 & DRL>0):     {cat_A}")
    print(f"    B (SLSQP>0 & DRL≤0):     {cat_B}  ← DRL fails to recover available gain")
    print(f"    C (SLSQP≈0 & DRL≈0):     {cat_C}  ← correctly passive")
    print(f"    D (SLSQP≈0 & DRL<0):     {cat_D}  ← DRL actively harmful")


if __name__ == "__main__":
    main()
