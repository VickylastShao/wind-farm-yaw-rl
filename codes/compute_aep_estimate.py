#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEP (Annual Energy Production) impact estimate.

Uses realistic wind frequency distributions to weight per-condition
gains from the DRL policy evaluation, producing an AEP gain estimate.

Wind distributions:
  - Speed: Weibull(k=2, A=11 m/s) — typical offshore
  - Direction: von Mises(mu=270°, kappa=1,2) — dominant westerly

Method:
  1. Sample 10,000 (phi, v) pairs from the joint distribution
  2. For each pair, look up the DRL gain from p0c_eval_randomized.json
     (nearest-neighbor interpolation)
  3. Compute AEP_gain = sum(G(phi,v) * P_baseline(phi,v) * f(phi,v)) * 8760 h
  4. Also compute a simplified estimate using the segmented regime data

Output:
  latex_draft/figures/aep_estimate.json
"""

import os
import json
import numpy as np
from scipy.stats import weibull_min
from scipy.special import i0  # modified Bessel function of order 0

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures")
DRL_JSON = os.path.join(FIG_DIR, "p0c_eval_randomized.json")


def von_mises_pdf(phi, mu, kappa):
    """von Mises PDF for circular variable phi (in radians)."""
    return np.exp(kappa * np.cos(phi - mu)) / (2 * np.pi * i0(kappa))


def main():
    # Load DRL eval data
    with open(DRL_JSON) as f:
        drl_data = json.load(f)

    # Per-condition gains (5-seed mean)
    n_seeds = drl_data["n_seeds"]
    rows = drl_data["per_seed_rows"][0]  # use first seed's phi,v
    phis_eval = np.array([r["phi"] for r in rows])
    vs_eval = np.array([r["v"] for r in rows])

    gains_stack = []
    for s in range(n_seeds):
        gains_stack.append([r["policy_gain_pct"] for r in drl_data["per_seed_rows"][s]])
    gains_stack = np.array(gains_stack)
    gains_mean = gains_stack.mean(axis=0)

    # Baseline power per condition (from the eval)
    # We need to reconstruct this from the gains:
    # gain_pct = (P_policy - P_baseline) / P_baseline * 100
    # P_policy = P_baseline * (1 + gain_pct/100)
    # We need P_baseline separately — use the gray-box model
    from windfarm_env import (create_wind_farm_layout_3x3,
                               calculate_inflow_speeds, power_output,
                               C_T, I, d_0, alpha_star, beta_star, alpha)

    positions, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions)

    # Wind distributions
    A_weibull = 11.0  # m/s, scale parameter
    k_weibull = 2.0   # shape parameter
    mu_vonmises = np.radians(270.0)  # dominant direction
    kappa_values = [1.0, 2.0]

    N_AEP_SAMPLES = 10000

    results = {}

    for kappa in kappa_values:
        key = f"kappa_{kappa}"
        print(f"\n## kappa={kappa}")

        # Sample from joint distribution
        np_rng = np.random.default_rng(20260605)

        # Sample wind speed from Weibull
        v_samples = weibull_min.rvs(k_weibull, scale=A_weibull,
                                     size=N_AEP_SAMPLES, random_state=np_rng)

        # Clip to evaluation range
        v_samples = np.clip(v_samples, 6.0, 16.0)

        # Sample wind direction from von Mises
        phi_samples_rad = np_rng.vonmises(mu_vonmises, kappa, size=N_AEP_SAMPLES)
        phi_samples = np.degrees(phi_samples_rad) % 360
        # Clip to evaluation range
        phi_samples = np.clip(phi_samples, 173.0, 353.0)

        # For each sample, find nearest evaluated condition and get gain
        # Use vectorized nearest-neighbor lookup
        from scipy.spatial import cKDTree
        # Normalize coordinates for tree search
        phi_norm = phis_eval / 180.0
        v_norm = vs_eval / 10.0
        tree = cKDTree(np.column_stack([phi_norm, v_norm]))

        phi_s_norm = phi_samples / 180.0
        v_s_norm = v_samples / 10.0
        _, indices = tree.query(np.column_stack([phi_s_norm, v_s_norm]))

        gains_aep = gains_mean[indices]

        # Compute baseline power for each sample
        baselines_aep = np.empty(N_AEP_SAMPLES)
        for i in range(N_AEP_SAMPLES):
            inflow = calculate_inflow_speeds(
                positions, phi_samples[i], C_T, I, d_0, v_samples[i],
                np.zeros(N_turb), alpha_star, beta_star, alpha)
            baselines_aep[i] = sum(power_output(inflow[j], 0.0)
                                    for j in range(N_turb)) / 1e6  # MW

        # AEP calculation
        # Weighted gain: sum over samples of (gain * baseline_power)
        # Normalized by sum of baseline_power (unweighted average baseline)
        annual_hours = 8760.0
        baseline_aep = np.mean(baselines_aep) * annual_hours  # MWh/yr
        policy_aep = np.mean(baselines_aep * (1 + gains_aep / 100.0)) * annual_hours  # MWh/yr
        aep_gain_mwh = policy_aep - baseline_aep
        aep_gain_pct = aep_gain_mwh / baseline_aep * 100.0

        # Revenue estimate (50 EUR/MWh offshore)
        revenue_gain = aep_gain_mwh * 50.0  # EUR/yr

        # By regime
        dphi_aep = np.abs(((phi_samples - 270.0 + 180.0) % 360.0) - 180.0)
        aligned_cube_aep = (dphi_aep < 15.0) & (v_samples < 11.4)

        regime_stats = {
            "aligned_cube": {
                "n": int(aligned_cube_aep.sum()),
                "fraction": float(aligned_cube_aep.mean()),
                "mean_gain_pct": float(gains_aep[aligned_cube_aep].mean())
                    if aligned_cube_aep.sum() > 0 else None,
                "aep_contribution_pct": float(
                    np.mean(baselines_aep[aligned_cube_aep] * gains_aep[aligned_cube_aep] / 100.0) /
                    np.mean(baselines_aep * gains_aep / 100.0) * 100.0)
                    if aligned_cube_aep.sum() > 0 and np.mean(baselines_aep * gains_aep / 100.0) != 0
                    else None,
            },
            "non_aligned": {
                "n": int((~aligned_cube_aep).sum()),
                "fraction": float((~aligned_cube_aep).mean()),
                "mean_gain_pct": float(gains_aep[~aligned_cube_aep].mean()),
            }
        }

        result = {
            "kappa": kappa,
            "weibull_A": A_weibull,
            "weibull_k": k_weibull,
            "n_samples": N_AEP_SAMPLES,
            "baseline_aep_mwh_yr": float(baseline_aep),
            "policy_aep_mwh_yr": float(policy_aep),
            "aep_gain_mwh_yr": float(aep_gain_mwh),
            "aep_gain_pct": float(aep_gain_pct),
            "revenue_gain_eur_yr": float(revenue_gain),
            "price_eur_per_mwh": 50.0,
            "marginal_gain_pct": float(gains_aep.mean()),
            "regime_breakdown": regime_stats,
        }
        results[key] = result

        print(f"  Baseline AEP:  {baseline_aep:.0f} MWh/yr")
        print(f"  Policy AEP:    {policy_aep:.0f} MWh/yr")
        print(f"  AEP gain:      {aep_gain_mwh:+.0f} MWh/yr ({aep_gain_pct:+.3f}%)")
        print(f"  Revenue gain:  {revenue_gain:+.0f} EUR/yr @ 50 EUR/MWh")
        print(f"  Marginal gain: {gains_aep.mean():+.3f}%")
        if regime_stats["aligned_cube"]["aep_contribution_pct"] is not None:
            print(f"  Aligned-cube AEP contribution: "
                  f"{regime_stats['aligned_cube']['aep_contribution_pct']:.1f}%")

    # Save
    out_path = os.path.join(FIG_DIR, "aep_estimate.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
