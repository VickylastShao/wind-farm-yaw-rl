#!/usr/bin/env python3
"""Parameter sensitivity analysis for the gray-box wake model.

Tests the sensitivity of farm power prediction to perturbations of
the calibrated parameters α* and α (the two most influential parameters
for wake superposition and yaw deflection).

For each condition in a representative grid, we:
  1. Compute baseline farm power with calibrated parameters
  2. Perturb α* and α by ±10%, ±20%, ±30%
  3. Report the power deviation and its regime dependence
"""

import sys, os, json, time
import numpy as np

sys.path.insert(0, '/home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes')

from windfarm_env import (
    calculate_inflow_speeds,
    power_output,
    create_wind_farm_layout_3x3,
    d_0, z_h, P_rated, C_T, I,
    alpha_star as alpha_star_0,
    beta_star,
    alpha as alpha_0,
)

# Build layout
positions, N_rows, N_cols = create_wind_farm_layout_3x3()
N = len(positions)

# Calibrated values
CALIBRATED = {"alpha_star": alpha_star_0, "alpha": alpha_0, "beta_star": beta_star, "I": I}
print(f"Calibrated parameters: α*={alpha_star_0:.4f}, α={alpha_0:.4f}, β*={beta_star:.4f}, I={I:.4f}")

def farm_power(phi, v, gammas, params):
    """Compute total farm power for given parameters."""
    p_alpha_star = params.get("alpha_star", alpha_star_0)
    p_alpha = params.get("alpha", alpha_0)
    p_beta_star = params.get("beta_star", beta_star)
    p_I = params.get("I", I)
    inflow = calculate_inflow_speeds(
        positions, phi, C_T, p_I, d_0, v, gammas,
        p_alpha_star, p_beta_star, p_alpha
    )
    return sum(power_output(inflow[i], gammas[i]) for i in range(N)) / 1e6

# ===========================================================================
# Evaluation conditions
# ===========================================================================
conditions = []
# Aligned-cube: where wake effects are strongest
for phi in np.arange(255, 286, 5):
    for v in [7, 8, 9, 10, 11]:
        conditions.append((float(phi), float(v), "aligned_cube"))
# Off-axis: where wake effects are weak
for phi in [200, 220, 240, 300, 320, 340]:
    for v in [9, 11.4]:
        conditions.append((float(phi), float(v), "off_axis"))
# Reference
conditions.append((270.0, 11.4, "reference"))

print(f"\nTotal conditions: {len(conditions)}")

# ===========================================================================
# Perturbation levels
# ===========================================================================
perturbations = [-0.30, -0.20, -0.10, 0.0, +0.10, +0.20, +0.30]

results = []

print("\n" + "=" * 90)
print("PARAMETER SENSITIVITY ANALYSIS")
print("=" * 90)

for idx, (phi, v, label) in enumerate(conditions):
    if idx % 20 == 0:
        print(f"  Progress: {idx}/{len(conditions)}...")

    # Baseline with calibrated parameters
    base_pwr = farm_power(phi, v, np.zeros(N), CALIBRATED)

    for pert in perturbations:
        # Perturb α* only
        p_a_star = alpha_star_0 * (1.0 + pert)
        pwr_as = farm_power(phi, v, np.zeros(N),
                            {"alpha_star": p_a_star, "alpha": alpha_0,
                             "beta_star": beta_star, "I": I})
        rel_err_as = (pwr_as - base_pwr) / base_pwr * 100 if base_pwr > 0 else 0

        # Perturb α only
        p_a = alpha_0 * (1.0 + pert)
        pwr_a = farm_power(phi, v, np.zeros(N),
                           {"alpha_star": alpha_star_0, "alpha": p_a,
                            "beta_star": beta_star, "I": I})
        rel_err_a = (pwr_a - base_pwr) / base_pwr * 100 if base_pwr > 0 else 0

        # Perturb both (correlated)
        pwr_both = farm_power(phi, v, np.zeros(N),
                              {"alpha_star": p_a_star, "alpha": p_a,
                               "beta_star": beta_star, "I": I})
        rel_err_both = (pwr_both - base_pwr) / base_pwr * 100 if base_pwr > 0 else 0

        results.append({
            "phi": phi, "v": v, "label": label,
            "perturbation": pert,
            "alpha_star_val": p_a_star, "alpha_val": p_a,
            "baseline_mw": base_pwr,
            "perturbed_as_mw": pwr_as, "rel_err_as_pct": rel_err_as,
            "perturbed_a_mw": pwr_a, "rel_err_a_pct": rel_err_a,
            "perturbed_both_mw": pwr_both, "rel_err_both_pct": rel_err_both,
        })

print(f"  Progress: {len(conditions)}/{len(conditions)} - done.")

# ===========================================================================
# Summary statistics
# ===========================================================================
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

# Separate by regime
aligned_results = [r for r in results if r["label"] == "aligned_cube"]
off_axis_results = [r for r in results if r["label"] == "off_axis"]
ref_results = [r for r in results if r["label"] == "reference"]

for regime_name, regime_results in [("Aligned-cube", aligned_results),
                                      ("Off-axis", off_axis_results),
                                      ("Reference", ref_results)]:
    print(f"\n--- {regime_name} (n={len(regime_results)} condition×perturbation pairs) ---")

    for pert_level in [-0.20, -0.10, +0.10, +0.20]:
        pert_data = [r for r in regime_results if abs(r["perturbation"] - pert_level) < 0.001]

        if pert_data:
            as_errs = [abs(r["rel_err_as_pct"]) for r in pert_data]
            a_errs = [abs(r["rel_err_a_pct"]) for r in pert_data]
            both_errs = [abs(r["rel_err_both_pct"]) for r in pert_data]

            print(f"  Perturb {pert_level:+3.0%}: "
                  f"|ΔP|_α* = {np.mean(as_errs):.2f}% (max {np.max(as_errs):.2f}%), "
                  f"|ΔP|_α = {np.mean(a_errs):.2f}% (max {np.max(a_errs):.2f}%), "
                  f"|ΔP|_both = {np.mean(both_errs):.2f}% (max {np.max(both_errs):.2f}%)")

# ===========================================================================
# Save
# ===========================================================================
out_path = '/home/gpu/sz_workspace/JAX-WFCOYAW-RL/latex_draft/figures/param_sensitivity.json'
with open(out_path, 'w') as f:
    json.dump({
        "description": "Wake model parameter sensitivity: α* and α perturbations",
        "calibrated_params": CALIBRATED,
        "perturbations": perturbations,
        "n_conditions": len(conditions),
        "results": results,
    }, f, indent=2)
print(f"\nSaved: {out_path}")
