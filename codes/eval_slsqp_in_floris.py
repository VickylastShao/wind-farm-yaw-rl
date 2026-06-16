#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate SLSQP-optimal yaw angles inside FLORIS.

Loads the 75-condition SLSQP results from slsqp_optimum_results.json,
evaluates the SLSQP yaw vectors in FLORIS (NREL 5MW, GCH wake model),
and compares FLORIS-validated SLSQP gains against gray-box SLSQP gains.

This determines whether the gray-box model bias is DRL-specific or a
global property of the wake model.

Output:
  latex_draft/figures/floris_slsqp_cross_eval.json
"""

import os
import sys
import json
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures")
SLSQP_JSON = os.path.join(FIG_DIR, "slsqp_optimum_results.json")


def make_floris_model():
    """Create FLORIS model with 3x3 NREL 5MW layout, GCH wake model."""
    from floris import FlorisModel

    fm = FlorisModel("defaults")

    # 3x3 layout matching windfarm_env.create_wind_farm_layout_3x3()
    d_0 = 126.0
    sx, sy = 7 * d_0, 7 * d_0
    tilt = np.radians(7.0)
    xs, ys = [], []
    for j in range(3):
        for i in range(3):
            xs.append(i * sx + j * sy * np.sin(tilt))
            ys.append(j * sy * np.cos(tilt))

    fm.set(layout_x=xs, layout_y=ys)
    return fm


def floris_farm_power(fm, phi, v, yaw_angles=None, ti=0.065):
    """Compute total farm power in FLORIS for a single condition."""
    n_turb = len(fm.layout_x)
    if yaw_angles is None:
        yaw_angles = np.zeros((1, n_turb))
    else:
        yaw_angles = np.asarray(yaw_angles).reshape(1, -1)

    fm.set(
        wind_directions=[float(phi)],
        wind_speeds=[float(v)],
        turbulence_intensities=[float(ti)],
        yaw_angles=yaw_angles,
    )
    fm.run()
    powers = fm.get_turbine_powers()
    return float(np.sum(powers)) / 1e6  # MW


def main():
    t_start = time.time()

    # Load SLSQP results
    with open(SLSQP_JSON) as f:
        slsqp_data = json.load(f)
    results = slsqp_data["results"]
    print(f"# FLORIS cross-evaluation of SLSQP-optimal yaw angles")
    print(f"# N conditions: {len(results)}")

    # Initialize FLORIS
    print("## Initializing FLORIS model...")
    fm = make_floris_model()

    # Evaluate each condition
    floris_results = []
    gb_gains = []
    floris_gains = []
    erosion_vals = []

    for idx, r in enumerate(results):
        phi, v = r["phi"], r["v"]
        slsqp_gammas = r["slsqp_gammas"]
        gb_baseline = r["baseline_mw"]
        gb_slsqp_gain = r["slsqp_gain_pct"]
        gb_slsqp_mw = r["slsqp_opt_mw"]

        # FLORIS zero-yaw baseline
        floris_base = floris_farm_power(fm, phi, v, yaw_angles=None)

        # FLORIS with SLSQP yaw
        floris_yawked = floris_farm_power(fm, phi, v, yaw_angles=slsqp_gammas)

        # FLORIS gain
        floris_gain = (floris_yawked - floris_base) / floris_base * 100.0

        # Erosion
        if gb_slsqp_gain > 0.01:
            erosion = (1 - floris_gain / gb_slsqp_gain) * 100
        else:
            erosion = None

        floris_results.append({
            "phi": phi,
            "v": v,
            "label": r["label"],
            "gb_baseline_mw": gb_baseline,
            "gb_slsqp_mw": gb_slsqp_mw,
            "gb_slsqp_gain_pct": gb_slsqp_gain,
            "floris_baseline_mw": floris_base,
            "floris_slsqp_mw": floris_yawked,
            "floris_slsqp_gain_pct": floris_gain,
            "erosion_pct": erosion,
            "slsqp_gammas": slsqp_gammas,
        })

        gb_gains.append(gb_slsqp_gain)
        floris_gains.append(floris_gain)
        if erosion is not None:
            erosion_vals.append(erosion)

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(results)} done ({time.time()-t_start:.0f}s)")

    # Aggregate
    gb_gains = np.array(gb_gains)
    floris_gains = np.array(floris_gains)

    # By regime
    regimes = {}
    for fr in floris_results:
        lab = fr["label"]
        regimes.setdefault(lab, {"gb": [], "floris": []})
        regimes[lab]["gb"].append(fr["gb_slsqp_gain_pct"])
        regimes[lab]["floris"].append(fr["floris_slsqp_gain_pct"])

    regime_summary = {}
    for lab, vals in regimes.items():
        gb_arr = np.array(vals["gb"])
        fl_arr = np.array(vals["floris"])
        regime_summary[lab] = {
            "n": len(gb_arr),
            "gb_mean_pct": float(gb_arr.mean()),
            "floris_mean_pct": float(fl_arr.mean()),
            "erosion_pct": float((1 - fl_arr.mean() / max(gb_arr.mean(), 0.01)) * 100)
                if gb_arr.mean() > 0.01 else None,
        }

    # Overall summary
    summary = {
        "description": "FLORIS cross-evaluation of SLSQP-optimal yaw angles",
        "n_conditions": len(results),
        "floris_version": __import__('floris').__version__,
        "overall_gb_slsqp_mean_pct": float(gb_gains.mean()),
        "overall_floris_slsqp_mean_pct": float(floris_gains.mean()),
        "overall_erosion_pct": float((1 - floris_gains.mean() / max(gb_gains.mean(), 0.01)) * 100)
            if gb_gains.mean() > 0.01 else None,
        "mean_erosion_where_positive_pct": float(np.mean(erosion_vals)) if erosion_vals else None,
        "per_regime": regime_summary,
        "per_condition": floris_results,
    }

    # Save
    out_path = os.path.join(FIG_DIR, "floris_slsqp_cross_eval.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"FLORIS SLSQP CROSS-EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Gray-box SLSQP mean gain:  {gb_gains.mean():+.3f}%")
    print(f"  FLORIS SLSQP mean gain:    {floris_gains.mean():+.3f}%")
    if gb_gains.mean() > 0.01:
        print(f"  Overall erosion:           {(1-floris_gains.mean()/gb_gains.mean())*100:.1f}%")
    if erosion_vals:
        print(f"  Mean erosion (where gb>0): {np.mean(erosion_vals):.1f}%")
    print(f"\n  By regime:")
    for lab, s in regime_summary.items():
        fl_s = f"{s['floris_mean_pct']:+.3f}" if s['floris_mean_pct'] is not None else "N/A"
        er_s = f"{s['erosion_pct']:.1f}" if s['erosion_pct'] is not None else "N/A"
        print(f"    {lab:25s}: gb={s['gb_mean_pct']:+.3f}%  floris={fl_s}%  erosion={er_s}%")

    # Key diagnostic: is erosion DRL-specific or global?
    aligned_cube = [fr for fr in floris_results if fr["label"] == "aligned_cube"]
    if aligned_cube:
        ac_gb = np.array([r["gb_slsqp_gain_pct"] for r in aligned_cube])
        ac_fl = np.array([r["floris_slsqp_gain_pct"] for r in aligned_cube])
        print(f"\n  Aligned-cube SLSQP (n={len(aligned_cube)}):")
        print(f"    Gray-box:  {ac_gb.mean():+.3f}%")
        print(f"    FLORIS:    {ac_fl.mean():+.3f}%")
        if ac_gb.mean() > 0.01:
            print(f"    Erosion:   {(1-ac_fl.mean()/ac_gb.mean())*100:.1f}%")
            print(f"    → Model bias is {'GLOBAL' if ac_fl.mean()/ac_gb.mean() < 0.8 else 'DRL-SPECIFIC'}")
            print(f"      (affects SLSQP equally → gray-box systematically overestimates)")

    print(f"\n  Total wall-clock: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
