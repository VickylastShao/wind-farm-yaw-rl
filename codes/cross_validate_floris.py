# -*- coding: utf-8 -*-
"""
Cross-validation of the proposed gray-box wake model against FLORIS (NREL)
on the Horns Rev I 8x10 offshore layout (Vestas V-80).

Two checks are performed:
  (1) Power vs wind direction sweep, 173 - 353 deg, V_inf = 8 m/s, no yaw.
      Curves: this work (calibrated Bastankhah--Porte-Agel), FLORIS Gauss,
              and (optionally) digitized LES reference if present in
              figures/hornsrev_les_reference.csv.
  (2) Two-turbine yaw sweep, gamma_1 in [0, 50], to check yaw-deflection
      consistency on the calibration setting used in Stage I.

Outputs:
  figures/fig_floris_hornsrev_compare.{pdf,jpg}
  figures/fig_floris_yaw_sweep.{pdf,jpg}
  figures/floris_validation_stats.json
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from windfarm_env import (
    calculate_inflow_speeds,
    power_output,
)


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "latex_draft", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# ----- Vestas V-80 parameters used by debug_eval.py / Horns Rev case -----
D_V80 = 80.0
ZH_V80 = 70.0
P_RATED_V80 = 2.0e6
U_INF_V80 = 8.0
I_V80 = 0.077
ALPHA_STAR_V80 = 2.32
BETA_STAR_V80 = 0.154
ALPHA_V80 = 0.8
C_T_V80 = 0.8


def horns_rev_layout():
    """8 columns x 10 rows, 7D spacing, ~6 deg lattice tilt (standard convention)."""
    n_cols, n_rows = 8, 10
    sx, sy = 7.0 * D_V80, 7.0 * D_V80
    tilt = np.radians(6.0)
    positions = []
    for j in range(n_rows):
        for i in range(n_cols):
            x = i * sx + j * sy * np.sin(tilt)
            y = j * sy * np.cos(tilt)
            positions.append((x, y, ZH_V80))
    return positions


def proposed_farm_power(positions, wind_dir_meteo, gammas=None):
    if gammas is None:
        gammas = np.zeros(len(positions))
    u = calculate_inflow_speeds(
        positions, wind_dir_meteo, C_T_V80, I_V80, D_V80, U_INF_V80,
        gammas, ALPHA_STAR_V80, BETA_STAR_V80, ALPHA_V80
    )
    p = np.array([power_output(ui, gi) for ui, gi in zip(u, gammas)])
    return p.sum(), p


def proposed_two_turbine_yaw(gamma1, gamma2=0.0, spacing=7 * D_V80):
    positions = [(0.0, 0.0, ZH_V80), (spacing, 0.0, ZH_V80)]
    u = calculate_inflow_speeds(
        positions, 270.0, C_T_V80, I_V80, D_V80, U_INF_V80,
        np.array([gamma1, gamma2]),
        ALPHA_STAR_V80, BETA_STAR_V80, ALPHA_V80,
    )
    return power_output(u[0], gamma1) + power_output(u[1], gamma2), u


def floris_horns_rev_sweep(directions):
    try:
        from floris import FlorisModel
    except Exception as exc:
        print(f"[floris] unavailable: {exc}. Skipping FLORIS curve.")
        return None
    try:
        fm = FlorisModel("gch.yaml")
    except Exception:
        fm = FlorisModel("defaults")

    positions = horns_rev_layout()
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    fm.set(layout_x=xs, layout_y=ys)

    out = np.empty(len(directions))
    for k, phi in enumerate(directions):
        fm.set(wind_directions=[float(phi)], wind_speeds=[U_INF_V80],
               turbulence_intensities=[I_V80])
        fm.run()
        powers = fm.get_turbine_powers()
        out[k] = float(np.sum(powers))
    return out


def floris_two_turbine_yaw(gammas):
    try:
        from floris import FlorisModel
    except Exception as exc:
        print(f"[floris] unavailable: {exc}. Skipping FLORIS yaw sweep.")
        return None
    try:
        fm = FlorisModel("gch.yaml")
    except Exception:
        fm = FlorisModel("defaults")
    fm.set(layout_x=[0.0, 7 * D_V80], layout_y=[0.0, 0.0])

    out = np.empty(len(gammas))
    for k, g in enumerate(gammas):
        fm.set(wind_directions=[270.0], wind_speeds=[U_INF_V80],
               turbulence_intensities=[I_V80],
               yaw_angles=np.array([[float(g), 0.0]]))
        fm.run()
        out[k] = float(np.sum(fm.get_turbine_powers()))
    return out


def maybe_load_les_reference():
    csv = os.path.join(OUT_DIR, "hornsrev_les_reference.csv")
    if not os.path.exists(csv):
        return None
    try:
        data = np.loadtxt(csv, delimiter=",", skiprows=1)
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"[LES] failed to load reference: {e}")
        return None


def main():
    print("# Cross-validation against FLORIS on Horns Rev I")

    # --- (1) wind-direction sweep ---
    directions = np.arange(173.0, 353.1, 2.0)
    positions = horns_rev_layout()

    proposed_p = np.empty(len(directions))
    for k, phi in enumerate(directions):
        ptot, _ = proposed_farm_power(positions, float(phi))
        proposed_p[k] = ptot

    floris_p = floris_horns_rev_sweep(directions)
    les = maybe_load_les_reference()

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    proposed_norm = proposed_p / proposed_p.max()
    ax.plot(directions, proposed_norm, "b-", lw=1.6, label="This work (gray-box)")
    if floris_p is not None:
        floris_norm = floris_p / floris_p.max()
        ax.plot(directions, floris_norm, "g--", lw=1.4, label="FLORIS (Gauss)")
        rmse = float(np.sqrt(np.mean((proposed_norm - floris_norm) ** 2)))
        rel = float(np.mean(np.abs(proposed_norm - floris_norm)) * 100)
        print(f"  vs FLORIS:  RMSE = {rmse:.4f}   mean abs diff = {rel:.2f} %")
    if les is not None:
        ax.plot(les[0], les[1], "k.", ms=3, label="LES reference")
    ax.set_xlabel(r"Wind direction $\phi$ [deg]")
    ax.set_ylabel("Normalized farm power")
    ax.set_title("Horns Rev I, $U_\\infty = 8$ m/s, $I = 7.7\\%$")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_floris_hornsrev_compare.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig_floris_hornsrev_compare.jpg"),
                dpi=200, bbox_inches="tight")

    # --- (2) two-turbine yaw sweep ---
    gammas = np.arange(0.0, 50.1, 2.5)
    proposed_y = np.empty(len(gammas))
    for k, g in enumerate(gammas):
        ptot, _ = proposed_two_turbine_yaw(float(g))
        proposed_y[k] = ptot
    proposed_y_norm = proposed_y / proposed_y[0]

    floris_y = floris_two_turbine_yaw(gammas)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(gammas, proposed_y_norm, "b-o", ms=3.5, lw=1.4, label="This work (gray-box)")
    if floris_y is not None:
        floris_y_norm = floris_y / floris_y[0]
        ax.plot(gammas, floris_y_norm, "g--s", ms=3.5, lw=1.4, label="FLORIS (Gauss)")
        rmse_y = float(np.sqrt(np.mean((proposed_y_norm - floris_y_norm) ** 2)))
        print(f"  yaw sweep RMSE vs FLORIS = {rmse_y:.4f}")
    ax.set_xlabel(r"Upstream yaw $\gamma_1$ [deg]")
    ax.set_ylabel(r"Normalized two-turbine power $P / P_{\gamma=0}$")
    ax.set_title("Two V-80, 7D spacing, $U_\\infty = 8$ m/s")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_floris_yaw_sweep.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig_floris_yaw_sweep.jpg"), dpi=200,
                bbox_inches="tight")

    stats = dict(
        directions=directions.tolist(),
        proposed_power=proposed_p.tolist(),
        floris_power=(floris_p.tolist() if floris_p is not None else None),
        gammas=gammas.tolist(),
        proposed_two_turbine=proposed_y.tolist(),
        floris_two_turbine=(floris_y.tolist() if floris_y is not None else None),
    )
    with open(os.path.join(OUT_DIR, "floris_validation_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved figures and stats to {OUT_DIR}")


if __name__ == "__main__":
    main()
