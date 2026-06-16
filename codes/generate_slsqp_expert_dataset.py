#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate SLSQP expert dataset for offline training (behavior cloning, regret reward).

SLSQP is the offline optimal control oracle. This script samples wind conditions
uniformly, runs multi-start SLSQP to find the optimal yaw vector for each condition,
and saves observations (from the JAX env reset) alongside SLSQP targets.

Output:
  codes/expert_datasets/slsqp_expert_3x3_seed{seed}_n{count}.npz
  codes/expert_datasets/slsqp_expert_3x3_seed{seed}_n{count}.json

Env vars:
  N_EXPERT         (default 2000)   – number of conditions to sample
  EVAL_SEED        (default 20260605)
  N_SLSQP_STARTS   (default 8)
  PHI_MIN, PHI_MAX (default 173.0, 353.0)
  V_MIN, V_MAX     (default 6.0, 16.0)
"""

import os
import sys
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

# Reuse SLSQP optimizer and NumPy physics from the existing regime comparison.
from eval_drl_vs_slsqp_regime import (
    optimize_slsqp, total_farm_power_np, N_SLSQP_STARTS,
)
from windfarm_env import create_wind_farm_layout_3x3
from windfarm_env_jax import env_reset, positions_to_jax

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, "expert_datasets")
os.makedirs(OUT_DIR, exist_ok=True)

N_EXPERT = int(os.environ.get("N_EXPERT", 2000))
EVAL_SEED = int(os.environ.get("EVAL_SEED", 20260605))
_PHI_RANGE = (float(os.environ.get("PHI_MIN", 173.0)),
              float(os.environ.get("PHI_MAX", 353.0)))
_V_RANGE = (float(os.environ.get("V_MIN", 6.0)),
            float(os.environ.get("V_MAX", 16.0)))
_N_SLSQP_STARTS = int(os.environ.get("N_SLSQP_STARTS", N_SLSQP_STARTS))
# Fraction of samples from aligned-cube (|dphi|<15°, v<11.4), where SLSQP
# headroom is highest.  0.0 = pure uniform, 0.5 = half aligned-cube.
ALIGNED_FRAC = float(os.environ.get("ALIGNED_FRAC", 0.0))


def main():
    t_start = time.time()

    positions_list, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions_list)
    positions_jax = positions_to_jax(positions_list)

    # obs_dim_per_step = gammas(N) + inflow(N) + cos/sin/v(3) + locked(N) = 3N+3
    obs_dim_per_step = 3 * N_turb + 3

    print(f"# SLSQP expert dataset generation")
    print(f"# N_EXPERT       : {N_EXPERT}")
    print(f"# EVAL_SEED      : {EVAL_SEED}")
    print(f"# N_SLSQP_STARTS : {_N_SLSQP_STARTS}")
    print(f"# phi range      : {_PHI_RANGE}")
    print(f"# v range        : {_V_RANGE}")
    print(f"# device         : {jax.devices()[0]}")
    print()

    # ---------- Sample conditions ----------
    rng = np.random.default_rng(EVAL_SEED)
    if ALIGNED_FRAC > 0:
        n_aligned = int(N_EXPERT * ALIGNED_FRAC)
        n_global = N_EXPERT - n_aligned
        # Aligned-cube: |dphi| < 15°, v < 11.4
        phis_aligned = rng.uniform(270.0 - 15.0, 270.0 + 15.0, size=n_aligned)
        vs_aligned = rng.uniform(_V_RANGE[0], 11.4, size=n_aligned)
        phis_global = rng.uniform(_PHI_RANGE[0], _PHI_RANGE[1], size=n_global)
        vs_global = rng.uniform(_V_RANGE[0], _V_RANGE[1], size=n_global)
        phis = np.concatenate([phis_aligned, phis_global])
        vs = np.concatenate([vs_aligned, vs_global])
        # Shuffle to mix aligned and global.
        perm = rng.permutation(N_EXPERT)
        phis = phis[perm]
        vs = vs[perm]
        print(f"# aligned-cube   : {n_aligned} conditions (|dphi|<15°, v<11.4)")
        print(f"# global         : {n_global} conditions (full range)")
    else:
        phis = rng.uniform(_PHI_RANGE[0], _PHI_RANGE[1], size=N_EXPERT)
        vs = rng.uniform(_V_RANGE[0], _V_RANGE[1], size=N_EXPERT)

    # ---------- Storage ----------
    obs_list = []           # (N_EXPERT, obs_dim)
    slsqp_yaw_list = []     # (N_EXPERT, N_turb)
    slsqp_power_list = []   # (N_EXPERT,)
    zero_power_list = []    # (N_EXPERT,)
    slsqp_gain_list = []    # (N_EXPERT,)
    downstream_masks = []   # (N_EXPERT, N_turb) – stored for diagnostics

    # ---------- Per-condition solve ----------
    for idx in range(N_EXPERT):
        phi, v = float(phis[idx]), float(vs[idx])

        # Zero-yaw baseline.
        base_mw = total_farm_power_np(np.zeros(N_turb), positions_list, phi, v, N_turb)

        # SLSQP optimal yaw.
        opt_mw, opt_gammas = optimize_slsqp(
            phi, v, positions_list, N_turb, n_starts=_N_SLSQP_STARTS, seed=EVAL_SEED + idx)
        gain = (opt_mw - base_mw) / base_mw * 100.0 if base_mw > 0 else 0.0

        # Observation from JAX env (j=1, deterministic wind, gammas=0).
        key = jax.random.key(EVAL_SEED + idx)
        state, obs = env_reset(key, positions_jax,
                               specific_wind_dir=jnp.float32(phi),
                               specific_wind_speed=jnp.float32(v),
                               randomize_wind=False, j=1, max_steps=200)

        obs_np = np.asarray(obs, dtype=np.float32)
        mask_np = np.asarray(state.downstream_mask, dtype=bool)

        obs_list.append(obs_np)
        slsqp_yaw_list.append(opt_gammas.astype(np.float32))
        slsqp_power_list.append(float(opt_mw))
        zero_power_list.append(float(base_mw))
        slsqp_gain_list.append(float(gain))
        downstream_masks.append(mask_np)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            gain_mean = np.mean(slsqp_gain_list[:idx+1])
            print(f"  {idx+1}/{N_EXPERT} done ({elapsed:.0f}s)  "
                  f"mean SLSQP gain so far: {gain_mean:+.3f}%")

    # ---------- Assemble & save ----------
    obs_arr = np.stack(obs_list, axis=0)            # (N, obs_dim)
    yaw_arr = np.stack(slsqp_yaw_list, axis=0)      # (N, N_turb)
    pwr_arr = np.array(slsqp_power_list, dtype=np.float32)
    z_pwr_arr = np.array(zero_power_list, dtype=np.float32)
    gain_arr = np.array(slsqp_gain_list, dtype=np.float32)
    mask_arr = np.stack(downstream_masks, axis=0)   # (N, N_turb)

    out_tag = f"slsqp_expert_3x3_seed{EVAL_SEED}_n{N_EXPERT}"
    npz_path = os.path.join(OUT_DIR, f"{out_tag}.npz")
    np.savez_compressed(
        npz_path,
        obs=obs_arr,
        phi=phis.astype(np.float32),
        v=vs.astype(np.float32),
        slsqp_yaw=yaw_arr,
        slsqp_power_mw=pwr_arr,
        zero_power_mw=z_pwr_arr,
        slsqp_gain_pct=gain_arr,
        downstream_mask=mask_arr,
    )
    print(f"\nSaved {npz_path}  ({os.path.getsize(npz_path)/1024/1024:.1f} MB)")

    # ---------- Summary JSON ----------
    summary = {
        "description": "SLSQP expert dataset for offline DRL training",
        "n_expert": N_EXPERT,
        "eval_seed": EVAL_SEED,
        "n_slsqp_starts": _N_SLSQP_STARTS,
        "n_turbines": N_turb,
        "obs_dim": obs_arr.shape[1],
        "phi_range": list(_PHI_RANGE),
        "v_range": list(_V_RANGE),
        "statistics": {
            "mean_slsqp_gain_pct": float(np.mean(gain_arr)),
            "std_slsqp_gain_pct": float(np.std(gain_arr)),
            "min_slsqp_gain_pct": float(np.min(gain_arr)),
            "max_slsqp_gain_pct": float(np.max(gain_arr)),
            "mean_abs_opt_yaw_deg": float(np.mean(np.abs(yaw_arr))),
            "max_abs_opt_yaw_deg": float(np.max(np.abs(yaw_arr))),
        },
        "wall_clock_s": float(time.time() - t_start),
    }

    json_path = os.path.join(OUT_DIR, f"{out_tag}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {json_path}")

    # ---------- Quick sanity print ----------
    print(f"\n{'='*60}")
    print(f"SLSQP Expert Dataset Summary")
    print(f"{'='*60}")
    print(f"  N conditions        : {N_EXPERT}")
    print(f"  obs shape           : {obs_arr.shape}")
    print(f"  yaw shape           : {yaw_arr.shape}")
    print(f"  mean SLSQP gain     : {np.mean(gain_arr):+.3f}%")
    print(f"  std  SLSQP gain     : {np.std(gain_arr):.3f}%")
    print(f"  min/max gain        : {np.min(gain_arr):+.3f} / {np.max(gain_arr):+.3f}%")
    print(f"  mean |yaw|          : {np.mean(np.abs(yaw_arr)):.2f}°")
    print(f"  max  |yaw|          : {np.max(np.abs(yaw_arr)):.2f}°")
    print(f"  wall-clock          : {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
