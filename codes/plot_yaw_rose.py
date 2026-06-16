#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-condition yaw angle visualization across the wind rose.

Generates a figure showing the learned yaw angles as a function of wind
direction and speed, for the 3x3 layout, seed-0 policy.

Output:
  latex_draft/figures/fig_yaw_rose.{pdf,jpg}
"""

import os, json
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx

from windfarm_env import create_wind_farm_layout_3x3
from windfarm_env_jax import (
    env_reset, env_step, positions_to_jax,
    find_downstream_mask_jax,
)
from train_3x3_nnx import ActorCritic
from cross_val_jaxenv_vs_numpyenv import load_nnx_policy, SETTLE_STEPS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PHI_RANGE = (173.0, 353.0)
V_RANGE = (6.0, 16.0)


def main():
    positions, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions)
    positions_jax = positions_to_jax(positions)

    # Create a fine wind-rose grid for visualization
    phis = jnp.arange(PHI_RANGE[0], PHI_RANGE[1] + 1, 5.0, dtype=jnp.float32)
    vs = jnp.array([8.0, 11.4], dtype=jnp.float32)  # low and rated speed

    # Load seed-0 policy
    obs_dim = 3 * N_turb + 3
    act_dim = N_turb
    ckpt = os.path.join(CKPT_DIR, "policy_seed0_p0c.pkl")
    model = load_nnx_policy(ckpt, obs_dim, act_dim)

    # For each wind speed, compute yaw angles across all directions
    all_yaws = {}  # {v: (n_phi, N_turb)}
    all_masks = {}

    for v_val in vs:
        yaws_list = []
        masks_list = []
        for phi_val in phis:
            key = jax.random.key(0)
            state, obs = env_reset(key, positions_jax,
                                    specific_wind_dir=float(phi_val),
                                    specific_wind_speed=float(v_val),
                                    randomize_wind=False,
                                    max_steps=SETTLE_STEPS + 10)

            for step in range(SETTLE_STEPS):
                mean, _, _ = model(obs.reshape(1, -1))
                action = jnp.clip(mean.reshape(N_turb), -5.0, 5.0)
                state, obs, reward, done = env_step(state, action, positions_jax,
                                                     max_steps=SETTLE_STEPS + 10)

            yaws_list.append(np.asarray(state.gammas))
            masks_list.append(np.asarray(state.downstream_mask))

        all_yaws[float(v_val)] = np.stack(yaws_list)  # (n_phi, N_turb)
        all_masks[float(v_val)] = np.stack(masks_list)

    # Create figure: 2 rows (v=8, v=11.4), 1 column (yaw angles vs phi)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    turb_colors = plt.cm.tab10(np.arange(N_turb))

    for row_idx, (v_val, ax) in enumerate(zip(vs, axes)):
        yaws = all_yaws[float(v_val)]
        masks = all_masks[float(v_val)]
        phis_np = np.asarray(phis)

        for t in range(N_turb):
            yaw_t = yaws[:, t]
            mask_t = masks[:, t]
            # Plot yaw angles, marking locked turbines with dashed line
            is_locked = mask_t > 0.5
            ax.plot(phis_np, yaw_t, '-', color=turb_colors[t], lw=1.2,
                    label=f'T{t}' if row_idx == 0 else None, alpha=0.8)
            # Mark locked conditions with a dot
            if is_locked.any():
                ax.scatter(phis_np[is_locked], yaw_t[is_locked],
                          marker='x', s=30, color=turb_colors[t], alpha=0.5, zorder=5)

        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(270, color='red', lw=0.8, ls=':', alpha=0.5, label='$\\phi=270°$ (aligned)')
        ax.set_ylabel(f'Yaw angle [°]\n($v={float(v_val):.1f}$ m/s)')
        ax.grid(alpha=0.3)
        if row_idx == 0:
            ax.legend(fontsize=7, ncol=5, loc='upper left', framealpha=0.8)

    axes[-1].set_xlabel('Wind direction $\\phi$ [°] (meteorological)')
    axes[0].set_title('Learned yaw angles across the wind rose (seed 0, $3\\times3$ layout)')

    # Add shaded region for aligned band
    for ax in axes:
        ax.axvspan(255, 285, alpha=0.08, color='red', label=None)
        ax.text(270, ax.get_ylim()[1] * 0.85, 'aligned', ha='center',
                fontsize=8, color='red', alpha=0.6)

    fig.tight_layout()

    # Save
    for ext in ['pdf', 'jpg']:
        path = os.path.join(FIG_DIR, f'fig_yaw_rose.{ext}')
        fig.savefig(path, dpi=300 if ext == 'jpg' else None, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
