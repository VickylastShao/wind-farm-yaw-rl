#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observation noise robustness evaluation.

Quantifies how sensor noise in wind direction and speed observations
erodes DRL policy performance.  The policy receives noisy observations
but is evaluated against the true (noise-free) environment dynamics.

Noise model:
  - phi_observed = phi_true + N(0, sigma_phi^2)
  - v_observed   = v_true + N(0, sigma_v^2)
  - cos(phi), sin(phi) computed from phi_observed (not perturbed independently)
  - gammas and inflow: NOT perturbed (SCADA-direct measurement assumption)

Sweep:
  sigma_phi in {0, 1, 2, 3, 5, 7, 10} degrees
  sigma_v   in {0, 0.5, 1.0, 2.0} m/s

Output:
  latex_draft/figures/obs_noise_robustness.json
  latex_draft/figures/fig_obs_noise_robustness.{pdf,jpg}
"""

import os
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx

from windfarm_env import create_wind_farm_layout_3x3
from windfarm_env_jax import (
    env_reset, env_step, positions_to_jax,
)
from train_3x3_nnx import ActorCritic
from cross_val_jaxenv_vs_numpyenv import load_nnx_policy, SETTLE_STEPS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_SEEDS = int(os.environ.get("N_SEEDS", 5))
PHI_RANGE = (173.0, 353.0)
V_RANGE = (6.0, 16.0)
N_CONDITIONS = int(os.environ.get("N_CONDITIONS", 3000))
EVAL_SEED = int(os.environ.get("EVAL_SEED", 20260604))
NOISE_SEED = int(os.environ.get("NOISE_SEED", 999))

SIGMA_PHI_VALUES = [0, 1, 2, 3, 5, 7, 10]   # degrees
SIGMA_V_VALUES = [0, 0.5, 1.0, 2.0]           # m/s


def evaluate_noisy(model, positions_jax, phis, vs, baselines, N_turb,
                   sigma_phi, sigma_v, noise_key):
    """Evaluate policy with noisy observations.

    The environment runs with TRUE (phi, v), but the policy receives
    noisy (phi_obs, v_obs) in the observation vector.
    """
    N_steps = SETTLE_STEPS
    B = phis.shape[0]

    # Pre-generate all noise: shape (N_steps, B)
    # We add noise to phi and v in the observation each step.
    noise_keys = jax.random.split(noise_key, N_steps + 1)
    phi_noise_all = jax.random.normal(noise_keys[0], (N_steps, B)) * sigma_phi
    v_noise_all = jax.random.normal(noise_keys[0], (N_steps, B)) * sigma_v

    @nnx.jit
    def run(m, phis, vs, phi_noise_all, v_noise_all):
        @jax.vmap
        def reset_one(phi, v):
            key = jax.random.key(0)
            state, obs = env_reset(key, positions_jax,
                                    specific_wind_dir=phi,
                                    specific_wind_speed=v,
                                    randomize_wind=False,
                                    max_steps=N_steps + 10)
            return state, obs

        states, obs_batch = reset_one(phis, vs)

        @jax.vmap
        def predict_one(o):
            mean, _, _ = m(o.reshape(1, -1))
            return mean.reshape(N_turb)

        @jax.vmap
        def step_one(s, a):
            return env_step(s, a, positions_jax, max_steps=N_steps + 10)

        def body(carry, noise_idx):
            states, obs = carry
            # Inject noise into the observation
            # obs layout: [gammas(N), inflow(N), cos(phi), sin(phi), v, locked(N)]
            # We perturb cos(phi), sin(phi), and v entries
            N = N_turb
            obs_noisy = obs

            if sigma_phi > 0:
                # phi_noise is per-condition (batch), broadcast across obs rows
                pn = phi_noise_all[noise_idx]  # (B,)
                # Reconstruct noisy phi from cos/sin, add noise, re-encode
                # Actually easier: just perturb cos and sin
                # phi_obs = phi_true + noise → cos(phi_obs), sin(phi_obs)
                # But we don't have phi_true in obs directly.
                # Instead: perturb cos/sin as if phi shifted
                cos_obs = obs_noisy[:, N + N]       # (B,)
                sin_obs = obs_noisy[:, N + N + 1]   # (B,)
                phi_true = jnp.arctan2(sin_obs, cos_obs)
                phi_noisy = phi_true + jnp.radians(pn)
                obs_noisy = obs_noisy.at[:, N + N].set(jnp.cos(phi_noisy))
                obs_noisy = obs_noisy.at[:, N + N + 1].set(jnp.sin(phi_noisy))

            if sigma_v > 0:
                vn = v_noise_all[noise_idx]  # (B,)
                v_obs = obs_noisy[:, N + N + 2]  # (B,)
                obs_noisy = obs_noisy.at[:, N + N + 2].set(v_obs + vn)

            actions = predict_one(obs_noisy)
            actions = jnp.clip(actions, -5.0, 5.0)
            new_states, new_obs, _, _ = step_one(states, actions)
            return (new_states, new_obs), None

        (final_states, _), _ = jax.lax.scan(
            body, (states, obs_batch), jnp.arange(N_steps))
        return final_states.total_mw

    total_mw = run(model, phis, vs, phi_noise_all, v_noise_all)
    gains = (total_mw - baselines) / baselines * 100.0
    return gains


def main():
    t_start = time.time()
    positions, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions)
    positions_jax = positions_to_jax(positions)

    print(f"# Observation noise robustness evaluation")
    print(f"# N_SEEDS={N_SEEDS}, N_CONDITIONS={N_CONDITIONS}")
    print(f"# sigma_phi ∈ {SIGMA_PHI_VALUES} deg")
    print(f"# sigma_v   ∈ {SIGMA_V_VALUES} m/s")
    print(f"# device={jax.devices()[0]}")

    # Build condition batch (same as eval_p0c_randomized)
    from eval_p0c_randomized import build_batch
    phis, vs, baselines = build_batch(positions_jax, EVAL_SEED)
    baselines_np = np.asarray(baselines)
    phis_np = np.asarray(phis)
    vs_np = np.asarray(vs)

    # Regime masks
    dphi_arr = np.abs(((phis_np - 270.0 + 180.0) % 360.0) - 180.0)
    aligned_cube = (dphi_arr < 15.0) & (vs_np < 11.4)

    # Run noise sweep
    results = {}

    for sigma_phi in SIGMA_PHI_VALUES:
        for sigma_v in SIGMA_V_VALUES:
            key_str = f"sphi_{sigma_phi}_sv_{sigma_v}"
            print(f"\n## sigma_phi={sigma_phi}°, sigma_v={sigma_v} m/s")

            seed_gains = []
            for s in range(N_SEEDS):
                ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_p0c.pkl")
                obs_dim = 3 * N_turb + 3
                act_dim = N_turb
                model = load_nnx_policy(ckpt, obs_dim, act_dim)

                noise_key = jax.random.key(NOISE_SEED + s)
                gains = evaluate_noisy(model, positions_jax, phis, vs, baselines,
                                       N_turb, sigma_phi, sigma_v, noise_key)
                seed_gains.append(np.asarray(gains))

            gains_stack = np.stack(seed_gains, axis=0)
            cond_mean = gains_stack.mean(axis=0)

            result = {
                "sigma_phi_deg": sigma_phi,
                "sigma_v_mps": sigma_v,
                "marginal_mean_pct": float(cond_mean.mean()),
                "aligned_cube_pct": float(cond_mean[aligned_cube].mean())
                    if aligned_cube.sum() > 0 else None,
                "per_seed_means": [float(sg.mean()) for sg in seed_gains],
            }
            results[key_str] = result
            print(f"  marginal={result['marginal_mean_pct']:+.3f}%  "
                  f"aligned-cube={result['aligned_cube_pct']:+.3f}%"
                  if result['aligned_cube_pct'] is not None else
                  f"  marginal={result['marginal_mean_pct']:+.3f}%")

    # Save
    out_path = os.path.join(FIG_DIR, "obs_noise_robustness.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Gain vs sigma_phi (sigma_v=0)
    ax = axes[0]
    sp_vals = SIGMA_PHI_VALUES
    marginal_y = [results[f"sphi_{sp}_sv_0"]["marginal_mean_pct"] for sp in sp_vals]
    aligned_y = [results[f"sphi_{sp}_sv_0"]["aligned_cube_pct"] for sp in sp_vals]
    ax.plot(sp_vals, marginal_y, 'b-o', ms=6, lw=1.5, label="Marginal mean")
    ax.plot(sp_vals, aligned_y, 'r-s', ms=6, lw=1.5, label="Aligned-cube")
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel(r"$\sigma_\phi$ [deg]")
    ax.set_ylabel("Mean gain [%]")
    ax.set_title("(a) Gain vs wind-dir noise")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Gain vs sigma_v (sigma_phi=0)
    ax = axes[1]
    sv_vals = SIGMA_V_VALUES
    marginal_y = [results[f"sphi_0_sv_{sv}"]["marginal_mean_pct"] for sv in sv_vals]
    aligned_y = [results[f"sphi_0_sv_{sv}"]["aligned_cube_pct"] for sv in sv_vals]
    ax.plot(sv_vals, marginal_y, 'b-o', ms=6, lw=1.5, label="Marginal mean")
    ax.plot(sv_vals, aligned_y, 'r-s', ms=6, lw=1.5, label="Aligned-cube")
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel(r"$\sigma_v$ [m/s]")
    ax.set_ylabel("Mean gain [%]")
    ax.set_title("(b) Gain vs wind-speed noise")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # (c) 2D heatmap: gain as function of (sigma_phi, sigma_v)
    ax = axes[2]
    grid = np.zeros((len(SIGMA_V_VALUES), len(SIGMA_PHI_VALUES)))
    for i, sv in enumerate(SIGMA_V_VALUES):
        for j, sp in enumerate(SIGMA_PHI_VALUES):
            grid[i, j] = results[f"sphi_{sp}_sv_{sv}"]["aligned_cube_pct"] \
                if results[f"sphi_{sp}_sv_{sv}"]["aligned_cube_pct"] is not None \
                else results[f"sphi_{sp}_sv_{sv}"]["marginal_mean_pct"]
    im = ax.imshow(grid, aspect='auto', origin='lower',
                   extent=[SIGMA_PHI_VALUES[0]-0.5, SIGMA_PHI_VALUES[-1]+0.5,
                           SIGMA_V_VALUES[0]-0.25, SIGMA_V_VALUES[-1]+0.25],
                   cmap='RdYlGn')
    ax.set_xlabel(r"$\sigma_\phi$ [deg]")
    ax.set_ylabel(r"$\sigma_v$ [m/s]")
    ax.set_title("(c) Aligned-cube gain")
    fig.colorbar(im, ax=ax, label="Gain [%]")

    fig.suptitle("Observation noise robustness of DRL yaw controller", fontsize=11)
    fig.tight_layout()
    for ext in ['pdf', 'jpg']:
        path = os.path.join(FIG_DIR, f"fig_obs_noise_robustness.{ext}")
        fig.savefig(path, dpi=300 if ext == 'jpg' else None, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)

    print(f"\nTotal wall-clock: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
