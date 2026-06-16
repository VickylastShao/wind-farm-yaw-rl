#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task C15: Evaluate trained DRL policy under wake-model parameter perturbation.

Perturbs alpha_star and alpha (the two most influential wake model
parameters) by +/-10%, +/-20%, +/-30% and re-evaluates the trained
Config-E PPO policy's aligned-cube gain.

This quantifies how sensitive the policy's *gain* (not just baseline
power) is to calibration uncertainty, addressing Reviewer 2's concern
that the FLORIS 9.1% deviation may have affected training.

Output:
  codes/param_perturbation_eval.json
"""

import sys
import os
import json
import pickle
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from flax import nnx

from train_3x3_nnx import ActorCritic, MLP, NET_ARCH
from windfarm_env_jax import (
    env_reset_batched, env_step_autoreset, positions_to_jax,
)
from windfarm_env import create_wind_farm_layout_3x3

# ---- Constants (must match windfarm_env.py) ----
# These are the default calibrated values
DEFAULT_ALPHA_STAR = 2.727612115052532
DEFAULT_BETA_STAR  = 0.1
DEFAULT_ALPHA      = 0.539933732451907
DEFAULT_I          = 0.065

# Globals that windfarm_env_jax reads; we override these for perturbation
import windfarm_env_jax as _wfj

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Load trained Config-E policy ----
def load_policy(ckpt_path, obs_dim, act_dim):
    """Load a Config-E PPO policy from checkpoint."""
    model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(0))
    graphdef, _ = nnx.split(model)
    with open(ckpt_path, "rb") as f:
        state = pickle.load(f)
    return nnx.merge(graphdef, state)


def evaluate_policy(policy, positions_j, n_conditions=50, seed=20260612):
    """Evaluate the policy on n_conditions aligned-cube conditions.

    Returns per-condition gain (%).
    """
    N_turb = positions_j.shape[0]
    obs_dim_per_step = 3 * N_turb + 3 + 2 * N_turb  # with positions
    J = 3
    obs_dim = J * obs_dim_per_step
    N_ENVS = 1  # single env for evaluation

    key = jax.random.PRNGKey(seed)
    gains = []

    for i in range(n_conditions):
        key, rk = jax.random.split(key)
        # Sample aligned-cube condition
        phi = 270.0 + float(np.random.uniform(-15, 15))
        v = float(np.random.uniform(6.0, 11.4))

        # Reset env with specific wind (JAX scalar arrays)
        reset_key = jax.random.split(rk, N_ENVS)
        state, obs = env_reset_batched(
            reset_key, positions_j,
            j=J, max_steps=200,
            randomize_wind=False,
            specific_wind_dir=jnp.array(phi),
            specific_wind_speed=jnp.array(v),
        )

        # Run policy for 100 steps to converge
        total_reward = 0.0
        for _ in range(100):
            key, sk = jax.random.split(key)
            mean, log_std, _ = policy(obs)
            action = mean  # deterministic (no exploration for eval)
            action = jnp.clip(action, -10.0, 10.0)
            reset_keys = jax.random.split(sk, N_ENVS)
            state, obs, reward, done = env_step_autoreset(
                state, action, reset_keys, positions_j,
                j=J, max_steps=200, randomize_wind=False,
            )
            total_reward += float(reward[0])

        # Gain = total_reward / 10.0 (per the reward scaling in windfarm_env_jax)
        gain_pct = total_reward / 10.0
        gains.append(gain_pct)

    return np.array(gains)


def set_wake_params(alpha_star, alpha):
    """Override wake model parameters in the JAX environment module."""
    _wfj.DEFAULT_ALPHA_STAR = alpha_star
    _wfj.DEFAULT_ALPHA = alpha
    # Also update the module-level constants in windfarm_env_jax
    import windfarm_env_jax
    windfarm_env_jax.alpha_star_default = alpha_star
    windfarm_env_jax.alpha_default = alpha


def restore_wake_params():
    """Restore default wake model parameters."""
    import windfarm_env_jax
    windfarm_env_jax.alpha_star_default = DEFAULT_ALPHA_STAR
    windfarm_env_jax.alpha_default = DEFAULT_ALPHA


def main():
    positions_list, _, _ = create_wind_farm_layout_3x3()
    positions_j = positions_to_jax(positions_list)
    N_turb = positions_j.shape[0]
    obs_dim_per_step = 3 * N_turb + 3 + 2 * N_turb
    obs_dim = 3 * obs_dim_per_step
    act_dim = N_turb

    # Load Config-E policy — path from env var CKPT_PATH or auto-discover
    ckpt_dir = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
    ckpt_path = os.environ.get("CKPT_PATH", "")
    if not ckpt_path or not os.path.exists(ckpt_path):
        # Auto-discover: try sens_act10 (Config-E) seeds 0→4
        for s in range(5):
            for tag in ["sens_act10", "full60m"]:
                p = os.path.join(ckpt_dir, f"policy_seed{s}_{tag}.pkl")
                if os.path.exists(p):
                    ckpt_path = p
                    break
            if ckpt_path:
                break
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("ERROR: No Config-E checkpoint found. Set CKPT_PATH env var.")
        print("Looked for: sens_act10, full60m in", ckpt_dir)
        sys.exit(1)

    policy = load_policy(ckpt_path, obs_dim, act_dim)
    print(f"Loaded policy from {ckpt_path}")

    # Evaluate under perturbations
    perturbations = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    n_conditions = 50
    results = []

    print(f"\n{'='*60}")
    print(f"Parameter perturbation evaluation (n={n_conditions} conditions each)")
    print(f"{'='*60}")

    # Also test alpha and alpha_star separately
    # Test 1: perturb alpha_star, alpha fixed at default
    print("\n--- Perturbing alpha* (alpha fixed) ---")
    for delta in perturbations:
        as_val = DEFAULT_ALPHA_STAR * (1.0 + delta)
        set_wake_params(as_val, DEFAULT_ALPHA)
        gains = evaluate_policy(policy, positions_j, n_conditions=n_conditions)
        results.append(dict(
            param="alpha_star", delta=delta, value=as_val,
            mean_gain=float(np.mean(gains)),
            std_gain=float(np.std(gains)),
            p5=float(np.percentile(gains, 5)),
            p95=float(np.percentile(gains, 95)),
        ))
        print(f"  alpha*={as_val:.4f} ({delta:+.0%}): "
              f"gain={np.mean(gains):+.2f}% +/- {np.std(gains):.2f}%")

    # Test 2: perturb alpha, alpha_star fixed at default
    print("\n--- Perturbing alpha (alpha* fixed) ---")
    for delta in perturbations:
        a_val = DEFAULT_ALPHA * (1.0 + delta)
        set_wake_params(DEFAULT_ALPHA_STAR, a_val)
        gains = evaluate_policy(policy, positions_j, n_conditions=n_conditions)
        results.append(dict(
            param="alpha", delta=delta, value=a_val,
            mean_gain=float(np.mean(gains)),
            std_gain=float(np.std(gains)),
            p5=float(np.percentile(gains, 5)),
            p95=float(np.percentile(gains, 95)),
        ))
        print(f"  alpha={a_val:.4f} ({delta:+.0%}): "
              f"gain={np.mean(gains):+.2f}% +/- {np.std(gains):.2f}%")

    # Restore defaults
    restore_wake_params()

    # Save results
    out_path = os.path.join(_SCRIPT_DIR, "param_perturbation_eval.json")
    with open(out_path, "w") as f:
        json.dump(dict(results=results, n_conditions=n_conditions,
                       checkpoint=ckpt_path), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
