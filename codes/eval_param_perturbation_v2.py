#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task M3: Lightweight parameter perturbation evaluation using NumPy environment.

Evaluates Config-E PPO policy's aligned-cube gain under ±10%, ±20%, ±30%
perturbations of alpha_star and alpha. Uses the NumPy gym environment to
avoid JAX JIT recompilation overhead.

Output: codes/param_perturbation_eval_v2.json
"""

import sys
import os
import json
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from flax import nnx

from train_3x3_nnx import ActorCritic
from windfarm_env import (
    WindFarmYawEnv, create_wind_farm_layout_3x3,
    calculate_inflow_speeds, power_output,
    alpha_star as _DEFAULT_AS, alpha as _DEFAULT_A,
    d_0, C_T, I,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Override globals in windfarm_env ----
import windfarm_env as _wfe


def set_wake_params(alpha_star, alpha):
    """Override wake params in the NumPy env module."""
    _wfe.alpha_star = alpha_star
    _wfe.alpha = alpha


def restore_wake_params():
    _wfe.alpha_star = _DEFAULT_AS
    _wfe.alpha = _DEFAULT_A


def load_policy(ckpt_path, obs_dim, act_dim):
    """Load NNX policy from checkpoint."""
    model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(0))
    graphdef, _ = nnx.split(model)
    with open(ckpt_path, "rb") as f:
        state = pickle.load(f)
    return nnx.merge(graphdef, state)


def compute_farm_power(gammas, positions, phi, v):
    """Compute total farm power for given yaw angles (NumPy)."""
    gammas_np = np.asarray(gammas, dtype=np.float64)
    inflow = calculate_inflow_speeds(
        positions, phi, C_T, I, d_0, v, gammas_np,
        _DEFAULT_AS, 0.1, _DEFAULT_A)  # Use default beta_star=0.1
    powers = np.array([power_output(u, g) for u, g in zip(inflow, gammas_np)])
    return np.sum(powers)


def evaluate_policy_gain(policy, env, n_steps=100):
    """Run policy for n_steps and return final gain (%)."""
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(n_steps):
        obs_j = jnp.asarray(obs[None, :])
        mean, _, _ = policy(obs_j)
        action = np.array(mean[0])  # deterministic
        action = np.clip(action, -10.0, 10.0)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward / 10.0  # scaled reward → gain %


def main():
    ckpt_dir = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
    ckpt_path = os.path.join(ckpt_dir, "policy_seed0_sens_act10.pkl")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Config-E checkpoint not found: {ckpt_path}")
        sys.exit(1)

    positions_list, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions_list)
    obs_dim = 3 * (3 * N_turb + 3 + 2 * N_turb)  # J=3, with positions
    act_dim = N_turb

    policy = load_policy(ckpt_path, obs_dim, act_dim)
    print(f"Loaded: {ckpt_path}")

    # Build Gym env (CPU, no JIT issues)
    env = WindFarmYawEnv(
        positions=positions_list,
        j=3, max_steps=200,
        randomize_wind=False,
        use_deficit=True,
        use_positions=True,
        act_bound=10.0,
        no_lock=False,
    )

    perturbations = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    n_conditions = 30  # reduced for speed
    seed = 20260613
    np.random.seed(seed)

    results = []

    # Test 1: perturb alpha_star
    print("\n--- Perturbing alpha* ---")
    for delta in perturbations:
        as_val = _DEFAULT_AS * (1.0 + delta)
        set_wake_params(as_val, _DEFAULT_A)
        gains = []
        for _ in range(n_conditions):
            phi = 270.0 + np.random.uniform(-15, 15)
            v = np.random.uniform(6.0, 11.4)
            env.unwrapped.specific_wind_dir = phi
            env.unwrapped.specific_wind_speed = v
            env.unwrapped.randomize_wind = False
            gain = evaluate_policy_gain(policy, env)
            gains.append(gain)
        results.append(dict(param="alpha_star", delta=delta, value=as_val,
                           mean_gain=float(np.mean(gains)),
                           std_gain=float(np.std(gains))))
        print(f"  α*={as_val:.4f} ({delta:+.0%}): gain={np.mean(gains):+.2f}% ± {np.std(gains):.2f}%")
    restore_wake_params()

    # Test 2: perturb alpha
    print("\n--- Perturbing alpha ---")
    for delta in perturbations:
        a_val = _DEFAULT_A * (1.0 + delta)
        set_wake_params(_DEFAULT_AS, a_val)
        gains = []
        for _ in range(n_conditions):
            phi = 270.0 + np.random.uniform(-15, 15)
            v = np.random.uniform(6.0, 11.4)
            env.unwrapped.specific_wind_dir = phi
            env.unwrapped.specific_wind_speed = v
            gain = evaluate_policy_gain(policy, env)
            gains.append(gain)
        results.append(dict(param="alpha", delta=delta, value=a_val,
                           mean_gain=float(np.mean(gains)),
                           std_gain=float(np.std(gains))))
        print(f"  α={a_val:.4f} ({delta:+.0%}): gain={np.mean(gains):+.2f}% ± {np.std(gains):.2f}%")
    restore_wake_params()

    # Save
    out_path = os.path.join(_SCRIPT_DIR, "param_perturbation_eval_v2.json")
    with open(out_path, "w") as f:
        json.dump(dict(results=results, n_conditions=n_conditions,
                       checkpoint=ckpt_path, params=dict(
                           alpha_star_default=_DEFAULT_AS,
                           alpha_default=_DEFAULT_A)), f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
