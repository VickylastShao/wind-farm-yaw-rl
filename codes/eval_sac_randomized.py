#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate trained SAC policies on the same 3000-condition protocol as PPO.

Loads SAC checkpoints, runs deterministic (mean) policy, and computes
per-condition gains using the same JAX batched evaluation pipeline.

Output:
  latex_draft/figures/sac_eval_randomized.json
"""

import os
import json
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from windfarm_env import create_wind_farm_layout_3x3
from windfarm_env_jax import (
    env_reset, env_step, inflow_speeds_jax, power_output_jax,
    find_downstream_mask_jax, positions_to_jax,
)
from cross_val_jaxenv_vs_numpyenv import SETTLE_STEPS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_sac_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_SEEDS = int(os.environ.get("N_SEEDS", 3))
PHI_RANGE = (173.0, 353.0)
V_RANGE = (6.0, 16.0)
N_CONDITIONS = int(os.environ.get("N_CONDITIONS", 3000))
EVAL_SEED = int(os.environ.get("EVAL_SEED", 20260604))
ACT_HIGH = 5.0


class MLP(nnx.Module):
    def __init__(self, din, hidden, dout, *, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(din, hidden[0], rngs=rngs)
        self.l2 = nnx.Linear(hidden[0], hidden[1], rngs=rngs)
        self.out = nnx.Linear(hidden[1], dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.tanh(self.l1(x))
        x = nnx.tanh(self.l2(x))
        return self.out(x)


class SACPolicy(nnx.Module):
    def __init__(self, obs_dim, act_dim, *, rngs: nnx.Rngs):
        self.backbone = MLP(obs_dim, (128, 128), 128, rngs=rngs)
        self.mean_head = nnx.Linear(128, act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(128, act_dim, rngs=rngs)

    def __call__(self, obs):
        h = nnx.tanh(self.backbone.l1(obs))
        h = nnx.tanh(self.backbone.l2(h))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        return mean, log_std

    def deterministic(self, obs):
        mean, _ = self(obs)
        return jnp.tanh(mean) * ACT_HIGH


def load_sac_policy(path, obs_dim, act_dim):
    """Load SAC policy from checkpoint."""
    model = SACPolicy(obs_dim, act_dim, rngs=nnx.Rngs(0))
    with open(path, "rb") as f:
        state_ck = pickle.load(f)
    nnx.update(model, state_ck)
    return model


def build_batch(positions_jax, rng):
    """Sample N_CONDITIONS (phi, v) pairs."""
    np_rng = np.random.default_rng(rng)
    phis = jnp.asarray(np_rng.uniform(PHI_RANGE[0], PHI_RANGE[1],
                                        size=N_CONDITIONS), dtype=jnp.float32)
    vs = jnp.asarray(np_rng.uniform(V_RANGE[0], V_RANGE[1],
                                      size=N_CONDITIONS), dtype=jnp.float32)

    @jax.jit
    def zero_yaw_baseline(phi, v):
        inflow_0 = inflow_speeds_jax(positions_jax, phi, v,
                                      jnp.zeros(positions_jax.shape[0]))
        baseline_mw = jnp.sum(power_output_jax(inflow_0,
                                                jnp.zeros(positions_jax.shape[0]))) / 1e6
        return baseline_mw

    baselines = jax.vmap(zero_yaw_baseline)(phis, vs)
    return phis, vs, baselines


def evaluate_batch(model, positions_jax, phis, vs, baselines, N_turb):
    """Evaluate SAC deterministic policy."""
    N_steps = SETTLE_STEPS

    @nnx.jit
    def run(m, phis, vs):
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
            action = m.deterministic(o.reshape(1, -1))
            return action.reshape(N_turb)

        @jax.vmap
        def step_one(s, a):
            return env_step(s, a, positions_jax, max_steps=N_steps + 10)

        def body(carry, _):
            states, obs = carry
            actions = predict_one(obs)
            actions = jnp.clip(actions, -5.0, 5.0)
            new_states, new_obs, _, _ = step_one(states, actions)
            return (new_states, new_obs), None

        (final_states, _), _ = jax.lax.scan(body, (states, obs_batch), None, length=N_steps)
        return final_states.total_mw, final_states.gammas

    total_mw, gammas = run(model, phis, vs)
    gains = (total_mw - baselines) / baselines * 100.0
    yaws = jnp.abs(gammas)
    return gains, yaws.max(axis=1), yaws.mean(axis=1)


def main():
    t_start = time.time()
    positions, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions)
    positions_jax = positions_to_jax(positions)

    print(f"# SAC randomized evaluation")
    print(f"# N_SEEDS={N_SEEDS}, N_CONDITIONS={N_CONDITIONS}")

    # Check if SAC checkpoints exist
    if not os.path.exists(CKPT_DIR):
        print(f"[ERROR] No SAC checkpoints found at {CKPT_DIR}")
        print(f"Run train_3x3_sac_jaxenv.py first.")
        return

    phis, vs, baselines = build_batch(positions_jax, EVAL_SEED)
    baselines_np = np.asarray(baselines)

    per_seed_gains = []
    summary = dict(n_seeds=N_SEEDS, n_conditions=N_CONDITIONS, per_seed=[])

    for s in range(N_SEEDS):
        ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_sac_n256.pkl")
        if not os.path.exists(ckpt):
            # Try alternate naming
            ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}.pkl")
        if not os.path.exists(ckpt):
            print(f"  seed {s}: checkpoint not found, skipping")
            continue

        obs_dim = 3 * N_turb + 3
        act_dim = N_turb
        model = load_sac_policy(ckpt, obs_dim, act_dim)

        gains, max_yaws, mean_yaws = evaluate_batch(
            model, positions_jax, phis, vs, baselines, N_turb)
        gains_np = np.asarray(gains)

        sg = dict(
            seed=s,
            mean_gain_pct=float(gains_np.mean()),
            std_gain_pct=float(gains_np.std()),
            median_gain_pct=float(np.median(gains_np)),
            p5_gain_pct=float(np.percentile(gains_np, 5)),
            p95_gain_pct=float(np.percentile(gains_np, 95)),
            n_positive=int((gains_np > 0).sum()),
        )
        per_seed_gains.append(gains_np)
        summary["per_seed"].append(sg)

        print(f"  seed {s}: mean={sg['mean_gain_pct']:+.3f}%  "
              f"median={sg['median_gain_pct']:+.3f}%  "
              f"p95={sg['p95_gain_pct']:+.2f}%")

    if not per_seed_gains:
        print("[ERROR] No SAC checkpoints evaluated. Exiting.")
        return

    # Across-seed aggregate
    g_stack = np.stack(per_seed_gains, axis=0)
    cond_mean = g_stack.mean(axis=0)

    phis_np = np.asarray(phis)
    vs_np = np.asarray(vs)
    dphi_arr = np.abs(((phis_np - 270.0 + 180.0) % 360.0) - 180.0)
    aligned_cube = (dphi_arr < 15.0) & (vs_np < 11.4)

    summary["marginal_mean_pct"] = float(cond_mean.mean())
    summary["aligned_cube_pct"] = float(cond_mean[aligned_cube].mean()) \
        if aligned_cube.sum() > 0 else None

    out_path = os.path.join(FIG_DIR, "sac_eval_randomized.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"  Marginal mean: {cond_mean.mean():+.3f}%")
    if aligned_cube.sum() > 0:
        print(f"  Aligned-cube:  {cond_mean[aligned_cube].mean():+.3f}%")
    print(f"  Total: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
