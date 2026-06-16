#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavior Cloning (BC) from SLSQP expert dataset.

Trains the ActorCritic policy's mean head to mimic SLSQP optimal yaw targets.
The target action is an incremental yaw command:
    target = clip(slsqp_yaw - current_gammas, -5°, +5°)

At reset time current_gammas=0, so the first-step target is simply
clip(slsqp_yaw, -5°, +5°). The dataset optionally synthesizes intermediate
yaw states (alpha * slsqp_yaw + noise) to improve robustness.

The saved checkpoint is structurally identical to a PPO ActorCritic checkpoint
and can be loaded by load_nnx_policy() for BC+PPO warm-start.

Output (in checkpoints_3x3_nnx_jaxenv/):
  policy_seed{seed}_bc.pkl
  metrics_seed{seed}_bc.json
  summary_bc.json

Env vars:
  DATASET          (required)  – path to .npz expert dataset
  N_SEEDS          (default 3)
  SEED_START       (default 0)
  BC_EPOCHS        (default 100)
  BATCH_SIZE       (default 1024)
  LEARNING_RATE    (default 3e-4)
  OUT_TAG          (default "bc")
  VAL_FRAC         (default 0.1)
  SYNTH_STATES     (default 3)  – number of synthetic intermediate states per expert point
"""

import os
import json
import time
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from train_3x3_nnx import ActorCritic, NET_ARCH, MLP
from windfarm_env_jax import inflow_speeds_jax, positions_to_jax
from windfarm_env import create_wind_farm_layout_3x3

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
os.makedirs(CKPT_DIR, exist_ok=True)

DATASET = os.environ.get("DATASET", "")
N_SEEDS = int(os.environ.get("N_SEEDS", 3))
SEED_START = int(os.environ.get("SEED_START", 0))
BC_EPOCHS = int(os.environ.get("BC_EPOCHS", 100))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1024))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 3e-4))
OUT_TAG = os.environ.get("OUT_TAG", "bc")
VAL_FRAC = float(os.environ.get("VAL_FRAC", 0.1))
SYNTH_STATES = int(os.environ.get("SYNTH_STATES", 3))
# BC_MODE: "inc" = incremental actions (for BC+PPO warm-start),
#          "abs" = absolute yaw targets (standalone BC, needs eval conversion).
BC_MODE = os.environ.get("BC_MODE", "abs")

# ---------------------------------------------------------------------------
# Data loading + synthetic intermediate-state generation (NumPy, host-side).
# ---------------------------------------------------------------------------
def load_expert_data(npz_path: str):
    """Load expert dataset; return dict of numpy arrays."""
    data = np.load(npz_path)
    N = data["obs"].shape[0]
    print(f"  Loaded {N} expert conditions from {npz_path}")
    print(f"    obs shape     : {data['obs'].shape}")
    print(f"    yaw shape     : {data['slsqp_yaw'].shape}")
    print(f"    mean SLSQP gain: {float(data['slsqp_gain_pct'].mean()):+.3f}%")
    return data


def build_bc_samples(data: dict, rng: np.random.Generator):
    """Build BC training samples from expert data.

    Trains the policy to output ABSOLUTE yaw targets (the SLSQP optimum).
    During evaluation, the rollout converts absolute targets to incremental
    actions via: action = clip(target_yaw - current_gammas, -5, 5).

    This avoids the distribution-shift problem of incremental-action BC,
    where multi-step consistency is hard to learn from static data.
    """
    N_expert = data["obs"].shape[0]
    N_turb = data["slsqp_yaw"].shape[1]
    obs_dim = data["obs"].shape[1]

    # Pre-compute positions for JAX inflow.
    positions_list, _, _ = create_wind_farm_layout_3x3()
    positions_j = positions_to_jax(positions_list)

    n_per_expert = 1 + SYNTH_STATES
    total_samples = N_expert * n_per_expert

    all_obs = np.zeros((total_samples, obs_dim), dtype=np.float32)
    all_targets = np.zeros((total_samples, N_turb), dtype=np.float32)

    for i in range(N_expert):
        orig_obs = data["obs"][i].copy()
        slsqp_yaw = data["slsqp_yaw"][i]
        phi = float(data["phi"][i])
        v = float(data["v"][i])

        # Sample 0: gammas = 0 (original reset state).
        # Convert inflow to wake deficit: v - inflow.
        base_idx = i * n_per_expert
        obs0 = orig_obs.copy()
        obs0[N_turb:2*N_turb] = v - obs0[N_turb:2*N_turb]  # deficit
        all_obs[base_idx] = obs0
        if BC_MODE == "inc":
            all_targets[base_idx] = np.clip(slsqp_yaw, -5.0, 5.0).astype(np.float32)
        else:
            all_targets[base_idx] = slsqp_yaw.astype(np.float32)

        if SYNTH_STATES == 0:
            continue

        # Synthetic intermediate states with correct inflow.
        synth_gammas_list = []
        for k in range(SYNTH_STATES):
            alpha = rng.uniform(0.0, 1.0)
            noise = rng.uniform(-2.0, 2.0, size=N_turb)
            sg = np.clip(alpha * slsqp_yaw + noise, -50.0, 50.0)
            synth_gammas_list.append(sg)

        synth_gammas_arr = np.stack(synth_gammas_list, axis=0)

        # Batch-compute correct inflow via JAX.
        sg_j = jnp.asarray(synth_gammas_arr, dtype=jnp.float32)
        phi_j = jnp.float32(phi)
        v_j = jnp.float32(v)

        @jax.jit
        def _batch_inflow(gammas_batch):
            return jax.vmap(
                lambda g: inflow_speeds_jax(positions_j, phi_j, v_j, g)
            )(gammas_batch)

        correct_inflow = np.asarray(_batch_inflow(sg_j))

        for k in range(SYNTH_STATES):
            sg = synth_gammas_arr[k]
            synth_obs = orig_obs.copy()
            synth_obs[:N_turb] = sg.astype(np.float32)
            synth_obs[N_turb:2*N_turb] = v - correct_inflow[k].astype(np.float32)
            all_obs[base_idx + 1 + k] = synth_obs
            if BC_MODE == "inc":
                all_targets[base_idx + 1 + k] = np.clip(
                    slsqp_yaw - sg, -5.0, 5.0).astype(np.float32)
            else:
                all_targets[base_idx + 1 + k] = slsqp_yaw.astype(np.float32)

    return all_obs, all_targets


def split_train_val(all_obs, all_targets, val_frac, rng):
    """Random train/val split."""
    N = all_obs.shape[0]
    idx = rng.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return (all_obs[train_idx], all_targets[train_idx],
            all_obs[val_idx], all_targets[val_idx])


# ---------------------------------------------------------------------------
# BC training (NNX + JAX).
# ---------------------------------------------------------------------------
@nnx.jit
def bc_train_step(model: ActorCritic, optimizer: nnx.Optimizer, obs_batch, target_batch):
    """One BC optimizer step. Returns MSE loss."""

    def loss_fn(m):
        mean, _, _ = m(obs_batch)
        return jnp.mean((mean - target_batch) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def bc_eval_loss(model: ActorCritic, obs_batch, target_batch):
    """Compute MSE loss (no grad)."""
    mean, _, _ = model(obs_batch)
    return jnp.mean((mean - target_batch) ** 2)


# ---------------------------------------------------------------------------
# Per-seed training.
# ---------------------------------------------------------------------------
def train_one_seed(seed: int, data: dict, rng: np.random.Generator) -> dict:
    print(f"\n{'='*60}\n# BC seed={seed}  device={jax.devices()[0]}\n{'='*60}")

    N_turb = data["slsqp_yaw"].shape[1]
    obs_dim = data["obs"].shape[1]
    act_dim = N_turb

    # Build augmented samples.
    all_obs, all_targets = build_bc_samples(data, rng)
    train_obs, train_targets, val_obs, val_targets = split_train_val(
        all_obs, all_targets, VAL_FRAC, rng)

    N_train = train_obs.shape[0]
    N_val = val_obs.shape[0]
    print(f"  Train samples : {N_train}")
    print(f"  Val samples   : {N_val}")

    # Model + optimizer.
    model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(seed))
    tx = optax.adam(LEARNING_RATE)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Training loop.
    best_val_mse = float("inf")
    best_state = None
    iterations_log = []

    t_start = time.time()

    try:
        for epoch in range(BC_EPOCHS):
            # Shuffle training data.
            perm = rng.permutation(N_train)
            train_obs_shuf = train_obs[perm]
            train_targets_shuf = train_targets[perm]

            # Minibatch training.
            epoch_losses = []
            for start in range(0, N_train, BATCH_SIZE):
                end = min(start + BATCH_SIZE, N_train)
                obs_b = jnp.asarray(train_obs_shuf[start:end])
                tgt_b = jnp.asarray(train_targets_shuf[start:end])
                loss = bc_train_step(model, optimizer, obs_b, tgt_b)
                epoch_losses.append(float(loss))

            train_mse = float(np.mean(epoch_losses))

            # Validation (on a single batch to save time; full pass at checkpoints).
            val_mse = float("nan")
            do_full_val = (epoch % 10 == 0) or (epoch == BC_EPOCHS - 1)
            if do_full_val and N_val > 0:
                val_losses = []
                for start in range(0, N_val, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, N_val)
                    obs_b = jnp.asarray(val_obs[start:end])
                    tgt_b = jnp.asarray(val_targets[start:end])
                    vloss = bc_eval_loss(model, obs_b, tgt_b)
                    val_losses.append(float(vloss))
                val_mse = float(np.mean(val_losses))

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    _, best_state = nnx.split(model)

            iterations_log.append(dict(
                epoch=epoch,
                train_mse=train_mse,
                val_mse=val_mse if do_full_val else None,
            ))

            if epoch % 20 == 0 or epoch == BC_EPOCHS - 1:
                print(f"  epoch {epoch:4d} | train MSE {train_mse:.6f} | "
                      f"val MSE {val_mse:.6f}" + (" *" if val_mse == best_val_mse else ""))
    finally:
        elapsed_total = time.time() - t_start

        # Restore best state if available.
        if best_state is not None:
            graphdef, _ = nnx.split(model)
            model = nnx.merge(graphdef, best_state)

        metrics = dict(
            seed=seed,
            dataset=os.path.basename(DATASET),
            n_train=N_train,
            n_val=N_val,
            synth_states_per_expert=SYNTH_STATES,
            bc_epochs=BC_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            final_train_mse=float(iterations_log[-1]["train_mse"]),
            best_val_mse=float(best_val_mse) if best_val_mse < float("inf") else None,
            wall_clock_s=elapsed_total,
            iterations=iterations_log,
        )

        metrics_path = os.path.join(CKPT_DIR, f"metrics_seed{seed}_{OUT_TAG}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  wrote {metrics_path}")

        try:
            _, state_ck = nnx.split(model)
            ckpt_path = os.path.join(CKPT_DIR, f"policy_seed{seed}_{OUT_TAG}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(state_ck, f)
            print(f"  wrote {ckpt_path}")
        except Exception as exc:
            print(f"  [warn] policy save failed: {type(exc).__name__}: {exc}")

    return metrics


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    if not DATASET:
        print("ERROR: DATASET env var is required. "
              "e.g. DATASET=codes/expert_datasets/slsqp_expert_3x3_seed20260605_n5.npz")
        sys.exit(1)
    if not os.path.exists(DATASET):
        print(f"ERROR: dataset not found: {DATASET}")
        sys.exit(1)

    print(f"# Behavior Cloning (BC) from SLSQP expert")
    print(f"# dataset      : {DATASET}")
    print(f"# N_SEEDS      : {N_SEEDS}")
    print(f"# BC_EPOCHS    : {BC_EPOCHS}")
    print(f"# BATCH_SIZE   : {BATCH_SIZE}")
    print(f"# LR           : {LEARNING_RATE}")
    print(f"# OUT_TAG      : {OUT_TAG}")
    print(f"# SYNTH_STATES : {SYNTH_STATES}")
    print(f"# device       : {jax.devices()[0]}")

    data = load_expert_data(DATASET)
    seed_rng = np.random.default_rng(20260605)

    all_metrics = []
    for s in range(SEED_START, N_SEEDS):
        rng = np.random.default_rng(seed_rng.integers(0, 2**31))
        all_metrics.append(train_one_seed(s, data, rng))

    # Write summary.
    summary_path = os.path.join(CKPT_DIR, f"summary_{OUT_TAG}.json")
    summary = dict(
        tag=OUT_TAG,
        dataset=os.path.basename(DATASET),
        n_seeds=len(all_metrics),
        per_seed=all_metrics,
        wall_clock_mean_s=float(np.mean([m["wall_clock_s"] for m in all_metrics])),
        best_val_mse_mean=float(np.mean(
            [m["best_val_mse"] for m in all_metrics if m["best_val_mse"] is not None])),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
