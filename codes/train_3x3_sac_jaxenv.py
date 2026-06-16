#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAC (Soft Actor-Critic) training with JAX/NNX on-device stack.

Uses nnx.split/update pattern for all jitted functions to avoid
trace-level issues.  No closures over NNX objects.

Serves as an alternative RL algorithm comparison to PPO.

Outputs:
  codes/checkpoints_3x3_sac_jaxenv/policy_seedN.pkl
  codes/checkpoints_3x3_sac_jaxenv/metrics_seedN.json
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

from train_3x3_nnx import NET_ARCH
from windfarm_env_jax import (
    env_reset_batched, env_step_autoreset, positions_to_jax,
)
from windfarm_env import create_wind_farm_layout_3x3

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_sac_jaxenv")
os.makedirs(CKPT_DIR, exist_ok=True)

N_SEEDS = int(os.environ.get("N_SEEDS", 3))
SEED_START = int(os.environ.get("SEED_START", 0))
N_ENVS = int(os.environ.get("N_ENVS", 256))
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 30_000_000)))
REPLAY_BUFFER_SIZE = int(os.environ.get("REPLAY_BUFFER_SIZE", 1_000_000))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA_LR = 3e-4
MAX_EPISODE_STEPS = 200
ACT_HIGH = 5.0
J = 1
OUT_TAG = os.environ.get("OUT_TAG", f"sac_n{N_ENVS}")


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
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
        self.backbone = MLP(obs_dim, NET_ARCH, NET_ARCH[-1], rngs=rngs)
        self.mean_head = nnx.Linear(NET_ARCH[-1], act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(NET_ARCH[-1], act_dim, rngs=rngs)

    def __call__(self, obs):
        h = nnx.tanh(self.backbone.l1(obs))
        h = nnx.tanh(self.backbone.l2(h))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        return mean, log_std


class TwinQ(nnx.Module):
    def __init__(self, obs_dim, act_dim, *, rngs: nnx.Rngs):
        self.q1 = MLP(obs_dim + act_dim, NET_ARCH, 1, rngs=rngs)
        self.q2 = MLP(obs_dim + act_dim, NET_ARCH, 1, rngs=rngs)

    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class AlphaParam(nnx.Module):
    def __init__(self, act_dim, *, rngs: nnx.Rngs):
        self.log_alpha = nnx.Param(jnp.array(0.0, dtype=jnp.float32))
        self.target_entropy = jnp.array(-float(act_dim), dtype=jnp.float32)  # constant, not trainable


# ---------------------------------------------------------------------------
# Pure-function helpers for jitted operations
# ---------------------------------------------------------------------------
def policy_sample(graphdef, state, obs, key):
    """Sample action from policy (pure function for JIT)."""
    model = nnx.merge(graphdef, state)
    mean, log_std = model(obs)
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mean.shape)
    pre_tanh = mean + std * eps
    action = jnp.tanh(pre_tanh) * ACT_HIGH
    # log_prob
    var = jnp.exp(2.0 * log_std)
    log_unnorm = -0.5 * ((pre_tanh - mean) ** 2) / var
    log_norm = -0.5 * jnp.log(2.0 * jnp.pi) - log_std
    log_tanh_corr = jnp.log(jnp.clip(1.0 - action ** 2 / (ACT_HIGH ** 2), 1e-6, 1.0))
    logp = jnp.sum(log_unnorm + log_norm - log_tanh_corr, axis=-1)
    mean_action = jnp.tanh(mean) * ACT_HIGH
    return action, logp, mean_action


def policy_deterministic(graphdef, state, obs):
    """Deterministic action from policy (pure function for JIT)."""
    model = nnx.merge(graphdef, state)
    mean, _ = model(obs)
    return jnp.tanh(mean) * ACT_HIGH


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        idx = self.ptr % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = done
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        indices = rng.integers(0, self.size, size=batch_size)
        return (self.obs[indices], self.actions[indices],
                self.rewards[indices], self.next_obs[indices],
                self.dones[indices])


# ---------------------------------------------------------------------------
# JIT-compiled collection step (no closure over NNX objects)
# ---------------------------------------------------------------------------
@nnx.jit
def collect_step(policy, state, obs, positions_j, key):
    """Collect one step: sample actions, step env, return transitions."""
    # Sample actions
    mean, log_std = policy(obs)
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mean.shape)
    pre_tanh = mean + std * eps
    actions = jnp.tanh(pre_tanh) * ACT_HIGH

    # Step environment
    reset_keys = jax.random.split(key, actions.shape[0])
    new_state, new_obs, reward, done = env_step_autoreset(
        state, actions, reset_keys, positions_j,
        j=J, max_steps=MAX_EPISODE_STEPS, randomize_wind=True)
    return actions, new_state, new_obs, reward, done


# ---------------------------------------------------------------------------
# JIT-compiled SAC update step
# ---------------------------------------------------------------------------
@nnx.jit
def sac_update(policy, q_net, q_target, alpha,
               policy_opt, q_opt, alpha_opt,
               obs_b, act_b, rew_b, next_obs_b, done_b, key1, key2):
    """One SAC update step. All NNX objects passed as arguments."""

    # --- Critic update ---
    next_act, next_logp, _ = policy.sample(next_obs_b, key1)
    q1_tgt, q2_tgt = q_target(next_obs_b, next_act)
    q_tgt_min = jnp.minimum(q1_tgt, q2_tgt)
    alpha_val = jnp.exp(alpha.log_alpha.value)
    td_target = rew_b + GAMMA * (1.0 - done_b) * (q_tgt_min - alpha_val * next_logp)
    td_target = jax.lax.stop_gradient(td_target)

    # Q loss + grad
    def q_loss_fn(m):
        q1, q2 = m(obs_b, act_b)
        return 0.5 * jnp.mean((q1 - td_target)**2) + 0.5 * jnp.mean((q2 - td_target)**2)

    q_loss, q_grads = nnx.value_and_grad(q_loss_fn)(q_net)
    q_opt.update(q_net, q_grads)

    # --- Actor update ---
    def actor_loss_fn(m):
        act_new, logp_new, _ = m.sample(obs_b, key2)
        q1, q2 = q_net(obs_b, act_new)
        q_min = jnp.minimum(q1, q2)
        alpha_v = jnp.exp(alpha.log_alpha.value)
        return jnp.mean(alpha_v * logp_new - q_min), logp_new

    (actor_loss, logp_val), policy_grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(policy)
    policy_opt.update(policy, policy_grads)

    # --- Alpha update ---
    def alpha_loss_fn(a):
        av = jnp.exp(a.log_alpha.value)
        return jnp.mean(-av * (logp_val + a.target_entropy))

    alpha_loss, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(alpha)
    alpha_opt.update(alpha, alpha_grads)

    # --- Target EMA (polyak average) ---
    _, state_online = nnx.split(q_net)
    _, state_target = nnx.split(q_target)
    state_target_new = jax.tree.map(
        lambda t, o: TAU * o + (1 - TAU) * t,
        state_target, state_online)
    nnx.update(q_target, state_target_new)

    return q_loss, actor_loss, alpha_loss


# ---------------------------------------------------------------------------
# Fix SACPolicy to have sample/deterministic as methods (used by sac_update)
# ---------------------------------------------------------------------------
def _policy_sample(self, obs, key):
    mean, log_std = self(obs)
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mean.shape)
    pre_tanh = mean + std * eps
    action = jnp.tanh(pre_tanh) * ACT_HIGH
    var = jnp.exp(2.0 * log_std)
    log_unnorm = -0.5 * ((pre_tanh - mean) ** 2) / var
    log_norm = -0.5 * jnp.log(2.0 * jnp.pi) - log_std
    log_tanh_corr = jnp.log(jnp.clip(1.0 - action ** 2 / (ACT_HIGH ** 2), 1e-6, 1.0))
    logp = jnp.sum(log_unnorm + log_norm - log_tanh_corr, axis=-1)
    mean_action = jnp.tanh(mean) * ACT_HIGH
    return action, logp, mean_action

SACPolicy.sample = _policy_sample


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_seed(seed: int) -> dict:
    print(f"\n{'='*60}\n# SAC seed={seed}  device={jax.devices()[0]}  N_ENVS={N_ENVS}\n{'='*60}")

    positions_list, _, _ = create_wind_farm_layout_3x3()
    positions_j = positions_to_jax(positions_list)
    N_turb = positions_j.shape[0]
    obs_dim = 3 * N_turb + 3
    act_dim = N_turb

    # Initialize networks
    key = jax.random.PRNGKey(seed)
    policy_key, q_key, reset_key = jax.random.split(key, 3)

    policy = SACPolicy(obs_dim, act_dim, rngs=nnx.Rngs(int(policy_key[0])))
    q_net = TwinQ(obs_dim, act_dim, rngs=nnx.Rngs(int(q_key[0])))
    q_target = TwinQ(obs_dim, act_dim, rngs=nnx.Rngs(int(q_key[0] + 1)))
    alpha = AlphaParam(act_dim, rngs=nnx.Rngs(0))

    # Optimizers
    policy_opt = nnx.Optimizer(policy, optax.chain(
        optax.clip_by_global_norm(0.5), optax.adam(LEARNING_RATE)), wrt=nnx.Param)
    q_opt = nnx.Optimizer(q_net, optax.chain(
        optax.clip_by_global_norm(0.5), optax.adam(LEARNING_RATE)), wrt=nnx.Param)
    alpha_opt = nnx.Optimizer(alpha, optax.adam(ALPHA_LR), wrt=nnx.Param)

    # Copy online Q weights to target
    _, state_online = nnx.split(q_net)
    nnx.update(q_target, state_online)

    # Replay buffer
    replay = ReplayBuffer(obs_dim, act_dim, REPLAY_BUFFER_SIZE)

    # Initial env reset
    reset_keys = jax.random.split(reset_key, N_ENVS)
    reset_jit = jax.jit(env_reset_batched, static_argnames=("j", "max_steps", "randomize_wind"))
    state, obs = reset_jit(reset_keys, positions_j, j=J, max_steps=MAX_EPISODE_STEPS, randomize_wind=True)

    # Training loop
    rng = np.random.default_rng(seed)
    key_counter = 1000  # avoid key=0 conflicts
    iterations_log = []
    total_env_steps = 0
    ep_returns = []
    running_returns = np.zeros(N_ENVS, dtype=np.float32)
    running_lens = np.zeros(N_ENVS, dtype=np.int32)
    t_train_start = time.time()

    LEARN_START = int(os.environ.get("LEARN_START", BATCH_SIZE * 10))
    LEARN_EVERY = int(os.environ.get("LEARN_EVERY", 1))  # learn every N collect steps

    step_count = 0
    try:
        while total_env_steps < TOTAL_STEPS:
            # --- Collect transition ---
            key_counter += 1
            collect_key = jax.random.key(key_counter)

            actions, state, next_obs, reward, done = collect_step(
                policy, state, obs, positions_j, collect_key)

            obs_np = np.asarray(obs)
            next_obs_np = np.asarray(next_obs)
            actions_np = np.asarray(actions)
            reward_np = np.asarray(reward)
            done_np = np.asarray(done)

            for i in range(N_ENVS):
                replay.add(obs_np[i], actions_np[i], reward_np[i],
                          next_obs_np[i], float(done_np[i]))
                running_returns[i] += reward_np[i]
                running_lens[i] += 1
                if done_np[i]:
                    ep_returns.append(float(running_returns[i]))
                    running_returns[i] = 0.0
                    running_lens[i] = 0

            obs = next_obs
            total_env_steps += N_ENVS
            step_count += 1

            # --- Learning step ---
            if replay.size >= LEARN_START and step_count % LEARN_EVERY == 0:
                batch = replay.sample(BATCH_SIZE, rng)
                key_counter += 1
                key1 = jax.random.key(key_counter)
                key_counter += 1
                key2 = jax.random.key(key_counter)

                q_loss, actor_loss, alpha_loss = sac_update(
                    policy, q_net, q_target, alpha,
                    policy_opt, q_opt, alpha_opt,
                    jnp.asarray(batch[0]), jnp.asarray(batch[1]),
                    jnp.asarray(batch[2]), jnp.asarray(batch[3]),
                    jnp.asarray(batch[4]), key1, key2)

            # Log
            if step_count % 100 == 0:
                elapsed = time.time() - t_train_start
                fps = total_env_steps / max(1e-9, elapsed)
                ep_rew_mean = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
                alpha_val = float(jnp.exp(alpha.log_alpha.value))
                iterations_log.append(dict(
                    step=step_count,
                    total_env_steps=total_env_steps,
                    elapsed_s=elapsed,
                    fps=fps,
                    ep_rew_mean=ep_rew_mean,
                    buffer_size=replay.size,
                    alpha=alpha_val,
                ))
                print(f"  step {step_count:5d} | env {total_env_steps:8d} | "
                      f"fps {fps:7.0f} | ep_rew {ep_rew_mean:+8.2f} | "
                      f"buf {replay.size:6d} | alpha {alpha_val:.4f}")

    finally:
        elapsed_total = time.time() - t_train_start
        metrics = dict(
            seed=seed,
            backend="nnx-sac-jaxenv-gpu",
            device=str(jax.devices()[0]),
            n_envs=N_ENVS,
            total_env_steps=total_env_steps,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            wall_clock_s=elapsed_total,
            fps=total_env_steps / max(1e-9, elapsed_total),
            final_ep_rew_mean=float(np.mean(ep_returns[-20:])) if ep_returns else None,
            num_completed_episodes=len(ep_returns),
            iterations=iterations_log,
        )
        out_path = os.path.join(CKPT_DIR, f"metrics_seed{seed}_{OUT_TAG}.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  wrote {out_path}")

        try:
            _, state_ck = nnx.split(policy)
            with open(os.path.join(CKPT_DIR, f"policy_seed{seed}_{OUT_TAG}.pkl"), "wb") as f:
                pickle.dump(state_ck, f)
        except Exception as exc:
            print(f"[warn] policy save failed: {type(exc).__name__}: {exc}")

    return metrics


def main():
    print(f"# SAC (JAX/NNX)  jax={jax.__version__}  device={jax.devices()[0]}")
    print(f"# seeds: {N_SEEDS}  N_ENVS: {N_ENVS}  TOTAL_STEPS: {TOTAL_STEPS}")

    all_metrics = []
    for s in range(SEED_START, N_SEEDS):
        all_metrics.append(train_one_seed(s))

    summary_path = os.path.join(CKPT_DIR, f"summary_{OUT_TAG}.json")
    with open(summary_path, "w") as f:
        json.dump({"per_seed": all_metrics}, f, indent=2)
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
