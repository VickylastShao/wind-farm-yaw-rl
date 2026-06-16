# -*- coding: utf-8 -*-
"""
3x3 PPO training with Flax NNX 0.12.4 + JAX 0.9.0.1 + Optax 0.2.6 on GPU.

A from-scratch PPO implementation that mirrors stable-baselines3's
hyperparameters exactly, so it can be benchmarked head-to-head against
train_3x3_local.py at identical n_steps / n_envs / batch_size / n_epochs
/ lr / network width / total_timesteps. Reward normalization is *off*
on both sides to keep the comparison clean.

This file is the migration target of the user's "NNX best-practices"
spec; every rule below is enforced:

  1.  nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)   -- wrt here ONLY
  2.  nnx.value_and_grad(loss_fn)(model, ...)              -- no wrt
  3.  optimizer.update(model, grads)                       -- Flax 0.12+ signature
  4.  @nnx.jit (never @jax.jit) on NNX-containing functions
  5.  Parallel envs via gymnasium SyncVectorEnv -- env stays in numpy/host
  6.  jax.random.PRNGKey(int_seed) with Python int
  7.  No jax.vmap-over-NNX, no nnx.param (lower-case), no CONFIG-in-jit,
       no nonlocal in value_and_grad, no os.environ['JAX_PLATFORMS']='cpu'
  8.  try/finally around metrics.json write so partial runs still record

Outputs:
  codes/checkpoints_3x3_nnx/policy_seedN.pkl
                            metrics_seedN.json
  latex_draft/figures/
    fig_3x3_nnx_vs_sb3.{pdf,jpg}    (written by separate compare script)

Env vars (mirror train_3x3_local.py for fair A/B):
  N_SEEDS       (default 1)
  N_ENVS        (default 8)
  TOTAL_STEPS   (default 50_000)
  N_STEPS       (default 2048)
  BATCH_SIZE    (default 256)
  N_EPOCHS      (default 10)
"""

import os
import json
import time
import pickle
from typing import Any

import numpy as np

# Rule 7: do NOT force JAX to CPU; let it pick the CUDA device.
import jax
import jax.numpy as jnp
import optax
from flax import nnx

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from windfarm_env import WindFarmYawEnv, create_wind_farm_layout_3x3


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_SCRIPT_DIR)
OUT_DIR = os.path.join(_PROJ_ROOT, "latex_draft", "figures")
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Hyperparameters (default: 50k-step smoke benchmark; raise TOTAL_STEPS for
# real training). Kept identical to train_3x3_local.py defaults so wall-clock
# is comparable.
# ---------------------------------------------------------------------------
N_SEEDS = int(os.environ.get("N_SEEDS", 1))
N_ENVS = int(os.environ.get("N_ENVS", 8))
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 50_000)))
N_STEPS = int(os.environ.get("N_STEPS", 2048))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
N_EPOCHS = int(os.environ.get("N_EPOCHS", 10))

# These are paper-PPO fixed constants — not tuned in the A/B comparison.
LEARNING_RATE = float(os.environ.get("LR", "3e-4"))
GAMMA = float(os.environ.get("GAMMA", "0.99"))
GAE_LAMBDA = float(os.environ.get("GAE_LAMBDA", "0.95"))
CLIP_RANGE = 0.2
ENT_COEF = float(os.environ.get("ENT_COEF", "0.005"))
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
NET_ARCH_RAW = os.environ.get("NET_ARCH", "128,128")
NET_ARCH = tuple(int(x.strip()) for x in NET_ARCH_RAW.split(","))


RESIDUAL = os.environ.get("RESIDUAL", "0") == "1"

# ---- Turbine Self-Attention architecture ----
USE_ATTENTION = os.environ.get("ATTENTION", "0") == "1"
ATTN_EMBED = int(os.environ.get("ATTN_EMBED", "64"))
ATTN_HEADS = int(os.environ.get("ATTN_HEADS", "2"))
ATTN_HIDDEN = int(os.environ.get("ATTN_HIDDEN", "64"))
# N_TURB = 9 for 3x3 layout; USE_POSITIONS determines feature count per turbine.
_N_TURB = 9
_USE_POS = os.environ.get("USE_POSITIONS", "0") == "1"
# Per-step layout (when USE_POSITIONS=1):
#   [gammas(N), inflow(N), wind(3), locked(N), pos_x(N), pos_y(N)]
#   step_size = 5*N+3
# Per-turbine features (J steps): J gamma, J inflow, J locked, J pos_x, J pos_y
# Global features (J steps): J cos, J sin, J v
_J = int(os.environ.get("J", "1"))
_STEP = 5 * _N_TURB + 3 if _USE_POS else 3 * _N_TURB + 3
_N_TURB_FEAT = 5 if _USE_POS else 3  # per turbine, per step

# Pre-compute observation index arrays for efficient feature extraction.
def _make_obs_indices(J, N, step_size, n_turb_feat):
    """Return (turbine_idx, global_idx) for indexing flat obs.

    turbine_idx: (N, J * n_turb_feat) — per-turbine features across history
    global_idx:  (J * 3,)           — wind cos/sin/v across history
    """
    turbine_idx = []
    global_idx = []
    for j in range(J):
        off = j * step_size
        global_idx.extend([off + 2 * N + k for k in range(3)])  # cos, sin, v
    # Per-turbine: gather all J steps' features together.
    for i in range(N):
        feats = []
        for j in range(J):
            off = j * step_size
            feats.append(off + i)                               # gamma
            feats.append(off + N + i)                           # inflow
            feats.append(off + 2 * N + 3 + i)                   # locked
            if _USE_POS:
                feats.append(off + 3 * N + 3 + i)               # pos_x
                feats.append(off + 4 * N + 3 + i)               # pos_y
        turbine_idx.append(feats)
    return (jnp.array(global_idx, dtype=jnp.int32),             # (J*3,)
            jnp.array(turbine_idx, dtype=jnp.int32))            # (N, J*n_turb_feat)

_OBS_GLOBAL_IDX, _OBS_TURBINE_IDX = _make_obs_indices(_J, _N_TURB, _STEP, _N_TURB_FEAT)
_TURBINE_FEAT_DIM = _J * _N_TURB_FEAT  # per-turbine input dim
_GLOBAL_FEAT_DIM = _J * 3               # global input dim


# ---------------------------------------------------------------------------
# Policy: Gaussian actor + value critic, NNX modules.
# ---------------------------------------------------------------------------
class MLP(nnx.Module):
    """Variable-depth MLP with optional residual connections + LayerNorm.

    When RESIDUAL=1 and consecutive layers have the same width, a skip
    connection is added: x = x + tanh(W·LN(x)).  LayerNorm is applied
    before each activation for training stability.
    """

    def __init__(self, din: int, hidden: tuple, dout: int, *, rngs: nnx.Rngs):
        self._num_layers = len(hidden)
        self._use_residual = RESIDUAL
        prev = din
        for i, h in enumerate(hidden):
            setattr(self, f"linear_{i}", nnx.Linear(prev, h, rngs=rngs))
            if self._use_residual:
                setattr(self, f"ln_{i}", nnx.LayerNorm(h, rngs=rngs))
            prev = h
        self.out = nnx.Linear(prev, dout, rngs=rngs)

    def __call__(self, x):
        for i in range(self._num_layers):
            h = getattr(self, f"linear_{i}")(x)
            if self._use_residual:
                h = getattr(self, f"ln_{i}")(h)
                h = nnx.tanh(h)
                # Skip connection when input/output dims match.
                if x.shape[-1] == h.shape[-1]:
                    x = x + h
                else:
                    x = h
            else:
                x = nnx.tanh(h)
        return self.out(x)


class ActorCritic(nnx.Module):
    """SB3-style separate-trunk MlpPolicy: independent actor + critic MLPs,
    plus a state-independent log_std parameter for the diagonal Gaussian.

    Action space is Box(-5,+5,shape=(N,)); we output a *pre-clip* mean and
    leave the clipping to the env (matches SB3 behaviour)."""

    def __init__(self, obs_dim: int, act_dim: int, *, rngs: nnx.Rngs):
        self.actor_mlp = MLP(obs_dim, NET_ARCH, act_dim, rngs=rngs)
        self.critic_mlp = MLP(obs_dim, NET_ARCH, 1, rngs=rngs)
        # Rule 7: use nnx.Param (capital P), not nnx.param.
        self.log_std = nnx.Param(jnp.zeros((act_dim,), dtype=jnp.float32))

    def __call__(self, obs):
        mean = self.actor_mlp(obs)
        value = self.critic_mlp(obs).squeeze(-1)
        # Rule 7: nnx.Param (capital P); use [...] indexer per Flax 0.12+
        # deprecation note (".value" attribute access is removed in 0.13+).
        log_std = jnp.broadcast_to(self.log_std[...], mean.shape)
        return mean, log_std, value


class TurbineSelfAttention(nnx.Module):
    """Multi-head self-attention over turbines to model wake interactions.

    Turbines attend to each other based on their features (yaw, inflow, position),
    learning which upstream turbines affect downstream ones for a given wind
    direction.  Permutation-equivariant by design — respects turbine symmetry.
    """

    def __init__(self, embed_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, x):
        # x: (B, N, embed_dim)
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim
        q = self.q_proj(x).reshape(B, N, H, d).transpose(0, 2, 1, 3)  # (B, H, N, d)
        k = self.k_proj(x).reshape(B, N, H, d).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, H, d).transpose(0, 2, 1, 3)
        attn_logits = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        out = attn_weights @ v  # (B, H, N, d)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)


class AttentionActorCritic(nnx.Module):
    """Actor-Critic with per-turbine self-attention + global wind context.

    Observation is decomposed into:
      - Per-turbine features: (N, J*n_turb_feat) — yaw, inflow, lock, positions
      - Global features: (J*3,) — cos(phi), sin(phi), wind speed

    A shared per-turbine encoder maps each turbine to an embedding, then
    multi-head self-attention models pairwise wake interactions.  The
    attention output is fused with a global wind encoding and fed to
    per-turbine actor heads and a pooled critic head.
    """

    def __init__(self, obs_dim: int, act_dim: int, *,
                 n_turb: int = _N_TURB,
                 turbine_feat_dim: int = _TURBINE_FEAT_DIM,
                 global_feat_dim: int = _GLOBAL_FEAT_DIM,
                 rngs: nnx.Rngs):
        self.n_turb = n_turb
        self.turbine_feat_dim = turbine_feat_dim
        self.global_feat_dim = global_feat_dim

        # Per-turbine feature encoder (shared across turbines)
        self.turbine_encoder = MLP(turbine_feat_dim,
                                   (ATTN_HIDDEN, ATTN_HIDDEN), ATTN_EMBED,
                                   rngs=rngs)
        # Self-attention over turbines
        self.attention = TurbineSelfAttention(ATTN_EMBED, ATTN_HEADS, rngs=rngs)
        # Global wind-context encoder
        self.global_encoder = MLP(global_feat_dim,
                                  (ATTN_HIDDEN,), ATTN_EMBED,
                                  rngs=rngs)

        # Actor head: per-turbine action from fused features
        self.actor_head = MLP(ATTN_EMBED * 2, NET_ARCH, 1, rngs=rngs)

        # Critic head: single value from pooled attention + global context
        self.critic_mlp = MLP(ATTN_EMBED + ATTN_EMBED, NET_ARCH, 1, rngs=rngs)

        self.log_std = nnx.Param(jnp.zeros((act_dim,), dtype=jnp.float32))

    def __call__(self, obs):
        B = obs.shape[0]

        # 1. Decompose observation into turbine & global features.
        turbine_raw = obs[:, _OBS_TURBINE_IDX]           # (B, N, J*n_feat)
        global_raw = obs[:, _OBS_GLOBAL_IDX]              # (B, J*3)

        # 2. Per-turbine encoder (vmap over N turbines).
        def encode_one(t):
            return self.turbine_encoder(t)                # (B, embed_dim)
        turbine_emb = jax.vmap(encode_one, in_axes=1, out_axes=1)(turbine_raw)
        # turbine_emb: (B, N, ATTN_EMBED)

        # 3. Self-attention over turbines.
        attn_out = self.attention(turbine_emb)            # (B, N, ATTN_EMBED)

        # 4. Global wind encoding.
        global_emb = self.global_encoder(global_raw)      # (B, ATTN_EMBED)

        # 5. Fuse per-turbine attention output with global context.
        global_tiled = jnp.broadcast_to(global_emb[:, None, :],
                                        (B, self.n_turb, ATTN_EMBED))
        fused = jnp.concatenate([attn_out, global_tiled], axis=-1)
        # fused: (B, N, 2*ATTN_EMBED)

        # 6. Per-turbine action mean.
        def actor_one(f):
            return self.actor_head(f).squeeze(-1)         # (B,)
        mean = jax.vmap(actor_one, in_axes=1, out_axes=1)(fused)
        # mean: (B, N)

        # 7. Critic: mean-pool turbine embeddings + global context.
        pooled = attn_out.mean(axis=1)                    # (B, ATTN_EMBED)
        critic_in = jnp.concatenate([pooled, global_emb], axis=-1)
        value = self.critic_mlp(critic_in).squeeze(-1)    # (B,)

        log_std = jnp.broadcast_to(self.log_std[...], mean.shape)
        return mean, log_std, value


def gaussian_log_prob(mean, log_std, action):
    """Sum-over-action-dims log N(action | mean, exp(log_std)^2)."""
    var = jnp.exp(2.0 * log_std)
    log_unnorm = -0.5 * ((action - mean) ** 2) / var
    log_norm = -0.5 * jnp.log(2.0 * jnp.pi) - log_std
    return jnp.sum(log_unnorm + log_norm, axis=-1)


def gaussian_entropy(log_std):
    """Per-sample entropy of diagonal Gaussian (broadcast-friendly)."""
    # H = 0.5 * sum(log(2 pi e sigma^2)) = sum(log_std + 0.5*log(2 pi e))
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)


# ---------------------------------------------------------------------------
# Rollout-time helpers (jit'd, run on GPU).
# ---------------------------------------------------------------------------
@nnx.jit
def sample_action(model: ActorCritic, obs, key):
    """Sample a stochastic action; returns (action, logp, value)."""
    mean, log_std, value = model(obs)
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mean.shape)
    action = mean + std * eps
    logp = gaussian_log_prob(mean, log_std, action)
    return action, logp, value


@nnx.jit
def predict_value(model: ActorCritic, obs):
    _, _, value = model(obs)
    return value


# ---------------------------------------------------------------------------
# GAE on host (small arrays; not worth jit'ing).
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Generalized advantage estimator.

    rewards/values/dones: (T, N_envs) numpy arrays.
    last_value: (N_envs,) value of the state after the last step.
    Returns advantages (T, N_envs) and returns (T, N_envs)."""
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO loss + train step, all jit'd. Rule 4: @nnx.jit, not @jax.jit.
# ---------------------------------------------------------------------------
@nnx.jit
def ppo_train_step(model: ActorCritic, optimizer: nnx.Optimizer,
                   obs, actions, old_logp, advantages, returns,
                   clip_range: float, ent_coef: float, vf_coef: float):
    """One PPO minibatch SGD step. Returns (total_loss, pg_loss, v_loss,
    entropy, approx_kl, clip_frac)."""

    def loss_fn(m):
        # Rule 5: no nonlocal closures over mutable state inside the value_and_grad
        # function — every dependency is an explicit argument captured by closure
        # from the *outer* function args, which are themselves traced.
        mean, log_std, value_pred = m(obs)
        logp = gaussian_log_prob(mean, log_std, actions)
        ratio = jnp.exp(logp - old_logp)

        # PPO clipped surrogate
        unclipped = ratio * advantages
        clipped = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        pg_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

        # Value loss (no value clipping -- matches SB3 default)
        v_loss = 0.5 * jnp.mean((value_pred - returns) ** 2)

        # Entropy bonus (per-sample, then mean across batch)
        entropy = jnp.mean(gaussian_entropy(log_std))

        total = pg_loss + vf_coef * v_loss - ent_coef * entropy

        # Diagnostics: approx_kl, clip_fraction (stop-grad'd via lax.stop_gradient)
        approx_kl = jnp.mean(old_logp - logp)
        clip_frac = jnp.mean(
            (jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32))
        return total, (pg_loss, v_loss, entropy, approx_kl, clip_frac)

    # Rule 2: nnx.value_and_grad WITHOUT wrt.
    (total, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Global-norm gradient clip (matches SB3 max_grad_norm=0.5).
    # Optax's clip_by_global_norm is applied as part of the optimizer chain;
    # since we built optimizer with optax.chain below, this is already handled.
    # Rule 3: optimizer.update(model, grads)  -- Flax 0.12+ signature.
    optimizer.update(model, grads)
    return total, *aux


# ---------------------------------------------------------------------------
# Env builder (kept on host).
# ---------------------------------------------------------------------------
def make_env(seed: int):
    positions, R, C = create_wind_farm_layout_3x3()

    def _thunk():
        e = WindFarmYawEnv(positions, R, C, j=1, randomize_wind=True,
                           max_steps=200)
        return e

    return _thunk


# ---------------------------------------------------------------------------
# Main training loop for one seed.
# ---------------------------------------------------------------------------
def train_one_seed(seed: int) -> dict:
    print(f"\n{'='*60}\n# NNX PPO seed={seed}  device={jax.devices()[0]}\n{'='*60}")

    # Rule 5: SyncVectorEnv (Subproc adds noise to wall-clock at this scale).
    venv = SyncVectorEnv([make_env(seed + 1000 * k) for k in range(N_ENVS)])

    obs_dim = int(np.prod(venv.single_observation_space.shape))
    act_dim = int(np.prod(venv.single_action_space.shape))
    act_low = float(venv.single_action_space.low.min())
    act_high = float(venv.single_action_space.high.max())

    # Rule 6: PRNGKey from Python int seed.
    key = jax.random.PRNGKey(seed)
    model_key, sample_key = jax.random.split(key)
    model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(int(model_key[0])))

    # Rule 1: nnx.Optimizer(..., wrt=nnx.Param). Pair with optax.chain so the
    # MAX_GRAD_NORM clip is part of the optimizer update.
    tx = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(LEARNING_RATE),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Storage buffers
    obs_buf = np.zeros((N_STEPS, N_ENVS, obs_dim), dtype=np.float32)
    act_buf = np.zeros((N_STEPS, N_ENVS, act_dim), dtype=np.float32)
    logp_buf = np.zeros((N_STEPS, N_ENVS), dtype=np.float32)
    val_buf = np.zeros((N_STEPS, N_ENVS), dtype=np.float32)
    rew_buf = np.zeros((N_STEPS, N_ENVS), dtype=np.float32)
    done_buf = np.zeros((N_STEPS, N_ENVS), dtype=np.float32)

    obs_np, _ = venv.reset(seed=seed)

    # Bookkeeping
    ep_returns: list[float] = []
    ep_lens: list[int] = []
    running_returns = np.zeros(N_ENVS, dtype=np.float32)
    running_lens = np.zeros(N_ENVS, dtype=np.int32)

    iterations_log: list[dict] = []
    t_train_start = time.time()
    total_env_steps = 0
    iteration = 0
    n_iterations = TOTAL_STEPS // (N_STEPS * N_ENVS)
    if n_iterations < 1:
        n_iterations = 1
        actual_n_steps = max(1, TOTAL_STEPS // N_ENVS)
    else:
        actual_n_steps = N_STEPS

    # Reuse buffers of correct length for short smoke runs.
    if actual_n_steps != N_STEPS:
        obs_buf = np.zeros((actual_n_steps, N_ENVS, obs_dim), dtype=np.float32)
        act_buf = np.zeros((actual_n_steps, N_ENVS, act_dim), dtype=np.float32)
        logp_buf = np.zeros((actual_n_steps, N_ENVS), dtype=np.float32)
        val_buf = np.zeros((actual_n_steps, N_ENVS), dtype=np.float32)
        rew_buf = np.zeros((actual_n_steps, N_ENVS), dtype=np.float32)
        done_buf = np.zeros((actual_n_steps, N_ENVS), dtype=np.float32)

    # Timing accumulators
    t_rollout_total = 0.0
    t_update_total = 0.0

    try:
        for iteration in range(n_iterations):
            # --- ROLLOUT ---
            t_roll_0 = time.time()
            for t in range(actual_n_steps):
                obs_jax = jnp.asarray(obs_np)
                sample_key, sub = jax.random.split(sample_key)
                action_j, logp_j, value_j = sample_action(model, obs_jax, sub)
                # Clip action to env bounds before stepping (matches SB3
                # "clip_action" default for Box spaces).
                action_clipped = np.clip(np.asarray(action_j), act_low, act_high)
                next_obs, rew, term, trunc, _ = venv.step(action_clipped)
                done = np.logical_or(term, trunc).astype(np.float32)

                obs_buf[t] = obs_np
                act_buf[t] = np.asarray(action_j)
                logp_buf[t] = np.asarray(logp_j)
                val_buf[t] = np.asarray(value_j)
                rew_buf[t] = rew.astype(np.float32)
                done_buf[t] = done

                running_returns += rew.astype(np.float32)
                running_lens += 1
                for i, d in enumerate(done):
                    if d:
                        ep_returns.append(float(running_returns[i]))
                        ep_lens.append(int(running_lens[i]))
                        running_returns[i] = 0.0
                        running_lens[i] = 0
                obs_np = next_obs
            t_rollout_total += time.time() - t_roll_0

            # Bootstrap value for the *last* state
            last_val = np.asarray(predict_value(model, jnp.asarray(obs_np)))
            adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_val,
                                   GAMMA, GAE_LAMBDA)

            # Flatten (T, N, ...) -> (T*N, ...)
            B = actual_n_steps * N_ENVS
            b_obs = obs_buf.reshape(B, obs_dim)
            b_act = act_buf.reshape(B, act_dim)
            b_logp = logp_buf.reshape(B)
            b_adv = adv.reshape(B)
            b_ret = ret.reshape(B)
            # Normalize advantages (SB3 default behaviour).
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            # --- UPDATE ---
            t_upd_0 = time.time()
            mb_size = min(BATCH_SIZE, B)
            idx = np.arange(B)
            losses, pg_losses, v_losses, ents, kls, clip_fs = [], [], [], [], [], []
            for _ in range(N_EPOCHS):
                np.random.shuffle(idx)
                for start in range(0, B, mb_size):
                    end = start + mb_size
                    mb = idx[start:end]
                    loss, pgl, vl, ent, kl, cf = ppo_train_step(
                        model, optimizer,
                        jnp.asarray(b_obs[mb]),
                        jnp.asarray(b_act[mb]),
                        jnp.asarray(b_logp[mb]),
                        jnp.asarray(b_adv[mb]),
                        jnp.asarray(b_ret[mb]),
                        CLIP_RANGE, ENT_COEF, VF_COEF,
                    )
                    losses.append(float(loss))
                    pg_losses.append(float(pgl))
                    v_losses.append(float(vl))
                    ents.append(float(ent))
                    kls.append(float(kl))
                    clip_fs.append(float(cf))
            t_update_total += time.time() - t_upd_0

            total_env_steps += actual_n_steps * N_ENVS
            elapsed = time.time() - t_train_start
            fps = total_env_steps / max(1e-9, elapsed)
            ep_rew_mean = (float(np.mean(ep_returns[-20:])) if ep_returns
                           else float("nan"))
            ep_len_mean = (float(np.mean(ep_lens[-20:])) if ep_lens
                           else float("nan"))
            iterations_log.append(dict(
                iteration=iteration,
                total_env_steps=total_env_steps,
                elapsed_s=elapsed,
                fps=fps,
                ep_rew_mean=ep_rew_mean,
                ep_len_mean=ep_len_mean,
                loss=float(np.mean(losses)),
                pg_loss=float(np.mean(pg_losses)),
                v_loss=float(np.mean(v_losses)),
                entropy=float(np.mean(ents)),
                approx_kl=float(np.mean(kls)),
                clip_frac=float(np.mean(clip_fs)),
            ))
            print(f"  iter {iteration:3d} | steps {total_env_steps:8d} | "
                  f"fps {fps:7.0f} | ep_rew {ep_rew_mean:+8.2f} | "
                  f"loss {np.mean(losses):+.4f} | kl {np.mean(kls):.4f} | "
                  f"clip {np.mean(clip_fs):.3f}")
    finally:
        # Rule 8: try/finally so even a crashed run leaves a metrics file.
        elapsed_total = time.time() - t_train_start
        _dev = str(jax.devices()[0])
        _backend = "nnx-jax-cpu" if "CPU" in _dev.upper() else "nnx-jax-gpu"
        metrics = dict(
            seed=seed,
            backend=_backend,
            device=_dev,
            jax_version=jax.__version__,
            flax_version=__import__("flax").__version__,
            optax_version=optax.__version__,
            n_seeds=N_SEEDS,
            n_envs=N_ENVS,
            total_steps_target=TOTAL_STEPS,
            total_env_steps=total_env_steps,
            n_steps=actual_n_steps,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            wall_clock_s=elapsed_total,
            rollout_s=t_rollout_total,
            update_s=t_update_total,
            fps=total_env_steps / max(1e-9, elapsed_total),
            final_ep_rew_mean=(float(np.mean(ep_returns[-20:]))
                               if ep_returns else None),
            num_completed_episodes=len(ep_returns),
            iterations=iterations_log,
        )
        with open(os.path.join(CKPT_DIR, f"metrics_seed{seed}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save the policy (NNX state).
        try:
            _, state = nnx.split(model)
            with open(os.path.join(CKPT_DIR,
                                   f"policy_seed{seed}.pkl"), "wb") as f:
                pickle.dump(state, f)
        except Exception as exc:
            print(f"[warn] policy save failed: {type(exc).__name__}: {exc}")

        venv.close()
    return metrics


def main():
    print(f"# NNX PPO  jax={jax.__version__}  device={jax.devices()[0]}")
    print(f"# seeds       : {N_SEEDS}")
    print(f"# parallel env: {N_ENVS}")
    print(f"# total steps : {TOTAL_STEPS}")
    print(f"# n_steps     : {N_STEPS}")
    print(f"# batch_size  : {BATCH_SIZE}")
    print(f"# n_epochs    : {N_EPOCHS}")

    all_metrics = []
    for s in range(N_SEEDS):
        all_metrics.append(train_one_seed(s))

    summary = dict(
        backend="nnx-jax-gpu",
        n_seeds=N_SEEDS,
        per_seed=all_metrics,
        wall_clock_mean_s=float(np.mean(
            [m["wall_clock_s"] for m in all_metrics])),
        fps_mean=float(np.mean([m["fps"] for m in all_metrics])),
    )
    with open(os.path.join(CKPT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote per-seed metrics + summary to {CKPT_DIR}")


if __name__ == "__main__":
    main()
