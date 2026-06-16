# -*- coding: utf-8 -*-
"""
3x3 SAC (Soft Actor-Critic) training with JAX/NNX + JAX-vec env.

Fair-comparison reimplementation of SAC on the same on-device stack as the
NNX PPO in train_3x3_nnx_jaxenv.py.  Uses twin Q-networks, automatic
entropy tuning, a Gaussian policy with tanh squashing, and a host-side
replay buffer.

Design goals:
  - Same env interface (windfarm_env_jax.py) as the PPO baseline.
  - Same observation / action spaces, reward (including SLSQP regret),
    downstream locking, and focused wind sampling.
  - Matched training budget: 6×10⁷ environment steps, 5 seeds.
  - [256, 256] MLP for policy and twin Q-networks (per review specs).

NNX best-practices (inherited from train_3x3_nnx.py docstring):
  1. nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
  2. nnx.value_and_grad(loss_fn)(model, ...)   — no wrt
  3. optimizer.update(model, grads)
  4. @nnx.jit on NNX-containing functions (never @jax.jit)
  5. jax.random.PRNGKey(int_seed) with Python int
  6. No jax.vmap-over-NNX, no nnx.param (lower-case)

Env vars:
  N_SEEDS      (default 5)
  N_ENVS       (default 128)
  TOTAL_STEPS  (default 60_000_000)  — matched to PPO Config-E
  N_STEPS      (default 64)          — per-iter rollout length
  BATCH_SIZE   (default 256)         — SAC minibatch
  N_EPOCHS     (default 8)           — SAC updates per rollout (UTD ratio)

Outputs:
  codes/checkpoints_3x3_sac_jaxenv_v2/
    metrics_seedN.json
    policy_seedN.pkl
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

from train_3x3_nnx import MLP, NET_ARCH
from windfarm_env_jax import (
    env_reset_batched, env_step_autoreset, positions_to_jax,
)
from windfarm_env import create_wind_farm_layout_3x3


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_sac_jaxenv_v2")
os.makedirs(CKPT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Hyperparameters from environment
# ---------------------------------------------------------------------------
N_SEEDS = int(os.environ.get("N_SEEDS", 5))
N_ENVS = int(os.environ.get("N_ENVS", 16))          # SAC is off-policy: fewer envs
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 60_000_000)))
N_STEPS = int(os.environ.get("N_STEPS", 512))        # longer rollout to compensate
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
N_EPOCHS = int(os.environ.get("N_EPOCHS", 4))        # fewer epochs per rollout
MAX_EPISODE_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", 200))

# SAC-specific hyperparameters
GAMMA = float(os.environ.get("GAMMA", "0.99"))
TAU = float(os.environ.get("TAU", "0.005"))          # Polyak averaging coefficient
ACTOR_LR = float(os.environ.get("ACTOR_LR", "3e-4"))
CRITIC_LR = float(os.environ.get("CRITIC_LR", "3e-4"))
ALPHA_LR = float(os.environ.get("ALPHA_LR", "3e-4"))
REPLAY_SIZE = int(os.environ.get("REPLAY_SIZE", "1000000"))
INITIAL_ALPHA = float(os.environ.get("ALPHA", "0.2"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "0.5"))

# Use the same weight decay and LR schedule as PPO Config-E
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
LR_DECAY = os.environ.get("LR_DECAY", "1") == "1"
LR_END = float(os.environ.get("LR_END", "3e-5"))

# Action bounds (matched to PPO Config-E)
_act_bound = float(os.environ.get("ACT_BOUND", "10.0"))
ACT_LOW, ACT_HIGH = -_act_bound, _act_bound

J = int(os.environ.get("J", 3))
OUT_TAG = os.environ.get("OUT_TAG", f"n{N_ENVS}")

# ---- Focused wind sampling ----
_WIND_MIX_RAW = os.environ.get("WIND_MIXTURE", "")
WIND_MIXTURE = None
if _WIND_MIX_RAW:
    parts = [float(x.strip()) for x in _WIND_MIX_RAW.split(",")]
    if len(parts) == 3:
        WIND_MIXTURE = tuple(parts)
        print(f"# wind mixture: aligned={parts[0]}, near={parts[1]}, global={parts[2]}")

# ---- SLSQP-regret reward ----
USE_REGRET = os.environ.get("USE_REGRET", "1") == "1"
SLSQP_LOOKUP = None
if USE_REGRET:
    import json as _json
    _lt_path = os.path.join(os.path.dirname(_SCRIPT_DIR),
                            "latex_draft", "figures", "lookup_table_baseline.json")
    if os.path.exists(_lt_path):
        with open(_lt_path) as _f:
            _lt = _json.load(_f)
        _phi_g = jnp.asarray(_lt["phi_grid"], dtype=jnp.float32)
        _v_g = jnp.asarray(_lt["v_grid"], dtype=jnp.float32)
        _gain_g = jnp.asarray(_lt["gain_table"], dtype=jnp.float32)
        SLSQP_LOOKUP = (_phi_g, _v_g, _gain_g)
        print(f"# regret reward : SLSQP lookup loaded ({len(_phi_g)}x{len(_v_g)})")


# ---------------------------------------------------------------------------
# SAC Network Definitions
# ---------------------------------------------------------------------------

class SACActor(nnx.Module):
    """Gaussian policy with tanh squashing.

    Outputs (mean, log_std_offset) for the action distribution.
    Includes per-dimension learnable log_std.

    Actions are sampled as: a = ACT_BOUND * tanh(mean + exp(log_std) * eps)
    """
    def __init__(self, obs_dim, act_dim, *, rngs: nnx.Rngs):
        self.net = MLP(obs_dim, NET_ARCH, act_dim, rngs=rngs)
        # Learnable per-dimension log_std (broadcast over batch)
        init_log_std = jnp.full((act_dim,), jnp.log(0.5), dtype=jnp.float32)
        self.log_std = nnx.Param(init_log_std)
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        self._act_bound = ACT_HIGH

    def __call__(self, obs):
        """Return deterministic action (mean after tanh squashing)."""
        mean = self.net(obs)
        return jnp.tanh(mean) * self._act_bound

    def sample(self, obs, key):
        """Sample action with reparameterization trick + tanh squashing.

        Returns:
          action: (B, act_dim) in [ACT_LOW, ACT_HIGH]
          log_prob: (B,) log probability of the sampled action
        """
        mean = self.net(obs)
        log_std = jnp.broadcast_to(
            jnp.clip(self.log_std[...], self.log_std_min, self.log_std_max),
            mean.shape)
        std = jnp.exp(log_std)

        eps = jax.random.normal(key, mean.shape)
        z = mean + std * eps                                  # raw sample
        u = jnp.tanh(z)                                       # (-1, 1)
        action = u * self._act_bound                          # scale to bounds

        # Log-probability of the squashed Gaussian
        # log π(a|s) = log N(z|μ,σ) - Σ log(1 - tanh²(z)) - Σ log(act_bound)
        log_prob_z = -0.5 * (eps ** 2 + jnp.log(2.0 * jnp.pi)) - log_std
        log_prob_z = log_prob_z.sum(axis=-1)
        log_det_jac = jnp.log(1.0 - u ** 2 + 1e-6).sum(axis=-1)
        log_act_scale = jnp.log(self._act_bound) * mean.shape[-1]
        log_prob = log_prob_z - log_det_jac - log_act_scale
        return action, log_prob


class SACCritic(nnx.Module):
    """Twin Q-networks: two critics for pessimism bias reduction.

    Forward pass returns both Q-values stacked as (..., 2).
    """
    def __init__(self, obs_dim, act_dim, *, rngs: nnx.Rngs):
        self.q1 = MLP(obs_dim + act_dim, NET_ARCH, 1, rngs=rngs)
        self.q2 = MLP(obs_dim + act_dim, NET_ARCH, 1, rngs=rngs)

    def __call__(self, obs, action):
        """Return stacked Q-values: shape (B, 2)."""
        x = jnp.concatenate([obs, action], axis=-1)
        q1 = self.q1(x).squeeze(-1)
        q2 = self.q2(x).squeeze(-1)
        return jnp.stack([q1, q2], axis=-1)           # (B, 2)


# ---------------------------------------------------------------------------
# Replay Buffer (host-side numpy arrays)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size ring buffer for off-policy experience replay."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self._size = 0

    def add(self, obs, actions, rewards, next_obs, dones):
        """Add a batch of transitions. All inputs have leading dim B."""
        B = obs.shape[0]
        if B >= self.capacity:
            obs = obs[-self.capacity:]
            actions = actions[-self.capacity:]
            rewards = rewards[-self.capacity:]
            next_obs = next_obs[-self.capacity:]
            dones = dones[-self.capacity:]
            B = obs.shape[0]

        end_idx = self.ptr + B
        if end_idx <= self.capacity:
            self.obs[self.ptr:end_idx] = obs
            self.actions[self.ptr:end_idx] = actions
            self.rewards[self.ptr:end_idx] = rewards
            self.next_obs[self.ptr:end_idx] = next_obs
            self.dones[self.ptr:end_idx] = dones
        else:
            remaining = self.capacity - self.ptr
            self.obs[self.ptr:] = obs[:remaining]
            self.actions[self.ptr:] = actions[:remaining]
            self.rewards[self.ptr:] = rewards[:remaining]
            self.next_obs[self.ptr:] = next_obs[:remaining]
            self.dones[self.ptr:] = dones[:remaining]
            self.obs[:B - remaining] = obs[remaining:]
            self.actions[:B - remaining] = actions[remaining:]
            self.rewards[:B - remaining] = rewards[remaining:]
            self.next_obs[:B - remaining] = next_obs[remaining:]
            self.dones[:B - remaining] = dones[remaining:]
        self.ptr = (self.ptr + B) % self.capacity
        self._size = min(self.capacity, self._size + B)

    def sample(self, batch_size: int, rng: np.random.Generator):
        """Sample a random minibatch of transitions."""
        indices = rng.integers(0, self._size, size=batch_size)
        return (self.obs[indices], self.actions[indices],
                self.rewards[indices], self.next_obs[indices],
                self.dones[indices])

    def __len__(self):
        return self._size


# ---------------------------------------------------------------------------
# Device-side rollout (on-device, lax.scan)
# ---------------------------------------------------------------------------

def make_rollout_fn(positions_j):
    """Return a closure that runs N_STEPS of rollout on-device."""

    def _rollout(model, init_state, init_obs, init_key, n_steps_static):
        def body(carry, _):
            state, obs, key = carry
            key, sub_sample = jax.random.split(key)
            action, _ = model.sample(obs, sub_sample)
            reset_keys = jax.random.split(sub_sample, action.shape[0])
            next_state, next_obs, reward, done = env_step_autoreset(
                state, action, reset_keys, positions_j,
                j=J, max_steps=MAX_EPISODE_STEPS, randomize_wind=True,
                wind_mixture=WIND_MIXTURE,
                slsqp_lookup=SLSQP_LOOKUP,
            )
            record = dict(obs=obs, action=action, reward=reward,
                          next_obs=next_obs, done=done)
            return (next_state, next_obs, key), record

        (final_state, final_obs, final_key), traj = jax.lax.scan(
            body, (init_state, init_obs, init_key), None, length=n_steps_static)
        return final_state, final_obs, final_key, traj

    return _rollout


# ---------------------------------------------------------------------------
# SAC Updates (JAX-compiled, on-device)
# ---------------------------------------------------------------------------

def _make_critic_loss_fn(actor_gd, actor_state,
                         target_critic_gd, target_critic_state,
                         alpha, obs_b, act_b, rew_b, nobs_b, dones_b, key):
    """Return a loss function for the critic that captures frozen dependencies."""
    def _loss(critic):
        actor = nnx.merge(actor_gd, actor_state)
        target = nnx.merge(target_critic_gd, target_critic_state)

        # Current Q estimates: (B, 2)
        q_values = critic(obs_b, act_b)

        # Target Q: sample next action, compute min Q
        next_key, _ = jax.random.split(key)
        next_actions, next_log_probs = actor.sample(nobs_b, next_key)
        target_q = target(nobs_b, next_actions)              # (B, 2)
        target_q_min = jnp.min(target_q, axis=-1)            # (B,)

        # Standard SAC target
        y = rew_b + GAMMA * (1.0 - dones_b) * (target_q_min - alpha * next_log_probs)

        # MSE loss: broadcast target over both critics
        return jnp.mean((q_values - y[:, None]) ** 2)
    return _loss


def _make_actor_loss_fn(critic_gd, critic_state, alpha, obs_b, key):
    """Return a loss function for the actor that captures frozen critic."""
    def _loss(actor):
        critic = nnx.merge(critic_gd, critic_state)

        actions, log_probs = actor.sample(obs_b, key)
        q_values = critic(obs_b, actions)                    # (B, 2)
        q_min = jnp.min(q_values, axis=-1)                   # (B,)

        # SAC actor loss: minimize α·log_prob - min Q
        actor_loss = jnp.mean(alpha * log_probs - q_min)
        return actor_loss, log_probs
    return _loss


def _make_alpha_loss_fn(target_entropy, log_probs):
    """Return a loss function for log_alpha (automatic entropy tuning)."""
    def _loss(log_alpha_val):
        alpha = jnp.exp(log_alpha_val)
        return jnp.mean(-alpha * (log_probs + target_entropy))
    return _loss


# ---------------------------------------------------------------------------
# Training loop for one seed
# ---------------------------------------------------------------------------

def train_one_seed(seed: int) -> dict:
    print(f"\n{'='*60}\n# JAX-SAC seed={seed}  "
          f"device={jax.devices()[0]}  N_ENVS={N_ENVS}\n{'='*60}")

    positions_list, _, _ = create_wind_farm_layout_3x3()
    positions_j = positions_to_jax(positions_list)
    N_turb = positions_j.shape[0]

    # Observation space (same as PPO Config-E with positions)
    obs_dim_per_step = 3 * N_turb + 3       # gammas + inflow + cos/sin/v + mask
    if os.environ.get("USE_POSITIONS", "1") == "1":
        obs_dim_per_step += 2 * N_turb       # normalized (x,y) per turbine
    obs_dim = J * obs_dim_per_step
    act_dim = N_turb
    target_entropy = -act_dim

    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  target_entropy={target_entropy:.1f}")

    # Compute number of iterations
    n_iterations = max(1, TOTAL_STEPS // (N_STEPS * N_ENVS))
    actual_total = n_iterations * N_STEPS * N_ENVS
    if actual_total < TOTAL_STEPS:
        n_iterations += 1
        actual_total = n_iterations * N_STEPS * N_ENVS
    print(f"  iterations={n_iterations}  per-iter env-steps={N_STEPS*N_ENVS}  "
          f"total budget={actual_total}")

    # PRNG
    key = jax.random.PRNGKey(seed)
    model_key, reset_key, rollout_key = jax.random.split(key, 3)

    # Build SAC models
    actor = SACActor(obs_dim, act_dim, rngs=nnx.Rngs(int(model_key[0])))
    critic = SACCritic(obs_dim, act_dim, rngs=nnx.Rngs(int(model_key[0]) + 1))

    # Target critic: deep copy of critic
    target_critic = SACCritic(obs_dim, act_dim, rngs=nnx.Rngs(int(model_key[0]) + 2))
    _, critic_state = nnx.split(critic)
    target_gd, _ = nnx.split(target_critic)
    target_critic = nnx.merge(target_gd, critic_state)

    # Entropy coefficient (learnable via SGD)
    log_alpha = jnp.array(jnp.log(INITIAL_ALPHA))

    # Optimizers
    def _make_opt(lr, model):
        if LR_DECAY:
            sched = optax.cosine_decay_schedule(
                init_value=lr, decay_steps=n_iterations * N_EPOCHS,
                alpha=LR_END / lr)
            adam = (optax.adamw(sched, weight_decay=WEIGHT_DECAY)
                    if WEIGHT_DECAY > 0 else optax.adam(sched))
        else:
            adam = (optax.adamw(lr, weight_decay=WEIGHT_DECAY)
                    if WEIGHT_DECAY > 0 else optax.adam(lr))
        return nnx.Optimizer(model, optax.chain(
            optax.clip_by_global_norm(MAX_GRAD_NORM), adam), wrt=nnx.Param)

    actor_opt = _make_opt(ACTOR_LR, actor)
    critic_opt = _make_opt(CRITIC_LR, critic)
    alpha_opt = optax.adam(ALPHA_LR)
    alpha_opt_state = alpha_opt.init(log_alpha)

    # Replay buffer
    replay = ReplayBuffer(REPLAY_SIZE, obs_dim, act_dim)
    np_rng = np.random.default_rng(seed)

    # Initial vec-env reset
    reset_keys = jax.random.split(reset_key, N_ENVS)
    _static_argnames = ("j", "max_steps", "randomize_wind", "wind_mixture")
    reset_batched_jit = jax.jit(env_reset_batched, static_argnames=_static_argnames)
    state, obs = reset_batched_jit(reset_keys, positions_j,
                                   j=J, max_steps=MAX_EPISODE_STEPS,
                                   randomize_wind=True,
                                   wind_mixture=WIND_MIXTURE,
                                   slsqp_lookup=SLSQP_LOOKUP)

    # JIT-compile rollout
    rollout_fn = make_rollout_fn(positions_j)
    rollout_jit = nnx.jit(rollout_fn, static_argnums=(4,))

    # Bookkeeping
    iterations_log = []
    total_env_steps = 0
    t_train_start = time.time()

    # Episode tracking (host-side)
    running_returns = np.zeros(N_ENVS, dtype=np.float32)
    running_lens = np.zeros(N_ENVS, dtype=np.int32)
    ep_returns: list[float] = []
    ep_lens: list[int] = []

    # LCB checkpoint tracking
    best_lcb = float("-inf")
    best_ckpt_path = None
    eval_interval = max(1, n_iterations // 50)

    try:
        for iteration in range(n_iterations):
            # ---- ROLLOUT ----
            t0 = time.time()
            state, obs, rollout_key, traj = rollout_jit(
                actor, state, obs, rollout_key, N_STEPS)
            jax.block_until_ready(traj["reward"])
            t_rollout = time.time() - t0

            # Pull to host and add to replay buffer
            traj_h = jax.tree.map(np.asarray, traj)
            T = N_STEPS
            for t in range(T):
                replay.add(
                    traj_h["obs"][t], traj_h["action"][t],
                    traj_h["reward"][t], traj_h["next_obs"][t],
                    traj_h["done"][t])
                running_returns += traj_h["reward"][t]
                running_lens += 1
                d = traj_h["done"][t]
                for i, di in enumerate(d):
                    if di:
                        ep_returns.append(float(running_returns[i]))
                        ep_lens.append(int(running_lens[i]))
                        running_returns[i] = 0.0
                        running_lens[i] = 0

            total_env_steps += T * N_ENVS

            if len(replay) < BATCH_SIZE:
                continue  # Wait for enough transitions

            # ---- SAC UPDATE ----
            t1 = time.time()
            c_losses, a_losses, al_losses, alphas, lp_means = [], [], [], [], []

            # Pre-extract frozen states for all updates in this iteration
            actor_gd, actor_state_frozen = nnx.split(actor)
            tcritic_gd, tcritic_state_frozen = nnx.split(target_critic)

            for epoch in range(N_EPOCHS):
                n_updates = max(1, (T * N_ENVS) // BATCH_SIZE)
                for _ in range(n_updates):
                    batch = replay.sample(BATCH_SIZE, np_rng)
                    obs_b = jnp.asarray(batch[0])
                    act_b = jnp.asarray(batch[1])
                    rew_b = jnp.asarray(batch[2])
                    nobs_b = jnp.asarray(batch[3])
                    dones_b = jnp.asarray(batch[4])

                    alpha_val = jnp.exp(log_alpha)
                    key, ck, ak = jax.random.split(key, 3)

                    # --- Critic update ---
                    def _c_loss(critic_m):
                        actor_m = nnx.merge(actor_gd, actor_state_frozen)
                        target_m = nnx.merge(tcritic_gd, tcritic_state_frozen)
                        qv = critic_m(obs_b, act_b)
                        na, nlp = actor_m.sample(nobs_b, ck)
                        tq = target_m(nobs_b, na)
                        y = rew_b + GAMMA * (1.0 - dones_b) * (tq.min(axis=-1) - alpha_val * nlp)
                        return jnp.mean((qv - y[:, None]) ** 2)
                    c_loss, c_grads = nnx.value_and_grad(_c_loss)(critic)
                    critic_opt.update(critic, c_grads)

                    # --- Actor update ---
                    critic_gd, critic_state_frozen = nnx.split(critic)
                    def _a_loss(actor_m):
                        critic_m = nnx.merge(critic_gd, critic_state_frozen)
                        actions, log_probs = actor_m.sample(obs_b, ak)
                        q_min = critic_m(obs_b, actions).min(axis=-1)
                        return (alpha_val * log_probs - q_min).mean(), log_probs
                    (a_loss, log_probs), a_grads = \
                        nnx.value_and_grad(_a_loss, has_aux=True)(actor)
                    actor_opt.update(actor, a_grads)
                    log_prob_mean = jnp.mean(log_probs)

                    # --- Alpha update ---
                    def _al_loss(la):
                        a = jnp.exp(la)
                        return jnp.mean(-a * (log_prob_mean + target_entropy))
                    al_loss, al_grads = jax.value_and_grad(_al_loss)(log_alpha)
                    alpha_updates, alpha_opt_state = alpha_opt.update(
                        al_grads, alpha_opt_state)
                    log_alpha = optax.apply_updates(log_alpha, alpha_updates)

                    # --- Polyak update target critic ---
                    _, c_st = nnx.split(critic)
                    t_gd, t_st = nnx.split(target_critic)
                    new_t_st = jax.tree.map(
                        lambda t, s: TAU * s + (1.0 - TAU) * t, t_st, c_st)
                    target_critic = nnx.merge(t_gd, new_t_st)

                    # Update frozen states for next critic/actor update
                    actor_gd, actor_state_frozen = nnx.split(actor)
                    tcritic_gd, tcritic_state_frozen = nnx.split(target_critic)

                    c_losses.append(float(c_loss))
                    a_losses.append(float(a_loss))
                    al_losses.append(float(al_loss))
                    alphas.append(float(alpha_val))
                    lp_means.append(float(log_prob_mean))

            t_update = time.time() - t1

            # ---- Logging ----
            n_log = max(1, len(c_losses) - n_updates) if c_losses else 0
            c_avg = float(np.mean(c_losses[-n_log:])) if n_log > 0 else 0.0
            a_avg = float(np.mean(a_losses[-n_log:])) if n_log > 0 else 0.0
            al_avg = float(np.mean(al_losses[-n_log:])) if n_log > 0 else 0.0
            alpha_avg = float(np.mean(alphas[-n_log:])) if n_log > 0 else INITIAL_ALPHA
            lp_avg = float(np.mean(lp_means[-n_log:])) if n_log > 0 else 0.0
            ep_avg = float(np.mean(ep_returns[-10:])) if ep_returns else 0.0

            iter_log = dict(
                iteration=iteration, total_steps=int(total_env_steps),
                ep_return_mean=ep_avg,
                critic_loss=c_avg, actor_loss=a_avg, alpha_loss=al_avg,
                alpha=alpha_avg, log_prob_mean=lp_avg,
                t_rollout=float(t_rollout), t_update=float(t_update),
                replay_size=int(len(replay)),
            )
            iterations_log.append(iter_log)

            if (iteration + 1) % max(1, n_iterations // 10) == 0:
                elapsed = time.time() - t_train_start
                fps = total_env_steps / max(elapsed, 1e-6)
                print(f"  iter {iteration+1:4d}/{n_iterations}  "
                      f"steps {total_env_steps/1e6:.1f}M  ep_ret {ep_avg:+.2f}  "
                      f"alpha {alpha_avg:.3f}  c_loss {c_avg:.4f}  a_loss {a_avg:.4f}  "
                      f"fps {fps:.0f}  replay {len(replay):.0f}")

            # ---- LCB Checkpoint ----
            if (iteration + 1) % eval_interval == 0 and len(ep_returns) >= 10:
                recent = ep_returns[-50:] if len(ep_returns) >= 50 else ep_returns[-10:]
                m, s, n = float(np.mean(recent)), float(np.std(recent)), len(recent)
                lcb = m - 1.96 * s / np.sqrt(n)
                if lcb > best_lcb:
                    best_lcb = lcb
                    best_ckpt_path = os.path.join(CKPT_DIR, f"policy_seed{seed}_best.pkl")
                    _, best_state = nnx.split(actor)
                    with open(best_ckpt_path, "wb") as f:
                        pickle.dump(best_state, f)
                    print(f"  [LCB] new best: lcb={lcb:.3f}  mean={m:.3f}  "
                          f"std={s:.3f}  n={n}")

    except KeyboardInterrupt:
        print("\n# Early stop (KeyboardInterrupt) — saving partial metrics")

    # ---- Final save ----
    actor_path = os.path.join(CKPT_DIR, f"policy_seed{seed}.pkl")
    critic_path = os.path.join(CKPT_DIR, f"critic_seed{seed}.pkl")
    metrics_path = os.path.join(CKPT_DIR, f"metrics_seed{seed}.json")

    _, actor_state = nnx.split(actor)
    with open(actor_path, "wb") as f:
        pickle.dump(actor_state, f)
    _, critic_state = nnx.split(critic)
    with open(critic_path, "wb") as f:
        pickle.dump(critic_state, f)

    result = dict(
        seed=seed, total_env_steps=int(total_env_steps),
        n_iterations=n_iterations, ep_returns=ep_returns, ep_lens=ep_lens,
        iterations=iterations_log, best_lcb=best_lcb,
        best_ckpt=best_ckpt_path,
        train_time_s=time.time() - t_train_start,
    )
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"# SAC seed={seed} done: {total_env_steps/1e6:.1f}M steps  "
          f"train_time={result['train_time_s']/60:.1f}min  best_lcb={best_lcb:.3f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    seed_start = 0
    seed_count = N_SEEDS
    if len(sys.argv) > 1:
        seed_start = int(sys.argv[1])
    if len(sys.argv) > 2:
        seed_count = int(sys.argv[2])

    all_results = []
    for s in range(seed_start, seed_start + seed_count):
        result = train_one_seed(s)
        all_results.append(result)

    if len(all_results) > 1:
        summary = dict(
            n_seeds=len(all_results),
            total_env_steps=[r["total_env_steps"] for r in all_results],
            train_times=[r["train_time_s"] for r in all_results],
            best_lcbs=[r["best_lcb"] for r in all_results],
        )
        with open(os.path.join(CKPT_DIR, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n# All {len(all_results)} seeds done.  Summary saved.")
