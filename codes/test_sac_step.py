#!/usr/bin/env python3
"""Minimal SAC update test to debug JIT compilation issues."""
import os, sys, time
os.environ['USE_POSITIONS'] = '1'
os.environ['USE_DEFICIT'] = '1'

import jax, jax.numpy as jnp
import optax
import numpy as np
from flax import nnx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_3x3_nnx import MLP, NET_ARCH

# ---- Simple SAC networks ----
class Actor(nnx.Module):
    def __init__(self, obs_dim, act_dim, rngs):
        self.net = MLP(obs_dim, NET_ARCH, 2*act_dim, rngs=rngs)
        self.log_std = nnx.Param(jnp.full((act_dim,), -0.5))
    def sample(self, obs, key):
        x = self.net(obs)
        mean, _ = jnp.split(x, 2, -1)
        log_std = jnp.broadcast_to(jnp.clip(self.log_std[...], -20., 2.), mean.shape)
        eps = jax.random.normal(key, mean.shape)
        z = mean + jnp.exp(log_std) * eps
        u = jnp.tanh(z)
        action = u * 10.0
        log_prob = -0.5*(eps**2 + jnp.log(2*jnp.pi)) - log_std
        log_prob = (log_prob - jnp.log(1-u**2+1e-6)).sum(-1)
        return action, log_prob

class Critic(nnx.Module):
    def __init__(self, obs_dim, act_dim, rngs):
        self.q1 = MLP(obs_dim+act_dim, NET_ARCH, 1, rngs=rngs)
        self.q2 = MLP(obs_dim+act_dim, NET_ARCH, 1, rngs=rngs)
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], -1)
        return jnp.stack([self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)], -1)

# ---- Test data ----
obs_dim, act_dim = 144, 9
B = 32
key = jax.random.PRNGKey(0)
obs_b = jnp.ones((B, obs_dim))
act_b = jnp.ones((B, act_dim))
rew_b = jnp.ones((B,))
nobs_b = jnp.ones((B, obs_dim))
dones_b = jnp.zeros((B,))

# ---- Build models ----
actor = Actor(obs_dim, act_dim, rngs=nnx.Rngs(0))
critic = Critic(obs_dim, act_dim, rngs=nnx.Rngs(1))
tcritic = Critic(obs_dim, act_dim, rngs=nnx.Rngs(2))
# Copy critic -> tcritic
_, cs = nnx.split(critic)
tg, _ = nnx.split(tcritic)
tcritic = nnx.merge(tg, cs)

opt = nnx.Optimizer(critic, optax.adam(3e-4), wrt=nnx.Param)
alpha = 0.2
GAMMA = 0.99

# ---- Test 1: Simple critic forward ----
print("Test 1: Critic forward...")
q = critic(obs_b, act_b)
print(f"  Q shape: {q.shape}")  # (32, 2)

# ---- Test 2: Actor sample ----
print("Test 2: Actor sample...")
a, lp = actor.sample(obs_b, jax.random.PRNGKey(42))
print(f"  Action shape: {a.shape}, log_prob shape: {lp.shape}")  # (32,9), (32,)

# ---- Test 3: Critic loss with frozen actor + target ----
print("Test 3: Critic loss + grad...")
actor_gd, actor_st = nnx.split(actor)
tcritic_gd, tcritic_st = nnx.split(tcritic)

def critic_loss(critic_model):
    a_model = nnx.merge(actor_gd, actor_st)
    t_model = nnx.merge(tcritic_gd, tcritic_st)
    q_values = critic_model(obs_b, act_b)
    nk = jax.random.PRNGKey(1)
    na, nlp = a_model.sample(nobs_b, nk)
    tq = t_model(nobs_b, na)
    tq_min = tq.min(axis=-1)
    target = rew_b + GAMMA * (1-dones_b) * (tq_min - alpha * nlp)
    return ((q_values - target[:, None])**2).mean()

t0 = time.time()
loss, grads = nnx.value_and_grad(critic_loss)(critic)
t1 = time.time()
print(f"  Critic loss: {loss:.4f}, grad time: {t1-t0:.2f}s")

# ---- Test 4: Full update step ----
print("Test 4: Full update step (critic + actor + alpha + polyak)...")
actor_opt = nnx.Optimizer(actor, optax.adam(3e-4), wrt=nnx.Param)

# Critic update
opt.update(critic, grads)

# Actor update
critic_gd, critic_st = nnx.split(critic)
def actor_loss(actor_model):
    c_model = nnx.merge(critic_gd, critic_st)
    actions, log_probs = actor_model.sample(obs_b, jax.random.PRNGKey(2))
    q_min = c_model(obs_b, actions).min(axis=-1)
    return (alpha * log_probs - q_min).mean(), log_probs

(a_loss, lps), a_grads = nnx.value_and_grad(actor_loss, has_aux=True)(actor)
actor_opt.update(actor, a_grads)

# Polyak update
_, critic_st2 = nnx.split(critic)
tcritic_gd2, tcritic_st2 = nnx.split(tcritic)
new_st = jax.tree.map(lambda t,s: 0.005*s + 0.995*t, tcritic_st2, critic_st2)
tcritic = nnx.merge(tcritic_gd2, new_st)

t2 = time.time()
print(f"  Full step time: {t2-t1:.2f}s")
print(f"  Actor loss: {a_loss:.4f}, log_prob mean: {lps.mean():.4f}")

# ---- Test 5: JIT the full step ----
print("Test 5: JIT compile full step...")
def full_step(actor, critic, tcritic, key):
    actor_gd, actor_st = nnx.split(actor)
    tcritic_gd, tcritic_st = nnx.split(tcritic)
    def c_loss(critic_m):
        a_m = nnx.merge(actor_gd, actor_st)
        t_m = nnx.merge(tcritic_gd, tcritic_st)
        qv = critic_m(obs_b, act_b)
        nk, _ = jax.random.split(key)
        na, nlp = a_m.sample(nobs_b, nk)
        tq = t_m(nobs_b, na)
        return ((qv - (rew_b+GAMMA*(1-dones_b)*(tq.min(-1)-alpha*nlp))[:,None])**2).mean()
    return nnx.value_and_grad(c_loss)(critic)

t0 = time.time()
l, g = full_step(actor, critic, tcritic, jax.random.PRNGKey(3))
t1 = time.time()
print(f"  JIT compile + first run: {t1-t0:.2f}s")
t0 = time.time()
l, g = full_step(actor, critic, tcritic, jax.random.PRNGKey(4))
t1 = time.time()
print(f"  Second run (cached): {t1-t0:.4f}s")
print("All tests passed!")
