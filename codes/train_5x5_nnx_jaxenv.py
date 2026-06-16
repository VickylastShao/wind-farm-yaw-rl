# -*- coding: utf-8 -*-
"""
5x5 PPO with EVERYTHING on the GPU: NNX policy + JAX-vec env.

Drop-in twin of train_3x3_nnx_jaxenv.py for the 5x5 NREL-5MW layout
(create_wind_farm_layout_5x5, same 7d_0 spacing + 7° row tilt).
All PPO maths, hyperparameters and metrics schema are shared with the
3x3 script so bench_compare.py / eval_p0c_randomized.py can ingest
both checkpoint families without code changes.

Env vars (mirrors train_3x3_nnx_jaxenv.py):
  N_SEEDS      (default 1)
  N_ENVS       (default 256)
  TOTAL_STEPS  (default 5e7 — P1 default)
  N_STEPS      (default 256)
  BATCH_SIZE   (default 4096)
  N_EPOCHS     (default 10)

Outputs:
  codes/checkpoints_5x5_nnx_jaxenv/
    metrics_seed{N}_{tag}.json
    policy_seed{N}_{tag}.pkl
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

from train_3x3_nnx import (
    ActorCritic, MLP, gaussian_log_prob, gaussian_entropy,
    sample_action, predict_value, ppo_train_step, compute_gae,
    LEARNING_RATE, GAMMA, GAE_LAMBDA, CLIP_RANGE,
    ENT_COEF, VF_COEF, MAX_GRAD_NORM, NET_ARCH,
)
from windfarm_env_jax import (
    env_reset_batched, env_step_autoreset, positions_to_jax,
)
from windfarm_env import create_wind_farm_layout_5x5


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_5x5_nnx_jaxenv")
os.makedirs(CKPT_DIR, exist_ok=True)


N_SEEDS = int(os.environ.get("N_SEEDS", 1))
N_ENVS = int(os.environ.get("N_ENVS", 256))
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 50_000_000)))
N_STEPS = int(os.environ.get("N_STEPS", 256))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4096))
N_EPOCHS = int(os.environ.get("N_EPOCHS", 10))
MAX_EPISODE_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", 200))

_act_bound = float(os.environ.get("ACT_BOUND", "5.0"))
ACT_LOW, ACT_HIGH = -_act_bound, _act_bound
J = int(os.environ.get("J", 1))

OUT_TAG = os.environ.get("OUT_TAG", "p1_5x5")

# ---- Advanced training features (mirror train_3x3_nnx_jaxenv.py) ----
_WIND_MIX_RAW = os.environ.get("WIND_MIXTURE", "")
WIND_MIXTURE = None
if _WIND_MIX_RAW:
    parts = [float(x.strip()) for x in _WIND_MIX_RAW.split(",")]
    if len(parts) == 3:
        WIND_MIXTURE = tuple(parts)

USE_REGRET = os.environ.get("USE_REGRET", "") == "1"
SLSQP_LOOKUP = None
if USE_REGRET:
    import json as _json
    _lt_path = os.path.join(os.path.dirname(__file__), "..", "latex_draft", "figures", "lookup_table_baseline.json")
    if os.path.exists(_lt_path):
        with open(_lt_path) as _f:
            _lt = _json.load(_f)
        SLSQP_LOOKUP = (jnp.asarray(_lt["phi_grid"], dtype=jnp.float32),
                         jnp.asarray(_lt["v_grid"], dtype=jnp.float32),
                         jnp.asarray(_lt["gain_table"], dtype=jnp.float32))

TARGET_KL = float(os.environ.get("TARGET_KL", "0"))
LR_DECAY = os.environ.get("LR_DECAY", "0") == "1"
LR_END = float(os.environ.get("LR_END", "3e-5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0"))
GAMMA = float(os.environ.get("GAMMA", "0.99"))
GAE_LAMBDA = float(os.environ.get("GAE_LAMBDA", "0.95"))


def make_rollout_fn(positions_j):
    """Same on-device rollout as train_3x3_nnx_jaxenv.make_rollout_fn,
    closing over positions_j so XLA can constant-fold the layout."""

    def _rollout(model, init_state, init_obs, init_key, n_steps_static):
        def body(carry, _):
            state, obs, key = carry
            key, sub_sample, sub_reset = jax.random.split(key, 3)
            mean, log_std, value = model(obs)
            std = jnp.exp(log_std)
            eps = jax.random.normal(sub_sample, mean.shape)
            action = mean + std * eps
            logp = gaussian_log_prob(mean, log_std, action)
            action_clipped = jnp.clip(action, ACT_LOW, ACT_HIGH)
            reset_keys = jax.random.split(sub_reset, action.shape[0])
            next_state, next_obs, reward, done = env_step_autoreset(
                state, action_clipped, reset_keys, positions_j,
                j=J, max_steps=MAX_EPISODE_STEPS, randomize_wind=True,
            )
            record = dict(obs=obs, action=action_clipped, logp=logp,
                          value=value, reward=reward, done=done)
            return (next_state, next_obs, key), record

        (final_state, final_obs, final_key), traj = jax.lax.scan(
            body, (init_state, init_obs, init_key), None, length=n_steps_static)
        return final_state, final_obs, final_key, traj

    return _rollout


def train_one_seed(seed: int) -> dict:
    print(f"\n{'='*60}\n# NNX-JAXEnv PPO 5x5 seed={seed}  "
          f"device={jax.devices()[0]}  N_ENVS={N_ENVS}\n{'='*60}")

    positions_list, _, _ = create_wind_farm_layout_5x5()
    positions_j = positions_to_jax(positions_list)
    N_turb = positions_j.shape[0]
    obs_dim_per_step = 3 * N_turb + 3
    if os.environ.get("USE_POSITIONS", "0") == "1":
        obs_dim_per_step += 2 * N_turb
    obs_dim = J * obs_dim_per_step
    act_dim = N_turb

    key = jax.random.PRNGKey(seed)
    model_key, reset_key, rollout_key = jax.random.split(key, 3)
    model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(int(model_key[0])))

    tx = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY) if WEIGHT_DECAY > 0 else optax.adam(LEARNING_RATE),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    reset_keys = jax.random.split(reset_key, N_ENVS)
    _static_argnames = ("j", "max_steps", "randomize_wind", "wind_mixture")
    reset_batched_jit = jax.jit(env_reset_batched,
                                static_argnames=_static_argnames)
    state, obs = reset_batched_jit(reset_keys, positions_j,
                                   j=J, max_steps=MAX_EPISODE_STEPS,
                                   randomize_wind=True,
                                   wind_mixture=WIND_MIXTURE,
                                   slsqp_lookup=SLSQP_LOOKUP)

    rollout_raw = make_rollout_fn(positions_j)
    rollout_jit = nnx.jit(rollout_raw, static_argnums=(4,))

    n_iterations = max(1, TOTAL_STEPS // (N_STEPS * N_ENVS))
    actual_n_steps = N_STEPS
    if n_iterations * N_STEPS * N_ENVS < TOTAL_STEPS:
        n_iterations += 1
    print(f"  iterations={n_iterations}  per-iter env-steps={N_STEPS*N_ENVS}  "
          f"total budget={n_iterations*N_STEPS*N_ENVS}  "
          f"obs_dim={obs_dim}  act_dim={act_dim}  J={J}  gamma={GAMMA}")

    if LR_DECAY:
        _total_updates = n_iterations * N_EPOCHS
        _lr_schedule = optax.cosine_decay_schedule(
            init_value=LEARNING_RATE, decay_steps=_total_updates,
            alpha=LR_END / LEARNING_RATE)
        _adam = (optax.adamw(_lr_schedule, weight_decay=WEIGHT_DECAY)
                 if WEIGHT_DECAY > 0 else optax.adam(_lr_schedule))
        tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), _adam)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        print(f"# lr schedule: cosine {LEARNING_RATE:.0e} -> {LR_END:.0e} over {_total_updates} updates")

    iterations_log: list[dict] = []
    total_env_steps = 0
    t_train_start = time.time()
    t_rollout_total = 0.0
    t_update_total = 0.0

    running_returns = np.zeros(N_ENVS, dtype=np.float32)
    running_lens = np.zeros(N_ENVS, dtype=np.int32)
    ep_returns: list[float] = []
    ep_lens: list[int] = []

    try:
        for iteration in range(n_iterations):
            t0 = time.time()
            state, obs, rollout_key, traj = rollout_jit(
                model, state, obs, rollout_key, actual_n_steps)
            jax.block_until_ready(traj["reward"])
            t_rollout_total += time.time() - t0

            traj_h = jax.tree.map(np.asarray, traj)
            rew_buf = traj_h["reward"]
            done_buf = traj_h["done"]
            val_buf = traj_h["value"]
            obs_buf = traj_h["obs"]
            act_buf = traj_h["action"]
            logp_buf = traj_h["logp"]

            for t in range(actual_n_steps):
                running_returns += rew_buf[t]
                running_lens += 1
                d = done_buf[t]
                for i, di in enumerate(d):
                    if di:
                        ep_returns.append(float(running_returns[i]))
                        ep_lens.append(int(running_lens[i]))
                        running_returns[i] = 0.0
                        running_lens[i] = 0

            last_val = np.asarray(predict_value(model, obs))
            adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_val,
                                   GAMMA, GAE_LAMBDA)

            B = actual_n_steps * N_ENVS
            b_obs = obs_buf.reshape(B, obs_dim)
            b_act = act_buf.reshape(B, act_dim)
            b_logp = logp_buf.reshape(B)
            b_adv = adv.reshape(B)
            b_ret = ret.reshape(B)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            t1 = time.time()
            mb_size = min(BATCH_SIZE, B)
            idx = np.arange(B)
            losses, pg_losses, v_losses, ents, kls, clip_fs = [], [], [], [], [], []
            early_stop = False
            for epoch in range(N_EPOCHS):
                np.random.shuffle(idx)
                epoch_kls = []
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
                    epoch_kls.append(float(kl))
                if TARGET_KL > 0 and np.mean(epoch_kls) > TARGET_KL * 1.5:
                    early_stop = True
                    break
            t_update_total += time.time() - t1

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
            if iteration % 10 == 0 or iteration == n_iterations - 1:
                print(f"  iter {iteration:4d} | steps {total_env_steps:9d} | "
                      f"fps {fps:7.0f} | ep_rew {ep_rew_mean:+8.2f} | "
                      f"loss {np.mean(losses):+.4f} | kl {np.mean(kls):.4f} | "
                      f"clip {np.mean(clip_fs):.3f}")
    finally:
        elapsed_total = time.time() - t_train_start
        _dev = str(jax.devices()[0])
        _backend = ("nnx-jaxenv-cpu" if "CPU" in _dev.upper()
                    else "nnx-jaxenv-gpu")
        metrics = dict(
            seed=seed,
            backend=_backend,
            device=_dev,
            jax_version=jax.__version__,
            flax_version=__import__("flax").__version__,
            optax_version=optax.__version__,
            n_seeds=N_SEEDS,
            n_envs=N_ENVS,
            farm="5x5",
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
        out_path = os.path.join(CKPT_DIR, f"metrics_seed{seed}_{OUT_TAG}.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  wrote {out_path}")

        try:
            _, state_ck = nnx.split(model)
            with open(os.path.join(CKPT_DIR,
                                   f"policy_seed{seed}_{OUT_TAG}.pkl"), "wb") as f:
                pickle.dump(state_ck, f)
        except Exception as exc:
            print(f"[warn] policy save failed: {type(exc).__name__}: {exc}")

    return metrics


def main():
    print(f"# NNX-JAXEnv PPO 5x5  jax={jax.__version__}  device={jax.devices()[0]}")
    print(f"# seeds       : {N_SEEDS}")
    print(f"# parallel env: {N_ENVS}    (out tag: {OUT_TAG})")
    print(f"# total steps : {TOTAL_STEPS}")
    print(f"# n_steps     : {N_STEPS}")
    print(f"# batch_size  : {BATCH_SIZE}")
    print(f"# n_epochs    : {N_EPOCHS}")

    all_metrics = []
    for s in range(N_SEEDS):
        all_metrics.append(train_one_seed(s))

    summary = dict(
        backend=all_metrics[0]["backend"],
        farm="5x5",
        n_envs=N_ENVS,
        n_seeds=N_SEEDS,
        per_seed=all_metrics,
        wall_clock_mean_s=float(np.mean(
            [m["wall_clock_s"] for m in all_metrics])),
        fps_mean=float(np.mean([m["fps"] for m in all_metrics])),
    )
    with open(os.path.join(CKPT_DIR, f"summary_{OUT_TAG}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote per-seed metrics + summary to {CKPT_DIR}")


if __name__ == "__main__":
    main()
