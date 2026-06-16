# -*- coding: utf-8 -*-
"""
Reward-design ablation entry: 3x3 PPO with optional yaw-magnitude and
yaw-rate penalties added to the baseline-aligned reward.

This is a thin twin of train_3x3_nnx_jaxenv.py — same network, same PPO
hyperparameters, same JAX vec env, same checkpoint format — but threads
LAMBDA_MAG / LAMBDA_RATE through env_step_autoreset so we can A/B the
"no-penalty" (default) and "with-penalty" variants on identical seeds.

Used by Phase 2.3 ablation in the paper §Reward design.

Env vars:
  LAMBDA_MAG    (default 0.0)     yaw-magnitude penalty weight, applied to
                                  sum(gammas**2) per step
  LAMBDA_RATE   (default 0.0)     yaw-rate penalty weight, applied to
                                  sum(action**2) per step
  OUT_TAG       (default "abl")   appended to metrics filename
  + all of N_SEEDS / N_ENVS / TOTAL_STEPS / N_STEPS / BATCH_SIZE / N_EPOCHS
    as in train_3x3_nnx_jaxenv.py

Outputs:
  codes/checkpoints_3x3_nnx_jaxenv/metrics_seed{N}_{OUT_TAG}.json
  codes/checkpoints_3x3_nnx_jaxenv/policy_seed{N}_{OUT_TAG}.pkl
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
from windfarm_env import create_wind_farm_layout_3x3


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
os.makedirs(CKPT_DIR, exist_ok=True)


N_SEEDS = int(os.environ.get("N_SEEDS", 1))
N_ENVS = int(os.environ.get("N_ENVS", 256))
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 50_000_000)))
N_STEPS = int(os.environ.get("N_STEPS", 256))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4096))
N_EPOCHS = int(os.environ.get("N_EPOCHS", 10))
MAX_EPISODE_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", 200))

LAMBDA_MAG = float(os.environ.get("LAMBDA_MAG", 0.0))
LAMBDA_RATE = float(os.environ.get("LAMBDA_RATE", 0.0))
OUT_TAG = os.environ.get("OUT_TAG", "abl")

ACT_LOW, ACT_HIGH = -5.0, 5.0
J = 1


def make_rollout_fn(positions_j, lambda_mag, lambda_rate):
    """Same as train_3x3_nnx_jaxenv.make_rollout_fn but with penalty
    weights baked into the env_step_autoreset closure."""

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
                lambda_mag=lambda_mag, lambda_rate=lambda_rate,
            )
            record = dict(obs=obs, action=action_clipped, logp=logp,
                          value=value, reward=reward, done=done)
            return (next_state, next_obs, key), record

        (final_state, final_obs, final_key), traj = jax.lax.scan(
            body, (init_state, init_obs, init_key), None, length=n_steps_static)
        return final_state, final_obs, final_key, traj

    return _rollout


def train_one_seed(seed: int) -> dict:
    print(f"\n{'='*60}\n# NNX-JAXEnv PPO 3x3 ABLATION seed={seed}  "
          f"lambda_mag={LAMBDA_MAG} lambda_rate={LAMBDA_RATE}  "
          f"device={jax.devices()[0]}  N_ENVS={N_ENVS}\n{'='*60}")

    positions_list, _, _ = create_wind_farm_layout_3x3()
    positions_j = positions_to_jax(positions_list)
    N_turb = positions_j.shape[0]
    obs_dim_per_step = 3 * N_turb + 3
    obs_dim = J * obs_dim_per_step
    act_dim = N_turb

    key = jax.random.PRNGKey(seed)
    model_key, reset_key, rollout_key = jax.random.split(key, 3)
    model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(int(model_key[0])))

    tx = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(LEARNING_RATE),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    reset_keys = jax.random.split(reset_key, N_ENVS)
    reset_batched_jit = jax.jit(env_reset_batched, static_argnames=(
        "j", "max_steps", "randomize_wind"))
    state, obs = reset_batched_jit(reset_keys, positions_j,
                                   j=J, max_steps=MAX_EPISODE_STEPS,
                                   randomize_wind=True)

    rollout_raw = make_rollout_fn(positions_j, LAMBDA_MAG, LAMBDA_RATE)
    rollout_jit = nnx.jit(rollout_raw, static_argnums=(4,))

    n_iterations = max(1, TOTAL_STEPS // (N_STEPS * N_ENVS))
    actual_n_steps = N_STEPS
    if n_iterations * N_STEPS * N_ENVS < TOTAL_STEPS:
        n_iterations += 1
    print(f"  iterations={n_iterations}  per-iter env-steps={N_STEPS*N_ENVS}  "
          f"total budget={n_iterations*N_STEPS*N_ENVS}  "
          f"obs_dim={obs_dim}  act_dim={act_dim}")

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
            farm="3x3",
            lambda_mag=LAMBDA_MAG,
            lambda_rate=LAMBDA_RATE,
            ablation_tag=OUT_TAG,
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
    print(f"# NNX-JAXEnv PPO 3x3 ABLATION  jax={jax.__version__}  "
          f"device={jax.devices()[0]}")
    print(f"# lambda_mag = {LAMBDA_MAG}    lambda_rate = {LAMBDA_RATE}")
    print(f"# out tag    : {OUT_TAG}")
    print(f"# seeds      : {N_SEEDS}")
    print(f"# parallel env: {N_ENVS}")
    print(f"# total steps : {TOTAL_STEPS}")

    all_metrics = []
    for s in range(N_SEEDS):
        all_metrics.append(train_one_seed(s))

    summary = dict(
        backend=all_metrics[0]["backend"],
        farm="3x3",
        lambda_mag=LAMBDA_MAG,
        lambda_rate=LAMBDA_RATE,
        ablation_tag=OUT_TAG,
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
