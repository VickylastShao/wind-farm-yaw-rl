# -*- coding: utf-8 -*-
"""
3x3 PPO with EVERYTHING on the GPU: NNX policy + JAX-vec env.

This is the "closed-loop on a single device" variant of
train_3x3_nnx.py — instead of importing a numpy gym env via
SyncVectorEnv (which forces a host<->device hop every step), we use
the pure-JAX env from windfarm_env_jax.py so the full rollout runs
inside one @nnx.jit'd lax.scan kernel.

Goal of this file: prove that on the 3x3 farm, once the env is also
on-device, vec width N_ENVS=128..256 lets the GPU actually outscale
SB3-CPU. The PPO maths, hyperparameters, and metrics schema are kept
*identical* to train_3x3_nnx.py so bench_compare.py can plot them
side-by-side without changes.

NNX best-practices (per train_3x3_nnx.py docstring -- same 8 rules)
are reused verbatim by importing from train_3x3_nnx.

Env vars:
  N_SEEDS      (default 1)
  N_ENVS       (default 128)        — wide vec width
  TOTAL_STEPS  (default 50_000)
  N_STEPS      (default 256)        — per-iter rollout length
  BATCH_SIZE   (default 4096)       — flat minibatch
  N_EPOCHS     (default 10)

Note on N_STEPS: with N_ENVS=128 and N_STEPS=2048 (the SB3 default), a
single iteration would store 256k transitions for a 50k-step budget,
i.e. exactly one iteration. We drop to N_STEPS=256 so the 50k budget
still gives ~1-2 iterations and the rollout buffer fits easily in
device memory.

Outputs:
  codes/checkpoints_3x3_nnx_jaxenv/
    metrics_seedN.json       (same schema as train_3x3_nnx.py)
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

# Reuse the PPO maths verbatim. See train_3x3_nnx.py for the rule-by-rule
# NNX compliance commentary; this file inherits every guarantee.
from train_3x3_nnx import (
    ActorCritic, AttentionActorCritic, USE_ATTENTION,
    MLP, gaussian_log_prob, gaussian_entropy,
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
# SEED_START lets us extend an existing N_SEEDS run without overwriting
# checkpoints. With SEED_START=3 N_SEEDS=5 we train seeds 3 and 4
# (since N_SEEDS is the *upper exclusive* bound when SEED_START is set,
# matching the existing semantics of range(SEED_START, N_SEEDS)).
SEED_START = int(os.environ.get("SEED_START", 0))
# INIT_TAG / INIT_CKPT support warm-start from a pre-trained checkpoint
# (e.g. behavior cloning).  INIT_TAG="bc" loads policy_seed{s}_bc.pkl
# for each seed; INIT_CKPT loads an explicit path (same for all seeds).
INIT_TAG = os.environ.get("INIT_TAG", "")
INIT_CKPT = os.environ.get("INIT_CKPT", "")
N_ENVS = int(os.environ.get("N_ENVS", 128))
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 50_000)))
N_STEPS = int(os.environ.get("N_STEPS", 256))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4096))
N_EPOCHS = int(os.environ.get("N_EPOCHS", 10))
MAX_EPISODE_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", 200))

DIRECT_YAW = os.environ.get("DIRECT_YAW", "0") == "1"
if DIRECT_YAW:
    ACT_LOW, ACT_HIGH = -50.0, 50.0
else:
    _act_bound = float(os.environ.get("ACT_BOUND", "5.0"))
    ACT_LOW, ACT_HIGH = -_act_bound, _act_bound
    if _act_bound != 5.0:
        print(f"# action bounds : [{-_act_bound}, {_act_bound}] (non-default)")
J = int(os.environ.get("J", 1))  # observation history length

OUT_TAG = os.environ.get("OUT_TAG", f"n{N_ENVS}")

# ---- PPO training dynamics improvements ----
TARGET_KL = float(os.environ.get("TARGET_KL", "0.015"))    # KL early-stop threshold
LR_DECAY = os.environ.get("LR_DECAY", "0") == "1"          # cosine LR decay
LR_END = float(os.environ.get("LR_END", "3e-5"))           # final LR when decaying
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0"))  # AdamW weight decay

# ---- Focused wind sampling ----
_WIND_MIX_RAW = os.environ.get("WIND_MIXTURE", "")
WIND_MIXTURE = None
if _WIND_MIX_RAW:
    parts = [float(x.strip()) for x in _WIND_MIX_RAW.split(",")]
    if len(parts) == 3:
        WIND_MIXTURE = tuple(parts)
        print(f"# wind mixture : aligned={parts[0]}, near={parts[1]}, global={parts[2]}")

# ---- SLSQP-regret reward ----
USE_REGRET = os.environ.get("USE_REGRET", "") == "1"
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
    else:
        print(f"# regret reward : WARNING lookup table not found, falling back to marginal")


# ---------------------------------------------------------------------------
# Warm-start helper: load a pre-trained policy state into a fresh model.
# ---------------------------------------------------------------------------
def _load_policy_state(path: str, obs_dim: int, act_dim: int):
    """Restore model from a pickle saved by nnx.split(model)."""
    if USE_ATTENTION:
        model = AttentionActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(0))
    else:
        model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(0))
    graphdef, _ = nnx.split(model)
    with open(path, "rb") as f:
        state = pickle.load(f)
    return nnx.merge(graphdef, state)


# ---------------------------------------------------------------------------
# Device-side rollout.  All of (sample action -> step env -> bookkeep) lives
# inside a single lax.scan call, so there is NO host<->device hop per step.
# ---------------------------------------------------------------------------
def make_rollout_fn(positions_j):
    """Return a closure that runs n_steps of rollout on-device.

    Closing over positions_j keeps it baked into the traced graph; we do not
    pass it through scan carry so the compiler can constant-fold its shape.
    """

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
                wind_mixture=WIND_MIXTURE,
                slsqp_lookup=SLSQP_LOOKUP,
            )
            record = dict(obs=obs, action=action_clipped, logp=logp,
                          value=value, reward=reward, done=done)
            return (next_state, next_obs, key), record

        (final_state, final_obs, final_key), traj = jax.lax.scan(
            body, (init_state, init_obs, init_key), None, length=n_steps_static)
        return final_state, final_obs, final_key, traj

    return _rollout


def train_one_seed(seed: int) -> dict:
    print(f"\n{'='*60}\n# NNX-JAXEnv PPO seed={seed}  "
          f"device={jax.devices()[0]}  N_ENVS={N_ENVS}\n{'='*60}")

    positions_list, _, _ = create_wind_farm_layout_3x3()
    positions_j = positions_to_jax(positions_list)
    N_turb = positions_j.shape[0]
    # obs row layout (per the jax env, mirrors the numpy env):
    #   gammas (N) + inflow (N) + cos/sin/v (3) + locked (N) = 3N + 3
    obs_dim_per_step = 3 * N_turb + 3  # gammas + inflow/deficit + cos/sin/v + mask
    if os.environ.get("USE_POSITIONS", "0") == "1":
        obs_dim_per_step += 2 * N_turb  # normalized (x, y) per turbine
    obs_dim = J * obs_dim_per_step
    act_dim = N_turb

    # PRNG plumbing -- Python int seed per the NNX best-practice rules.
    key = jax.random.PRNGKey(seed)
    model_key, reset_key, rollout_key = jax.random.split(key, 3)

    # --- Warm-start from pre-trained checkpoint (e.g. BC) ---
    init_path = None
    if INIT_CKPT:
        init_path = INIT_CKPT
    elif INIT_TAG:
        init_path = os.path.join(CKPT_DIR, f"policy_seed{seed}_{INIT_TAG}.pkl")
    if init_path and os.path.exists(init_path):
        print(f"  Warm-start from: {os.path.relpath(init_path)}")
        model = _load_policy_state(init_path, obs_dim, act_dim)
    else:
        if (INIT_CKPT or INIT_TAG) and init_path:
            print(f"  [warn] init checkpoint not found: {init_path}, "
                  f"falling back to random init")
        if USE_ATTENTION:
            model = AttentionActorCritic(obs_dim, act_dim,
                                         rngs=nnx.Rngs(int(model_key[0])))
        else:
            model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(int(model_key[0])))

    _adam = (optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY)
             if WEIGHT_DECAY > 0 else optax.adam(LEARNING_RATE))
    tx = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        _adam,
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Initial vec-env reset.
    reset_keys = jax.random.split(reset_key, N_ENVS)
    _static_argnames = ("j", "max_steps", "randomize_wind", "wind_mixture")
    reset_batched_jit = jax.jit(env_reset_batched,
                                static_argnames=_static_argnames)
    state, obs = reset_batched_jit(reset_keys, positions_j,
                                   j=J, max_steps=MAX_EPISODE_STEPS,
                                   randomize_wind=True,
                                   wind_mixture=WIND_MIXTURE,
                                   slsqp_lookup=SLSQP_LOOKUP)

    # JIT-compile the device-side rollout. `n_steps_static` is a static
    # arg so lax.scan can unroll into a single XLA kernel for the round.
    rollout_raw = make_rollout_fn(positions_j)
    rollout_jit = nnx.jit(rollout_raw, static_argnums=(4,))

    # Per-iteration plan.
    n_iterations = max(1, TOTAL_STEPS // (N_STEPS * N_ENVS))
    actual_n_steps = N_STEPS
    if n_iterations * N_STEPS * N_ENVS < TOTAL_STEPS:
        n_iterations += 1
    print(f"  iterations={n_iterations}  per-iter env-steps={N_STEPS*N_ENVS}  "
          f"total budget={n_iterations*N_STEPS*N_ENVS}")

    # Recreate optimizer with LR schedule if LR_DECAY is enabled (needs
    # n_iterations, which we just computed).
    if LR_DECAY:
        _total_updates = n_iterations * N_EPOCHS
        _lr_schedule = optax.cosine_decay_schedule(
            init_value=LEARNING_RATE,
            decay_steps=_total_updates,
            alpha=LR_END / LEARNING_RATE,
        )
        _adam = (optax.adamw(_lr_schedule, weight_decay=WEIGHT_DECAY)
                 if WEIGHT_DECAY > 0 else optax.adam(_lr_schedule))
        tx = optax.chain(
            optax.clip_by_global_norm(MAX_GRAD_NORM),
            _adam,
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        print(f"# lr schedule  : cosine {LEARNING_RATE:.0e} → {LR_END:.0e} "
              f"over {_total_updates} updates"
              f"{' + weight_decay=' + str(WEIGHT_DECAY) if WEIGHT_DECAY > 0 else ''}")

    # Bookkeeping accumulators.
    iterations_log: list[dict] = []
    total_env_steps = 0
    t_train_start = time.time()
    t_rollout_total = 0.0
    t_update_total = 0.0

    # Episode-return tracking with running per-env state (host-side).
    running_returns = np.zeros(N_ENVS, dtype=np.float32)
    running_lens = np.zeros(N_ENVS, dtype=np.int32)
    ep_returns: list[float] = []
    ep_lens: list[int] = []

    try:
        for iteration in range(n_iterations):
            # --- ROLLOUT (on-device, lax.scan) ---
            t0 = time.time()
            state, obs, rollout_key, traj = rollout_jit(
                model, state, obs, rollout_key, actual_n_steps)
            # Force materialization to time the kernel honestly.
            jax.block_until_ready(traj["reward"])
            t_rollout_total += time.time() - t0

            # Pull buffers to host for GAE + minibatch indexing.
            traj_h = jax.tree.map(np.asarray, traj)
            rew_buf = traj_h["reward"]                    # (T, N_envs)
            done_buf = traj_h["done"]
            val_buf = traj_h["value"]
            obs_buf = traj_h["obs"]                       # (T, N_envs, obs_dim)
            act_buf = traj_h["action"]                    # (T, N_envs, act_dim)
            logp_buf = traj_h["logp"]                     # (T, N_envs)

            # Update host-side episode-return tracker.
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

            # GAE bootstrap from current obs (post-scan).
            last_val = np.asarray(predict_value(model, obs))
            adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_val,
                                   GAMMA, GAE_LAMBDA)

            # Flatten (T, N_envs, ...) -> (T*N_envs, ...).
            B = actual_n_steps * N_ENVS
            b_obs = obs_buf.reshape(B, obs_dim)
            b_act = act_buf.reshape(B, act_dim)
            b_logp = logp_buf.reshape(B)
            b_adv = adv.reshape(B)
            b_ret = ret.reshape(B)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            # --- UPDATE ---
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
                # KL-targeted early stopping: if mean KL exceeds threshold,
                # stop further PPO epochs to prevent policy collapse.
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
                early_stop=early_stop,
            ))
            _es = " ES" if early_stop else ""
            print(f"  iter {iteration:3d} | steps {total_env_steps:8d} | "
                  f"fps {fps:7.0f} | ep_rew {ep_rew_mean:+8.2f} | "
                  f"loss {np.mean(losses):+.4f} | kl {np.mean(kls):.4f} | "
                  f"clip {np.mean(clip_fs):.3f}{_es}")
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
    print(f"# NNX-JAXEnv PPO  jax={jax.__version__}  device={jax.devices()[0]}")
    print(f"# seeds       : {N_SEEDS}  (start: {SEED_START})")
    print(f"# parallel env: {N_ENVS}    (out tag: {OUT_TAG})")
    print(f"# total steps : {TOTAL_STEPS}")
    print(f"# n_steps     : {N_STEPS}")
    print(f"# batch_size  : {BATCH_SIZE}")
    print(f"# n_epochs    : {N_EPOCHS}")
    if INIT_CKPT:
        print(f"# init        : {INIT_CKPT}")
    elif INIT_TAG:
        print(f"# init        : tag={INIT_TAG}")

    all_metrics = []
    for s in range(SEED_START, N_SEEDS):
        all_metrics.append(train_one_seed(s))

    # When extending an existing run (SEED_START > 0), splice the new
    # per-seed metrics into the existing summary instead of clobbering it.
    summary_path = os.path.join(CKPT_DIR, f"summary_{OUT_TAG}.json")
    merged: dict = {}
    if SEED_START > 0 and os.path.exists(summary_path):
        with open(summary_path) as f:
            old = json.load(f)
        by_seed = {m["seed"]: m for m in old.get("per_seed", [])}
        for m in all_metrics:
            by_seed[m["seed"]] = m  # new wins on collision
        merged_per_seed = [by_seed[k] for k in sorted(by_seed.keys())]
    else:
        merged_per_seed = list(all_metrics)

    summary = dict(
        backend=merged_per_seed[0]["backend"],
        n_envs=N_ENVS,
        n_seeds=len(merged_per_seed),
        per_seed=merged_per_seed,
        wall_clock_mean_s=float(np.mean(
            [m["wall_clock_s"] for m in merged_per_seed])),
        fps_mean=float(np.mean([m["fps"] for m in merged_per_seed])),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote per-seed metrics + summary to {CKPT_DIR}  "
          f"(summary now spans {len(merged_per_seed)} seeds)")


if __name__ == "__main__":
    main()
