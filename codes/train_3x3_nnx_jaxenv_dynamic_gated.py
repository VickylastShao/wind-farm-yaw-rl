#!/usr/bin/env python3
"""3x3 PPO training with AR(1) dynamic wind in the training environment.

Extends train_3x3_nnx_jaxenv.py: wind direction and speed evolve per step
following AR(1) dynamics instead of being fixed per episode.

This implements E1 from the peer review: re-train the steady-state-optimized
configuration under dynamic wind to test whether dynamic training resolves
the steady-state vs. dynamic responsiveness trade-off.

Key changes from the static-wind training:
1. Wind updates happen INSIDE the scan body, before each env_step call
2. downstream_mask, baseline_mw recomputed for each new wind condition
3. MAX_EPISODE_STEPS set very high to avoid autoreset disrupting AR(1)
4. All other hyperparameters identical to sens_act10 (the steady-state-opt config)
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
    ActorCritic, AttentionActorCritic, USE_ATTENTION,
    MLP, gaussian_log_prob, gaussian_entropy,
    sample_action, predict_value, ppo_train_step, compute_gae,
    LEARNING_RATE, GAMMA, GAE_LAMBDA, CLIP_RANGE,
    ENT_COEF, VF_COEF, MAX_GRAD_NORM, NET_ARCH,
)
from windfarm_env_jax import (
    env_reset_batched, env_step_autoreset, positions_to_jax,
    find_downstream_mask_jax, inflow_speeds_jax, power_output_jax,
    _build_obs_row, WindFarmJAXState,
    WIND_DIR_LOW, WIND_DIR_HIGH, WIND_SPEED_LOW, WIND_SPEED_HIGH,
)
from windfarm_env import create_wind_farm_layout_3x3

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Dynamic wind AR(1) params ──────────────────────────────────────────
AR1_MU_PHI = 263.0
AR1_RHO_PHI = 0.95
AR1_SIGMA_PHI = 2.0
AR1_MU_V = 11.0
AR1_RHO_V = 0.95
AR1_SIGMA_V = 1.0

# ── Training config (matches sens_act10) ────────────────────────────────
N_SEEDS = int(os.environ.get("N_SEEDS", 3))
SEED_START = int(os.environ.get("SEED_START", 0))
N_ENVS = int(os.environ.get("N_ENVS", 128))
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 30_000_000)))
N_STEPS = int(os.environ.get("N_STEPS", 256))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4096))
N_EPOCHS = int(os.environ.get("N_EPOCHS", 10))
# Very high max episode steps so done never triggers mid-scan,
# keeping AR(1) wind continuity across steps.
MAX_EPISODE_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", 999999))

ACT_BOUND = float(os.environ.get("ACT_BOUND", "10.0"))
ACT_LOW, ACT_HIGH = -ACT_BOUND, ACT_BOUND
J = int(os.environ.get("J", 3))

OUT_TAG = os.environ.get("OUT_TAG", "dynamic_wind")

# ── PPO training dynamics (identical to sens_act10) ────────────────────
TARGET_KL = float(os.environ.get("TARGET_KL", "0.015"))
LR_DECAY = os.environ.get("LR_DECAY", "0") == "1"
LR_END = float(os.environ.get("LR_END", "3e-5"))
LAMBDA_RATE = float(os.environ.get("LAMBDA_RATE", "0.0"))
GATE_ON = os.environ.get("GATE_ON", "0") == "1"
GATE_DPHI_IN = float(os.environ.get("GATE_DPHI_IN", "15.0"))
GATE_DPHI_OUT = float(os.environ.get("GATE_DPHI_OUT", "20.0"))
GATE_V_MAX = float(os.environ.get("GATE_V_MAX", "11.4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
GAMMA_DISC = float(os.environ.get("GAMMA_DISC", "0.995"))

# ── Focused wind sampling (for initial wind only) ──────────────────────
_WIND_MIX_RAW = os.environ.get("WIND_MIXTURE", "0.3,0.3,0.4")
WIND_MIXTURE = None
if _WIND_MIX_RAW:
    parts = [float(x.strip()) for x in _WIND_MIX_RAW.split(",")]
    if len(parts) == 3:
        WIND_MIXTURE = tuple(parts)

# ── SLSQP-regret reward ────────────────────────────────────────────────
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

# ── Pre-compile batched helpers ────────────────────────────────────────
# These are vmap'd for (N_ENVS,) batched calls inside the scan body.
_find_mask_batched = None   # lazily compiled
_inflow_batched = None
_power_batched = None
_slsqp_interp_batched = None

def _init_batched_helpers(positions_j):
    global _find_mask_batched, _inflow_batched, _power_batched, _slsqp_interp_batched
    N_t = positions_j.shape[0]
    _find_mask_batched = jax.vmap(find_downstream_mask_jax, in_axes=(None, 0, 0))
    _inflow_batched = jax.vmap(inflow_speeds_jax, in_axes=(None, 0, 0, 0))
    _power_batched = jax.vmap(power_output_jax, in_axes=(0, 0))

    if SLSQP_LOOKUP is not None:
        phi_g, v_g, gain_g = SLSQP_LOOKUP
        def _interp_one(phi, v):
            return _slsqp_gain_interp(phi, v, phi_g, v_g, gain_g)
        _slsqp_interp_batched = jax.vmap(_interp_one)

def _slsqp_gain_interp(phi, v, phi_grid, v_grid, gain_grid):
    n_phi = phi_grid.shape[0]; n_v = v_grid.shape[0]
    phi_idx = jnp.clip(jnp.searchsorted(phi_grid, phi) - 1, 0, n_phi - 2)
    v_idx = jnp.clip(jnp.searchsorted(v_grid, v) - 1, 0, n_v - 2)
    phi_lo, phi_hi = phi_grid[phi_idx], phi_grid[phi_idx + 1]
    v_lo, v_hi = v_grid[v_idx], v_grid[v_idx + 1]
    w_phi = jnp.clip((phi - phi_lo) / jnp.maximum(phi_hi - phi_lo, 1e-6), 0.0, 1.0)
    w_v = jnp.clip((v - v_lo) / jnp.maximum(v_hi - v_lo, 1e-6), 0.0, 1.0)
    gain = (gain_grid[phi_idx, v_idx] * (1 - w_phi) * (1 - w_v)
            + gain_grid[phi_idx + 1, v_idx] * w_phi * (1 - w_v)
            + gain_grid[phi_idx, v_idx + 1] * (1 - w_phi) * w_v
            + gain_grid[phi_idx + 1, v_idx + 1] * w_phi * w_v)
    return gain

# ── Rollout with dynamic wind ──────────────────────────────────────────
def make_rollout_fn_dynamic(positions_j):
    """Return a closure that runs n_steps of rollout with AR(1) wind updates.

    At each step, BEFORE calling env_step, the state's wind (phi, v) is
    updated via AR(1) and downstream_mask/baseline_mw/slsqp_opt_mw are
    recomputed for the new wind condition.
    """
    N_t = positions_j.shape[0]
    # Pre-extract lookup grids for the closure.
    _has_regret = SLSQP_LOOKUP is not None
    if _has_regret:
        _pg, _vg, _gg = SLSQP_LOOKUP

    def _rollout(model, init_state, init_obs, init_key, n_steps_static):
        def body(carry, _):
            state, obs, key = carry
            # Split keys: sample, reset, wind_phi, wind_v
            key, sub_sample, sub_reset, sub_wphi, sub_wv = jax.random.split(key, 5)

            # ── AR(1) wind update for THIS step ──
            new_phi = AR1_MU_PHI + AR1_RHO_PHI * (state.phi - AR1_MU_PHI) \
                      + AR1_SIGMA_PHI * jax.random.normal(sub_wphi, state.phi.shape)
            new_v   = AR1_MU_V   + AR1_RHO_V   * (state.v   - AR1_MU_V)   \
                      + AR1_SIGMA_V   * jax.random.normal(sub_wv,   state.v.shape)
            new_phi = jnp.clip(new_phi, WIND_DIR_LOW, WIND_DIR_HIGH)
            new_v   = jnp.clip(new_v,   WIND_SPEED_LOW, WIND_SPEED_HIGH)

            # Recompute wind-dependent state components
            new_downstream_mask = _find_mask_batched(positions_j, new_phi, new_v)
            zeros_g = jnp.zeros((state.gammas.shape[0], N_t), dtype=jnp.float32)
            new_inflow_0 = _inflow_batched(positions_j, new_phi, new_v, zeros_g)
            new_baseline_mw = jnp.sum(_power_batched(new_inflow_0, zeros_g), axis=1) / 1e6

            # SLSQP headroom for regret reward
            if _has_regret:
                new_slsqp_gain = _slsqp_interp_batched(new_phi, new_v)
                new_slsqp_opt_mw = new_baseline_mw * (1.0 + new_slsqp_gain / 100.0)
            else:
                new_slsqp_opt_mw = jnp.zeros_like(new_baseline_mw)

            # Update state with new wind
            state = state._replace(
                phi=new_phi, v=new_v,
                downstream_mask=new_downstream_mask,
                baseline_mw=new_baseline_mw,
                slsqp_opt_mw=new_slsqp_opt_mw,
            )

            # ── Model forward pass ──
            mean, log_std, value = model(obs)
            std = jnp.exp(log_std)
            eps = jax.random.normal(sub_sample, mean.shape)
            action = mean + std * eps
            logp = gaussian_log_prob(mean, log_std, action)
            action_clipped = jnp.clip(action, ACT_LOW, ACT_HIGH)

            # ── Gate logic: zero yaw outside aligned-cube regime ──
            dphi = jnp.minimum(jnp.abs(state.phi - 270.0), 360.0 - jnp.abs(state.phi - 270.0))
            in_gate = (dphi < GATE_DPHI_IN) & (state.v < GATE_V_MAX); in_gate = in_gate[:, None]
            action_clipped = jnp.where(GATE_ON & (~in_gate), jnp.zeros_like(action_clipped), action_clipped)

            # ── Env step (randomize_wind=False: wind is already updated) ──
            reset_keys = jax.random.split(sub_reset, action.shape[0])
            next_state, next_obs, reward, done = env_step_autoreset(
                state, action_clipped, reset_keys, positions_j,
                j=J, max_steps=MAX_EPISODE_STEPS,
                randomize_wind=False,  # wind updated manually
                wind_mixture=None,
                slsqp_lookup=None,     # regret handled manually above
                lambda_rate=LAMBDA_RATE,
            )

            record = dict(obs=obs, action=action_clipped, logp=logp,
                          value=value, reward=reward, done=done)
            return (next_state, next_obs, key), record

        (final_state, final_obs, final_key), traj = jax.lax.scan(
            body, (init_state, init_obs, init_key), None, length=n_steps_static)
        return final_state, final_obs, final_key, traj

    return _rollout

# ── Warm-start helper ──────────────────────────────────────────────────
def _load_policy_state(path, obs_dim, act_dim):
    if USE_ATTENTION:
        model = AttentionActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(0))
    else:
        model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(0))
    graphdef, _ = nnx.split(model)
    with open(path, "rb") as f:
        state = pickle.load(f)
    return nnx.merge(graphdef, state)

# ── Training loop ──────────────────────────────────────────────────────
def train_one_seed(seed: int) -> dict:
    print(f"\n{'='*60}\n# NNX-JAXEnv DYNAMIC-WIND PPO seed={seed}  "
          f"device={jax.devices()[0]}  N_ENVS={N_ENVS}\n{'='*60}")

    positions_list, _, _ = create_wind_farm_layout_3x3()
    positions_j = positions_to_jax(positions_list)
    N_turb = positions_j.shape[0]
    obs_dim_per_step = 3 * N_turb + 3
    if os.environ.get("USE_POSITIONS", "1") == "1":
        obs_dim_per_step += 2 * N_turb
    obs_dim = J * obs_dim_per_step
    act_dim = N_turb

    # Init batched helpers (lazy)
    _init_batched_helpers(positions_j)

    key = jax.random.PRNGKey(seed)
    model_key, reset_key, rollout_key = jax.random.split(key, 3)

    if USE_ATTENTION:
        model = AttentionActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(int(model_key[0])))
    else:
        model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(int(model_key[0])))

    _adam = (optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY)
             if WEIGHT_DECAY > 0 else optax.adam(LEARNING_RATE))
    tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), _adam)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Initial env reset (static wind, focused sampling for initial condition)
    reset_keys = jax.random.split(reset_key, N_ENVS)
    _static_argnames = ("j", "max_steps", "randomize_wind", "wind_mixture")
    reset_batched_jit = jax.jit(env_reset_batched, static_argnames=_static_argnames)
    state, obs = reset_batched_jit(reset_keys, positions_j,
                                   j=J, max_steps=MAX_EPISODE_STEPS,
                                   randomize_wind=True,
                                   wind_mixture=WIND_MIXTURE,
                                   slsqp_lookup=None)

    # JIT-compile dynamic-wind rollout
    rollout_raw = make_rollout_fn_dynamic(positions_j)
    rollout_jit = nnx.jit(rollout_raw, static_argnums=(4,))

    n_iterations = max(1, TOTAL_STEPS // (N_STEPS * N_ENVS))
    actual_n_steps = N_STEPS
    if n_iterations * N_STEPS * N_ENVS < TOTAL_STEPS:
        n_iterations += 1
    print(f"  iterations={n_iterations}  per-iter env-steps={N_STEPS*N_ENVS}  "
          f"total budget={n_iterations*N_STEPS*N_ENVS}")
    print(f"# dynamic wind : AR(1) ρ_φ={AR1_RHO_PHI} σ_φ={AR1_SIGMA_PHI}°  "
          f"ρ_v={AR1_RHO_V} σ_v={AR1_SIGMA_V} m/s")
    print(f"# action bounds: [{ACT_LOW}, {ACT_HIGH}]  J={J}  regret={USE_REGRET}")
    if WIND_MIXTURE:
        print(f"# initial wind : mixture aligned={WIND_MIXTURE[0]} "
              f"near={WIND_MIXTURE[1]} global={WIND_MIXTURE[2]}")

    # LR schedule
    if LR_DECAY:
        _total_updates = n_iterations * N_EPOCHS
        _lr_schedule = optax.cosine_decay_schedule(
            init_value=LEARNING_RATE, decay_steps=_total_updates,
            alpha=LR_END / LEARNING_RATE)
        _adam = (optax.adamw(_lr_schedule, weight_decay=WEIGHT_DECAY)
                 if WEIGHT_DECAY > 0 else optax.adam(_lr_schedule))
        tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), _adam)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        print(f"# lr schedule  : cosine {LEARNING_RATE:.0e} → {LR_END:.0e}")

    iterations_log = []
    total_env_steps = 0
    t_train_start = time.time()
    t_rollout_total = 0.0
    t_update_total = 0.0

    # Episode tracking
    running_returns = np.zeros(N_ENVS, dtype=np.float32)
    running_lens = np.zeros(N_ENVS, dtype=np.int32)
    ep_returns = []
    ep_lens = []

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

            # Episode tracking
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

            # GAE
            last_val = np.asarray(predict_value(model, obs))
            adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_val,
                                   GAMMA_DISC, GAE_LAMBDA)

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
                        jnp.asarray(b_obs[mb]), jnp.asarray(b_act[mb]),
                        jnp.asarray(b_logp[mb]), jnp.asarray(b_adv[mb]),
                        jnp.asarray(b_ret[mb]),
                        CLIP_RANGE, ENT_COEF, VF_COEF,
                    )
                    losses.append(float(loss)); pg_losses.append(float(pgl))
                    v_losses.append(float(vl)); ents.append(float(ent))
                    kls.append(float(kl)); clip_fs.append(float(cf))
                    epoch_kls.append(float(kl))
                if TARGET_KL > 0 and np.mean(epoch_kls) > TARGET_KL * 1.5:
                    early_stop = True; break
            t_update_total += time.time() - t1

            total_env_steps += actual_n_steps * N_ENVS
            elapsed = time.time() - t_train_start
            fps = total_env_steps / max(1e-9, elapsed)
            ep_rew_mean = (float(np.mean(ep_returns[-20:])) if ep_returns else float("nan"))
            ep_len_mean = (float(np.mean(ep_lens[-20:])) if ep_lens else float("nan"))
            iterations_log.append(dict(
                iteration=iteration, total_env_steps=total_env_steps,
                elapsed_s=elapsed, fps=fps, ep_rew_mean=ep_rew_mean,
                ep_len_mean=ep_len_mean, loss=float(np.mean(losses)),
                pg_loss=float(np.mean(pg_losses)), v_loss=float(np.mean(v_losses)),
                entropy=float(np.mean(ents)), approx_kl=float(np.mean(kls)),
                clip_frac=float(np.mean(clip_fs)), early_stop=early_stop,
            ))
            _es = " ES" if early_stop else ""
            print(f"  iter {iteration:3d} | steps {total_env_steps:8d} | "
                  f"fps {fps:7.0f} | ep_rew {ep_rew_mean:+8.2f} | "
                  f"loss {np.mean(losses):+.4f} | kl {np.mean(kls):.4f} | "
                  f"clip {np.mean(clip_fs):.3f}{_es}")
    finally:
        elapsed_total = time.time() - t_train_start
        _dev = str(jax.devices()[0])
        _backend = "nnx-jaxenv-gpu" if "GPU" in _dev.upper() else "nnx-jaxenv-cpu"
        metrics = dict(
            seed=seed, backend=_backend, device=_dev,
            jax_version=jax.__version__,
            total_env_steps=total_env_steps, n_envs=N_ENVS,
            n_steps=actual_n_steps, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
            wall_clock_s=elapsed_total, rollout_s=t_rollout_total,
            update_s=t_update_total,
            fps=total_env_steps / max(1e-9, elapsed_total),
            dynamic_wind=True,
            ar1_rho_phi=AR1_RHO_PHI, ar1_sigma_phi=AR1_SIGMA_PHI,
            ar1_rho_v=AR1_RHO_V, ar1_sigma_v=AR1_SIGMA_V,
            final_ep_rew_mean=(float(np.mean(ep_returns[-20:])) if ep_returns else None),
            num_completed_episodes=len(ep_returns),
            iterations=iterations_log,
        )
        out_path = os.path.join(CKPT_DIR, f"metrics_seed{seed}_{OUT_TAG}.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  wrote {out_path}")

        try:
            _, state_ck = nnx.split(model)
            with open(os.path.join(CKPT_DIR, f"policy_seed{seed}_{OUT_TAG}.pkl"), "wb") as f:
                pickle.dump(state_ck, f)
        except Exception as exc:
            print(f"[warn] policy save failed: {type(exc).__name__}: {exc}")

    return metrics


def main():
    print(f"# NNX-JAXEnv DYNAMIC-WIND PPO  jax={jax.__version__}  device={jax.devices()[0]}")
    print(f"# seeds: {N_SEEDS}  start: {SEED_START}")
    print(f"# N_ENVS: {N_ENVS}  N_STEPS: {N_STEPS}  TOTAL: {TOTAL_STEPS}")
    print(f"# OUT_TAG: {OUT_TAG}")

    all_metrics = []
    for s in range(SEED_START, N_SEEDS):
        all_metrics.append(train_one_seed(s))

    summary_path = os.path.join(CKPT_DIR, f"summary_{OUT_TAG}.json")
    summary = dict(
        backend=all_metrics[0]["backend"],
        n_envs=N_ENVS, n_seeds=len(all_metrics),
        dynamic_wind=True,
        per_seed=all_metrics,
        wall_clock_mean_s=float(np.mean([m["wall_clock_s"] for m in all_metrics])),
        fps_mean=float(np.mean([m["fps"] for m in all_metrics])),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
