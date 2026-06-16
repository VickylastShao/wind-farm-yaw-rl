#!/usr/bin/env python3
"""Fast eval: batch metric computation in JIT to avoid eager per-step calls."""
import os, sys, json, pickle, time
import numpy as np
import jax, jax.numpy as jnp
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
from windfarm_env import create_wind_farm_layout_3x3
from windfarm_env_jax import (env_reset, env_step, positions_to_jax,
                               inflow_speeds_jax, power_output_jax)
from train_3x3_nnx import ActorCritic

DYN_TAG = "dynamic_wind"
N_TRAJ, TRAJ_LEN, T_STEP = 1000, 200, 10.0
CKPT_DIR = "checkpoints_3x3_nnx_jaxenv"
FIG_DIR = "../latex_draft/figures"; os.makedirs(FIG_DIR, exist_ok=True)

positions, _, _ = create_wind_farm_layout_3x3()
N = len(positions); positions_j = positions_to_jax(positions)

# Load policies
models = []
for s in range(3):
    ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_{DYN_TAG}.pkl")
    model = ActorCritic(3*(5*N+3), N, rngs=nnx.Rngs(0))
    graphdef, _ = nnx.split(model)
    with open(ckpt, "rb") as f:
        model = nnx.merge(graphdef, pickle.load(f))
    models.append(model)
print(f"Loaded {len(models)} seeds")

# JIT: rollout (TRAJ_LEN is static so lax.scan traces correctly)
@nnx.jit
def rollout(m, init_state, init_obs):
    def body(carry, _):
        s, o = carry
        mean, _, _ = m(o.reshape(1, -1))
        a = jnp.clip(mean.reshape(N), -10.0, 10.0)
        s, o, _, _ = env_step(s, a, positions_j)
        return (s, o), (s.gammas, a)
    return jax.lax.scan(body, (init_state, init_obs), None, length=TRAJ_LEN)

# JIT: batch metric computation using fori_loop
@jax.jit
def batch_metrics(gammas_seq, actions_seq, phis_arr, vs_arr):
    """Compute gain, travel over (T,) steps using fori_loop."""
    zeros_N = jnp.zeros(N, dtype=jnp.float32)
    T = gammas_seq.shape[0]
    def body(t, carry):
        total_gain, prev_g, total_travel = carry
        g = gammas_seq[t]; a = actions_seq[t]
        phi = phis_arr[t]; v = vs_arr[t]
        inflow = inflow_speeds_jax(positions_j, phi, v, g)
        pwr = jnp.sum(power_output_jax(inflow, g)) / 1e6
        base_in = inflow_speeds_jax(positions_j, phi, v, zeros_N)
        base_pwr = jnp.sum(power_output_jax(base_in, zeros_N)) / 1e6
        step_gain = jnp.where(base_pwr > 1e-6, (pwr - base_pwr) / base_pwr * 100.0, 0.0)
        step_travel = jnp.sum(jnp.abs(actions_seq[t]))
        return (total_gain + step_gain, g, total_travel + step_travel)
    sum_gain, _, sum_travel = jax.lax.fori_loop(
        0, T, body, (0.0, jnp.zeros(N, dtype=jnp.float32), 0.0))
    return sum_gain / jnp.float32(T), sum_travel

# Generate AR(1) trajectories
rng = np.random.default_rng(20260609)
all_trajs = []
for traj in range(N_TRAJ):
    phi0 = rng.uniform(173, 353); v0 = rng.uniform(6, 16)
    phis = [phi0]; vs = [v0]
    for t in range(1, TRAJ_LEN):
        phis.append(0.95*phis[-1] + 0.05*270.0 + rng.normal(0, 2.0))
        vs.append(0.95*vs[-1] + 0.05*11.4 + rng.normal(0, 1.0))
    all_trajs.append((np.clip(phis, 173, 353).astype(np.float32),
                       np.clip(vs, 6, 16).astype(np.float32)))

# Evaluate
print(f"Evaluating {len(models)} seeds on {N_TRAJ} AR(1) trajectories...")
all_seed_results = []
for seed_idx, model in enumerate(models):
    t0 = time.time()
    gains = np.zeros(N_TRAJ); travels = np.zeros(N_TRAJ)
    for traj in range(N_TRAJ):
        phis, vs = all_trajs[traj]
        key = jax.random.key(traj)
        sj, obs = env_reset(key, positions_j,
                             specific_wind_dir=jnp.float32(phis[0]),
                             specific_wind_speed=jnp.float32(vs[0]),
                             randomize_wind=False, j=3)
        _, (gammas_seq, actions_seq) = rollout(model, sj, obs)
        gain, travel = batch_metrics(
            gammas_seq, actions_seq,
            jnp.asarray(phis), jnp.asarray(vs))
        gains[traj] = float(gain); travels[traj] = float(travel)
    elapsed = time.time() - t0
    mean_rate = travels.mean() / TRAJ_LEN / T_STEP
    eff = gains.mean()/travels.mean()*100 if travels.mean()>0 else 0
    print(f"  seed {seed_idx}: gain={gains.mean():+.4f}%, travel={travels.mean():.1f}°, "
          f"rate={mean_rate:.4f}°/s, eff={eff:.4f}/100°, time={elapsed:.0f}s")
    all_seed_results.append({"gain_mean": float(gains.mean()),
                              "travel_mean": float(travels.mean()),
                              "rate_mean": float(mean_rate)})

# Summary
mean_g = np.mean([r["gain_mean"] for r in all_seed_results])
mean_t = np.mean([r["travel_mean"] for r in all_seed_results])
mean_r = np.mean([r["rate_mean"] for r in all_seed_results])
print(f"\n  MEAN (3 seeds): gain={mean_g:+.4f}%, travel={mean_t:.1f}°, rate={mean_r:.4f}°/s")

# Comparison
print(f"\n{'='*60}")
print("COMPARISON: Dynamic-trained vs Static-trained under AR(1) wind")
print(f"{'='*60}")
print(f"{'Configuration':<30s} {'Gain':>8s} {'Travel':>8s} {'Rate':>8s}")
print("-" * 55)
comparisons = {
    "DRL dynamic-trained (E1)": (mean_g, mean_t, mean_r),
    "DRL static-opt (sens_act10)": (-0.14, 29.7, 0.30),
    "DRL baseline (p0c)": (0.18, 123.0, 0.13),
    "Lookup unlimited": (4.82, 7894.0, 37.5),
    "Lookup 0.1 deg/s": (1.34, 817.0, 0.60),
}
for name, (g, t, r) in comparisons.items():
    print(f"{name:<30s} {g:+8.4f}% {t:8.1f}° {r:8.4f}°/s")

# Save
output = {
    "description": "Dynamic-wind-trained DRL policy evaluation (E1, fast)",
    "tag": DYN_TAG, "n_seeds": len(models),
    "n_trajectories": N_TRAJ, "traj_length": TRAJ_LEN, "T_step_s": T_STEP,
    "per_seed": all_seed_results,
    "mean": {"gain_mean": float(mean_g), "travel_mean": float(mean_t), "rate_mean": float(mean_r)},
}
out_path = os.path.join(FIG_DIR, "dynamic_wind_trained_eval.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
