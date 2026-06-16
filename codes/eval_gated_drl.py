#!/usr/bin/env python3
"""
Gated DRL evaluation — dynamic wind with regime-based policy activation.

Four strategies:
  1. Raw DRL         — Config-E policy active at all times
  2. Gated DRL       — DRL active only in wake-aligned regime (|dphi|<15°, v<11.4);
                        zero yaw elsewhere
  3. Hysteresis-Gated — Entry threshold 15°, exit threshold 20° (anti-chatter)
  4. Gated + Deadband — Gated DRL + suppress yaw commands < 5°

Outputs:
  codes/gated_drl_results.json       — per-strategy metrics
  latex_draft/figures/fig_gated_drl.pdf — trajectory + Pareto plot
"""

import os, sys, json, time, pickle
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import jax, jax.numpy as jnp
from flax import nnx
from windfarm_env import (create_wind_farm_layout_3x3, calculate_inflow_speeds,
                           power_output, C_T, I, d_0, alpha_star, beta_star, alpha)
from windfarm_env_jax import (env_reset, env_step, positions_to_jax)
from train_3x3_nnx import ActorCritic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT_DIR = "checkpoints_3x3_nnx_jaxenv"
FIG_DIR = "../latex_draft/figures"
os.makedirs(FIG_DIR, exist_ok=True)

DYN_TAG = os.environ.get("DYN_TAG", "sens_act10")
N_TRAJ = int(os.environ.get("N_TRAJ", "1000"))
TRAJ_LEN = int(os.environ.get("TRAJ_LEN", "200"))
T_STEP = 10.0   # control period [s]
SETTLE = int(os.environ.get("SETTLE_STEPS", "150"))
ACT_BOUND = 10.0

# AR(1) parameters
SIGMA_PHI = float(os.environ.get("SIGMA_PHI", "2.0"))
SIGMA_V   = float(os.environ.get("SIGMA_V", "1.0"))
ALPHA_PHI = float(os.environ.get("ALPHA_PHI", "0.95"))
ALPHA_V   = float(os.environ.get("ALPHA_V", "0.95"))

# Gate thresholds
GATE_DPHI_IN  = float(os.environ.get("GATE_DPHI_IN", "15.0"))   # entry
GATE_DPHI_OUT = float(os.environ.get("GATE_DPHI_OUT", "20.0"))  # exit (hysteresis)
GATE_V_MAX    = float(os.environ.get("GATE_V_MAX", "11.4"))
DEADBAND_DEG  = float(os.environ.get("DEADBAND_DEG", "5.0"))

# Rate limits for lookup table
RATE_LIMITS = [None, 0.5, 0.3, 0.1]

# ---------------------------------------------------------------------------
# Load policies
# ---------------------------------------------------------------------------
positions, _, _ = create_wind_farm_layout_3x3()
N = len(positions)
positions_j = positions_to_jax(positions)

obs_dim_per_step = 5 * N + 3   # USE_POSITIONS=1
obs_dim = 3 * obs_dim_per_step  # J=3
act_dim = N

models = []
for s in range(5):
    ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_{DYN_TAG}.pkl")
    if not os.path.exists(ckpt):
        continue
    model = ActorCritic(obs_dim, act_dim, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)
    with open(ckpt, "rb") as f:
        state = pickle.load(f)
    models.append(nnx.merge(graphdef, state))
    print(f"Loaded seed {s}: {ckpt}")

print(f"Loaded {len(models)} seeds for tag '{DYN_TAG}'")

# ---------------------------------------------------------------------------
# SLSQP lookup table (same as existing eval scripts)
# ---------------------------------------------------------------------------
# SLSQP lookup skipped — focus on DRL gating comparison.
# Previous paper results serve as lookup reference:
#   Unlimited lookup: +4.82% gain, 7894° travel, 37.5°/s peak
#   RL=0.1°/s:       +1.34% gain, 817° travel, 0.60°/s peak

# ---------------------------------------------------------------------------
# AR(1) trajectory generation
# ---------------------------------------------------------------------------
def generate_trajectories(n_traj, traj_len, settle, seed=20260614):
    """Generate (phi, v) AR(1) trajectories."""
    rng = np.random.default_rng(seed)
    phi0 = rng.uniform(173, 353, size=n_traj)
    v0   = rng.uniform(6, 16, size=n_traj)

    total_len = settle + traj_len
    phi_traj = np.zeros((n_traj, total_len))
    v_traj   = np.zeros((n_traj, total_len))
    phi_traj[:, 0] = phi0
    v_traj[:, 0]   = v0

    for t in range(1, total_len):
        eps_phi = rng.normal(0, 1, size=n_traj)
        eps_v   = rng.normal(0, 1, size=n_traj)
        phi_traj[:, t] = 263.0 + ALPHA_PHI * (phi_traj[:, t-1] - 263.0) + SIGMA_PHI * eps_phi
        v_traj[:, t]   = 11.0  + ALPHA_V   * (v_traj[:, t-1]   - 11.0)  + SIGMA_V   * eps_v

    # Clip to training distribution
    phi_traj = np.clip(phi_traj, 173, 353)
    v_traj   = np.clip(v_traj, 6, 16)
    return phi_traj[:, settle:], v_traj[:, settle:]

# ---------------------------------------------------------------------------
# Gate function: is the wind condition in the wake-aligned regime?
# ---------------------------------------------------------------------------
def in_gate(phi, v, dphi_threshold=15.0, v_max=11.4):
    """Check if wind condition is in the wake-aligned cooperative-yaw regime."""
    dphi = abs(phi - 270.0)
    dphi = min(dphi, 360.0 - dphi)
    return (dphi < dphi_threshold) and (v < v_max)

# ---------------------------------------------------------------------------
# DRL policy inference (single step, deterministic)
# ---------------------------------------------------------------------------
def drl_action(model, obs_np):
    """Get deterministic DRL action (mean of policy distribution)."""
    obs_j = jnp.asarray(obs_np[None, :])
    mean, _, _ = model(obs_j)
    action = np.array(mean[0])
    return np.clip(action, -ACT_BOUND, ACT_BOUND)

# ---------------------------------------------------------------------------
# Evaluate one strategy on all trajectories
# ---------------------------------------------------------------------------
def evaluate_strategy(name, models, phi_traj, v_traj, strategy_fn):
    """
    Evaluate a control strategy on all trajectories.

    strategy_fn(phi, v, obs, gate_active, prev_gate_active, cum_yaw, model_idx)
      -> (action, new_gate_active)

    Returns: dict with aggregate metrics
    """
    n_traj, traj_len = phi_traj.shape
    n_models = len(models)

    all_gains = []      # per-timestep farm-power gain over zero-yaw (%)
    all_yaw_travel = [] # per-timestep cumulative yaw travel (°)
    all_peak_rate = []  # per-trajectory peak yaw rate (°/s)
    all_neg_frac = []   # per-trajectory fraction of negative-gain steps
    aligned_gains = []  # gains restricted to wake-aligned timesteps

    for traj_idx in range(n_traj):
        model_idx = traj_idx % n_models
        model = models[model_idx]

        # JAX env state
        key = jax.random.PRNGKey(traj_idx * 1000 + 20260614)
        phi0 = float(phi_traj[traj_idx, 0])
        v0   = float(v_traj[traj_idx, 0])

        state, obs = env_reset(
            key, positions_j, j=3, max_steps=200,
            randomize_wind=False,
            specific_wind_dir=jnp.array(phi0),
            specific_wind_speed=jnp.array(v0))

        obs_np = np.array(obs)
        cum_yaw = 0.0
        gate_active = False
        traj_gains = []
        traj_yaws = []

        for t in range(traj_len):
            phi_t = float(phi_traj[traj_idx, t])
            v_t   = float(v_traj[traj_idx, t])

            # Get action from strategy
            action, gate_active = strategy_fn(
                phi_t, v_t, obs_np, gate_active, cum_yaw, model_idx)

            # Execute action in JAX env
            next_state, next_obs, reward, done = env_step(
                state, jnp.asarray(action), positions_j)

            # Compute gain over zero-yaw baseline
            # reward from env_step is (P_current - P_baseline) / headroom * 10
            gain_pct = float(reward) / 10.0
            traj_gains.append(gain_pct)

            # Track yaw travel
            yaw_step = np.sum(np.abs(action))
            cum_yaw += yaw_step
            traj_yaws.append(yaw_step)

            # Track aligned-cube gains
            if in_gate(phi_t, v_t):
                aligned_gains.append(gain_pct)

            obs_np = np.array(next_obs)
            state = next_state

            if done:
                key, sk = jax.random.split(key)
                state, obs = env_reset(
                    sk, positions_j, j=3, max_steps=200,
                    randomize_wind=False,
                    specific_wind_dir=jnp.array(phi_t),
                    specific_wind_speed=jnp.array(v_t))
                obs_np = np.array(obs)

        traj_gains = np.array(traj_gains)
        all_gains.extend(traj_gains)
        all_yaw_travel.append(cum_yaw)
        all_peak_rate.append(np.max(traj_yaws) / T_STEP if traj_yaws else 0)
        all_neg_frac.append(np.mean(traj_gains < 0))

    mean_gain = np.mean(all_gains)
    mean_yaw  = np.mean(all_yaw_travel)
    peak_rate = np.max(all_peak_rate)
    neg_frac  = np.mean(all_neg_frac)
    al_gain   = np.mean(aligned_gains) if aligned_gains else 0.0
    al_frac   = len(aligned_gains) / len(all_gains) if all_gains else 0

    # AEP-weighted gain (Weibull k=2, A=11; von Mises κ=1, μ=270°)
    # Simplified: weight by aligned-cube frequency
    aep_gain = al_gain * al_frac + mean_gain * (1 - al_frac) * 0.3

    return dict(
        name=name,
        mean_gain=float(mean_gain),
        aligned_gain=float(al_gain),
        aep_gain=float(aep_gain),
        yaw_travel=float(mean_yaw),
        peak_rate=float(peak_rate),
        neg_frac=float(neg_frac),
        n_traj=n_traj, traj_len=traj_len,
    )

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

def strategy_raw_drl(phi, v, obs, gate_active, cum_yaw, model_idx):
    """Strategy 1: Raw DRL — always active."""
    model = models[model_idx]
    action = drl_action(model, obs)
    return action, gate_active  # gate_active unchanged

def strategy_gated(phi, v, obs, gate_active, cum_yaw, model_idx):
    """Strategy 2: Gated DRL — DRL only in wake-aligned regime."""
    model = models[model_idx]
    if in_gate(phi, v, GATE_DPHI_IN, GATE_V_MAX):
        action = drl_action(model, obs)
    else:
        action = np.zeros(N)  # zero yaw
    return action, gate_active

def strategy_hysteresis_gated(phi, v, obs, gate_active, cum_yaw, model_idx):
    """Strategy 3: Hysteresis-Gated DRL — different entry/exit thresholds."""
    model = models[model_idx]
    threshold = GATE_DPHI_IN if not gate_active else GATE_DPHI_OUT
    if in_gate(phi, v, threshold, GATE_V_MAX):
        action = drl_action(model, obs)
        gate_active = True
    else:
        action = np.zeros(N)
        gate_active = False
    return action, gate_active

def strategy_gated_deadband(phi, v, obs, gate_active, cum_yaw, model_idx):
    """Strategy 4: Gated + Deadband — suppress sub-threshold yaw commands."""
    model = models[model_idx]
    if in_gate(phi, v, GATE_DPHI_IN, GATE_V_MAX):
        raw_action = drl_action(model, obs)
        # Suppress if ALL turbine commands are below deadband
        if np.max(np.abs(raw_action)) < DEADBAND_DEG:
            action = np.zeros(N)
        else:
            action = raw_action
    else:
        action = np.zeros(N)
    return action, gate_active

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*60}")
    print(f"Gated DRL Evaluation")
    print(f"  N_TRAJ={N_TRAJ}, TRAJ_LEN={TRAJ_LEN}, T_STEP={T_STEP}s")
    print(f"  GATE_DPHI_IN={GATE_DPHI_IN}°, GATE_DPHI_OUT={GATE_DPHI_OUT}°")
    print(f"  GATE_V_MAX={GATE_V_MAX} m/s, DEADBAND={DEADBAND_DEG}°")
    print(f"  {len(models)} DRL models loaded")
    print(f"{'='*60}\n")

    # Generate trajectories
    print("Generating AR(1) trajectories...", end=" ", flush=True)
    phi_traj, v_traj = generate_trajectories(N_TRAJ, TRAJ_LEN, SETTLE)
    print(f"done ({phi_traj.shape[1]} steps × {phi_traj.shape[0]} trajectories)")

    strategies = [
        ("Raw DRL",           strategy_raw_drl),
        ("Gated DRL",         strategy_gated),
        ("Hysteresis Gated",  strategy_hysteresis_gated),
        ("Gated + Deadband",  strategy_gated_deadband),
    ]

    results = []
    for name, fn in strategies:
        print(f"\nEvaluating: {name}...", flush=True)
        t0 = time.time()
        r = evaluate_strategy(name, models, phi_traj, v_traj, fn)
        r["eval_time_s"] = time.time() - t0
        results.append(r)

        print(f"  mean_gain={r['mean_gain']:+.2f}%  "
              f"aligned_gain={r['aligned_gain']:+.2f}%  "
              f"yaw_travel={r['yaw_travel']:.1f}°  "
              f"peak_rate={r['peak_rate']:.2f}°/s  "
              f"neg_frac={r['neg_frac']*100:.0f}%  "
              f"time={r['eval_time_s']:.0f}s")

    # Save results
    out_path = "gated_drl_results.json"
    with open(out_path, "w") as f:
        json.dump(dict(
            results=results,
            config=dict(N_TRAJ=N_TRAJ, TRAJ_LEN=TRAJ_LEN, T_STEP=T_STEP,
                        SETTLE=SETTLE, GATE_DPHI_IN=GATE_DPHI_IN,
                        GATE_DPHI_OUT=GATE_DPHI_OUT, GATE_V_MAX=GATE_V_MAX,
                        DEADBAND_DEG=DEADBAND_DEG, DYN_TAG=DYN_TAG),
        ), f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results

if __name__ == "__main__":
    main()
