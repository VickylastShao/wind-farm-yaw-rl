# -*- coding: utf-8 -*-
"""
SAC (Soft Actor-Critic) training for wind farm yaw control.

Wraps WindFarmYawEnv with all RL improvements ported from the JAX PPO pipeline:
  - J-step history observation (J=3)
  - Wake deficit (v - inflow) instead of absolute inflow
  - Turbine (x,y) positions normalized by 7*d_0
  - Regret reward: (P - P_baseline) / (P_SLSQP - P_baseline) with clamping
  - Focused wind sampling (aligned-cube / near-aligned / global mixture)

SB3 SAC uses off-policy replay, so sample efficiency is much higher than PPO.
"""

import os
import json
import time
import pickle

import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from windfarm_env import (
    WindFarmYawEnv, create_wind_farm_layout_3x3,
    calculate_inflow_speeds, power_output,
    C_T, I, d_0, alpha_star, beta_star, alpha,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_SCRIPT_DIR)
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_sac")
FIG_DIR = os.path.join(_PROJ_ROOT, "latex_draft", "figures")
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------
J = int(os.environ.get("J", 3))
USE_DEFICIT = os.environ.get("USE_DEFICIT", "1") == "1"
USE_POSITIONS = os.environ.get("USE_POSITIONS", "1") == "1"
USE_REGRET = os.environ.get("USE_REGRET", "1") == "1"
NO_LOCK = os.environ.get("NO_LOCK", "1") == "1"

N_SEEDS = int(os.environ.get("N_SEEDS", 5))
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 1_000_000)))
OUT_TAG = os.environ.get("OUT_TAG", "sac")

# Wind mixture: (aligned_cube, near_aligned, global)
_WIND_MIX_RAW = os.environ.get("WIND_MIXTURE", "0.5,0.3,0.2")
WIND_MIXTURE = tuple(float(x.strip()) for x in _WIND_MIX_RAW.split(","))

# SB3 SAC hyperparameters
LEARNING_RATE = float(os.environ.get("LR", "3e-4"))
GAMMA = float(os.environ.get("GAMMA", "0.99"))
TAU = float(os.environ.get("TAU", "0.005"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
BUFFER_SIZE = int(os.environ.get("BUFFER_SIZE", "300000"))
NET_ARCH_SAC = [int(x) for x in os.environ.get("NET_ARCH", "256,256").split(",")]

# ---------------------------------------------------------------------------
# SLSQP lookup table (for regret reward)
# ---------------------------------------------------------------------------
SLSQP_LOOKUP = None
if USE_REGRET:
    _lt_path = os.path.join(FIG_DIR, "lookup_table_baseline.json")
    if os.path.exists(_lt_path):
        with open(_lt_path) as f:
            _lt = json.load(f)
        _phi_g = np.array(_lt["phi_grid"], dtype=np.float32)
        _v_g = np.array(_lt["v_grid"], dtype=np.float32)
        _gain_g = np.array(_lt["gain_table"], dtype=np.float32)
        SLSQP_LOOKUP = (_phi_g, _v_g, _gain_g)
        print(f"# regret reward : SLSQP lookup loaded ({len(_phi_g)}x{len(_v_g)})")


def _slsqp_gain_interp(phi, v):
    """Bilinear interpolation of SLSQP gain."""
    phi_g, v_g, gain_g = SLSQP_LOOKUP
    n_phi, n_v = len(phi_g), len(v_g)
    pi = np.clip(np.searchsorted(phi_g, phi) - 1, 0, n_phi - 2)
    vi = np.clip(np.searchsorted(v_g, v) - 1, 0, n_v - 2)
    w_phi = np.clip((phi - phi_g[pi]) / max(phi_g[pi + 1] - phi_g[pi], 1e-6), 0, 1)
    w_v = np.clip((v - v_g[vi]) / max(v_g[vi + 1] - v_g[vi], 1e-6), 0, 1)
    g = (gain_g[pi, vi] * (1 - w_phi) * (1 - w_v)
         + gain_g[pi + 1, vi] * w_phi * (1 - w_v)
         + gain_g[pi, vi + 1] * (1 - w_phi) * w_v
         + gain_g[pi + 1, vi + 1] * w_phi * w_v)
    return g


# ---------------------------------------------------------------------------
# Custom environment wrapper with all RL improvements
# ---------------------------------------------------------------------------
class ImprovedWindFarmEnv(gym.Env):
    """Gym env wrapping WindFarmYawEnv with J-step history, deficit obs,
    turbine positions, regret reward, and focused wind sampling."""

    def __init__(self, positions, N_rows, N_cols):
        super().__init__()
        self.N = len(positions)
        self.positions = positions
        self.N_rows = N_rows
        self.N_cols = N_cols

        # Pre-compute normalized turbine positions (matching JAX env).
        self._pos_xy_flat = None
        if USE_POSITIONS:
            xy = np.array([[p[0], p[1]] for p in positions], dtype=np.float32)
            self._pos_xy_flat = (xy / 882.0).reshape(-1)

        # Observation dimension calculation.
        obs_dim_per_step = 3 * self.N + 3  # gammas + inflow + wind + locked
        if USE_POSITIONS:
            obs_dim_per_step += 2 * self.N  # pos_x + pos_y
        self._obs_dim_per_step = obs_dim_per_step
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(J * obs_dim_per_step,), dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(self.N,), dtype=np.float32)

        # Internal state.
        self._base_env = None
        self._history_buf = np.zeros((J, obs_dim_per_step), dtype=np.float32)
        self._baseline_mw = 0.0
        self._slsqp_opt_mw = 0.0

    def _build_obs(self, gammas, inflow, phi, v, locked):
        """Build one observation row matching the JAX env."""
        if USE_DEFICIT:
            inflow = v - inflow  # wake deficit
        phi_rad = np.radians(phi)
        wind_info = np.array([np.cos(phi_rad), np.sin(phi_rad), v], dtype=np.float32)
        row = np.concatenate([gammas, inflow, wind_info, locked.astype(np.float32)])
        if USE_POSITIONS:
            row = np.concatenate([row, self._pos_xy_flat])
        return row

    def _sample_wind(self):
        """Focused wind sampling (aligned-cube / near / global mixture)."""
        aw, nw, gw = WIND_MIXTURE
        total_w = aw + nw + gw
        r = np.random.uniform(0, total_w)
        if r < aw:
            phi = np.random.uniform(255, 285)
            v = np.random.uniform(6, 11.4)
        elif r < aw + nw:
            phi = np.random.uniform(235, 305)
            v = np.random.uniform(6, 16)
        else:
            phi = np.random.uniform(173, 353)
            v = np.random.uniform(6, 16)
        return phi, v

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Sample wind.
        specific_phi = options.get('specific_wind_dir') if options else None
        specific_v = options.get('specific_wind_speed') if options else None

        if specific_phi is not None:
            phi, v = specific_phi, specific_v or 11.4
        else:
            phi, v = self._sample_wind()

        # Create a one-step base env for physics computation.
        self._base_env = WindFarmYawEnv(
            self.positions, self.N_rows, self.N_cols,
            j=1, max_steps=200, randomize_wind=False)

        # Reset base env with specific wind.
        obs_base, info = self._base_env.reset(options={
            'specific_wind_dir': phi,
            'specific_wind_speed': v,
        })
        self._baseline_mw = info['baseline_mw']

        # SLSQP optimum for regret reward.
        if SLSQP_LOOKUP is not None:
            slsqp_gain = _slsqp_gain_interp(phi, v)
            self._slsqp_opt_mw = self._baseline_mw * (1.0 + slsqp_gain / 100.0)
        else:
            self._slsqp_opt_mw = 0.0

        # Build downstream mask.
        locked = np.zeros(self.N, dtype=np.float32)
        if not NO_LOCK:
            from windfarm_env import find_downstream_turbines as _fdt
            ds = _fdt(self.positions, phi, v)
            for idx in ds:
                if 0 <= idx < self.N:
                    locked[idx] = 1.0

        # Initialize history buffer.
        gammas = np.zeros(self.N, dtype=np.float32)
        inflow_0 = calculate_inflow_speeds(
            self.positions, phi, C_T, I, d_0, v,
            gammas, alpha_star, beta_star, alpha)
        obs_row = self._build_obs(gammas, inflow_0, phi, v, locked)
        self._history_buf = np.broadcast_to(obs_row, (J, self._obs_dim_per_step))

        return self._history_buf.reshape(-1).astype(np.float32), info

    def step(self, action):
        # Step the base env.
        obs_base, marginal_r, term, trunc, info = self._base_env.step(action)

        gammas = self._base_env.current_gammas
        inflow = self._base_env.current_inflow_speeds
        phi = self._base_env.current_phi
        v = self._base_env.current_v
        total_mw = self._base_env.current_total_mw

        # Compute reward.
        if USE_REGRET and self._slsqp_opt_mw > 0:
            headroom = self._slsqp_opt_mw - self._baseline_mw
            if headroom > 0.5:
                delta_mw = total_mw - self._baseline_mw
                regret_r = np.clip(delta_mw / headroom, -2.0, 2.0)
                reward = regret_r * 10.0
            else:
                reward = marginal_r
        else:
            reward = marginal_r

        # Build observation with downstream mask.
        locked = np.zeros(self.N, dtype=np.float32)
        if not NO_LOCK:
            ds = self._base_env.downstream_turbines
            for idx in ds:
                if 0 <= idx < self.N:
                    locked[idx] = 1.0

        new_row = self._build_obs(gammas, inflow, phi, v, locked)
        self._history_buf = np.roll(self._history_buf, shift=-1, axis=0)
        self._history_buf[-1] = new_row

        obs = self._history_buf.reshape(-1).astype(np.float32)
        return obs, float(reward), term, trunc, info


# ---------------------------------------------------------------------------
# Callback: log metrics each rollout.
# ---------------------------------------------------------------------------
from collections import defaultdict as _defaultdict
from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback

class LoggingCallback(_BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.records = _defaultdict(list)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if buf and len(buf):
            recent = list(buf)[-20:]
            self.records["ep_rew_mean"].append(
                float(np.mean([e["r"] for e in recent])))
            self.records["total_steps"].append(self.num_timesteps)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def train_one_seed(seed: int) -> dict:
    print(f"\n{'='*60}\n# SB3 SAC  seed={seed}\n{'='*60}")

    # Build env factory.
    positions, R, C = create_wind_farm_layout_3x3()
    def _make_env():
        return ImprovedWindFarmEnv(positions, R, C)

    venv = DummyVecEnv([_make_env for _ in range(8)])
    venv = VecMonitor(venv)

    model = SAC(
        "MlpPolicy", venv,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        policy_kwargs=dict(net_arch=dict(pi=NET_ARCH_SAC, qf=NET_ARCH_SAC)),
        seed=seed,
        verbose=1,
        device="cuda",
    )

    cb = LoggingCallback()
    t0 = time.time()
    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=cb,
                    log_interval=100)
    finally:
        elapsed = time.time() - t0

    fps = TOTAL_STEPS / max(1e-9, elapsed)
    final_rew = (cb.records["ep_rew_mean"][-1]
                 if cb.records["ep_rew_mean"] else None)

    metrics = dict(
        seed=seed, backend="sb3-sac-cuda",
        total_steps=TOTAL_STEPS,
        wall_clock_s=elapsed, fps=fps,
        final_ep_rew_mean=final_rew,
    )
    with open(os.path.join(CKPT_DIR, f"metrics_seed{seed}_{OUT_TAG}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save policy (SB3 format).
    model.save(os.path.join(CKPT_DIR, f"policy_seed{seed}_{OUT_TAG}"))
    venv.close()

    print(f"  wall-clock: {elapsed:.1f}s   fps: {fps:.0f}   "
          f"final_ep_rew: {final_rew}")
    return metrics


def main():
    print(f"# SB3 SAC  wind-farm yaw control")
    print(f"# J={J}  deficit={USE_DEFICIT}  positions={USE_POSITIONS}  "
          f"regret={USE_REGRET}  nolock={NO_LOCK}")
    print(f"# seeds={N_SEEDS}  steps={TOTAL_STEPS}  tag={OUT_TAG}")
    print(f"# wind={WIND_MIXTURE}  gamma={GAMMA}  lr={LEARNING_RATE}")

    all_metrics = []
    for s in range(N_SEEDS):
        all_metrics.append(train_one_seed(s))

    summary = dict(
        backend="sb3-sac-cuda", n_seeds=N_SEEDS,
        per_seed=all_metrics,
    )
    with open(os.path.join(CKPT_DIR, f"summary_{OUT_TAG}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote checkpoints to {CKPT_DIR}")


if __name__ == "__main__":
    main()
