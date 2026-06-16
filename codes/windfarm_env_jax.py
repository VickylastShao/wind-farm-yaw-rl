# -*- coding: utf-8 -*-
"""
JAX-vectorized re-implementation of `windfarm_env.WindFarmYawEnv`.

The original gym env keeps everything in numpy / Python, which means
SyncVectorEnv runs N env copies serially on the host — this is what
dominates wall-clock at 3x3 scale (see bench_3x3/ab_report.json).

This module exposes a **functional**, **vmap-friendly** equivalent so
the entire rollout (env step + actor sample) can stay on the GPU. The
public API is:

  WindFarmJAXState                — NamedTuple holding all per-env state
  env_reset(key, positions, ...)  → (state, obs)
  env_step(state, action, ...)    → (state, obs, reward, done)
  env_reset_batched(keys, ...)    → vmap'd reset over N envs
  env_step_autoreset(states, actions, keys, positions, ...)
                                  → vmap'd step with done-triggered reset

Numerical contract: for every (positions, wind_dir, wind_speed, gammas)
combination, `inflow_speeds_jax` must agree with the numpy
`windfarm_env.calculate_inflow_speeds` to within ~1e-3 m/s. See
`compare_with_numpy_env()` in the __main__ block for the cross-check.

Design notes:
  - Constants imported from windfarm_env to avoid drift (only the
    NREL-5MW parameter set is mirrored here; the Vestas V-80 set used in
    debug scripts is intentionally NOT touched -- see CLAUDE.md).
  - `find_downstream_turbines` is reimplemented matrix-style (N x N
    pair checks) with jnp.where instead of Python control flow.
  - Coordinate rotation, deflection y_d, deficit, RSS sum, power curve
    are all expressed as jnp.where / jnp.clip so the whole graph traces
    under `jax.jit` and `jax.vmap`.
  - Observation layout exactly matches the numpy env:
        [gammas (N), inflow (N), cos(phi), sin(phi), v, locked (N)]
        history buffer of length j, flattened -> obs.
"""

import os
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Re-use the canonical NREL-5MW parameter set from the numpy env. Do NOT
# fork these values -- see project CLAUDE.md.
from windfarm_env import (
    U_infinity, d_0, z_h, P_rated, rho, S, C_P, C_T, I,
    alpha_star, beta_star, alpha, u_cut_in, u_rated, u_cut_out,
)


# ---------------------------------------------------------------------------
# Pair-physics helpers (all scalar-in, scalar-out, jit-friendly).
# ---------------------------------------------------------------------------
def _calc_k_star(I_: float) -> jnp.ndarray:
    return 0.3837 * I_ + 0.003678


def _calc_x_0(C_T_: float, gamma_deg: jnp.ndarray) -> jnp.ndarray:
    gamma_rad = jnp.deg2rad(gamma_deg)
    num = jnp.cos(gamma_rad) * (1.0 + jnp.sqrt(1.0 - C_T_))
    den = jnp.sqrt(2.0) * (alpha_star * I + beta_star * (1.0 - jnp.sqrt(1.0 - C_T_)))
    return (num / den) * d_0


def _calc_theta_0(C_T_: float, gamma_rad: jnp.ndarray) -> jnp.ndarray:
    term = C_T_ * jnp.cos(gamma_rad)
    return (0.3 * gamma_rad / jnp.cos(gamma_rad)) * (1.0 - jnp.sqrt(1.0 - term))


def _calc_y_d_jax(x: jnp.ndarray, gamma_deg: jnp.ndarray) -> jnp.ndarray:
    """Wake centerline deflection at distance x (m) for a turbine yawed
    by gamma_deg. Matches calculate_y_d in windfarm_env.py."""
    gamma_rad = jnp.deg2rad(gamma_deg)
    x_0 = _calc_x_0(C_T, gamma_deg)
    k_star = _calc_k_star(I)
    theta_0 = _calc_theta_0(C_T, gamma_rad)

    # Near-field linear branch (x <= x_0).
    near = theta_0 * x

    # Far-field log branch — compute with safe guards even when masked out.
    delta_x = jnp.maximum(x - x_0, 0.0)
    sigma_y = k_star * delta_x + (jnp.cos(gamma_rad) / jnp.sqrt(8.0)) * d_0
    sigma_z = k_star * delta_x + (1.0 / jnp.sqrt(8.0)) * d_0

    term1 = 2.9 + 1.3 * jnp.sqrt(1.0 - C_T) - C_T
    term2 = jnp.sqrt(jnp.cos(gamma_rad) / (k_star ** 2 * C_T))
    base = 8.0 * sigma_y * sigma_z / (d_0 ** 2 * jnp.cos(gamma_rad))
    base = jnp.maximum(base, 1e-9)
    sqrt_base = 1.6 * jnp.sqrt(base)
    sqrt_CT = jnp.sqrt(C_T)
    log_arg = ((1.6 + sqrt_CT) * (sqrt_base - sqrt_CT)) \
              / ((1.6 - sqrt_CT) * (sqrt_base + sqrt_CT))
    log_arg = jnp.maximum(log_arg, 1e-6)
    far = theta_0 * x_0 + (theta_0 / 14.7) * term1 * term2 * jnp.log(log_arg) * d_0

    return jnp.where(x <= x_0, near, far)


def _calc_velocity_deficit_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    z_h_j: jnp.ndarray,
    gamma_deg: jnp.ndarray,
) -> jnp.ndarray:
    """Wake speed deficit at point (x,y,z) downstream of a turbine of
    hub-height z_h_j yawed by gamma_deg. Returns Δu in m/s.

    Mirrors calculate_velocity_deficit in windfarm_env.py with all the
    Python conditionals lifted into jnp.where so it traces under jit."""
    gamma_rad = jnp.deg2rad(gamma_deg)
    k_star = _calc_k_star(I)
    x_0 = _calc_x_0(C_T, gamma_deg)

    sigma_y = k_star * jnp.maximum(x - x_0, 0.0) + (jnp.cos(gamma_rad) / jnp.sqrt(8.0)) * d_0
    sigma_z = k_star * jnp.maximum(x - x_0, 0.0) + (1.0 / jnp.sqrt(8.0)) * d_0

    denom = 8.0 * (sigma_y * sigma_z / d_0 ** 2)
    denom_safe = jnp.maximum(denom, 1e-9)

    C_T_eff = C_T * jnp.cos(gamma_rad) / denom_safe
    C_T_eff = jnp.clip(C_T_eff, 0.0, 1.0)

    term1 = 1.0 - jnp.sqrt(1.0 - C_T_eff)

    exp_y = jnp.where(sigma_y > 1e-9, (y / sigma_y) ** 2, 0.0)
    exp_z = jnp.where(sigma_z > 1e-9, ((z - z_h_j) / sigma_z) ** 2, 0.0)
    term2 = jnp.exp(-0.5 * (exp_y + exp_z))

    deficit = term1 * term2 * U_infinity

    # Mask out: x <= x_0 (no wake yet) and denom near zero.
    deficit = jnp.where(x <= x_0, 0.0, deficit)
    deficit = jnp.where(denom <= 1e-9, 0.0, deficit)
    return deficit


# ---------------------------------------------------------------------------
# Whole-field inflow: vectorized over the N*N (receiver, source) pairs.
# ---------------------------------------------------------------------------
def _rotate_to_wind_frame(positions: jnp.ndarray,
                          wind_dir_meteo: jnp.ndarray) -> jnp.ndarray:
    """positions: (N, 3). Returns coords in a frame where +x is the
    along-wind direction (matches numpy env's transform)."""
    angle_rad = jnp.deg2rad(270.0 - wind_dir_meteo)
    cos_a, sin_a = jnp.cos(angle_rad), jnp.sin(angle_rad)
    x = positions[:, 0] * cos_a - positions[:, 1] * sin_a
    y = positions[:, 0] * sin_a + positions[:, 1] * cos_a
    z = positions[:, 2]
    return jnp.stack([x, y, z], axis=-1)


def inflow_speeds_jax(
    positions: jnp.ndarray,        # (N, 3) absolute coords
    wind_dir_meteo: jnp.ndarray,   # scalar, degrees, meteo convention
    wind_speed: jnp.ndarray,       # scalar, m/s (free-stream U_inf for this episode)
    gammas: jnp.ndarray,           # (N,) yaw angles, degrees
) -> jnp.ndarray:
    """N-turbine inflow speeds under the Bastankhah–Porté-Agel wake
    model with the RSS deficit aggregation used in windfarm_env.py.

    Returns inflow (N,) in m/s. Traces under jax.jit / jax.vmap."""
    rot = _rotate_to_wind_frame(positions, wind_dir_meteo)
    xs = rot[:, 0]
    ys = rot[:, 1]
    zs = rot[:, 2]

    # Broadcast to (N_receiver, N_source) pair arrays.
    x_i = xs[:, None]; y_i = ys[:, None]; z_i = zs[:, None]
    x_j = xs[None, :]; y_j = ys[None, :]; z_j = zs[None, :]
    gamma_j = gammas[None, :]

    delta_x = x_i - x_j                            # >0 means j is upstream of i

    y_d = _calc_y_d_jax(delta_x, gamma_j)
    delta_y = (y_i - y_j) - y_d

    delta_u = _calc_velocity_deficit_jax(
        delta_x, delta_y, z_i, z_j, gamma_j,
    )

    # Mask out self-pair (i==j) and non-upstream sources (delta_x <= 0).
    eye = jnp.eye(positions.shape[0], dtype=bool)
    mask = (~eye) & (delta_x > 0.0)
    delta_u = jnp.where(mask, delta_u, 0.0)

    deficit_sq_sum = jnp.sum(delta_u ** 2, axis=1)  # over source dim
    # RSS aggregation with alpha weight, only applied where there is wake.
    total_deficit = alpha * jnp.sqrt(jnp.maximum(deficit_sq_sum, 0.0))
    return wind_speed - total_deficit


# ---------------------------------------------------------------------------
# Per-turbine power (NREL-5MW power curve with cos(γ)^1.88 yaw loss).
# ---------------------------------------------------------------------------
def power_output_jax(u_eff: jnp.ndarray, gamma_deg: jnp.ndarray) -> jnp.ndarray:
    """Per-turbine power in watts, vectorized. Mirrors power_output()."""
    yaw_loss = jnp.cos(jnp.deg2rad(gamma_deg)) ** 1.88
    p_partial = 0.5 * rho * C_P * S * (u_eff ** 3) * yaw_loss
    p_partial = jnp.minimum(p_partial, P_rated)

    # Three-segment piecewise:
    #   u <= u_cut_in  -> 0
    #   u_cut_in <= u <= u_rated -> 0.5 rho C_P S u^3 cos^1.88, capped at P_rated
    #   u_rated < u <= u_cut_out -> P_rated
    #   u >= u_cut_out -> 0
    p = jnp.where(u_eff <= u_rated, p_partial, P_rated)
    p = jnp.where(u_eff <= u_cut_in, 0.0, p)
    p = jnp.where(u_eff >= u_cut_out, 0.0, p)
    return p


# ---------------------------------------------------------------------------
# Downstream-turbine identification (one call per reset; not jit-critical).
# ---------------------------------------------------------------------------
def find_downstream_mask_jax(
    positions: jnp.ndarray,        # (N, 3)
    wind_dir_meteo: jnp.ndarray,   # scalar
    wind_speed: jnp.ndarray,       # scalar
    threshold: float = 0.01,
) -> jnp.ndarray:
    """Returns a bool (N,) mask, True where the turbine is at the
    most-downstream position (its wake does not significantly hit any
    other turbine). Matches find_downstream_turbines() in the numpy env.

    Used only at reset time, then stored in env state and reused every
    step, so this never runs inside the @jit'd rollout."""
    N = positions.shape[0]
    theta_math_rad = jnp.deg2rad(jnp.mod(270.0 - wind_dir_meteo, 360.0))
    vx_flow, vy_flow = jnp.cos(theta_math_rad), jnp.sin(theta_math_rad)

    # Pair offsets (j relative to i) in global coords.
    dx = positions[None, :, 0] - positions[:, None, 0]   # (i, j)
    dy = positions[None, :, 1] - positions[:, None, 1]
    z_recv = positions[None, :, 2]                       # (1, j) -> recv = j
    z_src  = positions[:,  None, 2]                      # (i, 1) -> src  = i

    delta_x_aligned = dx * vx_flow + dy * vy_flow        # j downstream of i if > 0.1 d_0
    delta_y_aligned = dx * (-vy_flow) + dy * vx_flow

    # Source has gamma=0 in the gating logic of numpy env.
    y_def_at_j = _calc_y_d_jax(delta_x_aligned, jnp.zeros_like(delta_x_aligned))
    effective_dy = delta_y_aligned - y_def_at_j

    k_star_ = _calc_k_star(I)
    sigma_y_gate = k_star_ * jnp.maximum(delta_x_aligned, 0.0) + (1.0 / jnp.sqrt(8.0)) * d_0

    deficit_at_j = _calc_velocity_deficit_jax(
        delta_x_aligned, effective_dy, z_recv, z_src,
        jnp.zeros_like(delta_x_aligned),
    )

    is_pair_valid = (delta_x_aligned > 0.1 * d_0) \
                    & (jnp.abs(effective_dy) <= 3.0 * sigma_y_gate) \
                    & (deficit_at_j / wind_speed > threshold)

    eye = jnp.eye(N, dtype=bool)
    is_pair_valid = is_pair_valid & (~eye)

    # i is NOT "most downstream" if any j is actually waked by i.
    waking_any = jnp.any(is_pair_valid, axis=1)
    return ~waking_any   # True => locked (downstream)


# ---------------------------------------------------------------------------
# Env state + step / reset.
# ---------------------------------------------------------------------------
class WindFarmJAXState(NamedTuple):
    gammas: jnp.ndarray            # (N,) yaw angles, degrees
    phi: jnp.ndarray               # () meteo wind direction, degrees
    v: jnp.ndarray                 # () wind speed, m/s
    baseline_mw: jnp.ndarray       # () zero-yaw farm power, MW
    slsqp_opt_mw: jnp.ndarray      # () SLSQP optimum farm power, MW (or 0 if N/A)
    downstream_mask: jnp.ndarray   # (N,) bool, True = locked
    inflow: jnp.ndarray            # (N,) current inflow speeds
    history_buf: jnp.ndarray       # (j, obs_dim_per_step)
    step_count: jnp.ndarray        # () int32
    total_mw: jnp.ndarray          # () current farm power, MW


MAX_YAW = 50.0
ACT_LOW, ACT_HIGH = -5.0, 5.0
WIND_DIR_LOW, WIND_DIR_HIGH = 173.0, 353.0
WIND_SPEED_LOW, WIND_SPEED_HIGH = 6.0, 16.0

# When True, all turbines are free to yaw (no downstream locking).
_NO_LOCK = os.environ.get("NO_LOCK", "0") == "1"
if _NO_LOCK:
    print("# NO_LOCK=1: downstream locking DISABLED")
# When True, action is absolute yaw angle (not incremental).  This lets the
# RL agent work in the same decision space as SLSQP — direct yaw optimization.
_DIRECT_YAW = os.environ.get("DIRECT_YAW", "0") == "1"
if _DIRECT_YAW:
    print("# DIRECT_YAW=1: actions are absolute yaw angles (not incremental)")
# When True, observation uses wake deficit (v - inflow) instead of absolute inflow.
_USE_DEFICIT = os.environ.get("USE_DEFICIT", "0") == "1"
if _USE_DEFICIT:
    print("# USE_DEFICIT=1: observation uses wake deficit (v - inflow)")

# When True, observation includes normalized (x,y) turbine positions.
# Positions are lazily computed per layout (3×3 or 5×5) via the helper
# below rather than at module-load time, so the same env module works
# for both layouts without a hardcoded 3×3 dependency.
_USE_POSITIONS = os.environ.get("USE_POSITIONS", "0") == "1"
_POS_CACHE = {}  # keyed by (N,), stores jnp array

def _get_pos_xy_flat(N: int, positions_list: list = None):
    """Return normalized (x, y) positions for N turbines, normalizing by 7*d_0."""
    if N not in _POS_CACHE:
        if positions_list is not None:
            xy = np.array([[p[0], p[1]] for p in positions_list], dtype=np.float32)
        else:
            from windfarm_env import create_wind_farm_layout_3x3
            plist, _, _ = create_wind_farm_layout_3x3()
            xy = np.array([[p[0], p[1]] for p in plist], dtype=np.float32)
        _POS_CACHE[N] = jnp.asarray((xy / 882.0).reshape(-1))
    return _POS_CACHE[N]

if _USE_POSITIONS:
    print("# USE_POSITIONS=1: turbine coordinates in observation (lazy per-layout)")


def _build_obs_row(gammas, inflow, phi, v, locked_mask, positions_j=None):
    phi_rad = jnp.deg2rad(phi)
    wind_info = jnp.stack([jnp.cos(phi_rad), jnp.sin(phi_rad), v])
    if _USE_DEFICIT:
        inflow = v - inflow  # wake deficit
    row = jnp.concatenate([gammas, inflow, wind_info, locked_mask.astype(jnp.float32)])
    if _USE_POSITIONS:
        if positions_j is not None:
            # Extract (x, y) from full positions (N, 3), normalize, flatten.
            pos_xy = positions_j[:, :2].reshape(-1) / 882.0
        else:
            pos_xy = _get_pos_xy_flat(gammas.shape[0])
        row = jnp.concatenate([row, pos_xy])
    return row


def _slsqp_gain_interp(phi, v, phi_grid, v_grid, gain_grid):
    """Bilinear interpolation of SLSQP gain on a grid. JAX-compatible."""
    n_phi = phi_grid.shape[0]
    n_v = v_grid.shape[0]
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


def env_reset(
    key: jnp.ndarray,
    positions: jnp.ndarray,
    *,
    j: int = 1,
    max_steps: int = 200,
    randomize_wind: bool = True,
    specific_wind_dir: jnp.ndarray = None,
    specific_wind_speed: jnp.ndarray = None,
    wind_mixture: tuple = None,  # (aligned_w, near_w, global_w) for focused sampling
    slsqp_lookup: tuple = None,  # (phi_grid, v_grid, gain_grid) for regret reward
) -> Tuple[WindFarmJAXState, jnp.ndarray]:
    """Functional reset. Returns (state, obs).

    wind_mixture: optional (aligned, near, global) weights for focused wind
    sampling.  aligned = |dphi|<15°, v<11.4; near = |dphi|<35°; global = full
    range.  When None, uses uniform sampling over the full range.

    slsqp_lookup: optional (phi_grid, v_grid, gain_grid) for regret reward.
    When provided, the SLSQP optimum power is computed and stored in the state.
    """
    N = positions.shape[0]
    k_phi, k_v, k_regime = jax.random.split(key, 3)
    if specific_wind_dir is None:
        if randomize_wind:
            if wind_mixture is not None:
                aw, nw, gw = wind_mixture
                total_w = aw + nw + gw
                r = jax.random.uniform(k_regime, ()) * total_w
                # aligned-cube: |dphi|<15°, v<11.4
                phi_aligned = jax.random.uniform(k_phi, (), minval=255.0, maxval=285.0)
                v_aligned = jax.random.uniform(k_v, (), minval=WIND_SPEED_LOW, maxval=11.4)
                # near-aligned: |dphi|<35°
                phi_near = jax.random.uniform(jax.random.split(k_phi)[0], (), minval=235.0, maxval=305.0)
                v_near = jax.random.uniform(jax.random.split(k_v)[0], (), minval=WIND_SPEED_LOW, maxval=WIND_SPEED_HIGH)
                # global
                phi_global = jax.random.uniform(jax.random.split(k_phi)[1], (), minval=WIND_DIR_LOW, maxval=WIND_DIR_HIGH)
                v_global = jax.random.uniform(jax.random.split(k_v)[1], (), minval=WIND_SPEED_LOW, maxval=WIND_SPEED_HIGH)
                phi = jnp.where(r < aw, phi_aligned,
                        jnp.where(r < aw + nw, phi_near, phi_global))
                v = jnp.where(r < aw, v_aligned,
                      jnp.where(r < aw + nw, v_near, v_global))
            else:
                phi = jax.random.uniform(k_phi, (), minval=WIND_DIR_LOW, maxval=WIND_DIR_HIGH)
                v = jax.random.uniform(k_v, (), minval=WIND_SPEED_LOW, maxval=WIND_SPEED_HIGH)
        else:
            phi = jnp.float32(270.0)
            v = jnp.float32(U_infinity)
    else:
        phi = jnp.asarray(specific_wind_dir, dtype=jnp.float32)
        v = jnp.asarray(specific_wind_speed, dtype=jnp.float32)

    gammas = jnp.zeros((N,), dtype=jnp.float32)
    downstream_mask = find_downstream_mask_jax(positions, phi, v)
    if _NO_LOCK:
        downstream_mask = jnp.zeros((N,), dtype=bool)  # all free

    inflow_0 = inflow_speeds_jax(positions, phi, v, jnp.zeros((N,), jnp.float32))
    baseline_mw = jnp.sum(power_output_jax(inflow_0, jnp.zeros((N,), jnp.float32))) / 1e6

    # SLSQP optimum power (for regret reward, optional).
    if slsqp_lookup is not None:
        phi_g, v_g, gain_g = slsqp_lookup
        slsqp_gain = _slsqp_gain_interp(phi, v, phi_g, v_g, gain_g)
        slsqp_opt_mw = baseline_mw * (1.0 + slsqp_gain / 100.0)
    else:
        slsqp_opt_mw = jnp.float32(0.0)

    # First-step current values (gammas == 0 here so inflow == inflow_0).
    inflow = inflow_0
    total_mw = baseline_mw

    obs_row = _build_obs_row(gammas, inflow, phi, v, downstream_mask, positions)
    obs_dim_per_step = obs_row.shape[0]
    history_buf = jnp.broadcast_to(obs_row, (j, obs_dim_per_step))
    obs = history_buf.reshape(-1)

    state = WindFarmJAXState(
        gammas=gammas, phi=phi, v=v, baseline_mw=baseline_mw,
        slsqp_opt_mw=slsqp_opt_mw,
        downstream_mask=downstream_mask, inflow=inflow,
        history_buf=history_buf, step_count=jnp.int32(0), total_mw=total_mw,
    )
    return state, obs


def env_step(
    state: WindFarmJAXState,
    action: jnp.ndarray,
    positions: jnp.ndarray,
    *,
    max_steps: int = 200,
    lambda_mag: float = 0.0,
    lambda_rate: float = 0.0,
) -> Tuple[WindFarmJAXState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Functional step. action is the yaw-increment vector (N,).

    Optional reward penalties (used by reward-design ablation; default 0):
      - lambda_mag  weights sum(gammas**2)   — yaw magnitude
      - lambda_rate weights sum(action**2)   — yaw rate-of-change (Delta gamma)
    """
    # Mask out actions for locked turbines, accumulate, clip, force-zero again.
    if _DIRECT_YAW:
        # Absolute yaw mode: action IS the target yaw angle (like SLSQP).
        a = jnp.where(state.downstream_mask, 0.0, action)
        new_gammas = jnp.clip(a, -MAX_YAW, MAX_YAW)
        new_gammas = jnp.where(state.downstream_mask, 0.0, new_gammas)
    else:
        # Incremental mode: action is the yaw *change* (±5°/step by default).
        a = jnp.where(state.downstream_mask, 0.0, action)
        new_gammas = jnp.clip(state.gammas + a, -MAX_YAW, MAX_YAW)
        new_gammas = jnp.where(state.downstream_mask, 0.0, new_gammas)

    inflow = inflow_speeds_jax(positions, state.phi, state.v, new_gammas)
    powers = power_output_jax(inflow, new_gammas)
    total_mw = jnp.sum(powers) / 1e6

    N = positions.shape[0]
    # Regret reward: normalize by SLSQP headroom when available.
    headroom = state.slsqp_opt_mw - state.baseline_mw
    # Only use regret when headroom is meaningful (> 0.5 MW, ~1% of farm).
    use_regret = headroom > 0.5
    delta_mw = total_mw - state.baseline_mw
    # Clamp regret ratio to [-2, 2] to prevent numerical explosion.
    regret_r = jnp.clip(jnp.where(use_regret, delta_mw / headroom, 0.0), -2.0, 2.0)
    marginal_r = delta_mw / N
    reward = jnp.where(use_regret, regret_r * 10.0, marginal_r * 10.0)
    # Penalty terms (zero by default -> unchanged behavior).
    penalty = (lambda_mag * jnp.sum(new_gammas ** 2)
               + lambda_rate * jnp.sum(a ** 2))
    reward = (reward - penalty).astype(jnp.float32)

    new_row = _build_obs_row(new_gammas, inflow, state.phi, state.v,
                             state.downstream_mask, positions)
    new_history = jnp.roll(state.history_buf, shift=-1, axis=0)
    new_history = new_history.at[-1].set(new_row)

    new_step = state.step_count + 1
    done = (new_step >= max_steps).astype(jnp.float32)

    new_state = WindFarmJAXState(
        gammas=new_gammas, phi=state.phi, v=state.v,
        baseline_mw=state.baseline_mw,
        slsqp_opt_mw=state.slsqp_opt_mw,
        downstream_mask=state.downstream_mask,
        inflow=inflow, history_buf=new_history, step_count=new_step,
        total_mw=total_mw,
    )
    return new_state, new_history.reshape(-1), reward, done


# ---------------------------------------------------------------------------
# Batched / auto-reset wrappers (vmap'd; safe inside lax.scan inside jit).
# ---------------------------------------------------------------------------
def env_reset_batched(
    keys: jnp.ndarray,             # (N_envs, 2)
    positions: jnp.ndarray,        # (N_turbines, 3)
    *,
    j: int = 1,
    max_steps: int = 200,
    randomize_wind: bool = True,
    specific_wind_dir: jnp.ndarray = None,
    specific_wind_speed: jnp.ndarray = None,
    wind_mixture: tuple = None,
    slsqp_lookup: tuple = None,
):
    def _reset_one(k):
        return env_reset(k, positions, j=j, max_steps=max_steps,
                         randomize_wind=randomize_wind,
                         specific_wind_dir=specific_wind_dir,
                         specific_wind_speed=specific_wind_speed,
                         wind_mixture=wind_mixture,
                         slsqp_lookup=slsqp_lookup)
    return jax.vmap(_reset_one)(keys)


def env_step_autoreset(
    states: WindFarmJAXState,
    actions: jnp.ndarray,
    reset_keys: jnp.ndarray,
    positions: jnp.ndarray,
    *,
    j: int = 1,
    max_steps: int = 200,
    randomize_wind: bool = True,
    wind_mixture: tuple = None,
    slsqp_lookup: tuple = None,
    lambda_mag: float = 0.0,
    lambda_rate: float = 0.0,
):
    """Step every env; for envs that hit done, immediately reset using
    the matching key. Returns (new_states, obs, reward, done).

    lambda_mag / lambda_rate are forwarded to env_step (default 0 = no penalty).
    """

    def _one(state, action, k):
        next_state, next_obs, reward, done = env_step(
            state, action, positions, max_steps=max_steps,
            lambda_mag=lambda_mag, lambda_rate=lambda_rate)
        reset_state, reset_obs = env_reset(
            k, positions, j=j, max_steps=max_steps,
            randomize_wind=randomize_wind,
            wind_mixture=wind_mixture,
            slsqp_lookup=slsqp_lookup)

        def _select(a, b):
            d = done
            # Broadcast `done` (scalar) to the shape of a/b.
            return jnp.where(d.astype(bool), b, a)

        out_state = jax.tree.map(_select, next_state, reset_state)
        out_obs = jnp.where(done.astype(bool), reset_obs, next_obs)
        return out_state, out_obs, reward, done

    return jax.vmap(_one)(states, actions, reset_keys)


def positions_to_jax(positions_list) -> jnp.ndarray:
    """Convert the layout returned by create_wind_farm_layout_3x3() (a
    list of (x,y,z) tuples) into a (N, 3) jnp.float32 array."""
    return jnp.asarray(np.array(positions_list, dtype=np.float32))


# ---------------------------------------------------------------------------
# Cross-check vs numpy env. Run as `python windfarm_env_jax.py`.
# ---------------------------------------------------------------------------
def compare_with_numpy_env(verbose: bool = True) -> bool:
    """Assert numerical agreement between jax and numpy physics on a
    grid of (phi, v, gammas) cases. Returns True if all pass."""
    from windfarm_env import (
        calculate_inflow_speeds, power_output,
        find_downstream_turbines, create_wind_farm_layout_3x3,
    )

    positions_list, R, Cn = create_wind_farm_layout_3x3()
    positions_j = positions_to_jax(positions_list)
    N = len(positions_list)

    # Choose probe cases that exercise: head-on wind, partial waking,
    # off-axis, low/high wind speed, yawed-source deflection.
    rng = np.random.default_rng(2026)
    cases = []
    for phi in (200.0, 240.0, 270.0, 300.0, 340.0):
        for v in (7.0, 11.4, 15.0):
            for trial in range(3):
                if trial == 0:
                    gammas = np.zeros(N, dtype=np.float32)
                elif trial == 1:
                    gammas = np.full(N, 25.0, dtype=np.float32)
                else:
                    gammas = rng.uniform(-30.0, 30.0, size=N).astype(np.float32)
                cases.append((phi, v, gammas))

    inflow_jit = jax.jit(inflow_speeds_jax)
    n_fail_inflow = 0
    max_inflow_err = 0.0
    for phi, v, gammas in cases:
        u_np = calculate_inflow_speeds(
            positions_list, phi, C_T, I, d_0, v, gammas,
            alpha_star, beta_star, alpha,
        )
        u_jx = np.asarray(inflow_jit(positions_j, jnp.float32(phi),
                                     jnp.float32(v), jnp.asarray(gammas)))
        err = np.max(np.abs(u_np - u_jx))
        max_inflow_err = max(max_inflow_err, err)
        if err > 1e-3:
            n_fail_inflow += 1
            if verbose:
                print(f"  [inflow miss] phi={phi:.1f} v={v:.1f} err={err:.4e}")

    # Power.
    pow_jit = jax.jit(power_output_jax)
    n_fail_pow = 0
    max_pow_err = 0.0
    for u in np.linspace(2.0, 27.0, 26, dtype=np.float32):
        for g in np.linspace(-30.0, 30.0, 7, dtype=np.float32):
            p_np = power_output(float(u), float(g))
            p_jx = float(pow_jit(jnp.float32(u), jnp.float32(g)))
            err = abs(p_np - p_jx)
            max_pow_err = max(max_pow_err, err)
            if err > 1.0:                   # Watts -- ~1e-7 relative
                n_fail_pow += 1

    # Downstream mask.
    mask_jit = jax.jit(find_downstream_mask_jax)
    n_fail_mask = 0
    n_mask_cases = 0
    for phi in (180.0, 200.0, 240.0, 270.0, 300.0, 330.0, 350.0):
        for v in (8.0, 11.4, 15.0):
            ds_np = sorted(find_downstream_turbines(positions_list, phi, v))
            mask_jx = np.asarray(mask_jit(positions_j, jnp.float32(phi),
                                          jnp.float32(v)))
            ds_jx = sorted(np.flatnonzero(mask_jx).tolist())
            n_mask_cases += 1
            if ds_np != ds_jx:
                n_fail_mask += 1
                if verbose:
                    print(f"  [mask miss] phi={phi:.1f} v={v:.1f}: "
                          f"np={ds_np} jx={ds_jx}")

    print(f"inflow:    {len(cases)} cases, fails={n_fail_inflow}, "
          f"max err={max_inflow_err:.4e} m/s")
    print(f"power:     {26*7} cases, fails={n_fail_pow}, "
          f"max err={max_pow_err:.4e} W")
    print(f"downstream:{n_mask_cases} cases, mask mismatches={n_fail_mask}")

    ok = (n_fail_inflow == 0 and n_fail_pow == 0 and n_fail_mask == 0)
    print("VERDICT:", "OK" if ok else "FAIL")
    return ok


def _smoke_rollout():
    """Tiny rollout to confirm reset/step/autoreset trace under jit."""
    from windfarm_env import create_wind_farm_layout_3x3
    positions_list, _, _ = create_wind_farm_layout_3x3()
    positions_j = positions_to_jax(positions_list)
    N_envs = 4
    keys = jax.random.split(jax.random.PRNGKey(0), N_envs)
    states, obs = jax.jit(env_reset_batched, static_argnames=(
        "j", "max_steps", "randomize_wind"))(keys, positions_j)
    print("reset:  obs shape =", obs.shape, "states.gammas shape =", states.gammas.shape)

    step_jit = jax.jit(env_step_autoreset, static_argnames=(
        "j", "max_steps", "randomize_wind"))

    key = jax.random.PRNGKey(1)
    for t in range(5):
        key, sub_a, sub_r = jax.random.split(key, 3)
        action = jax.random.uniform(sub_a, (N_envs, positions_j.shape[0]),
                                    minval=-5.0, maxval=5.0)
        reset_keys = jax.random.split(sub_r, N_envs)
        states, obs, reward, done = step_jit(states, action, reset_keys,
                                             positions_j)
        print(f"  step {t}: reward={np.asarray(reward)} done={np.asarray(done)}")


if __name__ == "__main__":
    print("# windfarm_env_jax cross-check\n")
    ok = compare_with_numpy_env()
    print()
    _smoke_rollout()
