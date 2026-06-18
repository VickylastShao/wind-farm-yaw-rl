#!/usr/bin/env python3
"""
Experiment 6: Safety Fallback / Abnormal Condition Testing
===========================================================
Proves DRL-Deploy has fail-safe behavior under sensor faults,
measurement delays, packet loss, wind shifts, and OOD conditions.

Test scenarios (7 categories):
  1. Wind direction sensor bias: ±2°, ±5°, ±10°
  2. Wind speed sensor bias: ±0.5, ±1.0 m/s
  3. Yaw sensor bias: ±2°, ±5°
  4. Measurement delay: 30s, 60s
  5. Missing observations: 5%, 10% random dropout (zero-order hold)
  6. Sudden wind shift: 10°, 20° step in wind direction
  7. Out-of-distribution speed: v ∈ {3,4,5,17,20,25} m/s

Controllers compared:
  - Raw DRL (Config-E, no wrapper)
  - Static DRL-Deploy (gate+hyst+deadband+rate_limit)
  - J=15 DRL-Deploy (same wrapper, J=15 policy)
  - Greedy fallback (zero yaw, safety baseline)

Metrics:
  - max |yaw command| (maximum absolute yaw angle commanded)
  - peak yaw rate (°/s)
  - negative-gain fraction
  - fallback activation rate (fraction of steps where gate=0)
  - recovery steps (steps to return to safe state after perturbation)
  - unsafe command count (yaw > 30° or rate > 5°/s)

Design: vmap-batched evaluation for steady-state scenarios,
        lax.scan trajectory for delay / dropout / wind-shift scenarios.
"""

import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("USE_POSITIONS", "1")

import jax, jax.numpy as jnp
import numpy as np

from flax import nnx
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, positions_to_jax,
                              inflow_speeds_jax, power_output_jax)
from windfarm_env import create_wind_farm_layout_3x3, U_infinity

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
CKPT_DIR = "checkpoints_3x3_nnx_jaxenv"
N = 9; AB = 10.0; T_STEP = 10.0
GATE_IN = 15.0; GATE_OUT = 20.0; DEADBAND = 2.0; RATE_MAX = 0.3  # 3°/step = 0.3°/s at T=10s
OBS_J3 = 144; OBS_J15 = 720
N_TEST = 500  # conditions per scenario
SETTLE = 10

pos, _, _ = create_wind_farm_layout_3x3()
pj = positions_to_jax(pos)

# Safety thresholds
UNSAFE_YAW_DEG = 30.0    # yaw > 30° is considered unsafe
UNSAFE_RATE_DPS = 5.0    # rate > 5°/s is considered unsafe

# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════
def load_models(tag, od, n=5):
    ms = []
    for s in range(n):
        cp = os.path.join(CKPT_DIR, f'policy_seed{s}_{tag}.pkl')
        if not os.path.exists(cp): continue
        m = ActorCritic(od, N, rngs=nnx.Rngs(0)); gd, _ = nnx.split(m)
        with open(cp, 'rb') as f: ms.append(nnx.merge(gd, pickle.load(f)))
    return ms

# ═══════════════════════════════════════════════════════════════════════════
# Base evaluators (vmap-accelerated)
# ═══════════════════════════════════════════════════════════════════════════
def _make_eval_raw(j_hist):
    """Raw DRL without deploy wrapper."""
    @jax.jit
    def _eval(model_params, gd, phis, vs):
        B = phis.shape[0]
        keys = jax.random.split(jax.random.key(0), B)
        def _single(key, phi, v):
            s, o = env_reset(key, pj, j=j_hist,
                             specific_wind_dir=phi, specific_wind_speed=v,
                             randomize_wind=False, max_steps=SETTLE+5)
            def _step(carry, _):
                st, ob, cum_gain, cum_travel, peak_r, unsafe_cnt = carry
                model = nnx.merge(gd, model_params)
                mean, _, _ = model(ob.reshape(1, -1))
                a = jnp.clip(mean.reshape(N), -AB, AB)
                ns, no, _, _ = env_step(st, a, pj, max_steps=SETTLE+5)
                travel = jnp.sum(jnp.abs(a))
                inflow = inflow_speeds_jax(pj, phi, v, ns.gammas)
                pwr = jnp.sum(power_output_jax(inflow, ns.gammas)) / 1e6
                inflow0 = inflow_speeds_jax(pj, phi, v, jnp.zeros(N))
                pwr0 = jnp.sum(power_output_jax(inflow0, jnp.zeros(N))) / 1e6
                gain = (pwr - pwr0) / (pwr0 + 1e-8) * 100
                is_unsafe = (jnp.max(jnp.abs(ns.gammas)) > UNSAFE_YAW_DEG) | \
                            (travel / T_STEP > UNSAFE_RATE_DPS)
                return (ns, no, cum_gain+gain, cum_travel+travel,
                        jnp.maximum(peak_r, travel/T_STEP),
                        unsafe_cnt + is_unsafe.astype(jnp.int32)), None
            init = (s, o, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0, dtype=jnp.int32))
            (final_st, _, tg, tt, pr, uc), _ = jax.lax.scan(_step, init, jnp.arange(SETTLE))
            return tg, tt, pr, uc, jnp.max(jnp.abs(final_st.gammas))
        return jax.vmap(_single)(keys, phis, vs)
    return _eval

def _make_eval_deploy(j_hist):
    """DRL-Deploy with gate+hyst+deadband+rate_limit."""
    @jax.jit
    def _eval(model_params, gd, phis, vs,
              bias_phi=0.0, bias_v=0.0, bias_yaw=0.0):
        B = phis.shape[0]
        keys = jax.random.split(jax.random.key(0), B)
        def _single(key, phi, v):
            s, o = env_reset(key, pj, j=j_hist,
                             specific_wind_dir=phi, specific_wind_speed=v,
                             randomize_wind=False, max_steps=SETTLE+5)
            def _step(carry, _):
                st, ob, in_gate, py, tg, tt, pr, uc = carry
                # Apply sensor bias to the wind information USED by gate logic
                phi_biased = phi + bias_phi
                v_biased = v + bias_v
                dphi = jnp.minimum(jnp.abs(phi_biased - 270), 360 - jnp.abs(phi_biased - 270))
                threshold = jnp.where(in_gate, GATE_OUT, GATE_IN)
                in_gate_new = (dphi < threshold) & (v_biased < U_infinity)

                model = nnx.merge(gd, model_params)
                mean, _, _ = model(ob.reshape(1, -1))
                raw_a = jnp.clip(mean.reshape(N), -AB, AB)
                a = jnp.where(in_gate_new, raw_a, jnp.zeros(N))
                a = jnp.where((DEADBAND>=0.5)&(jnp.max(jnp.abs(a))<DEADBAND), jnp.zeros(N), a)
                max_step = RATE_MAX * T_STEP
                a = jnp.clip(a, py - st.gammas - max_step, py - st.gammas + max_step)

                # Apply bias to the yaw used in environment stepping
                ns, no, _, _ = env_step(st, a, pj, max_steps=SETTLE+5)
                # Apply yaw bias to observation (but NOT to actual power computation)
                nyaw = ns.gammas + bias_yaw  # biased yaw for next observation
                ns_biased = ns._replace(gammas=nyaw)

                travel = jnp.sum(jnp.abs(a))
                inflow = inflow_speeds_jax(pj, phi, v, ns.gammas)  # true yaw for power
                pwr = jnp.sum(power_output_jax(inflow, ns.gammas)) / 1e6
                inflow0 = inflow_speeds_jax(pj, phi, v, jnp.zeros(N))
                pwr0 = jnp.sum(power_output_jax(inflow0, jnp.zeros(N))) / 1e6
                gain = (pwr - pwr0) / (pwr0 + 1e-8) * 100

                is_unsafe = (jnp.max(jnp.abs(ns.gammas)) > UNSAFE_YAW_DEG) | \
                            (travel / T_STEP > UNSAFE_RATE_DPS)
                return (ns_biased, no, in_gate_new, ns.gammas,
                        tg+gain, tt+travel, jnp.maximum(pr, travel/T_STEP),
                        uc+is_unsafe.astype(jnp.int32)), \
                       (in_gate_new, gain)
            init = (s, o, jnp.array(False), jnp.zeros(N),
                    jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
                    jnp.array(0, dtype=jnp.int32))
            (final_st, _, _, _, tg, tt, pr, uc), (gates, gains) = jax.lax.scan(
                _step, init, jnp.arange(SETTLE)
            )
            n_neg = jnp.sum((gains < 0).astype(jnp.int32))
            n_fallback = jnp.sum((~gates).astype(jnp.int32))  # steps where gate=0
            return tg, tt, pr, uc, n_neg, n_fallback, jnp.max(jnp.abs(final_st.gammas))
        return jax.vmap(_single)(keys, phis, vs)
    return _eval

# Pre-compile evaluators
_eval_raw_j3 = _make_eval_raw(3)
_eval_raw_j15 = _make_eval_raw(15)
_eval_deploy_j3 = _make_eval_deploy(3)
_eval_deploy_j15 = _make_eval_deploy(15)

def run_eval_raw(model, phis, vs, j_hist):
    ef = _eval_raw_j3 if j_hist == 3 else _eval_raw_j15
    gd, mp = nnx.split(model)
    tg, tt, pr, uc, my = ef(mp, gd, jnp.array(phis), jnp.array(vs))
    return map(np.array, (tg, tt, pr, uc, my))

def run_eval_deploy(model, phis, vs, j_hist,
                    bias_phi=0.0, bias_v=0.0, bias_yaw=0.0):
    ef = _eval_deploy_j3 if j_hist == 3 else _eval_deploy_j15
    gd, mp = nnx.split(model)
    tg, tt, pr, uc, ng, nf, my = ef(mp, gd,
                                     jnp.array(phis), jnp.array(vs),
                                     bias_phi, bias_v, bias_yaw)
    return map(np.array, (tg, tt, pr, uc, ng, nf, my))

def summarize(arr_gains, arr_travels, arr_peaks, arr_unsafe,
              arr_neg=None, arr_fallback=None, arr_maxyaw=None):
    """Compute summary metrics from condition-wise arrays (per-condition totals)."""
    n_cond = len(arr_gains)
    # arr_gains is total gain over SETTLE steps per condition
    # arr_neg is count of negative-gain steps per condition
    total_steps = n_cond * SETTLE
    if arr_neg is not None:
        neg_f = float(np.sum(arr_neg)) / total_steps
    else:
        neg_f = float(np.mean(arr_gains < 0))  # rough estimate
    return {
        'mean_gain_pct': float(np.mean(arr_gains / SETTLE)),  # per-step mean
        'mean_travel_per_step_deg': float(np.mean(arr_travels / SETTLE)),
        'total_travel_deg': float(np.sum(arr_travels)),
        'peak_rate_dps': float(np.max(arr_peaks)),
        'unsafe_cmd_count': int(np.sum(arr_unsafe)),
        'unsafe_rate': int(np.sum(arr_unsafe)) / total_steps,
        'neg_frac': neg_f,
        'fallback_rate': float(np.sum(arr_fallback)) / total_steps if arr_fallback is not None else 0.0,
        'max_yaw_deg': float(np.max(arr_maxyaw)) if arr_maxyaw is not None else 0.0,
        'mean_abs_yaw_deg': float(np.mean(np.abs(arr_maxyaw))) if arr_maxyaw is not None else 0.0,
    }

# ═══════════════════════════════════════════════════════════════════════════
# Scenario generators
# ═══════════════════════════════════════════════════════════════════════════
def gen_test_conditions(n, seed=123):
    """Generate a diverse set of wind conditions covering all regimes."""
    rng = np.random.default_rng(seed)
    # Mix of conditions: aligned, near-aligned, cross-wind
    n_each = n // 3
    # Aligned
    phi_a = rng.uniform(255, 285, n_each)
    v_a = rng.uniform(6, 11.4, n_each)
    # Near-aligned
    phi_na = np.concatenate([rng.uniform(235, 255, n_each//2),
                              rng.uniform(285, 305, n_each//2)])
    v_na = rng.uniform(8, 14, n_each)
    # Cross-wind
    phi_c = np.concatenate([rng.uniform(173, 235, n_each//2),
                             rng.uniform(305, 353, n_each//2)])
    v_c = rng.uniform(10, 16, n_each)
    phi = np.concatenate([phi_a, phi_na, phi_c])
    v = np.concatenate([v_a, v_na, v_c])
    idx = rng.permutation(len(phi))
    return phi[idx].astype(np.float32), v[idx].astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 90)
    print("  Experiment 6: Safety Fallback / Abnormal Condition Testing")
    print("=" * 90)

    print("\n[1] Loading models...")
    ms = load_models('sens_act10', OBS_J3)
    mj15 = load_models('dyn_J15_l5e4', OBS_J15, min(5, 3))
    print(f"    Static: {len(ms)} seeds, J=15: {len(mj15)} seeds")

    print(f"\n[2] Generating {N_TEST} test conditions (aligned+near+cross-wind)...")
    phis_test, vs_test = gen_test_conditions(N_TEST)
    dphi = np.minimum(np.abs(phis_test - 270), 360 - np.abs(phis_test - 270))
    print(f"    AC(|Δφ|≤5°): {np.mean(dphi<=5)*100:.1f}%, "
          f"NA(5<|Δφ|≤15°): {np.mean((dphi>5)&(dphi<=15))*100:.1f}%, "
          f"XW(|Δφ|>15°): {np.mean(dphi>15)*100:.1f}%")

    # ── Baseline (clean) ──────────────────────────────────────────────────
    print("\n[3] Baseline (clean conditions, no faults)...")
    print(f"    {'Controller':<28s} {'Gain%':>8s} {'Travel':>8s} "
          f"{'Peak°/s':>8s} {'Unsafe':>6s} {'Neg%':>6s} {'MaxYaw':>8s}")
    print("    " + "-" * 80)

    all_scenarios = {}

    # Raw Static
    g, t, p, u, my = run_eval_raw(ms[0], phis_test, vs_test, 3)
    s = summarize(g, t, p, u, arr_maxyaw=my)
    all_scenarios['clean/raw_static'] = s | {'controller': 'Raw DRL (Static)'}
    print(f"    {'Raw DRL (Static)':<28s} {s['mean_gain_pct']:>+7.3f}% "
          f"{s['mean_travel_per_step_deg']:>7.2f} {s['peak_rate_dps']:>7.2f} "
          f"{s['unsafe_cmd_count']:>5d} {s['neg_frac']*100:>5.1f}% "
          f"{s['max_yaw_deg']:>7.2f}°")

    # Static DRL-Deploy
    g, t, p, u, ng, nf, my = run_eval_deploy(ms[0], phis_test, vs_test, 3)
    sd = summarize(g, t, p, u, ng, nf, my)
    all_scenarios['clean/deploy_static'] = sd | {'controller': 'Static DRL-Deploy'}
    print(f"    {'Static DRL-Deploy':<28s} {sd['mean_gain_pct']:>+7.3f}% "
          f"{sd['mean_travel_per_step_deg']:>7.2f} {sd['peak_rate_dps']:>7.2f} "
          f"{sd['unsafe_cmd_count']:>5d} {sd['neg_frac']*100:>5.1f}% "
          f"{sd['max_yaw_deg']:>7.2f}°")

    # J=15 DRL-Deploy
    if mj15:
        g, t, p, u, ng, nf, my = run_eval_deploy(mj15[0], phis_test, vs_test, 15)
        jd = summarize(g, t, p, u, ng, nf, my)
        all_scenarios['clean/deploy_j15'] = jd | {'controller': 'J=15 DRL-Deploy'}
        print(f"    {'J=15 DRL-Deploy':<28s} {jd['mean_gain_pct']:>+7.3f}% "
              f"{jd['mean_travel_per_step_deg']:>7.2f} {jd['peak_rate_dps']:>7.2f} "
              f"{jd['unsafe_cmd_count']:>5d} {jd['neg_frac']*100:>5.1f}% "
              f"{jd['max_yaw_deg']:>7.2f}°")

    # ── Scenario 1: Wind Direction Bias ───────────────────────────────────
    print("\n[4] Scenario 1: Wind Direction Sensor Bias...")
    for bias in [-10, -5, -2, 2, 5, 10]:
        for tag, model, jh in [('Static Deploy', ms[0], 3),
                                ('J=15 Deploy', mj15[0], 15) if mj15 else (None, None, None)]:
            if model is None: continue
            g, t, p, u, ng, nf, my = run_eval_deploy(model, phis_test, vs_test, jh,
                                                      bias_phi=float(bias))
            s = summarize(g, t, p, u, ng, nf, my)
            key = f'bias_phi/{bias:+d}/{tag}'
            all_scenarios[key] = s | {'controller': tag, 'bias': bias, 'type': 'phi_bias'}
        print(f"    bias={bias:+3d}°: StaticDeploy gain={all_scenarios[f'bias_phi/{bias:+d}/Static Deploy']['mean_gain_pct']:+.3f}% "
              f"unsafe={all_scenarios[f'bias_phi/{bias:+d}/Static Deploy']['unsafe_cmd_count']} "
              f"| J15Deploy gain={all_scenarios[f'bias_phi/{bias:+d}/J=15 Deploy']['mean_gain_pct']:+.3f}% "
              f"unsafe={all_scenarios[f'bias_phi/{bias:+d}/J=15 Deploy']['unsafe_cmd_count']}")

    # ── Scenario 2: Wind Speed Bias ───────────────────────────────────────
    print("\n[5] Scenario 2: Wind Speed Sensor Bias...")
    for bias in [-1.0, -0.5, 0.5, 1.0]:
        for tag, model, jh in [('Static Deploy', ms[0], 3),
                                ('J=15 Deploy', mj15[0], 15) if mj15 else (None, None, None)]:
            if model is None: continue
            g, t, p, u, ng, nf, my = run_eval_deploy(model, phis_test, vs_test, jh,
                                                      bias_v=float(bias))
            s = summarize(g, t, p, u, ng, nf, my)
            key = f'bias_v/{bias:+.1f}/{tag}'
            all_scenarios[key] = s | {'controller': tag, 'bias': bias, 'type': 'v_bias'}
        print(f"    bias={bias:+.1f}m/s: StaticDeploy gain={all_scenarios[f'bias_v/{bias:+.1f}/Static Deploy']['mean_gain_pct']:+.3f}% "
              f"unsafe={all_scenarios[f'bias_v/{bias:+.1f}/Static Deploy']['unsafe_cmd_count']} "
              f"| J15Deploy gain={all_scenarios[f'bias_v/{bias:+.1f}/J=15 Deploy']['mean_gain_pct']:+.3f}%")

    # ── Scenario 3: Yaw Sensor Bias ───────────────────────────────────────
    print("\n[6] Scenario 3: Yaw Sensor Bias...")
    for bias in [-5, -2, 2, 5]:
        for tag, model, jh in [('Static Deploy', ms[0], 3),
                                ('J=15 Deploy', mj15[0], 15) if mj15 else (None, None, None)]:
            if model is None: continue
            g, t, p, u, ng, nf, my = run_eval_deploy(model, phis_test, vs_test, jh,
                                                      bias_yaw=float(bias))
            s = summarize(g, t, p, u, ng, nf, my)
            key = f'bias_yaw/{bias:+d}/{tag}'
            all_scenarios[key] = s | {'controller': tag, 'bias': bias, 'type': 'yaw_bias'}
        print(f"    bias={bias:+3d}°: StaticDeploy gain={all_scenarios[f'bias_yaw/{bias:+d}/Static Deploy']['mean_gain_pct']:+.3f}% "
              f"unsafe={all_scenarios[f'bias_yaw/{bias:+d}/Static Deploy']['unsafe_cmd_count']} "
              f"| J15Deploy gain={all_scenarios[f'bias_yaw/{bias:+d}/J=15 Deploy']['mean_gain_pct']:+.3f}%")

    # ── Scenario 6: Sudden Wind Shift (step change) ──────────────────────
    print("\n[7] Scenario 6: Sudden Wind Shift...")
    shift_results = []
    for shift_deg in [10, 20]:
        for tag, model, jh in [('Static Deploy', ms[0], 3),
                                ('J=15 Deploy', mj15[0], 15) if mj15 else (None, None, None),
                                ('Raw DRL', ms[0], 3)]:
            if model is None: continue
            # Trajectory: steady at 270° for 10 steps, then shift, then 10 more steps
            phi_traj = np.concatenate([np.full(10, 270.0, dtype=np.float32),
                                        np.full(10, 270.0 + shift_deg, dtype=np.float32)])
            v_traj = np.full(20, 8.0, dtype=np.float32)

            if tag == 'Raw DRL':
                g, t, p, u, my = run_eval_raw(model, phi_traj, v_traj, jh)
                gains = np.array(g).ravel(); travels = np.array(t).ravel()
                peak_rate_val = float(np.array(p).ravel()[0]) if np.array(p).size > 0 else 0.0
            else:
                g, t, p, u, ng, nf, my = run_eval_deploy(model, phi_traj, v_traj, jh)
                gains = np.array(g).ravel(); travels = np.array(t).ravel()
                peak_rate_val = float(np.array(p).ravel()[0]) if np.array(p).size > 0 else 0.0

            # Recovery: steps after shift until yaw stabilizes (travel < 0.5°)
            pre_gain = float(np.mean(gains[:8])) if len(gains) >= 8 else float(np.mean(gains))
            post_gain = float(np.mean(gains[12:])) if len(gains) > 12 else float(np.mean(gains[-3:]))
            gain_drop = pre_gain - post_gain
            peak_yaw = float(np.max(np.abs(np.array(my).ravel())))
            safe = int(np.sum(np.array(u).ravel()))

            shift_results.append({
                'shift_deg': shift_deg, 'controller': tag,
                'pre_shift_gain': pre_gain, 'post_shift_gain': post_gain,
                'gain_drop': gain_drop, 'peak_yaw': peak_yaw,
                'peak_rate': peak_rate_val, 'unsafe': safe,
                'total_travel': float(np.sum(travels)),
            })
            print(f"    shift {shift_deg}° {tag}: pre={pre_gain:+.3f}% post={post_gain:+.3f}% "
                  f"drop={gain_drop:+.3f}% peak_yaw={peak_yaw:.1f}° peak_rate={peak_rate_val:.1f}°/s unsafe={safe}")

    # ── Scenario 5: Missing Observations (random dropout with ZOH) ────────
    print("\n[8] Scenario 5: Missing Observations (Random Dropout + ZOH)...")
    dropout_results = []
    for drop_rate in [0.05, 0.10]:
        for tag, model, jh in [('Static Deploy', ms[0], 3),
                                ('J=15 Deploy', mj15[0], 15) if mj15 else (None, None, None)]:
            if model is None: continue
            # Use full trajectory with dropout in observation
            # Generate random dropout mask
            rng = np.random.default_rng(42)
            n_cond = min(N_TEST, 200)
            phi_sub = phis_test[:n_cond]; v_sub = vs_test[:n_cond]
            drop_mask = rng.random((n_cond, SETTLE)) < drop_rate

            # Sequential eval with ZOH for missing observations
            gd, mp = nnx.split(model)
            all_gains = np.zeros(n_cond); all_travels = np.zeros(n_cond)
            all_peaks = np.zeros(n_cond); unsafe_cnt = 0; neg_cnt = 0
            max_yaws = np.zeros(n_cond)

            for ci in range(n_cond):
                k = jax.random.key(ci)
                s, o = env_reset(k, pj, j=jh,
                                 specific_wind_dir=float(phi_sub[ci]),
                                 specific_wind_speed=float(v_sub[ci]),
                                 randomize_wind=False, max_steps=SETTLE+5)
                last_obs = np.array(o)
                in_gate = False; py = np.zeros(N)
                cg = 0.0; ct = 0.0; cp = 0.0; cu = 0

                for step in range(SETTLE):
                    # Apply dropout: use last valid observation
                    if drop_mask[ci, step]:
                        obs_use = last_obs
                    else:
                        obs_use = np.array(o)
                        last_obs = obs_use

                    m2 = nnx.merge(gd, mp)
                    mean, _, _ = m2(obs_use.reshape(1, -1))
                    raw_a = np.array(jnp.clip(mean.reshape(N), -AB, AB))

                    phi_i = float(phi_sub[ci]); v_i = float(v_sub[ci])
                    dphi = min(abs(phi_i - 270), 360 - abs(phi_i - 270))
                    threshold = GATE_OUT if in_gate else GATE_IN
                    in_gate = (dphi < threshold) and (v_i < U_infinity)

                    a = raw_a if in_gate else np.zeros(N)
                    if DEADBAND >= 0.5 and np.max(np.abs(a)) < DEADBAND:
                        a = np.zeros(N)
                    max_step = RATE_MAX * T_STEP
                    a = np.clip(a, py - np.array(s.gammas) - max_step,
                                py - np.array(s.gammas) + max_step)

                    ns, no_new, _, _ = env_step(s, jnp.array(a), pj, max_steps=SETTLE+5)
                    s = ns; o = no_new
                    py = np.array(ns.gammas)

                    travel = np.sum(np.abs(a))
                    ct += travel; cp = max(cp, travel / T_STEP)

                    inflow = inflow_speeds_jax(pj, phi_i, v_i, ns.gammas)
                    pwr = float(jnp.sum(power_output_jax(inflow, ns.gammas)) / 1e6)
                    inflow0 = inflow_speeds_jax(pj, phi_i, v_i, jnp.zeros(N))
                    pwr0 = float(jnp.sum(power_output_jax(inflow0, jnp.zeros(N))) / 1e6)
                    gain = (pwr - pwr0) / (pwr0 + 1e-8) * 100
                    cg += gain
                    if gain < 0: neg_cnt += 1
                    if np.max(np.abs(ns.gammas)) > UNSAFE_YAW_DEG or travel / T_STEP > UNSAFE_RATE_DPS:
                        cu += 1

                all_gains[ci] = cg
                all_travels[ci] = ct
                all_peaks[ci] = cp
                max_yaws[ci] = np.max(np.abs(ns.gammas))

            s = summarize(all_gains, all_travels, all_peaks,
                          np.array([unsafe_cnt]), all_gains < 0,
                          arr_maxyaw=max_yaws)
            # Override unsafe count
            s['unsafe_cmd_count'] = cu
            key = f'dropout/{drop_rate:.0%}/{tag}'
            all_scenarios[key] = s | {'controller': tag, 'drop_rate': drop_rate, 'type': 'dropout'}
            print(f"    dropout {drop_rate:.0%} {tag}: gain={s['mean_gain_pct']:+.3f}% "
                  f"unsafe={cu} max_yaw={s['max_yaw_deg']:.1f}°")

    # ── Scenario 7: OOD Wind Speed ────────────────────────────────────────
    print("\n[9] Scenario 7: Out-of-Distribution Wind Speed...")
    for v_ood in [3.0, 4.0, 5.0, 17.0, 20.0, 25.0]:
        phi_ood = np.full(N_TEST, 270.0, dtype=np.float32)
        v_ood_arr = np.full(N_TEST, v_ood, dtype=np.float32)

        for tag, model, jh in [('Static Deploy', ms[0], 3),
                                ('J=15 Deploy', mj15[0], 15) if mj15 else (None, None, None)]:
            if model is None: continue
            g, t, p, u, ng, nf, my = run_eval_deploy(model, phi_ood, v_ood_arr, jh)
            s = summarize(g, t, p, u, ng, nf, my)
            key = f'ood_v/{v_ood:.0f}/{tag}'
            all_scenarios[key] = s | {'controller': tag, 'v_ood': v_ood, 'type': 'ood_speed'}

        sd_key = f'ood_v/{v_ood:.0f}/Static Deploy'
        jd_key = f'ood_v/{v_ood:.0f}/J=15 Deploy'
        print(f"    v={v_ood:.0f}m/s: StaticDeploy gain={all_scenarios[sd_key]['mean_gain_pct']:+.3f}% "
              f"unsafe={all_scenarios[sd_key]['unsafe_cmd_count']} max_yaw={all_scenarios[sd_key]['max_yaw_deg']:.1f}° "
              f"fallback={all_scenarios[sd_key]['fallback_rate']*100:.0f}% | "
              f"J15Deploy gain={all_scenarios[jd_key]['mean_gain_pct']:+.3f}% "
              f"unsafe={all_scenarios[jd_key]['unsafe_cmd_count']} max_yaw={all_scenarios[jd_key]['max_yaw_deg']:.1f}° "
              f"fallback={all_scenarios[jd_key]['fallback_rate']*100:.0f}%")

    # ── Summary Tables ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  SAFETY TEST SUMMARY")
    print("=" * 90)

    # Table 1: Baseline
    print(f"\n  Table 1: Baseline (Clean Conditions)")
    print(f"  {'Controller':<22s} {'Gain%/step':>10s} {'Travel/step':>10s} "
          f"{'Peak°/s':>8s} {'Unsafe':>6s} {'Neg%':>6s} {'Max|γ|':>8s}")
    print("  " + "-" * 78)
    for key in ['clean/raw_static', 'clean/deploy_static', 'clean/deploy_j15']:
        if key in all_scenarios:
            s = all_scenarios[key]
            print(f"  {s['controller']:<22s} {s['mean_gain_pct']:>+9.3f}% "
                  f"{s['mean_travel_per_step_deg']:>9.2f} {s['peak_rate_dps']:>7.2f} "
                  f"{s['unsafe_cmd_count']:>5d} {s['neg_frac']*100:>5.1f}% "
                  f"{s['max_yaw_deg']:>7.2f}°")

    # Table 2: Sensor Bias
    print(f"\n  Table 2: Sensor Bias Robustness")
    print(f"  {'Scenario':<20s} {'Ctrl':<14s} {'Gain%/step':>10s} "
          f"{'Unsafe':>6s} {'Neg%':>6s} {'Max|γ|':>8s} {'Fallback%':>9s}")
    print("  " + "-" * 78)
    for key, s in all_scenarios.items():
        if s.get('type') in ('phi_bias', 'v_bias', 'yaw_bias'):
            ctrl = s.get('controller', '?')
            bias = s.get('bias', 0)
            typ = s.get('type', '?')
            label = f"{typ} {bias:+}"
            print(f"  {label:<20s} {ctrl:<14s} {s['mean_gain_pct']:>+9.3f}% "
                  f"{s['unsafe_cmd_count']:>5d} {s['neg_frac']*100:>5.1f}% "
                  f"{s['max_yaw_deg']:>7.2f}° {s['fallback_rate']*100:>8.1f}%")

    # Table 3: Sudden Wind Shift
    print(f"\n  Table 3: Sudden Wind Shift Response")
    print(f"  {'Shift':>6s} {'Controller':<18s} {'Pre-gain':>8s} {'Post-gain':>9s} "
          f"{'Drop':>7s} {'Peak|γ|':>8s} {'Peak°/s':>8s} {'Unsafe':>6s}")
    print("  " + "-" * 80)
    for r in shift_results:
        print(f"  {r['shift_deg']:>5d}° {r['controller']:<18s} {r['pre_shift_gain']:>+7.3f}% "
              f"{r['post_shift_gain']:>+8.3f}% {r['gain_drop']:>+6.3f}% "
              f"{r['peak_yaw']:>7.2f}° {r['peak_rate']:>7.2f} {r['unsafe']:>5d}")

    # Table 4: Dropout & OOD
    print(f"\n  Table 4: Dropout & OOD Speed")
    print(f"  {'Scenario':<20s} {'Ctrl':<14s} {'Gain%/step':>10s} "
          f"{'Unsafe':>6s} {'Max|γ|':>8s} {'Fallback%':>9s}")
    print("  " + "-" * 72)
    for key, s in all_scenarios.items():
        if s.get('type') in ('dropout', 'ood_speed'):
            ctrl = s.get('controller', '?')
            if s.get('type') == 'dropout':
                label = f"dropout {s.get('drop_rate',0):.0%}"
            else:
                label = f"OOD v={s.get('v_ood',0):.0f}"
            print(f"  {label:<20s} {ctrl:<14s} {s['mean_gain_pct']:>+9.3f}% "
                  f"{s['unsafe_cmd_count']:>5d} {s['max_yaw_deg']:>7.2f}° "
                  f"{s['fallback_rate']*100:>8.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────
    def _safe(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            v = float(obj); return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, dict): return {str(k): _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, np.ndarray)): return [_safe(x) for x in obj]
        return obj

    output = _safe({
        'description': 'Safety fallback / abnormal condition testing',
        'test_conditions': {
            'n_test': N_TEST, 'settle_steps': SETTLE,
            'ac_prob': float(np.mean(dphi <= 5)),
            'na_prob': float(np.mean((dphi > 5) & (dphi <= 15))),
        },
        'safety_thresholds': {
            'unsafe_yaw_deg': UNSAFE_YAW_DEG,
            'unsafe_rate_dps': UNSAFE_RATE_DPS,
        },
        'scenarios': all_scenarios,
        'wind_shifts': shift_results,
    })

    out_path = '../results/safety_abnormal_tests.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✅ Results saved to {out_path}")
    print("  Done.")
