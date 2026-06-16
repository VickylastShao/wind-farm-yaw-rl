#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the offline yaw-optimization baseline using SLSQP and differential_evolution
on the gray-box Bastankhah-Porte-Agel wake model.

For each wind condition, solve:
    argmax_gamma  P_farm(gamma; phi, v)
subject to:  -50 <= gamma_i <= +50  for all turbines i

Compares the optimal farm power to:
  - zero-yaw baseline
  - DRL policy gains from p0c_eval_randomized.json
  - DRL policy gains evaluated at exact conditions (using JAX env)

IMPORTANT MODEL NOTE:
  The power_output function has a hard threshold at u_rated (11.4 m/s):
    - u <= u_rated: P = 0.5*rho*C_P*S*u^3*cos(gamma)^1.88  (yaw loss applies)
    - u > u_rated:  P = P_rated  (NO yaw loss, power is capped)
  This creates a discontinuity in the optimization landscape: for v > u_rated,
  upstream turbines at free-stream have u_eff > u_rated, so yaw is "free"
  (it deflects the wake without reducing the upstream turbine's power).
  This leads to much larger optimal gains just above rated speed.
"""

import sys
import os
import json
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Add codes directory to path for imports
sys.path.insert(0, '/home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes')

from windfarm_env import (
    calculate_inflow_speeds,
    power_output,
    create_wind_farm_layout_3x3,
    U_infinity as U_infinity_global,
    u_rated,
    d_0, z_h, P_rated, rho, S, C_P, C_T, I,
    alpha_star, beta_star, alpha,
)

# ==============================================================================
# Build layout
# ==============================================================================
positions, N_rows, N_cols = create_wind_farm_layout_3x3()
N = len(positions)
print(f"Layout: {N_rows}x{N_cols} = {N} turbines, u_rated = {u_rated} m/s")

# ==============================================================================
# Objective function
# ==============================================================================

def total_farm_power(gammas, phi, v):
    """Compute total farm power (MW) for given yaw angles and wind condition."""
    gammas = np.asarray(gammas, dtype=np.float64)
    inflow = calculate_inflow_speeds(
        positions, phi, C_T, I, d_0, v, gammas, alpha_star, beta_star, alpha
    )
    total = sum(power_output(inflow[i], gammas[i]) for i in range(N))
    return total / 1e6  # MW


def neg_total_power(gammas, phi, v):
    """Negative total farm power (for minimization)."""
    return -total_farm_power(gammas, phi, v)


def baseline_power(phi, v):
    """Zero-yaw baseline power (MW)."""
    return total_farm_power(np.zeros(N), phi, v)


def optimize_slsqp(phi, v, n_starts=8, seed=42):
    """Multi-start SLSQP optimization."""
    rng = np.random.default_rng(seed)
    bounds = [(-50.0, 50.0)] * N

    best_result = None
    best_power = -np.inf

    # Always include zero start
    starts = [np.zeros(N)]
    for k in range(n_starts - 1):
        starts.append(rng.uniform(-30, 30, size=N))

    for k, x0 in enumerate(starts):
        try:
            res = minimize(
                neg_total_power, x0,
                args=(phi, v),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-13},
            )
            pwr = -res.fun
            if pwr > best_power:
                best_power = pwr
                best_result = res
        except Exception as e:
            print(f"    SLSQP start {k} failed: {e}")

    return best_result, best_power


def optimize_de(phi, v, seed=42, maxiter=1000, tol=1e-12):
    """Differential evolution global optimizer."""
    bounds = [(-50.0, 50.0)] * N
    try:
        res = differential_evolution(
            neg_total_power,
            bounds,
            args=(phi, v),
            seed=seed,
            maxiter=maxiter,
            tol=tol,
            polish=True,
            popsize=20,
            mutation=(0.5, 1.5),
            recombination=0.9,
        )
        return res, -res.fun
    except Exception as e:
        print(f"  DE failed: {e}")
        return None, -np.inf


# ==============================================================================
# Load DRL eval data
# ==============================================================================
eval_json_path = '/home/gpu/sz_workspace/JAX-WFCOYAW-RL/latex_draft/figures/p0c_eval_randomized.json'
with open(eval_json_path) as f:
    eval_data = json.load(f)

# Collect per-condition DRL gains across all seeds
all_rows = []
for seed_rows in eval_data['per_seed_rows']:
    all_rows.extend(seed_rows)

print(f"Loaded {len(all_rows)} DRL eval condition records")


def drl_gain_near(phi, v, phi_tol=3.0, v_tol=1.0):
    """Average DRL policy gain (%) for conditions near (phi, v).
    Returns (mean, n_matches, std).
    """
    matches = [
        r['policy_gain_pct'] for r in all_rows
        if abs(r['phi'] - phi) <= phi_tol and abs(r['v'] - v) <= v_tol
    ]
    if not matches:
        return None, 0, None
    return np.mean(matches), len(matches), np.std(matches)


# ==============================================================================
# DRL policy evaluation at exact conditions
# ==============================================================================
def evaluate_drl_exact(phi_val, v_val):
    """Evaluate all 5 DRL policy seeds at exact (phi, v) and return mean gain."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from cross_val_jaxenv_vs_numpyenv import load_nnx_policy, SETTLE_STEPS
        from windfarm_env_jax import (
            env_reset, env_step, positions_to_jax,
        )
    except ImportError:
        return None

    positions_jax = positions_to_jax(positions)
    N_turb = len(positions)
    obs_dim = 3 * N_turb + 3
    act_dim = N_turb
    ckpt_dir = os.path.join('/home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes',
                             'checkpoints_3x3_nnx_jaxenv')

    gains = []
    for s in range(5):
        ckpt = os.path.join(ckpt_dir, f'policy_seed{s}_p0c.pkl')
        if not os.path.exists(ckpt):
            continue
        model = load_nnx_policy(ckpt, obs_dim, act_dim)

        phi = jnp.float32(phi_val)
        v = jnp.float32(v_val)
        key = jax.random.key(42)
        state, obs = env_reset(key, positions_jax,
                               specific_wind_dir=phi,
                               specific_wind_speed=v,
                               randomize_wind=False,
                               max_steps=SETTLE_STEPS + 10)

        for step in range(SETTLE_STEPS):
            mean, _, _ = model(obs.reshape(1, -1))
            action = jnp.clip(mean.reshape(N_turb), -5.0, 5.0)
            state, obs, reward, done = env_step(state, action, positions_jax,
                                                max_steps=SETTLE_STEPS + 10)

        gain = float((state.total_mw - state.baseline_mw) / state.baseline_mw * 100)
        gains.append(gain)

    if not gains:
        return None
    return np.mean(gains)


# ==============================================================================
# Define conditions to evaluate
# ==============================================================================
conditions = []

# 1. Reference condition
conditions.append((270.0, 11.4, "reference"))

# 2. Grid of conditions (expanded to include above-rated regime)
for phi in [240, 255, 270, 285, 300]:
    for v in [8, 11.4, 11.5, 12, 14]:
        if phi == 270 and v == 11.4:
            continue  # already the reference
        conditions.append((float(phi), float(v), "grid"))

# 3. Fine grid around the u_rated transition for phi=270
for v in np.arange(11.0, 12.6, 0.2):
    conditions.append((270.0, float(v), "urated_transition"))

# 4. Aligned-cube conditions: |phi-270|<15 AND v<11.4
cube_phis = np.arange(255, 286, 5.0)   # 255, 260, ..., 285
cube_vs = np.arange(6.0, 11.5, 1.0)    # 6, 7, 8, 9, 10, 11
for phi_c in cube_phis:
    for v_c in cube_vs:
        conditions.append((phi_c, v_c, "aligned_cube"))

print(f"\nTotal conditions to optimize: {len(conditions)}")
print(f"  Reference: 1")
print(f"  Grid: {sum(1 for _,_,l in conditions if l=='grid')}")
print(f"  u_rated transition: {sum(1 for _,_,l in conditions if l=='urated_transition')}")
print(f"  Aligned-cube: {sum(1 for _,_,l in conditions if l=='aligned_cube')}")

# ==============================================================================
# Run optimization
# ==============================================================================
results = []

print("\n" + "="*120)
print("OPTIMIZATION RESULTS")
print("="*120)

header = (f"{'Condition':<20} {'Base(MW)':>9} {'SLSQP':>9} {'Best':>9} "
          f"{'SLSQP%':>8} {'Best%':>8} {'DRL_ex%':>8} {'DRL_nb%':>7} "
          f"{'DRL/Opt':>8}")
print(header)
print("-"*120)

# Evaluate DRL at exact conditions for a subset of key conditions
drl_exact_conditions = set()
for phi, v, label in conditions:
    if label in ('reference', 'grid', 'urated_transition'):
        drl_exact_conditions.add((phi, v))

# Pre-compute DRL exact gains
print("Pre-computing DRL policy gains at key conditions...")
drl_exact_cache = {}
for phi, v in sorted(drl_exact_conditions):
    gain = evaluate_drl_exact(phi, v)
    drl_exact_cache[(phi, v)] = gain
    print(f"  ({phi:.1f}, {v:.1f}): DRL exact = {gain:.3f}%" if gain is not None else f"  ({phi:.1f}, {v:.1f}): DRL exact = N/A")

print("\nRunning SLSQP/DE optimization...")

for idx, (phi, v, label) in enumerate(conditions):
    print(f"  [{idx+1}/{len(conditions)}] phi={phi:.1f}, v={v:.1f} ({label})...", end="", flush=True)
    t0 = time.time()

    # Baseline
    base = baseline_power(phi, v)

    # SLSQP (8 multi-starts)
    slsqp_res, slsqp_pwr = optimize_slsqp(phi, v, n_starts=8, seed=42)

    # Differential evolution (only for non-trivial SLSQP results to save time)
    slsqp_gain = (slsqp_pwr - base) / base * 100 if base > 0 else 0
    if slsqp_gain > 0.01:
        de_res, de_pwr = optimize_de(phi, v, seed=42, maxiter=1000, tol=1e-12)
    else:
        de_res, de_pwr = None, slsqp_pwr

    elapsed = time.time() - t0
    print(f" done ({elapsed:.1f}s)")

    # Gains
    de_gain = (de_pwr - base) / base * 100 if base > 0 else 0
    best_gain = max(slsqp_gain, de_gain)
    best_pwr = max(slsqp_pwr, de_pwr)

    # DRL gain (exact from policy)
    drl_exact = drl_exact_cache.get((phi, v))
    drl_exact_str = f"{drl_exact:.2f}" if drl_exact is not None else "-"

    # DRL gain (from neighborhood lookup)
    drl_nb, n_drl, drl_std = drl_gain_near(phi, v)
    drl_nb_str = f"{drl_nb:.2f}" if drl_nb is not None else "-"

    # DRL/Opt ratio (using exact DRL gain)
    drl_for_ratio = drl_exact if drl_exact is not None else drl_nb
    if drl_for_ratio is not None and best_gain > 0.01:
        drl_opt_ratio = drl_for_ratio / best_gain * 100
        ratio_str = f"{drl_opt_ratio:.1f}%"
    else:
        ratio_str = "-"

    cond_str = f"({phi:.0f},{v:.1f})"
    print(f"  {cond_str:<20} {base:>9.3f} {slsqp_pwr:>9.3f} {best_pwr:>9.3f} "
          f"{slsqp_gain:>7.3f}% {best_gain:>7.3f}% "
          f"{drl_exact_str:>8} {drl_nb_str:>7} "
          f"{ratio_str:>8}")

    # Store results
    opt_gammas = slsqp_res.x if slsqp_res is not None else None
    de_gammas = de_res.x if de_res is not None else None

    results.append({
        'phi': phi,
        'v': v,
        'label': label,
        'baseline_mw': base,
        'slsqp_opt_mw': slsqp_pwr,
        'slsqp_gain_pct': slsqp_gain,
        'slsqp_gammas': opt_gammas.tolist() if opt_gammas is not None else None,
        'de_opt_mw': de_pwr,
        'de_gain_pct': de_gain,
        'de_gammas': de_gammas.tolist() if de_gammas is not None else None,
        'best_opt_mw': best_pwr,
        'best_opt_gain_pct': best_gain,
        'drl_exact_gain_pct': drl_exact,
        'drl_nb_gain_pct': drl_nb,
        'drl_opt_ratio_pct': float(ratio_str.replace('%','')) if ratio_str not in ['-', 'inf%'] else None,
    })

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*120)
print("SUMMARY")
print("="*120)

# Reference condition
ref = [r for r in results if r['label'] == 'reference'][0]
print(f"\n--- Reference condition (270, 11.4) ---")
print(f"  Baseline:     {ref['baseline_mw']:.3f} MW")
print(f"  SLSQP opt:    {ref['slsqp_opt_mw']:.3f} MW  (+{ref['slsqp_gain_pct']:.3f}%)")
print(f"  Best opt:     {ref['best_opt_mw']:.3f} MW  (+{ref['best_opt_gain_pct']:.3f}%)")
if ref['drl_exact_gain_pct'] is not None:
    print(f"  DRL exact:    +{ref['drl_exact_gain_pct']:.3f}%")
    print(f"  DRL/opt:      {ref['drl_exact_gain_pct']/ref['best_opt_gain_pct']*100:.1f}%")
print(f"  Optimal yaws: {[f'{g:+.1f}' for g in ref['slsqp_gammas']]}")

# u_rated transition analysis
print(f"\n--- u_rated transition analysis (phi=270) ---")
print(f"  NOTE: power_output() returns P_rated for u>u_rated={u_rated} m/s,")
print(f"  making yaw 'free' for upstream turbines at v>u_rated.")
transition_results = [r for r in results if r['label'] == 'urated_transition']
for r in sorted(transition_results, key=lambda x: x['v']):
    drl_str = f"DRL={r['drl_exact_gain_pct']:+.2f}%" if r['drl_exact_gain_pct'] is not None else ""
    print(f"  v={r['v']:.1f}: opt=+{r['best_opt_gain_pct']:.3f}%  {drl_str}  "
          f"base={r['baseline_mw']:.2f} MW")

# Grid conditions
grid_results = [r for r in results if r['label'] == 'grid']
print(f"\n--- Grid conditions (sorted by opt gain) ---")
for r in sorted(grid_results, key=lambda x: -x['best_opt_gain_pct']):
    drl_str = f"DRL={r['drl_exact_gain_pct']:+.2f}%" if r['drl_exact_gain_pct'] is not None else "DRL=-"
    print(f"  ({r['phi']:.0f}, {r['v']:.1f}): Opt +{r['best_opt_gain_pct']:.3f}%, {drl_str}")

# Aligned-cube conditions
cube_results = [r for r in results if r['label'] == 'aligned_cube']
if cube_results:
    avg_best = np.mean([r['best_opt_gain_pct'] for r in cube_results])
    drl_vals = [r['drl_exact_gain_pct'] for r in cube_results if r['drl_exact_gain_pct'] is not None]
    avg_drl = np.mean(drl_vals) if drl_vals else 0

    valid_ratios = [r['drl_exact_gain_pct'] / r['best_opt_gain_pct'] * 100
                    for r in cube_results
                    if r['drl_exact_gain_pct'] is not None and r['best_opt_gain_pct'] > 0.1]

    print(f"\n--- Aligned-cube region (|phi-270|<15, v<11.4, n={len(cube_results)}) ---")
    print(f"  Mean best-opt gain:  +{avg_best:.3f}%")
    print(f"  Mean DRL gain:       +{avg_drl:.3f}%")
    if valid_ratios:
        print(f"  Mean DRL/opt (where opt>0.1%): {np.mean(valid_ratios):.1f}%  (n={len(valid_ratios)})")

    paper_drl = eval_data.get('paper_headline_aligned_cube', {}).get('mean_pct', None)
    if paper_drl is not None:
        print(f"  Paper DRL headline:  +{paper_drl:.3f}% (from eval JSON)")
        if avg_best > 0:
            print(f"  Paper DRL / opt:     {paper_drl / avg_best * 100:.1f}%")

# Cross-regime breakdown
print(f"\n--- Cross-regime breakdown ---")
for regime_name, phi_lo, phi_hi, v_lo, v_hi in [
    ("Aligned (|dphi|<15, v<11.4)", 255, 285, 6, 11.4),
    ("Aligned (|dphi|<15, v>=11.4)", 255, 285, 11.4, 16),
    ("Near-aligned (15<|dphi|<35)", 235, 255, 6, 16),
    ("Cross-wind (|dphi|>35)", 0, 235, 6, 16),
]:
    regime = [r for r in results
              if phi_lo <= r['phi'] <= phi_hi and v_lo <= r['v'] <= v_hi]
    if not regime:
        continue
    avg_opt = np.mean([r['best_opt_gain_pct'] for r in regime])
    drl_vals = [r['drl_exact_gain_pct'] for r in regime if r['drl_exact_gain_pct'] is not None]
    avg_drl = np.mean(drl_vals) if drl_vals else 0
    print(f"  {regime_name}: n={len(regime)}, opt=+{avg_opt:.3f}%, DRL=+{avg_drl:.3f}%")

# ==============================================================================
# Save results
# ==============================================================================
output_path = '/home/gpu/sz_workspace/JAX-WFCOYAW-RL/latex_draft/figures/slsqp_optimum_results.json'
with open(output_path, 'w') as f:
    json.dump({
        'description': 'SLSQP and DE yaw optimization on gray-box wake model',
        'layout': '3x3',
        'N_turbines': N,
        'bounds_deg': [-50, 50],
        'slsqp_n_starts': 8,
        'de_maxiter': 1000,
        'drl_lookup_window': {'phi_tol': 3.0, 'v_tol': 1.0},
        'note': (
            'The power_output function has a hard threshold at u_rated (11.4 m/s): '
            'for u > u_rated it returns P_rated regardless of yaw, making yaw "free" '
            'for upstream turbines. This creates a discontinuity in the optimization '
            'landscape: gains jump from ~4% at v=11.4 to ~9% at v=11.5.'
        ),
        'results': results,
    }, f, indent=2)
print(f"\nResults saved to {output_path}")
