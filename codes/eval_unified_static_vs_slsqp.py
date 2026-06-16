#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified static evaluation: compare Zero-yaw, SLSQP oracle, Lookup table,
and multiple DRL policy tags on the same wind conditions.

Evaluates each policy tag across multiple seeds and reports:
  - Mean/median/min gain vs zero-yaw
  - Recovery relative to SLSQP oracle
  - Negative-gain fraction
  - Aligned-cube subset summary

Output:
  latex_draft/figures/unified_static_vs_slsqp.json
  latex_draft/figures/fig_unified_static_vs_slsqp_scatter.{pdf,jpg}

Env vars:
  N_COMPARE       (default 500)
  EVAL_SEED       (default 20260605)
  POLICY_TAGS     (default "p0c")         – comma-separated
  N_SEEDS         (default 5)             – max seeds per tag (clipped to available)
  SETTLE_STEPS    (default 150)
  N_SLSQP_STARTS  (default 8)
  SKIP_SLSQP      (default "")            – set to "1" to reuse cached SLSQP data
"""

import os
import sys
import json
import time
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from flax import nnx

from windfarm_env import (
    create_wind_farm_layout_3x3,
    calculate_inflow_speeds, power_output,
    C_T, I, d_0, alpha_star, beta_star, alpha,
)
from windfarm_env_jax import (
    env_reset, env_step, positions_to_jax,
    inflow_speeds_jax, power_output_jax,
)
from train_3x3_nnx import ActorCritic, AttentionActorCritic, USE_ATTENTION, RESIDUAL as _NNX_RESIDUAL

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_COMPARE = int(os.environ.get("N_COMPARE", 500))
EVAL_SEED = int(os.environ.get("EVAL_SEED", 20260605))
_POLICY_TAGS = os.environ.get("POLICY_TAGS", "p0c")
N_SEEDS = int(os.environ.get("N_SEEDS", 5))
SETTLE_STEPS = int(os.environ.get("SETTLE_STEPS", 150))
N_SLSQP_STARTS = int(os.environ.get("N_SLSQP_STARTS", 8))
SKIP_SLSQP = os.environ.get("SKIP_SLSQP", "") == "1"
_EVAL_J = int(os.environ.get("J", 1))  # must match training J

# Parse policy tags into list of (tag, n_seeds) tuples.
# Format: "p0c,bc,bc_ppo" uses N_SEEDS for all; "p0c:5,bc:3" overrides per-tag.
def _parse_tags(raw, default_n):
    result = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            tag, n_str = item.split(":", 1)
            result.append((tag.strip(), int(n_str)))
        else:
            result.append((item, default_n))
    return result

POLICY_SPECS = _parse_tags(_POLICY_TAGS, N_SEEDS)


# ---------------------------------------------------------------------------
# SLSQP / physics helpers (NumPy).
# ---------------------------------------------------------------------------
def total_farm_power_np(gammas, positions, phi, v, N):
    gammas = np.asarray(gammas, dtype=np.float64)
    inflow = calculate_inflow_speeds(
        positions, phi, C_T, I, d_0, v, gammas, alpha_star, beta_star, alpha)
    total = sum(power_output(inflow[i], gammas[i]) for i in range(N))
    return total / 1e6


def optimize_slsqp(phi, v, positions, N, n_starts=N_SLSQP_STARTS, seed=42):
    rng = np.random.default_rng(seed)
    bounds = [(-50.0, 50.0)] * N
    best_power = -np.inf
    best_gammas = np.zeros(N)
    starts = [np.zeros(N)]
    for _ in range(n_starts - 1):
        starts.append(rng.uniform(-30, 30, size=N))
    for x0 in starts:
        try:
            res = minimize(
                lambda g: -total_farm_power_np(g, positions, phi, v, N),
                x0, method='SLSQP', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-13})
            pwr = -res.fun
            if pwr > best_power:
                best_power = pwr
                best_gammas = res.x.copy()
        except Exception:
            pass
    return best_power, best_gammas


def load_nnx_policy(path: str, obs_dim: int, act_dim: int):
    """Load an NNX policy checkpoint, auto-detecting arch + residual variant.

    Tries all combinations of model class (Attention / MLP) and residual
    setting until the leaf count matches.
    """
    with open(path, "rb") as f:
        state = pickle.load(f)

    import train_3x3_nnx as _nnx
    for residual in [False, True]:
        _nnx.RESIDUAL = residual  # override module-level flag
        for model_cls, name in [(AttentionActorCritic, "attention"),
                                 (ActorCritic, "mlp")]:
            try:
                model = model_cls(obs_dim, act_dim, rngs=nnx.Rngs(0))
                graphdef, _ = nnx.split(model)
                merged = nnx.merge(graphdef, state)
                _nnx.RESIDUAL = _NNX_RESIDUAL  # restore
                return merged
            except ValueError:
                continue
    _nnx.RESIDUAL = _NNX_RESIDUAL  # restore
    raise ValueError(
        f"Cannot load checkpoint; no model variant matches. "
        f"obs_dim={obs_dim}, act_dim={act_dim}")


# ---------------------------------------------------------------------------
# Lookup table loading.
# ---------------------------------------------------------------------------
def load_lookup():
    """Load precomputed lookup table; return None if missing."""
    lt_path = os.path.join(FIG_DIR, "lookup_table_baseline.json")
    yaw_path = os.path.join(FIG_DIR, "lookup_table_yaw.npy")
    if not os.path.exists(lt_path) or not os.path.exists(yaw_path):
        return None
    with open(lt_path) as f:
        lt = json.load(f)
    phi_grid = np.array(lt["phi_grid"], dtype=np.float32)
    v_grid = np.array(lt["v_grid"], dtype=np.float32)
    yaw_table = np.load(yaw_path)  # (n_phi, n_v, N)
    gain_table = np.array(lt["gain_table"], dtype=np.float32)
    return phi_grid, v_grid, yaw_table, gain_table


def lookup_interpolate_batch(phis, vs, phi_grid, v_grid, yaw_table):
    """Bilinear interpolation for a batch of queries."""
    n_phi, n_v = len(phi_grid), len(v_grid)
    phi_idx = np.searchsorted(phi_grid, phis) - 1
    v_idx = np.searchsorted(v_grid, vs) - 1
    phi_idx = np.clip(phi_idx, 0, n_phi - 2)
    v_idx = np.clip(v_idx, 0, n_v - 2)

    phi_lo = phi_grid[phi_idx]
    phi_hi = phi_grid[phi_idx + 1]
    v_lo = v_grid[v_idx]
    v_hi = v_grid[v_idx + 1]

    w_phi = np.clip((phis - phi_lo) / np.maximum(phi_hi - phi_lo, 1e-6), 0, 1)
    w_v = np.clip((vs - v_lo) / np.maximum(v_hi - v_lo, 1e-6), 0, 1)

    yaw_interp = (
        yaw_table[phi_idx, v_idx] * (1 - w_phi)[:, None] * (1 - w_v)[:, None]
        + yaw_table[phi_idx + 1, v_idx] * w_phi[:, None] * (1 - w_v)[:, None]
        + yaw_table[phi_idx, v_idx + 1] * (1 - w_phi)[:, None] * w_v[:, None]
        + yaw_table[phi_idx + 1, v_idx + 1] * w_phi[:, None] * w_v[:, None]
    )
    return yaw_interp


# ---------------------------------------------------------------------------
# DRL policy evaluation on JAX env (vmap'd).
# ---------------------------------------------------------------------------
def evaluate_policy_tag(tag: str, n_seeds: int, positions_j, positions_np, N_turb,
                        phis_j, vs_j, obs_dim, act_dim):
    """Evaluate a policy tag across seeds. Returns per-condition mean gains.

    Tags starting with 'bc' (but not 'bc_ppo') are treated as absolute-yaw
    policies: the model output is the target yaw, converted to incremental
    action via action = clip(target - current_gammas, -5, 5).
    """
    # Only pure BC (absolute yaw) policies need conversion.  Incremental
    # BC ("bc_inc") and BC+PPO ("bc_ppo*") output incremental actions directly.
    use_absolute_yaw = (tag.startswith("bc")
                        and "_inc" not in tag
                        and "_ppo" not in tag)
    all_seed_gains = []
    found = 0
    for s in range(n_seeds):
        ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_{tag}.pkl")
        if not os.path.exists(ckpt):
            continue
        model = load_nnx_policy(ckpt, obs_dim, act_dim)
        found += 1

        # Compute baselines for this seed.
        @jax.jit
        def zero_yaw_baseline(phi, v):
            inflow_0 = inflow_speeds_jax(positions_j, phi, v,
                                          jnp.zeros(N_turb, dtype=jnp.float32))
            return jnp.sum(power_output_jax(inflow_0,
                           jnp.zeros(N_turb, dtype=jnp.float32))) / 1e6

        baselines = jax.vmap(zero_yaw_baseline)(phis_j, vs_j)
        baselines_np = np.asarray(baselines)

        # Run policy.  For absolute-yaw policies (BC), convert model output
        # from target yaw to incremental action each step.  Also convert
        # absolute inflow to wake deficit (v - inflow) to match BC training.
        _use_abs = use_absolute_yaw  # capture for closure
        _use_deficit = use_absolute_yaw  # BC policies use deficit observation
        _Nt = N_turb  # capture for closure

        @nnx.jit
        def run_policy(m, phis_j, vs_j):
            @jax.vmap
            def reset_one(phi, v):
                key = jax.random.key(0)
                state, obs = env_reset(key, positions_j,
                                        specific_wind_dir=phi,
                                        specific_wind_speed=v,
                                        randomize_wind=False,
                                        j=_EVAL_J,
                                        max_steps=SETTLE_STEPS + 10)
                return state, obs

            states, obs_batch = reset_one(phis_j, vs_j)

            @jax.vmap
            def predict_one(o):
                mean, _, _ = m(o.reshape(1, -1))
                return mean.reshape(_Nt)

            @jax.vmap
            def step_one(s, a):
                return env_step(s, a, positions_j, max_steps=SETTLE_STEPS + 10)

            def _preprocess_deficit(o):
                """Replace absolute inflow with wake deficit v - inflow."""
                # obs: [gammas(N), inflow(N), cos, sin, v, locked(N)]
                # v is at index 2*N_turb + 2
                v_vals = o[:, 2 * _Nt + 2:2 * _Nt + 3]  # (B, 1)
                return o.at[:, _Nt:2 * _Nt].set(v_vals - o[:, _Nt:2 * _Nt])

            def body(carry, _):
                states, obs = carry
                if _use_deficit:
                    obs = _preprocess_deficit(obs)
                output = predict_one(obs)
                if _use_abs:
                    # BC policy outputs absolute yaw target; convert to incremental.
                    actions = jnp.clip(output - states.gammas, -5.0, 5.0)
                else:
                    actions = jnp.clip(output, -5.0, 5.0)
                new_states, new_obs, _, _ = step_one(states, actions)
                return (new_states, new_obs), None

            (final_states, _), _ = jax.lax.scan(
                body, (states, obs_batch), None, length=SETTLE_STEPS)
            return final_states.total_mw, final_states.gammas

        total_mw, gammas = run_policy(model, phis_j, vs_j)
        gains = (total_mw - baselines) / baselines * 100.0
        all_seed_gains.append(np.asarray(gains))

    if found == 0:
        print(f"  [{tag}] WARNING: no checkpoints found (looked for "
              f"policy_seed0_{tag}.pkl .. policy_seed{n_seeds-1}_{tag}.pkl)")
        return None, 0

    gains_stack = np.stack(all_seed_gains, axis=0)
    mean_gains = gains_stack.mean(axis=0)
    std_gains = gains_stack.std(axis=0) if found > 1 else np.zeros_like(mean_gains)

    print(f"  [{tag}] {found} seeds | mean gain = {float(mean_gains.mean()):+.3f}% "
          f"| aligned-cube = {_aligned_cube_mean(mean_gains, phis_j, vs_j):+.3f}%"
          f"{' (abs-yaw->inc)' if use_absolute_yaw else ''}")
    return mean_gains, found


def _aligned_cube_mean(gains, phis_j, vs_j):
    dphi = np.abs(((np.asarray(phis_j) - 270.0 + 180.0) % 360.0) - 180.0)
    aligned = (dphi < 15.0) & (np.asarray(vs_j) < 11.4)
    if aligned.sum() > 0:
        return float(gains[aligned].mean())
    return float("nan")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    positions, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions)
    positions_j = positions_to_jax(positions)
    obs_dim_per_step = 3 * N_turb + 3
    if os.environ.get("USE_POSITIONS", "0") == "1":
        obs_dim_per_step += 2 * N_turb
    obs_dim = _EVAL_J * obs_dim_per_step
    act_dim = N_turb

    print(f"# Unified Static DRL vs SLSQP Evaluation")
    print(f"# N_COMPARE      : {N_COMPARE}")
    print(f"# EVAL_SEED      : {EVAL_SEED}")
    print(f"# SETTLE_STEPS   : {SETTLE_STEPS}")
    print(f"# POLICY_TAGS    : {_POLICY_TAGS}")
    print(f"# N_SEEDS        : {N_SEEDS}")
    print(f"# device         : {jax.devices()[0]}")

    # ---------- Sample conditions ----------
    rng = np.random.default_rng(EVAL_SEED)
    phis = rng.uniform(173.0, 353.0, size=N_COMPARE).astype(np.float32)
    vs = rng.uniform(6.0, 16.0, size=N_COMPARE).astype(np.float32)
    phis_j = jnp.asarray(phis)
    vs_j = jnp.asarray(vs)

    # ---------- Zero-yaw baseline ----------
    print("\n## Zero-yaw baseline")
    @jax.jit
    def zero_power(phi, v):
        inflow_0 = inflow_speeds_jax(positions_j, phi, v,
                                      jnp.zeros(N_turb, dtype=jnp.float32))
        return jnp.sum(power_output_jax(inflow_0,
                       jnp.zeros(N_turb, dtype=jnp.float32))) / 1e6
    baselines = jax.vmap(zero_power)(phis_j, vs_j)
    baselines_np = np.asarray(baselines)
    zero_gains = np.zeros(N_COMPARE, dtype=np.float32)  # zero-yaw has gain=0 by definition

    # ---------- SLSQP oracle ----------
    print(f"\n## SLSQP oracle ({N_COMPARE} conditions, {N_SLSQP_STARTS} starts each)")
    slsqp_gains = np.empty(N_COMPARE, dtype=np.float32)
    slsqp_mws = np.empty(N_COMPARE, dtype=np.float32)
    for idx in range(N_COMPARE):
        phi, v = float(phis[idx]), float(vs[idx])
        base = baselines_np[idx]
        opt_mw, _ = optimize_slsqp(phi, v, positions, N_turb, seed=EVAL_SEED + idx)
        gain = (opt_mw - base) / base * 100.0 if base > 0 else 0.0
        slsqp_gains[idx] = gain
        slsqp_mws[idx] = opt_mw
        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{N_COMPARE} done ({time.time()-t_start:.0f}s)")

    slsqp_mean = float(np.mean(slsqp_gains))
    print(f"  SLSQP marginal mean : {slsqp_mean:+.3f}%")
    print(f"  SLSQP aligned-cube  : {_aligned_cube_mean(slsqp_gains, phis_j, vs_j):+.3f}%")

    # ---------- Lookup table ----------
    print(f"\n## Lookup table")
    lt = load_lookup()
    lookup_gains = np.zeros(N_COMPARE, dtype=np.float32)
    if lt is not None:
        phi_grid, v_grid, yaw_table, _ = lt
        lookup_yaws = lookup_interpolate_batch(phis, vs, phi_grid, v_grid, yaw_table)
        for idx in range(N_COMPARE):
            pwr = total_farm_power_np(lookup_yaws[idx], positions,
                                       float(phis[idx]), float(vs[idx]), N_turb)
            base = float(baselines_np[idx])
            lookup_gains[idx] = (pwr - base) / base * 100.0 if base > 0 else 0.0
        print(f"  Lookup marginal mean : {float(np.mean(lookup_gains)):+.3f}%")
        print(f"  Lookup aligned-cube  : {_aligned_cube_mean(lookup_gains, phis_j, vs_j):+.3f}%")
    else:
        print("  Lookup table not found; skipping.")

    # ---------- DRL policies ----------
    print(f"\n## DRL policy evaluation")
    policy_results = {}
    for tag, n_seeds in POLICY_SPECS:
        mean_gains, found = evaluate_policy_tag(
            tag, n_seeds, positions_j, positions, N_turb,
            phis_j, vs_j, obs_dim, act_dim)
        if mean_gains is not None:
            policy_results[tag] = {"gains": mean_gains, "n_seeds": found}

    # ---------- Analysis ----------
    dphi_arr = np.abs(((phis - 270.0 + 180.0) % 360.0) - 180.0)
    aligned_cube = (dphi_arr < 15.0) & (vs < 11.4)

    # Recovery relative to SLSQP.
    valid_mask = slsqp_gains > 0.1

    result = {
        "description": "Unified static DRL vs SLSQP evaluation",
        "n_conditions": N_COMPARE,
        "eval_seed": EVAL_SEED,
        "settle_steps": SETTLE_STEPS,
        "slsqp_n_starts": N_SLSQP_STARTS,
        "summary": {
            "zero_yaw": {
                "marginal_mean_pct": 0.0,
                "aligned_cube_pct": 0.0,
            },
            "slsqp_oracle": {
                "marginal_mean_pct": slsqp_mean,
                "aligned_cube_pct": _aligned_cube_mean(slsqp_gains, phis_j, vs_j),
            },
            "lookup_table": {
                "marginal_mean_pct": float(np.mean(lookup_gains)) if lt else None,
                "aligned_cube_pct": _aligned_cube_mean(lookup_gains, phis_j, vs_j) if lt else None,
            },
            "policies": {},
        },
    }

    for tag, pres in policy_results.items():
        gains = pres["gains"]
        recovery = gains[valid_mask] / slsqp_gains[valid_mask] * 100.0 \
            if valid_mask.sum() > 0 else np.array([0.0])
        neg_frac = float((gains < -0.01).mean())
        result["summary"]["policies"][tag] = {
            "n_seeds": pres["n_seeds"],
            "marginal_mean_pct": float(np.mean(gains)),
            "aligned_cube_pct": _aligned_cube_mean(gains, phis_j, vs_j),
            "mean_recovery_pct": float(np.mean(recovery)),
            "median_recovery_pct": float(np.median(recovery)),
            "negative_gain_fraction": neg_frac,
            "mean_oracle_gap_pp": float(np.mean(slsqp_gains - gains)),
        }

    # Print summary table.
    print(f"\n{'='*70}")
    print(f"SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Controller':<20s} {'Marginal':>10s} {'AlignedCube':>12s} {'Recovery':>10s} {'NegFrac':>8s}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")
    print(f"  {'Zero-yaw':<20s} {'0.000%':>10s} {'0.000%':>12s} {'---':>10s} {'---':>8s}")
    print(f"  {'SLSQP oracle':<20s} {slsqp_mean:>+10.3f}% "
          f"{_aligned_cube_mean(slsqp_gains, phis_j, vs_j):>+12.3f}% {'100.0%':>10s} {'---':>8s}")
    if lt:
        print(f"  {'Lookup table':<20s} {float(np.mean(lookup_gains)):>+10.3f}% "
              f"{_aligned_cube_mean(lookup_gains, phis_j, vs_j):>+12.3f}% "
              f"{float(np.mean(lookup_gains[valid_mask]/slsqp_gains[valid_mask]*100) if valid_mask.sum()>0 else 0):>10.1f}% {'---':>8s}")
    for tag, pres in policy_results.items():
        s = result["summary"]["policies"][tag]
        print(f"  {tag:<20s} {s['marginal_mean_pct']:>+10.3f}% "
              f"{s['aligned_cube_pct']:>+12.3f}% "
              f"{s['mean_recovery_pct']:>10.1f}% {s['negative_gain_fraction']:>8.1%}")

    # ---------- Save JSON ----------
    out_path = os.path.join(FIG_DIR, "unified_static_vs_slsqp.json")
    # Convert numpy arrays to lists for JSON.
    per_condition = []
    for idx in range(N_COMPARE):
        row = {
            "idx": idx,
            "phi": float(phis[idx]),
            "v": float(vs[idx]),
            "dphi": float(dphi_arr[idx]),
            "zero_mw": float(baselines_np[idx]),
            "slsqp_mw": float(slsqp_mws[idx]),
            "slsqp_gain_pct": float(slsqp_gains[idx]),
            "lookup_gain_pct": float(lookup_gains[idx]) if lt else None,
            "regime": "aligned_cube" if aligned_cube[idx] else "other",
            "policies": {},
        }
        for tag, pres in policy_results.items():
            row["policies"][tag] = {
                "gain_pct": float(pres["gains"][idx]),
            }
        per_condition.append(row)
    result["per_condition"] = per_condition

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---------- Scatter plot ----------
    _make_scatter_fig(policy_results, slsqp_gains, aligned_cube,
                       phis, vs, dphi_arr, valid_mask)

    print(f"\n  Total wall-clock: {time.time()-t_start:.0f}s")


def _make_scatter_fig(policy_results, slsqp_gains, aligned_cube,
                      phis, vs, dphi_arr, valid_mask):
    n_policies = len(policy_results)
    if n_policies == 0:
        return

    ncols = min(3, n_policies + 1)
    nrows = 1 if n_policies <= 2 else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)

    colors = np.where(aligned_cube, '#E45756', '#4C78A8')

    col = 0
    # (a) SLSQP gain histogram
    ax = axes[0][col]
    ax.hist(slsqp_gains, bins=40, alpha=0.5, color='#E45756', edgecolor='black', lw=0.3)
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel("SLSQP gain [%]")
    ax.set_ylabel("Count")
    ax.set_title("(a) SLSQP oracle gain distribution")
    ax.grid(alpha=0.3)
    col += 1

    # One scatter per policy.
    for tag, pres in policy_results.items():
        if col >= ncols * nrows:
            break
        row_idx = col // ncols
        col_idx = col % ncols
        ax = axes[row_idx][col_idx]
        gains = pres["gains"]
        ax.scatter(slsqp_gains, gains, s=10, alpha=0.5, c=colors)
        lims = [min(slsqp_gains.min(), gains.min()),
                max(slsqp_gains.max(), gains.max())]
        ax.plot(lims, lims, 'k--', lw=0.8, label='1:1')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.axvline(0, color='gray', lw=0.5, ls=':')
        ax.set_xlabel("SLSQP gain [%]")
        ax.set_ylabel(f"{tag} gain [%]")
        ax.set_title(f"DRL ({tag}) vs SLSQP\n(red = aligned-cube)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
        col += 1

    # Hide unused axes.
    for idx in range(col, ncols * nrows):
        row_idx = idx // ncols
        col_idx = idx % ncols
        axes[row_idx][col_idx].set_visible(False)

    fig.suptitle("Unified Static DRL vs SLSQP Evaluation", fontsize=11)
    fig.tight_layout()
    for ext in ['pdf', 'jpg']:
        path = os.path.join(FIG_DIR, f"fig_unified_static_vs_slsqp_scatter.{ext}")
        fig.savefig(path, dpi=300 if ext == 'jpg' else None, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
