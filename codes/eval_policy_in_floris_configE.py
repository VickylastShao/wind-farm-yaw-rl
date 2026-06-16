#!/usr/bin/env python3
"""FLORIS cross-evaluation of Config-E DRL policies (3×3 NREL-5MW)."""
import os, sys, json, time
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, inflow_speeds_jax,
    power_output_jax, positions_to_jax)
from windfarm_env import create_wind_farm_layout_3x3

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_nnx_jaxenv")
FIG_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Config-E settings
J = 3; N_TURB = 9; ACT_BOUND = 10.0
OBS_DIM_PER_STEP = 5*N_TURB + 3  # with USE_POSITIONS=1
OBS_DIM = J * OBS_DIM_PER_STEP  # 144
SETTLE_STEPS = 200
N_CONDITIONS = 3000
N_SEEDS = 5
EVAL_SEED = int(os.environ.get("EVAL_SEED", "20260604"))
TI = 0.065

def make_floris_model():
    from floris import FlorisModel
    fm = FlorisModel("gch")
    d_0 = 126.0; sx, sy = 7*d_0, 7*d_0; tilt = np.radians(7.0)
    xs, ys = [], []
    for j in range(3):
        for i in range(3):
            xs.append(i*sx + j*sy*np.sin(tilt))
            ys.append(j*sy*np.cos(tilt))
    fm.set(layout_x=xs, layout_y=ys)
    return fm

def floris_farm_power(fm, phi, v, yaw_angles=None):
    n_turb = len(fm.layout_x)
    yaw_angles = np.zeros((1, n_turb)) if yaw_angles is None else np.asarray(yaw_angles).reshape(1, -1)
    fm.set(wind_directions=[float(phi)], wind_speeds=[float(v)],
           turbulence_intensities=[float(TI)], yaw_angles=yaw_angles)
    fm.run()
    return float(np.sum(fm.get_turbine_powers())) / 1e6

def load_configE_policy(seed):
    ckpt = os.path.join(CKPT_DIR, f"policy_seed{seed}_sens_act10.pkl")
    if not os.path.exists(ckpt): return None
    model = ActorCritic(OBS_DIM, N_TURB, rngs=nnx.Rngs(0))
    graphdef, _ = nnx.split(model)
    with open(ckpt, "rb") as f: state = pickle.load(f)
    return nnx.merge(graphdef, state)

def evaluate_batch_get_yaws(model, positions_jax, phis, vs):
    """Run Config-E policy on batched conditions via vmap, return final yaw vectors."""
    import pickle
    N = N_CONDITIONS
    
    @nnx.jit
    def run(m, phis, vs):
        @jax.vmap
        def reset_one(phi, v):
            key = jax.random.key(0)
            state, obs = env_reset(key, positions_jax, j=J,
                specific_wind_dir=phi, specific_wind_speed=v,
                randomize_wind=False, max_steps=SETTLE_STEPS+10)
            return state, obs
        
        states, obs_batch = reset_one(phis, vs)
        
        @jax.vmap
        def predict_one(o):
            mean, _, _ = m(o.reshape(1, -1))
            return mean.reshape(N_TURB)
        
        @jax.vmap
        def step_one(s, a):
            return env_step(s, a, positions_jax, max_steps=SETTLE_STEPS+10)
        
        def body(carry, _):
            states, obs = carry
            actions = predict_one(obs)
            actions = jnp.clip(actions, -ACT_BOUND, ACT_BOUND)
            new_states, new_obs, _, _ = step_one(states, actions)
            return (new_states, new_obs), None
        
        (final_states, _), _ = jax.lax.scan(body, (states, obs_batch), None, length=SETTLE_STEPS)
        return final_states.total_mw, final_states.gammas
    
    total_mw, gammas = run(model, phis, vs)
    return np.asarray(gammas)

def main():
    import pickle
    t_start = time.time()
    positions, _, _ = create_wind_farm_layout_3x3()
    N_turb = len(positions)
    positions_jax = positions_to_jax(positions)
    
    # Sample conditions
    np.random.seed(EVAL_SEED)
    phis = np.random.uniform(173, 353, size=N_CONDITIONS)
    vs   = np.random.uniform(6, 16, size=N_CONDITIONS)
    phis_j = jnp.asarray(phis, dtype=jnp.float32)
    vs_j   = jnp.asarray(vs, dtype=jnp.float32)
    
    # Gray-box zero-yaw baselines
    @jax.jit
    def gb_baseline_fn(phi, v):
        inflow = inflow_speeds_jax(positions_jax, phi, v, jnp.zeros(N_turb))
        return jnp.sum(power_output_jax(inflow, jnp.zeros(N_turb))) / 1e6
    gb_baselines = jax.vmap(gb_baseline_fn)(phis_j, vs_j)
    gb_baselines_np = np.asarray(gb_baselines)
    
    # Regime masks
    dphi = np.abs(((phis - 270 + 180) % 360) - 180)
    aligned_cube = (dphi < 15) & (vs < 11.4)
    
    # FLORIS
    print("Init FLORIS...", flush=True)
    fm = make_floris_model()
    print("Computing FLORIS baselines...", flush=True)
    floris_baselines = np.empty(N_CONDITIONS)
    for i in range(N_CONDITIONS):
        floris_baselines[i] = floris_farm_power(fm, phis[i], vs[i])
        if (i+1) % 500 == 0: print(f"  {i+1}/{N_CONDITIONS}", flush=True)
    print(f"  GB baseline: {gb_baselines_np.mean():.2f} MW, FLORIS: {floris_baselines.mean():.2f} MW")
    
    # Evaluate each Config-E seed
    results = []
    for s in range(N_SEEDS):
        model = load_configE_policy(s)
        if model is None: print(f"Seed {s}: missing"); continue
        
        print(f"Seed {s}: rollout...", flush=True)
        yaws = evaluate_batch_get_yaws(model, positions_jax, phis_j, vs_j)
        
        print(f"Seed {s}: FLORIS eval...", flush=True)
        floris_yawked = np.empty(N_CONDITIONS)
        for i in range(N_CONDITIONS):
            floris_yawked[i] = floris_farm_power(fm, phis[i], vs[i], yaw_angles=yaws[i])
            if (i+1) % 500 == 0: print(f"  {i+1}/{N_CONDITIONS}", flush=True)
        
        gb_gains = (np.asarray([floris_farm_power.__wrapped__ if False else 0]) + 1) * 0  # placeholder
        # Actually compute gray-box gains
        gb_yawked = np.empty(N_CONDITIONS)
        for i in range(N_CONDITIONS):
            inflow = np.asarray(inflow_speeds_jax(positions_jax, phis_j[i], vs_j[i], jnp.asarray(yaws[i])))
            pwr = np.asarray(power_output_jax(jnp.asarray(inflow), jnp.asarray(yaws[i])))
            gb_yawked[i] = np.sum(pwr) / 1e6
        
        gb_gains = (gb_yawked - gb_baselines_np) / gb_baselines_np * 100
        floris_gains = (floris_yawked - floris_baselines) / floris_baselines * 100
        
        r = {
            "seed": s,
            "gb_marginal_pct": float(gb_gains.mean()),
            "gb_aligned_cube_pct": float(gb_gains[aligned_cube].mean()) if aligned_cube.sum()>0 else None,
            "floris_marginal_pct": float(floris_gains.mean()),
            "floris_aligned_cube_pct": float(floris_gains[aligned_cube].mean()) if aligned_cube.sum()>0 else None,
        }
        results.append(r)
        print(f"  GB: marginal={r['gb_marginal_pct']:+.3f}% aligned={r['gb_aligned_cube_pct']:+.3f}%")
        print(f"  FL: marginal={r['floris_marginal_pct']:+.3f}% aligned={r['floris_aligned_cube_pct']:+.3f}%")
    
    # Aggregate
    gb_marg = np.mean([r['gb_marginal_pct'] for r in results])
    fl_marg = np.mean([r['floris_marginal_pct'] for r in results])
    gb_ac = np.mean([r['gb_aligned_cube_pct'] for r in results if r['gb_aligned_cube_pct'] is not None])
    fl_ac = np.mean([r['floris_aligned_cube_pct'] for r in results if r['floris_aligned_cube_pct'] is not None])
    
    summary = {
        "n_seeds": len(results), "n_conditions": N_CONDITIONS,
        "gb_baseline_mw": float(gb_baselines_np.mean()),
        "floris_baseline_mw": float(floris_baselines.mean()),
        "gb_marginal_mean_pct": float(gb_marg), "floris_marginal_mean_pct": float(fl_marg),
        "gb_aligned_cube_pct": float(gb_ac), "floris_aligned_cube_pct": float(fl_ac),
        "n_aligned_cube": int(aligned_cube.sum()),
        "per_seed": results,
    }
    
    with open(os.path.join(RESULTS_DIR, "configE_floris_cross_validation.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to results/configE_floris_cross_validation.json")
    print(f"  GB: marg={gb_marg:+.3f}%, ac={gb_ac:+.3f}%")
    print(f"  FL: marg={fl_marg:+.3f}%, ac={fl_ac:+.3f}%")
    print(f"  Erosion: {(1-fl_ac/gb_ac)*100:.1f}%" if gb_ac > 0 else "  N/A")
    print(f"  Time: {(time.time()-t_start)/60:.1f} min")

if __name__ == "__main__":
    main()
