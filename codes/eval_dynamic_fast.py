#!/usr/bin/env python3
"""Fast dynamic wind eval using vmap batching. Evaluates DRL sens_act10 vs lookup."""
import os, sys, json, pickle, time, numpy as np
import jax, jax.numpy as jnp
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from windfarm_env import (create_wind_farm_layout_3x3, calculate_inflow_speeds,
                           power_output, C_T, I, d_0, alpha_star, beta_star, alpha)
from windfarm_env_jax import (env_reset, env_step, positions_to_jax,
                               inflow_speeds_jax, power_output_jax)
from train_3x3_nnx import ActorCritic

CKPT_DIR = "checkpoints_3x3_nnx_jaxenv"
FIG_DIR = "../latex_draft/figures"
DYN_TAG = os.environ.get("DYN_TAG", "sens_act10")
N_TRAJ = int(os.environ.get("N_TRAJ", "200"))  # reduced for speed
TRAJ_LEN = int(os.environ.get("TRAJ_LEN", "200"))
T_STEP = 10.0

positions, _, _ = create_wind_farm_layout_3x3()
N = len(positions)
pj = positions_to_jax(positions)

# Load models
obs_dim = 3 * (5 * N + 3)  # J=3, USE_POSITIONS=1
models = []
for s in range(3):
    ckpt = os.path.join(CKPT_DIR, f"policy_seed{s}_{DYN_TAG}.pkl")
    if os.path.exists(ckpt):
        m = ActorCritic(obs_dim, N, rngs=nnx.Rngs(0))
        gd, _ = nnx.split(m)
        with open(ckpt, "rb") as f:
            state = pickle.load(f)
        models.append(nnx.merge(gd, state))
print(f"Loaded {len(models)} DRL seeds")

# Generate AR(1) trajectories (all at once)
rng = np.random.default_rng(42)
phis = np.zeros((N_TRAJ, TRAJ_LEN))
vs = np.zeros((N_TRAJ, TRAJ_LEN))
for i in range(N_TRAJ):
    phis[i, 0] = rng.uniform(173, 353)
    vs[i, 0] = rng.uniform(6, 16)
    for t in range(1, TRAJ_LEN):
        phis[i, t] = 0.95 * phis[i, t-1] + 0.05 * 270.0 + rng.normal(0, 2.0)
        vs[i, t] = 0.95 * vs[i, t-1] + 0.05 * 11.4 + rng.normal(0, 1.0)
phis = np.clip(phis, 173, 353)
vs = np.clip(vs, 6, 16)

# Batched DRL evaluation
@nnx.jit
def eval_drl(m, phi0, v0, traj_phis, traj_vs):
    """Evaluate one trajectory: reset at (phi0, v0), then scan over trajectory."""
    key = jax.random.key(0)
    s, obs = env_reset(key, pj, specific_wind_dir=phi0, specific_wind_speed=v0,
                        randomize_wind=False, j=3)
    def body(carry, xs):
        s, obs, prev_g = carry
        phi_t, v_t = xs
        mean, _, _ = m(obs.reshape(1, -1))
        a = jnp.clip(mean.reshape(N), -10.0, 10.0)
        s, obs, _, _ = env_step(s, a, pj)
        g = s.gammas
        inflow = inflow_speeds_jax(pj, phi_t, v_t, g)
        pwr = jnp.sum(power_output_jax(inflow, g)) / 1e6
        base_in = inflow_speeds_jax(pj, phi_t, v_t, jnp.zeros(N, dtype=jnp.float32))
        base_pwr = jnp.sum(power_output_jax(base_in, jnp.zeros(N, dtype=jnp.float32))) / 1e6
        gain = jnp.where(base_pwr > 0, (pwr - base_pwr) / base_pwr * 100, 0.0)
        travel = jnp.abs(g - prev_g).sum()
        rate = travel / T_STEP
        return (s, obs, g), (gain, travel, rate)
    (_, _, _), (gains, travels, rates) = jax.lax.scan(body, (s, obs, jnp.zeros(N)),
                                                        (traj_phis, traj_vs))
    return gains.mean(), travels.sum(), rates.max()

# Evaluate all DRL seeds
drl_results = []
for seed_idx, model in enumerate(models):
    print(f"DRL seed {seed_idx}...")
    t0 = time.time()
    seed_gains, seed_travels, seed_rates = [], [], []
    for i in range(N_TRAJ):
        g, t, r = eval_drl(model, jnp.float32(phis[i, 0]), jnp.float32(vs[i, 0]),
                            jnp.array(phis[i], dtype=jnp.float32),
                            jnp.array(vs[i], dtype=jnp.float32))
        seed_gains.append(float(g)); seed_travels.append(float(t)); seed_rates.append(float(r))
    mg, mt, mr = np.mean(seed_gains), np.mean(seed_travels), np.mean(seed_rates)
    print(f"  gain={mg:+.4f}% travel={mt:.1f}° rate={mr:.4f}°/s "
          f"eff={mg/mt*100:.4f} per 100° ({time.time()-t0:.0f}s)")
    drl_results.append((mg, mt, mr))

drl_g = np.mean([r[0] for r in drl_results])
drl_t = np.mean([r[1] for r in drl_results])
drl_r = np.mean([r[2] for r in drl_results])

# Lookup table
lt_path = os.path.join(FIG_DIR, "lookup_table_baseline.json")
yaw_path = os.path.join(FIG_DIR, "lookup_table_yaw.npy")
with open(lt_path) as f:
    lt = json.load(f)
phi_g = np.array(lt["phi_grid"], dtype=np.float32)
v_g = np.array(lt["v_grid"], dtype=np.float32)
yaw_table = np.load(yaw_path)

def lookup_yaw(phi, v):
    pi = np.clip(np.searchsorted(phi_g, phi) - 1, 0, len(phi_g)-2)
    vi = np.clip(np.searchsorted(v_g, v) - 1, 0, len(v_g)-2)
    wp = np.clip((phi-phi_g[pi])/max(phi_g[pi+1]-phi_g[pi],1e-6), 0, 1)
    wv = np.clip((v-v_g[vi])/max(v_g[vi+1]-v_g[vi],1e-6), 0, 1)
    return (yaw_table[pi,vi]*(1-wp)*(1-wv)+yaw_table[pi+1,vi]*wp*(1-wv)
            +yaw_table[pi,vi+1]*(1-wp)*wv+yaw_table[pi+1,vi+1]*wp*wv)

print("\nLookup table...")
for rate_limit, rl_name in [(None, "unlimited"), (0.5, "0.5°/s"), (0.3, "0.3°/s"), (0.1, "0.1°/s")]:
    max_dg = 999 if rate_limit is None else rate_limit * T_STEP
    gains, travels, rates_arr = np.zeros(N_TRAJ), np.zeros(N_TRAJ), np.zeros(N_TRAJ)
    for i in range(N_TRAJ):
        total_g, total_t, max_r = 0.0, 0.0, 0.0
        prev_g = np.zeros(N)
        for t in range(TRAJ_LEN):
            target = lookup_yaw(phis[i, t], vs[i, t])
            if rate_limit is not None:
                dg = np.clip(target - prev_g, -max_dg, max_dg)
                g = prev_g + dg
            else:
                g = target
            inflow = calculate_inflow_speeds(positions, phis[i,t], C_T, I, d_0, vs[i,t], g, alpha_star, beta_star, alpha)
            pwr = sum(power_output(inflow[j], g[j]) for j in range(N)) / 1e6
            base_in = calculate_inflow_speeds(positions, phis[i,t], C_T, I, d_0, vs[i,t], np.zeros(N), alpha_star, beta_star, alpha)
            base_pwr = sum(power_output(base_in[j], 0.0) for j in range(N)) / 1e6
            if base_pwr > 0:
                total_g += (pwr - base_pwr) / base_pwr * 100
            travel = np.abs(g - prev_g).sum()
            total_t += travel
            max_r = max(max_r, travel / T_STEP)
            prev_g = g.copy()
        gains[i] = total_g / TRAJ_LEN
        travels[i] = total_t
        rates_arr[i] = max_r
    print(f"  {rl_name:>10s}: gain={gains.mean():+.4f}% travel={travels.mean():.1f}° "
          f"rate={rates_arr.mean():.4f}°/s eff={gains.mean()/travels.mean()*100:.4f}")

# Comparison
print(f"\n=== SUMMARY ===")
print(f"{'Controller':<25s} {'Gain':>8s} {'Travel':>8s} {'Rate':>8s} {'Eff':>8s}")
print("-" * 60)
print(f"{'DRL '+DYN_TAG:<25s} {drl_g:+8.4f}% {drl_t:8.1f}° {drl_r:8.4f}°/s {drl_g/drl_t*100:8.4f}")
print(f"{'OLD DRL p0c':<25s} {+0.183:+8.4f}% {123.0:8.1f}° {0.126:8.4f}°/s {0.183/123*100:8.4f}")
print(f"{'OLD Lookup unltd':<25s} {+1.895:+8.4f}% {2151.5:8.1f}° {3.320:8.4f}°/s {1.895/2151.5*100:8.4f}")

out = {
    "drl_gain": drl_g, "drl_travel": drl_t, "drl_rate": drl_r,
    "drl_efficiency": drl_g/drl_t*100,
    "n_traj": N_TRAJ, "traj_len": TRAJ_LEN, "tag": DYN_TAG,
}
with open(os.path.join(FIG_DIR, "dynamic_wind_optimized.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {FIG_DIR}/dynamic_wind_optimized.json")
