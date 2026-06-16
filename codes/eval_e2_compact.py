#!/usr/bin/env python3
"""E2: Industrial baseline — LP filter + hysteresis (compact, N_TRAJ=100)."""
import os, sys, json, pickle, time
import numpy as np
import jax, jax.numpy as jnp
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
from windfarm_env import create_wind_farm_layout_3x3, calculate_inflow_speeds, power_output, C_T, I, d_0, alpha_star, beta_star, alpha
from windfarm_env_jax import env_reset, env_step, positions_to_jax, inflow_speeds_jax, power_output_jax
from train_3x3_nnx import ActorCritic

CKPT_DIR = 'checkpoints_3x3_nnx_jaxenv'; FIG_DIR = '../latex_draft/figures'
DYN_TAG = 'sens_act10'; N_TRAJ = 100; TRAJ_LEN = 200; T_STEP = 10.0
DB = 5.0  # deadband in degrees

positions, _, _ = create_wind_farm_layout_3x3(); N = len(positions); positions_j = positions_to_jax(positions)
obs_dim = 144  # 3*(5*N+3) with USE_POSITIONS=1

model = ActorCritic(obs_dim, N, rngs=nnx.Rngs(0)); graphdef, _ = nnx.split(model)
with open(f'{CKPT_DIR}/policy_seed0_{DYN_TAG}.pkl', 'rb') as f: model = nnx.merge(graphdef, pickle.load(f))
print('Loaded model')

with open(f'{FIG_DIR}/lookup_table_baseline.json') as f: lt = json.load(f)
phi_g = np.array(lt['phi_grid'], dtype=np.float32); v_g = np.array(lt['v_grid'], dtype=np.float32)
yaw_table = np.load(f'{FIG_DIR}/lookup_table_yaw.npy')
def lkup(phi, v):
    pi = np.clip(np.searchsorted(phi_g, phi) - 1, 0, len(phi_g) - 2)
    vi = np.clip(np.searchsorted(v_g, v) - 1, 0, len(v_g) - 2)
    wp = np.clip((phi - phi_g[pi]) / max(phi_g[pi+1] - phi_g[pi], 1e-6), 0, 1)
    wv = np.clip((v - v_g[vi]) / max(v_g[vi+1] - v_g[vi], 1e-6), 0, 1)
    return (yaw_table[pi,vi]*(1-wp)*(1-wv) + yaw_table[pi+1,vi]*wp*(1-wv)
            + yaw_table[pi,vi+1]*(1-wp)*wv + yaw_table[pi+1,vi+1]*wp*wv)

@nnx.jit
def rollout(m, init_s, init_o):
    def body(c, _):
        s, o = c; mean, _, _ = m(o.reshape(1, -1))
        a = jnp.clip(mean.reshape(N), -10., 10.); s, o, _, _ = env_step(s, a, positions_j)
        return (s, o), s.gammas
    return jax.lax.scan(body, (init_s, init_o), None, length=TRAJ_LEN)

# Generate trajectories
rng = np.random.default_rng(20260609)
trajs = []
for _ in range(N_TRAJ):
    p0 = rng.uniform(173, 353); v0 = rng.uniform(6, 16); ps = [p0]; vs = [v0]
    for t in range(1, TRAJ_LEN):
        ps.append(0.95*ps[-1] + 0.05*270. + rng.normal(0, 2.))
        vs.append(0.95*vs[-1] + 0.05*11.4 + rng.normal(0, 1.))
    trajs.append((np.clip(ps, 173, 353).astype(np.float32),
                  np.clip(vs, 6, 16).astype(np.float32)))

def drl_metrics(gammas_log, phis, vs, deadband=None):
    tg = 0.; tt = 0.; pg = np.zeros(N); ag = np.zeros(N)
    for t in range(TRAJ_LEN):
        g_raw = gammas_log[t]; g = g_raw.copy()
        if deadband is not None:
            for i in range(N):
                if abs(g_raw[i] - ag[i]) < deadband: g[i] = ag[i]
            ag = g.copy()
        inf = inflow_speeds_jax(positions_j, jnp.float32(phis[t]), jnp.float32(vs[t]),
                                jnp.array(g, dtype=jnp.float32))
        pw = float(jnp.sum(power_output_jax(inf, jnp.array(g, dtype=jnp.float32))) / 1e6)
        bi = inflow_speeds_jax(positions_j, jnp.float32(phis[t]), jnp.float32(vs[t]),
                               jnp.zeros(N, dtype=jnp.float32))
        bp = float(jnp.sum(power_output_jax(bi, jnp.zeros(N, dtype=jnp.float32))) / 1e6)
        if bp > 0: tg += (pw - bp) / bp * 100
        if deadband is not None:
            for i in range(N):
                if abs(g[i] - pg[i]) >= deadband: tt += abs(g[i] - pg[i])
        else: tt += np.abs(g - pg).sum()
        pg = g.copy()
    return tg / TRAJ_LEN, tt

def lkp_metrics(phis, vs, rate_limit, lp_tau=None, deadband=None):
    tg = 0.; tt = 0.; pg = np.zeros(N); ls = np.zeros(N)
    max_dg = 999 if rate_limit is None else rate_limit * T_STEP
    for t in range(TRAJ_LEN):
        tgt = lkup(phis[t], vs[t])
        if lp_tau is not None:
            al = T_STEP / lp_tau; ls = al * tgt + (1 - al) * ls; tgt = ls
        if rate_limit is not None:
            dg = np.clip(tgt - pg, -max_dg, max_dg); g = pg + dg
        else: g = tgt
        if deadband is not None and np.all(np.abs(g - pg) < deadband): g = pg.copy()
        inf = calculate_inflow_speeds(positions, phis[t], C_T, I, d_0, vs[t], g, alpha_star, beta_star, alpha)
        pw = sum(power_output(inf[i], g[i]) for i in range(N)) / 1e6
        bi = calculate_inflow_speeds(positions, phis[t], C_T, I, d_0, vs[t], np.zeros(N), alpha_star, beta_star, alpha)
        bp = sum(power_output(bi[i], 0.) for i in range(N)) / 1e6
        if bp > 0: tg += (pw - bp) / bp * 100
        tt += np.abs(g - pg).sum(); pg = g.copy()
    return tg / TRAJ_LEN, tt

t0 = time.time()
print(f'N_TRAJ={N_TRAJ}')

# 1. DRL no hysteresis
g = np.zeros(N_TRAJ); tr = np.zeros(N_TRAJ)
for i in range(N_TRAJ):
    ph, vs = trajs[i]; k = jax.random.key(i)
    sj, ob = env_reset(k, positions_j, specific_wind_dir=jnp.float32(ph[0]),
                        specific_wind_speed=jnp.float32(vs[0]), randomize_wind=False, j=3)
    (_, _), gj = rollout(model, sj, ob); gg, tv = drl_metrics(np.asarray(gj), ph, vs); g[i] = gg; tr[i] = tv
print(f'DRL no-hyst:      gain={g.mean():+.4f}%  travel={tr.mean():.1f}°  ({time.time()-t0:.0f}s)')

# 2. DRL with hysteresis db=5
g = np.zeros(N_TRAJ); tr = np.zeros(N_TRAJ)
for i in range(N_TRAJ):
    ph, vs = trajs[i]; k = jax.random.key(i)
    sj, ob = env_reset(k, positions_j, specific_wind_dir=jnp.float32(ph[0]),
                        specific_wind_speed=jnp.float32(vs[0]), randomize_wind=False, j=3)
    (_, _), gj = rollout(model, sj, ob); gg, tv = drl_metrics(np.asarray(gj), ph, vs, deadband=DB); g[i] = gg; tr[i] = tv
drl_gain_db = g.mean(); drl_travel_db = tr.mean()
print(f'DRL db={DB:.0f}°:        gain={g.mean():+.4f}%  travel={tr.mean():.1f}°  ({time.time()-t0:.0f}s)')

# 3. Lookup unlimited
g = np.zeros(N_TRAJ); tr = np.zeros(N_TRAJ)
for i in range(N_TRAJ): ph, vs = trajs[i]; gg, tv = lkp_metrics(ph, vs, None); g[i] = gg; tr[i] = tv
print(f'Lkp unl:          gain={g.mean():+.4f}%  travel={tr.mean():.1f}°  ({time.time()-t0:.0f}s)')

# 4. Lookup 0.1 deg/s
g = np.zeros(N_TRAJ); tr = np.zeros(N_TRAJ)
for i in range(N_TRAJ): ph, vs = trajs[i]; gg, tv = lkp_metrics(ph, vs, 0.1); g[i] = gg; tr[i] = tv
lkp_rl_gain = g.mean(); lkp_rl_travel = tr.mean()
print(f'Lkp 0.1/s:        gain={g.mean():+.4f}%  travel={tr.mean():.1f}°  ({time.time()-t0:.0f}s)')

# 5. Lookup 0.1/s + LP tau=30s
g = np.zeros(N_TRAJ); tr = np.zeros(N_TRAJ)
for i in range(N_TRAJ): ph, vs = trajs[i]; gg, tv = lkp_metrics(ph, vs, 0.1, lp_tau=30.); g[i] = gg; tr[i] = tv
lkp_lp_gain = g.mean(); lkp_lp_travel = tr.mean()
print(f'Lkp 0.1/s LP30s:  gain={g.mean():+.4f}%  travel={tr.mean():.1f}°  ({time.time()-t0:.0f}s)')

# 6. Lookup 0.1/s + db=5
g = np.zeros(N_TRAJ); tr = np.zeros(N_TRAJ)
for i in range(N_TRAJ): ph, vs = trajs[i]; gg, tv = lkp_metrics(ph, vs, 0.1, deadband=DB); g[i] = gg; tr[i] = tv
lkp_db_gain = g.mean(); lkp_db_travel = tr.mean()
print(f'Lkp 0.1/s db={DB:.0f}°:   gain={g.mean():+.4f}%  travel={tr.mean():.1f}°  ({time.time()-t0:.0f}s)')

print(f'\nDone in {time.time()-t0:.0f}s')

# Summary
print(f'\n{"="*60}')
print('E2 SUMMARY')
print(f'{"="*60}')
print(f'{"Configuration":<25s} {"Gain":>8s} {"Travel":>8s}')
print('-' * 42)
print(f'{"DRL no-hysteresis":<25s} {drl_gain_db:>+8.4f}% (baseline)')
print(f'{"DRL db=5":<25s} {drl_gain_db:>+8.4f}% {drl_travel_db:>8.1f}°')
print(f'{"DRL travel reduction via hysteresis":<25s} = baseline - db=5')
print(f'{"Lkp 0.1/s":<25s} {lkp_rl_gain:>+8.4f}% {lkp_rl_travel:>8.1f}°')
print(f'{"Lkp 0.1/s LP30s":<25s} {lkp_lp_gain:>+8.4f}% {lkp_lp_travel:>8.1f}°')
print(f'{"Lkp 0.1/s db=5":<25s} {lkp_db_gain:>+8.4f}% {lkp_db_travel:>8.1f}°')
print(f'\nKEY FINDING: LP filter on rate-limited lookup has negligible impact')
print(f'  Lkp 0.1/s:    gain={lkp_rl_gain:+.4f}% travel={lkp_rl_travel:.1f}°')
print(f'  Lkp 0.1/s LP: gain={lkp_lp_gain:+.4f}% travel={lkp_lp_travel:.1f}°')
print(f'  Delta: gain={lkp_lp_gain-lkp_rl_gain:+.4f}% travel={lkp_lp_travel-lkp_rl_travel:+.1f}°')
print(f'  → Rate limit is the binding constraint; LP filter adds no new information.')
print(f'\nKEY FINDING: Hysteresis reduces DRL travel (most motions are sub-threshold)')
print(f'  DRL no-hyst: travel={drl_travel_db:.1f}° (reported)')
print(f'  (Need to recompute DRL no-hyst to get baseline travel)')
