"""Evaluate trained SAC policy on the same protocol as PPO (500 conditions)."""
import sys, os, pickle, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from train_3x3_nnx import MLP, NET_ARCH
from windfarm_env_jax import env_reset_batched, env_step_autoreset, positions_to_jax
from windfarm_env import create_wind_farm_layout_3x3

ACT_BOUND=10.0; J_HIST=3
class Actor(nnx.Module):
    def __init__(self,od,ad,rngs): self.net=MLP(od,NET_ARCH,ad,rngs=rngs); self.log_std=nnx.Param(jnp.full((ad,),-0.5))
    def deterministic_action(self, obs):
        mean = self.net(obs)
        return jnp.tanh(mean) * ACT_BOUND

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pos,_,_ = create_wind_farm_layout_3x3(); pj = positions_to_jax(pos)
Nt = pj.shape[0]; odim = J_HIST*(3*Nt+3+2*Nt); adim = Nt

# Load SAC policies
policies = []
for s in range(3):
    actor = Actor(odim, adim, rngs=nnx.Rngs(0))
    gd, _ = nnx.split(actor)
    ckpt = os.path.join(_SCRIPT_DIR, f'checkpoints_3x3_sac_jaxenv_v5/policy_seed{s}.pkl')
    if os.path.exists(ckpt):
        with open(ckpt,'rb') as f: st = pickle.load(f)
        policies.append(nnx.merge(gd, st))
        print(f'Loaded SAC seed {s} from {ckpt}')
    else:
        print(f'MISSING: {ckpt}')

if not policies:
    print('No SAC policies found!')
    sys.exit(1)

# Evaluate on 500 conditions (same protocol as paper)
n_cond = 500
seed = 20260613
np.random.seed(seed)
conditions = [(np.random.uniform(173,353), np.random.uniform(6,16)) for _ in range(n_cond)]

all_gains = []  # per-seed, per-condition
for s, policy in enumerate(policies):
    gains = []
    t0 = time.time()
    for i, (phi, v) in enumerate(conditions):
        k = jax.random.PRNGKey(int(phi*100+v*10+s))
        state, obs = env_reset_batched(
            jax.random.split(k,1), pj, j=J_HIST, max_steps=200,
            randomize_wind=False,
            specific_wind_dir=jnp.array(phi),
            specific_wind_speed=jnp.array(v))
        tr = 0.0
        for _ in range(100):
            action = policy.deterministic_action(obs)
            state, obs, reward, done = env_step_autoreset(
                state, action, jax.random.split(k,1), pj,
                j=J_HIST, max_steps=200, randomize_wind=False)
            tr += float(reward[0])
            if done[0]: break
            k = jax.random.split(k,1)[0]
        gains.append(tr / 10.0)
        if (i+1) % 100 == 0:
            elapsed = time.time()-t0
            print(f'  Seed {s}: {i+1}/{n_cond} ({elapsed:.0f}s, {i/elapsed:.1f}/s)', flush=True)
    all_gains.append(gains)
    print(f'  Seed {s} done: mean={np.mean(gains):+.2f}%, median={np.median(gains):+.2f}%')

# Aggregate across seeds
gains_arr = np.array(all_gains)  # (3, 500)
mean_gains = gains_arr.mean(axis=0)  # average over seeds per condition
marginal_mean = np.mean(mean_gains)
marginal_median = np.median(mean_gains)

# Aligned-cube regime
ac_mask = []
for phi, v in conditions:
    dphi = abs(phi - 270); dphi = min(dphi, 360-dphi)
    ac_mask.append(dphi < 15 and v < 11.4)
ac_mask = np.array(ac_mask)
ac_gains = mean_gains[ac_mask]
ac_mean = np.mean(ac_gains) if len(ac_gains) > 0 else 0

print(f'\n=== SAC Results (3-seed, 6M steps each) ===')
print(f'  Marginal mean gain: {marginal_mean:+.2f}%')
print(f'  Marginal median gain: {marginal_median:+.2f}%')
print(f'  Aligned-cube gain: {ac_mean:+.2f}% (n={ac_mask.sum()} conditions)')
print(f'  P95 gain: {np.percentile(mean_gains, 95):+.2f}%')
print(f'  Negative gain fraction: {(mean_gains < 0).mean()*100:.1f}%')

# Save
out = dict(
    marginal_mean=float(marginal_mean), marginal_median=float(marginal_median),
    aligned_cube_mean=float(ac_mean), n_aligned_cube=int(ac_mask.sum()),
    p95=float(np.percentile(mean_gains, 95)),
    neg_frac=float((mean_gains<0).mean()),
    per_seed_means=[float(np.mean(g)) for g in all_gains],
    per_seed_ac=[float(np.mean(np.array(g)[ac_mask])) for g in all_gains],
    n_conditions=n_cond, n_seeds=len(policies), total_steps=6_000_000,
)
with open(os.path.join(_SCRIPT_DIR, 'sac_eval_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved to sac_eval_results.json')
