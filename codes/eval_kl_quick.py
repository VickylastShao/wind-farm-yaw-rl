"""Quick evaluation of KL ON/OFF policies to fill Supplementary table."""
import os, sys, pickle, json
import jax, jax.numpy as jnp, numpy as np
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, inflow_speeds_jax,
    power_output_jax, positions_to_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT = "checkpoints_3x3_nnx_jaxenv"
# KL policies use Config-D settings: J=3, positions, deficit, ACT_BOUND=5.0
J = 3; N = 9; AB = 5.0
obs_dim_per_step = 5 * N + 3  # with USE_POSITIONS=1, USE_DEFICIT=1
OBS_DIM = J * obs_dim_per_step  # 144
SETTLE = 200; N_COND = 300

pos,_,_ = create_wind_farm_layout_3x3(); pj = positions_to_jax(pos)

np.random.seed(20260616)
phis = np.random.uniform(173, 353, N_COND)
vs   = np.random.uniform(6, 16, N_COND)
dphi = np.minimum(np.abs(phis-270), 360-np.abs(phis-270))
ac_mask = (dphi < 15) & (vs < 11.4)

results = {}
for tag in ['kl_on', 'kl_off']:
    gains_all = []
    for s in range(3):
        ckpt = os.path.join(CKPT, f"policy_seed{s}_{tag}.pkl")
        if not os.path.exists(ckpt):
            print(f"{tag} seed{s}: MISSING")
            continue
        m = ActorCritic(OBS_DIM, N, rngs=nnx.Rngs(0)); gd, _ = nnx.split(m)
        with open(ckpt, "rb") as f: st = pickle.load(f)
        model = nnx.merge(gd, st)
        
        gains = []
        for i in range(N_COND):
            k = jax.random.key(i)
            s_state, o = env_reset(k, pj, j=J, specific_wind_dir=jnp.array(phis[i]),
                specific_wind_speed=jnp.array(vs[i]), randomize_wind=False, max_steps=SETTLE+10)
            tr = 0.0
            for _ in range(SETTLE):
                mean, _, _ = model(o.reshape(1, -1))
                a = jnp.clip(mean.reshape(N), -AB, AB)
                s_state, o, r, _ = env_step(s_state, a, pj, max_steps=SETTLE+10)
                tr += float(r)
            gains.append(tr / 10.0)
        
        gains = np.array(gains)
        marg = float(gains.mean())
        ac = float(gains[ac_mask].mean()) if ac_mask.sum() > 0 else 0.0
        print(f"{tag} seed{s}: marg={marg:+.2f}%, ac={ac:+.2f}%")
        gains_all.append(marg)
    
    if gains_all:
        results[tag] = dict(
            mean_gain=float(np.mean(gains_all)),
            std_gain=float(np.std(gains_all)),
            n_seeds=len(gains_all),
        )

print()
for tag, r in results.items():
    print(f"{tag}: {r['n_seeds']} seeds, mean_gain={r['mean_gain']:+.2f}% ± {r['std_gain']:.2f}%")

with open("../results/kl_ablation_eval.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/kl_ablation_eval.json")
