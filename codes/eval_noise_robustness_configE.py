#!/usr/bin/env python3
"""Config-E observation-noise robustness evaluation."""
import os, sys, json, time, pickle
import jax, jax.numpy as jnp, numpy as np
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, inflow_speeds_jax,
    power_output_jax, positions_to_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT_DIR = "checkpoints_3x3_nnx_jaxenv"
RESULTS_DIR = "../results"; os.makedirs(RESULTS_DIR, exist_ok=True)

J=3; N=9; AB=10.0; OBS_DIM=3*(5*N+3); SETTLE=200
N_COND=500; EVAL_SEED=20260614
NOISE_PHI = [0, 1, 2, 5, 10]
NOISE_V   = [0, 0.1, 0.3, 0.5, 1.0]
NOISE_YAW = [0, 0.5, 1.0, 2.0]

pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

# Sample conditions
np.random.seed(EVAL_SEED)
phis=np.random.uniform(173,353,N_COND); vs=np.random.uniform(6,16,N_COND)
dphi=np.minimum(np.abs(phis-270),360-np.abs(phis-270))
ac_mask=(dphi<15)&(vs<11.4)

@jax.jit
def gb_baseline(phi,v):
    inf=inflow_speeds_jax(pj,phi,v,jnp.zeros(N)); return jnp.sum(power_output_jax(inf,jnp.zeros(N)))/1e6

gb_bl=np.asarray(jax.vmap(gb_baseline)(jnp.asarray(phis,jnp.float32),jnp.asarray(vs,jnp.float32)))

def eval_noisy(model, noise_phi, noise_v, noise_yaw):
    """Evaluate model with observation noise on all conditions."""
    gains=[]
    for i in range(N_COND):
        phi=phis[i]; v=vs[i]
        # Add noise to observed wind
        obs_phi = phi + np.random.normal(0, noise_phi)
        obs_v   = v   + np.random.normal(0, noise_v)
        obs_phi = np.clip(obs_phi, 173, 353); obs_v = np.clip(obs_v, 6, 16)
        k=jax.random.key(i)
        s,o=env_reset(k,pj,j=J,specific_wind_dir=jnp.array(phi),specific_wind_speed=jnp.array(v),
            randomize_wind=False,max_steps=SETTLE+10)
        o_np=np.array(o)
        tr=0.0
        for _ in range(SETTLE):
            # Apply noise to observation
            o_noisy = o_np.copy()
            # Noisy cos/sin of wind direction (indices N*2 to N*2+2)
            rad = np.radians(obs_phi)
            o_noisy[N*2] = np.cos(rad); o_noisy[N*2+1] = np.sin(rad)
            o_noisy[N*2+2] = obs_v
            # Noisy yaw readings (indices 0 to N-1)
            if noise_yaw > 0:
                o_noisy[:N] += np.random.normal(0, noise_yaw, N)
            o_j = jnp.asarray(o_noisy[None,:])
            mean,_,_ = model(o_j); a = np.clip(np.array(mean[0]), -AB, AB)
            s,o,r,_ = env_step(s, jnp.asarray(a), pj, max_steps=SETTLE+10)
            tr += float(r); o_np = np.array(o)
        gains.append(tr/10.0)
    gains=np.array(gains)
    return float(gains.mean()), float(gains[ac_mask].mean()) if ac_mask.sum()>0 else 0, float((gains<0).mean())

results=[]
for s in range(5):
    ckpt=os.path.join(CKPT_DIR,f"policy_seed{s}_sens_act10.pkl")
    if not os.path.exists(ckpt): continue
    model=ActorCritic(OBS_DIM,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(model)
    with open(ckpt,"rb") as f: st=pickle.load(f); model=nnx.merge(gd,st)
    print(f"\nSeed {s}:")
    
    # Direction noise
    for np_val in NOISE_PHI:
        mg,ac,ng=eval_noisy(model,np_val,0,0)
        print(f"  σφ={np_val:3d}°: marg={mg:+.2f}% ac={ac:+.2f}% neg={ng*100:.0f}%")
    
    # Speed noise
    for nv in NOISE_V[1:]:
        mg,ac,ng=eval_noisy(model,0,nv,0)
        print(f"  σv={nv:.1f}m/s: marg={mg:+.2f}% ac={ac:+.2f}% neg={ng*100:.0f}%")
    
    # Yaw noise
    for ny in NOISE_YAW[1:]:
        mg,ac,ng=eval_noisy(model,0,0,ny)
        print(f"  σγ={ny:.1f}°: marg={mg:+.2f}% ac={ac:+.2f}% neg={ng*100:.0f}%")
    
    # Combined
    mg,ac,ng=eval_noisy(model,2,0.3,1.0)
    print(f"  combined: marg={mg:+.2f}% ac={ac:+.2f}% neg={ng*100:.0f}%")
    
    break  # Single seed for quick test

print("\nDone (single seed quick test). Full multi-seed eval pending.")
