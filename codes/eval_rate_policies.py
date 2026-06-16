#!/usr/bin/env python3
"""Evaluate rate-penalized Config-A policies under dynamic wind."""
import os, sys, json, time, pickle
import jax, jax.numpy as jnp, numpy as np
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, inflow_speeds_jax,
    power_output_jax, positions_to_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT_DIR="checkpoints_3x3_nnx_jaxenv"; RESULTS_DIR="../results"
os.makedirs(RESULTS_DIR,exist_ok=True)
J=1; N=9; AB=5.0; OBS_DIM=J*(3*N+3)  # J=1, no positions (Config-A)
N_TRAJ=1000; TRAJ_LEN=200; SETTLE=150; T_STEP=10.0
ALPHA_PHI=0.95; SIGMA_PHI=2.0; ALPHA_V=0.95; SIGMA_V=1.0

pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

# AR(1) trajectories
np.random.seed(20260614)
phi=np.random.uniform(173,353,N_TRAJ); v=np.random.uniform(6,16,N_TRAJ)
phi_hist=[phi.copy()]; v_hist=[v.copy()]
for t in range(SETTLE+TRAJ_LEN-1):
    phi=263+ALPHA_PHI*(phi-263)+SIGMA_PHI*np.random.normal(0,1,N_TRAJ)
    v=11+ALPHA_V*(v-11)+SIGMA_V*np.random.normal(0,1,N_TRAJ)
    phi=np.clip(phi,173,353); v=np.clip(v,6,16)
    phi_hist.append(phi.copy()); v_hist.append(v.copy())
phi_traj=np.column_stack(phi_hist)[:,SETTLE:]; v_traj=np.column_stack(v_hist)[:,SETTLE:]
print(f"Trajectories: {phi_traj.shape}")

@jax.jit
def eval_configA_trajectory(model, phi_arr, v_arr):
    key=jax.random.key(0)
    phi0=phi_arr[0]; v0=v_arr[0]
    state,obs=env_reset(key,pj,j=J,specific_wind_dir=phi0,specific_wind_speed=v0,randomize_wind=False,max_steps=TRAJ_LEN+10)
    def body(carry,t):
        state,obs,key,cum_yaw,cum_rew=carry
        mean,_,_=model(obs.reshape(1,-1)); action=jnp.clip(mean.reshape(N),-AB,AB)
        key,sk=jax.random.split(key)
        new_state,new_obs,reward,_=env_step(state,action,pj,max_steps=TRAJ_LEN+10)
        cum_yaw+=jnp.sum(jnp.abs(action)); cum_rew+=reward
        return (new_state,new_obs,key,cum_yaw,cum_rew),None
    init=(state,obs,key,jnp.array(0.0),jnp.array(0.0))
    (_,_,_,cum_yaw,cum_rew),__=jax.lax.scan(body,init,jnp.arange(TRAJ_LEN))
    return cum_rew,cum_yaw

tags=["rate_med","rate_high","rate_extreme","p0c"]  # p0c = Config-A baseline (λ=0)
for tag in tags:
    models=[]
    for s in range(3):
        ckpt=os.path.join(CKPT_DIR,f"policy_seed{s}_{tag}.pkl")
        if not os.path.exists(ckpt): continue
        m=ActorCritic(OBS_DIM,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
        with open(ckpt,"rb") as f: st=pickle.load(f)
        models.append(nnx.merge(gd,st))
    if not models: print(f"{tag}: no models"); continue
    
    # Compile
    _=eval_configA_trajectory(models[0],jnp.asarray(phi_traj[0]),jnp.asarray(v_traj[0]))
    jax.block_until_ready(True)
    
    t0=time.time(); gains=[]; travels=[]
    for ti in range(min(N_TRAJ,200)):
        m=models[ti%len(models)]
        tr,ty=eval_configA_trajectory(m,jnp.asarray(phi_traj[ti]),jnp.asarray(v_traj[ti]))
        gains.append(float(tr)/10.0); travels.append(float(ty))
    print(f"{tag:15s}: gain={np.mean(gains):+.2f}% travel={np.mean(travels):.1f}° ({time.time()-t0:.0f}s)")

print("\nDone. Config-A rate-penalized dynamic wind comparison.")
