#!/usr/bin/env python3
"""GPU-batched Config-E noise robustness — vmap + lax.scan."""
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
J=3; N=9; AB=10.0; OBS_DIM=3*(5*N+3); SETTLE=200; N_COND=500
EVAL_SEED=20260614; N_SEEDS=5

pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

# Sample conditions
np.random.seed(EVAL_SEED)
phis=np.random.uniform(173,353,N_COND); vs=np.random.uniform(6,16,N_COND)

@jax.jit
def gb_baseline(phi,v):
    inf=inflow_speeds_jax(pj,phi,v,jnp.zeros(N))
    return jnp.sum(power_output_jax(inf,jnp.zeros(N)))/1e6
gb_bl=np.asarray(jax.vmap(gb_baseline)(jnp.asarray(phis,jnp.float32),jnp.asarray(vs,jnp.float32)))

@jax.jit
def eval_batch_noisy(model, phis, vs, key, noise_phi, noise_v, noise_yaw):
    """Batch-evaluate with observation noise. Returns per-condition total_reward."""
    @jax.vmap
    def reset_one(phi,v,k):
        s,o=env_reset(k,pj,j=J,specific_wind_dir=phi,specific_wind_speed=v,randomize_wind=False,max_steps=SETTLE+10)
        return s,o,k
    
    keys=jax.random.split(key,N_COND)
    states,obs_batch,keys=reset_one(phis,vs,keys)
    
    def body(carry,_):
        states,obs,keys=carry
        # Add noise to obs
        noisy_keys=jax.random.split(jax.random.fold_in(key,0),N_COND)
        phi_noise=jax.random.normal(noisy_keys[0],(N_COND,))*noise_phi
        v_noise=jax.random.normal(noisy_keys[1],(N_COND,))*noise_v
        yaw_noise=jax.random.normal(noisy_keys[2],(N_COND,N))*noise_yaw
        
        # Modified obs: noisy yaw + noisy wind
        obs_noisy=obs.at[:,:N].add(yaw_noise)
        # cos/sin/v at indices N*2 to N*2+2
        obs_phi=phis+phi_noise; obs_v=vs+v_noise
        rad=jnp.radians(obs_phi)
        obs_noisy=obs_noisy.at[:,N*2].set(jnp.cos(rad))
        obs_noisy=obs_noisy.at[:,N*2+1].set(jnp.sin(rad))
        obs_noisy=obs_noisy.at[:,N*2+2].set(obs_v)
        
        @jax.vmap
        def pred(o): mean,_,_=model(o.reshape(1,-1)); return jnp.clip(mean.reshape(N),-AB,AB)
        actions=pred(obs_noisy)
        
        @jax.vmap
        def step(s,a): return env_step(s,a,pj,max_steps=SETTLE+10)
        new_states,new_obs,rewards,_=step(states,actions)
        
        return (new_states,new_obs,keys),(rewards,)
    
    (_,_,_), (all_rewards,)=jax.lax.scan(body,(states,obs_batch,keys),None,length=SETTLE)
    total_reward=jnp.sum(all_rewards,axis=0)  # (N_COND,)
    return total_reward

# Load models
models=[]
for s in range(N_SEEDS):
    ckpt=os.path.join(CKPT_DIR,f"policy_seed{s}_sens_act10.pkl")
    if not os.path.exists(ckpt): continue
    m=ActorCritic(OBS_DIM,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
    with open(ckpt,"rb") as f: st=pickle.load(f)
    models.append(nnx.merge(gd,st))
print(f"Loaded {len(models)} models")

phis_j=jnp.asarray(phis,jnp.float32); vs_j=jnp.asarray(vs,jnp.float32)

# Evaluate for each noise level (use first model only for speed)
model=models[seed_idx]
results=[]
print("Evaluating noise robustness (GPU batched)...")
for noise_name, np_val, nv_val, ny_val in [
    ("clean",0,0,0), ("σφ=1°",1,0,0), ("σφ=2°",2,0,0), ("σφ=5°",5,0,0), ("σφ=10°",10,0,0),
    ("σv=0.3",0,0.3,0), ("σv=0.5",0,0.5,0), ("σv=1.0",0,1.0,0),
    ("σγ=0.5°",0,0,0.5), ("σγ=1°",0,0,1.0), ("σγ=2°",0,0,2.0),
    ("combined",2,0.3,1.0),
]:
    t0=time.time()
    tr=eval_batch_noisy(model,phis_j,vs_j,jax.random.key(0),
        jnp.array(np_val,dtype=jnp.float32),jnp.array(nv_val,dtype=jnp.float32),jnp.array(ny_val,dtype=jnp.float32))
    gains=np.asarray(tr)/10.0
    dphi_arr=np.minimum(np.abs(phis-270),360-np.abs(phis-270))
    ac_mask=(dphi_arr<15)&(vs<11.4)
    mg=float(gains.mean()); ac=float(gains[ac_mask].mean()) if ac_mask.sum()>0 else 0
    ng=float((gains<0).mean())
    print(f"  {noise_name:12s}: marg={mg:+.2f}% ac={ac:+.2f}% neg={ng*100:.0f}% ({time.time()-t0:.0f}s)")
    results.append(dict(noise=noise_name,sigma_phi=np_val,sigma_v=nv_val,sigma_yaw=ny_val,
        marginal=mg,aligned_cube=ac,neg_frac=ng))

with open(os.path.join(RESULTS_DIR,"configE_noise_robustness.json"),"w") as f: json.dump(results,f,indent=2)
print(f"Saved to results/configE_noise_robustness.json")
