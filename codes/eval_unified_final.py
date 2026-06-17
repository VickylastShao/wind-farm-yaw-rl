#!/usr/bin/env python3
"""Unified dynamic-wind benchmark — Final working version."""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, positions_to_jax, inflow_speeds_jax, power_output_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT="checkpoints_3x3_nnx_jaxenv"; J=3; N=9; AB=10.0; OBS_DIM=144
TRAJ_LEN=200; N_TRAJ=200; T_STEP=10.0
pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

# Generate trajectories
np.random.seed(20260617)
P=np.zeros((N_TRAJ,TRAJ_LEN)); V=np.zeros((N_TRAJ,TRAJ_LEN))
P[:,0]=np.random.uniform(173,353,N_TRAJ); V[:,0]=np.random.uniform(6,16,N_TRAJ)
for t in range(1,TRAJ_LEN):
    P[:,t]=263+0.95*(P[:,t-1]-263)+2.0*np.random.normal(0,1,N_TRAJ)
    V[:,t]=11+0.95*(V[:,t-1]-11)+1.0*np.random.normal(0,1,N_TRAJ)
    P[:,t]=np.clip(P[:,t],173,353); V[:,t]=np.clip(V[:,t],6,16)
print(f"Trajectories: {N_TRAJ}×{TRAJ_LEN}")

# Core eval functions
@jax.jit
def eval_zero(phi,v):
    def body(c,t):
        return c,None
    tg,_=jax.lax.scan(body,jnp.array(0.0),jnp.arange(TRAJ_LEN))
    return tg

@jax.jit
def eval_drl(model,phi,v):
    k=jax.random.key(0)
    s,o=env_reset(k,pj,j=J,specific_wind_dir=phi[0],specific_wind_speed=v[0],randomize_wind=False,max_steps=TRAJ_LEN+10)
    def body(c,t):
        st,ob,tg,tt,pr,nc=c
        mean,_,_=model(ob.reshape(1,-1)); a=jnp.clip(mean.reshape(N),-AB,AB)
        ns,no,_,_=env_step(st,a,pj,max_steps=TRAJ_LEN+10)
        inf_z=inflow_speeds_jax(pj,phi[t],v[t],jnp.zeros(N))
        pwr_z=jnp.sum(power_output_jax(inf_z,jnp.zeros(N)))/1e6
        inf_y=inflow_speeds_jax(pj,phi[t],v[t],ns.gammas)
        pwr_y=jnp.sum(power_output_jax(inf_y,ns.gammas))/1e6
        gain=(pwr_y-pwr_z)/(pwr_z+1e-6)*100
        st=jnp.sum(jnp.abs(a))
        return (ns,no,tg+gain,tt+st,jnp.maximum(pr,st/T_STEP),nc+(gain<0).astype(jnp.int32)),None
    (_,_,tg,tt,pr,nc),_=jax.lax.scan(body,(s,o,jnp.array(0.0),jnp.array(0.0),jnp.array(0.0),jnp.array(0,dtype=jnp.int32)),jnp.arange(TRAJ_LEN))
    return tg,tt,pr,nc

def load(tag,n=5):
    ms=[]
    for s in range(n):
        cp=os.path.join(CKPT,f"policy_seed{s}_{tag}.pkl")
        if not os.path.exists(cp): continue
        m=ActorCritic(OBS_DIM,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
        with open(cp,'rb') as f: st=pickle.load(f); ms.append(nnx.merge(gd,st))
    return ms

controllers=[
    ("Zero yaw","sens_act10",True),  # tag=any, use_zero=True
    ("Static Config-E","sens_act10",False),
    ("Dynamic λ=0","dyn_lambda0_60M",False),
    ("Dynamic λ=5e-4","dyn_lambda5_60M",False),
    ("Dynamic λ=2e-3","dyn_lambda2_60M",False),
]

all_r=[]
for name,tag,use_zero in controllers:
    if use_zero:
        _=eval_zero(jnp.array(P[0]),jnp.array(V[0]))
        jax.block_until_ready(True)
        print(f"  {name}...",end=" ",flush=True); t0=time.time()
        G=[]; T=[]; Pk=[]; N=[]
        for ti in range(N_TRAJ):
            g=eval_zero(jnp.array(P[ti]),jnp.array(V[ti]))
            G.append(0.0); T.append(0.0); Pk.append(0.0); N.append(0.0)
    else:
        models=load(tag)
        if not models: print(f"  {name}: SKIP"); continue
        _=eval_drl(models[0],jnp.array(P[0]),jnp.array(V[0]))
        jax.block_until_ready(True)
        print(f"  {name}...",end=" ",flush=True); t0=time.time()
        G=[]; T=[]; Pk=[]; N=[]
        for ti in range(N_TRAJ):
            mi=ti%len(models)
            g,t,p,n=eval_drl(models[mi],jnp.array(P[ti]),jnp.array(V[ti]))
            G.append(float(g)); T.append(float(t)); Pk.append(float(p)); N.append(float(n)/TRAJ_LEN)
    
    r=dict(name=name,mean_gain=float(np.mean(G)),std_gain=float(np.std(G)),
        mean_travel=float(np.mean(T)),peak_rate=float(np.max(Pk)),
        neg_frac=float(np.mean(N)),gain_per_travel=float(np.mean(G)/(np.mean(T)+1e-6)))
    all_r.append(r)
    print(f"gain={r['mean_gain']:+.2f}% travel={r['mean_travel']:.0f}° peak={r['peak_rate']:.2f}°/s neg={r['neg_frac']*100:.0f}% ({time.time()-t0:.0f}s)")

with open("../results/unified_dynamic_final.json","w") as f: json.dump(all_r,f,indent=2)
print(f"\n{'Controller':25s} {'Gain%':>7s} {'Travel':>7s} {'Peak':>6s} {'Neg%':>5s}")
for r in all_r:
    print(f"{r['name']:25s} {r['mean_gain']:>+6.2f} {r['mean_travel']:>6.0f}° {r['peak_rate']:>5.2f} {r['neg_frac']*100:>4.0f}%")
