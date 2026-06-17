#!/usr/bin/env python3
"""Unified dynamic-wind benchmark V2: correct per-step percentage gain."""
import os, sys, json, time, pickle
import jax, jax.numpy as jnp, numpy as np
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, positions_to_jax,
    inflow_speeds_jax, power_output_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT="checkpoints_3x3_nnx_jaxenv"; RESULTS="../results"; os.makedirs(RESULTS,exist_ok=True)
J=3; N=9; AB=10.0; OBS_DIM=3*(5*N+3); T_STEP=10.0
N_TRAJ=200; TRAJ_LEN=200
AP=0.95; SP=2.0; AV=0.95; SV=1.0

pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

np.random.seed(20260617)
P=np.zeros((N_TRAJ,TRAJ_LEN)); V=np.zeros((N_TRAJ,TRAJ_LEN))
P[:,0]=np.random.uniform(173,353,N_TRAJ); V[:,0]=np.random.uniform(6,16,N_TRAJ)
for t in range(1,TRAJ_LEN):
    P[:,t]=263+AP*(P[:,t-1]-263)+SP*np.random.normal(0,1,N_TRAJ)
    V[:,t]=11+AV*(V[:,t-1]-11)+SV*np.random.normal(0,1,N_TRAJ)
    P[:,t]=np.clip(P[:,t],173,353); V[:,t]=np.clip(V[:,t],6,16)
print(f"Trajectories: {N_TRAJ}×{TRAJ_LEN}")

@jax.jit
def eval_zero(phi,v):
    def body(c,t):
        tg=c
        inf=inflow_speeds_jax(pj,phi[t],v[t],jnp.zeros(N))
        pwr=jnp.sum(power_output_jax(inf,jnp.zeros(N)))/1e6
        return tg,None
    tg,_=jax.lax.scan(body,jnp.array(0.0),jnp.arange(TRAJ_LEN))
    return tg,jnp.array(0.0),jnp.array(0.0)

@jax.jit
def eval_drl(model,phi,v,gate_on,gate_in,gate_out,deadband):
    k=jax.random.key(0)
    s,o=env_reset(k,pj,j=J,specific_wind_dir=phi[0],specific_wind_speed=v[0],randomize_wind=False,max_steps=TRAJ_LEN+10)
    ga=jnp.array(False)
    def body(c,t):
        st,ob,tg,tt,pr,nc,ga,ag_cnt=c
        pt=phi[t]; vt=v[t]
        dphi=jnp.minimum(jnp.abs(pt-270),360-jnp.abs(pt-270))
        in_regime=(dphi<gate_in)&(vt<11.4)
        threshold=jnp.where(ga,gate_out,gate_in)
        in_gate=jnp.where(gate_on,(dphi<threshold)&(vt<11.4),True)
        ga_new=jnp.where(gate_on,in_gate,True)
        mean,_,_=model(ob.reshape(1,-1)); raw=jnp.clip(mean.reshape(N),-AB,AB)
        a=jnp.where(ga_new,raw,jnp.zeros(N))
        a=jnp.where((deadband>0.5)&(jnp.max(jnp.abs(a))<deadband),jnp.zeros(N),a)
        ns,no,_,_=env_step(st,a,pj,max_steps=TRAJ_LEN+10)
        # Per-step zero-yaw baseline
        inf_z=inflow_speeds_jax(pj,pt,vt,jnp.zeros(N))
        pwr_z=jnp.sum(power_output_jax(inf_z,jnp.zeros(N)))/1e6
        inf_y=inflow_speeds_jax(pj,pt,vt,ns.gammas)
        pwr_y=jnp.sum(power_output_jax(inf_y,ns.gammas))/1e6
        gain=(pwr_y-pwr_z)/(pwr_z+1e-6)*100
        st=jnp.sum(jnp.abs(a))
        al_g=jnp.where(in_regime,gain,0.0); al_n=in_regime.astype(jnp.int32)
        return (ns,no,tg+gain,tt+st,jnp.maximum(pr,st/T_STEP),nc+(gain<0).astype(jnp.int32),ga_new,ag_cnt+al_g+al_n*0),None
    init=(s,o,jnp.array(0.0),jnp.array(0.0),jnp.array(0.0),jnp.array(0,dtype=jnp.int32),ga,jnp.array(0.0))
    (_,_,tg,tt,pr,nc,_,ag_cnt),_=jax.lax.scan(body,init,jnp.arange(TRAJ_LEN))
    return tg,tt,pr,nc,ag_cnt

def load(tag,n=5):
    ms=[]; 
    for s in range(n):
        cp=os.path.join(CKPT,f"policy_seed{s}_{tag}.pkl")
        if not os.path.exists(cp): continue
        m=ActorCritic(OBS_DIM,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
        with open(cp,'rb') as f: st=pickle.load(f); ms.append(nnx.merge(gd,st))
    return ms

controllers=[
    ("Zero yaw",None,None),
    ("Static Config-E","sens_act10",(False,15,15,0)),
    ("Static + Gate","sens_act10",(True,15,15,0)),
    ("Static + Hyst Gate","sens_act10",(True,15,20,0)),
    ("Static + Hyst + 2dB","sens_act10",(True,15,20,2)),
    ("Dynamic λ=0","dyn_lambda0_60M",(False,15,15,0)),
    ("Dynamic λ=5e-4","dyn_lambda5_60M",(False,15,15,0)),
    ("Dynamic λ=5e-4 + Gate + 2dB","dyn_lambda5_60M",(True,15,20,2)),
]

print("\n=== Unified Dynamic-Wind Benchmark V2 ===")
all_r=[]
for name,tag,gate in controllers:
    models=load(tag) if tag else None
    if tag and not models: print(f"  {name}: SKIP"); continue
    
    # Compile
    if name=="Zero yaw":
        _=eval_zero(jnp.array(P[0]),jnp.array(V[0]))
        jax.block_until_ready(True)
    else:
        go,gi,gg,db=gate; m=models[0]
        _=eval_drl(m,jnp.array(P[0]),jnp.array(V[0]),jnp.array(go),jnp.array(float(gi)),jnp.array(float(gg)),jnp.array(float(db)))
        jax.block_until_ready(True)
    
    print(f"  {name}...",end=" ",flush=True); t0=time.time()
    G=[]; T=[]; Pk=[]; N=[]; AG=[]; AN=[]
    for ti in range(N_TRAJ):
        if name=="Zero yaw":
            g,t,p=eval_zero(jnp.array(P[ti]),jnp.array(V[ti])); n=0
        else:
            mi=ti%len(models); go,gi,gg,db=gate
            g,t,p,n,ac=eval_drl(models[mi],jnp.array(P[ti]),jnp.array(V[ti]),jnp.array(go),jnp.array(float(gi)),jnp.array(float(gg)),jnp.array(float(db)))
        G.append(float(g)); T.append(float(t)); Pk.append(float(p)); N.append(float(n)/TRAJ_LEN)
    r=dict(name=name,mean_gain=float(np.mean(G)),std_gain=float(np.std(G)),
        mean_travel=float(np.mean(T)),peak_rate=float(np.max(Pk)),
        neg_frac=float(np.mean(N)),gain_per_travel=float(np.mean(G)/(np.mean(T)+1e-6)))
    all_r.append(r)
    print(f"gain={r['mean_gain']:+.2f}% travel={r['mean_travel']:.0f}° peak={r['peak_rate']:.2f}°/s neg={r['neg_frac']*100:.0f}% ({time.time()-t0:.0f}s)")

with open(f"{RESULTS}/unified_dynamic_benchmark_v2.json","w") as f: json.dump(all_r,f,indent=2)
print(f"\n{'Controller':30s} {'Gain%':>7s} {'Travel':>7s} {'Peak':>6s} {'Neg%':>5s} {'G/Tr':>6s}")
for r in all_r:
    print(f"{r['name']:30s} {r['mean_gain']:>+6.2f} {r['mean_travel']:>6.0f}° {r['peak_rate']:>5.2f} {r['neg_frac']*100:>4.0f}% {r['gain_per_travel']:>+5.3f}")
