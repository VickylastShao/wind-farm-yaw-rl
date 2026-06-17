#!/usr/bin/env python3
"""J=15 + DRL-Deploy evaluation (separate J=15 env)."""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, positions_to_jax, inflow_speeds_jax, power_output_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT="checkpoints_3x3_nnx_jaxenv"; N=9; AB=10.0; TL=200; NT=200; T_STEP=10.0
J15=15; OD15=720
pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

np.random.seed(20260617)
P=np.zeros((NT,TL)); V=np.zeros((NT,TL)); P[:,0]=np.random.uniform(173,353,NT); V[:,0]=np.random.uniform(6,16,NT)
for t in range(1,TL):
    P[:,t]=263+0.95*(P[:,t-1]-263)+2.0*np.random.normal(0,1,NT); V[:,t]=np.clip(11+0.95*(V[:,t-1]-11)+1.0*np.random.normal(0,1,NT),6,16)
    P[:,t]=np.clip(P[:,t],173,353)

def load(tag,odim,n=2):
    ms=[]
    for s in range(n):
        cp=os.path.join(CKPT,f'policy_seed{s}_{tag}.pkl')
        if not os.path.exists(cp): continue
        m=ActorCritic(odim,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
        with open(cp,'rb') as f: st=pickle.load(f); ms.append(nnx.merge(gd,st))
    return ms

@jax.jit
def eval_j15_deploy(model,phi,v):
    """J=15 + gate + hysteresis + 2deg deadband + 3deg/s rate limit."""
    k=jax.random.key(0); ga=jnp.array(False); py0=jnp.zeros(N)
    s,o=env_reset(k,pj,j=J15,specific_wind_dir=phi[0],specific_wind_speed=v[0],randomize_wind=False,max_steps=TL+10)
    def body(c,t):
        st,ob,tg,tt,pr,nc,ga,py=c; pt=phi[t]; vt=v[t]
        dphi=jnp.minimum(jnp.abs(pt-270),360-jnp.abs(pt-270)); th=jnp.where(ga,20.0,15.0)
        in_gate=(dphi<th)&(vt<11.4); ga_new=jnp.where(True,in_gate,False)
        mean,_,_=model(ob.reshape(1,-1)); raw_a=jnp.clip(mean.reshape(N),-AB,AB)
        a=jnp.where(ga_new,raw_a,jnp.zeros(N))
        a=jnp.where((2.0>0.5)&(jnp.max(jnp.abs(a))<2.0),jnp.zeros(N),a)
        ms=3.0*T_STEP; a=jnp.clip(a,py-ms,py+ms)
        ns,no,_,_=env_step(st,a,pj,max_steps=TL+10)
        i0=inflow_speeds_jax(pj,pt,vt,jnp.zeros(N)); p0=jnp.sum(power_output_jax(i0,jnp.zeros(N)))/1e6
        iy=inflow_speeds_jax(pj,pt,vt,ns.gammas); pw=jnp.sum(power_output_jax(iy,ns.gammas))/1e6
        g=(pw-p0)/(p0+1e-6)*100; st=jnp.sum(jnp.abs(a-py))
        return (ns,no,tg+g,tt+st,jnp.maximum(pr,st/T_STEP),nc+(g<0).astype(jnp.int32),ga_new,a),None
    (_,_,tg,tt,pr,nc,_,_),_=jax.lax.scan(body,(s,o,jnp.array(0.0),jnp.array(0.0),jnp.array(0.0),jnp.array(0,dtype=jnp.int32),ga,py0),jnp.arange(TL))
    return tg,tt,pr,nc

ms=load('dyn_J15_l5e4',OD15,2)
if not ms: print('J=15 models not found'); sys.exit(1)
_=eval_j15_deploy(ms[0],jnp.array(P[0]),jnp.array(V[0])); jax.block_until_ready(True)
print('J=15+Deploy compiled. Evaluating...')
t0=time.time(); G=[];T_=[];Pk=[];Ng=[]
for ti in range(NT):
    mi=ti%len(ms); g,t,p,n=eval_j15_deploy(ms[mi],jnp.array(P[ti]),jnp.array(V[ti]))
    G.append(float(g)/TL);T_.append(float(t)/N);Pk.append(float(p));Ng.append(float(n)/TL)
r=dict(name='J=15 Dyn + DRL-Deploy',gain=np.mean(G),travel=np.mean(T_),peak=np.max(Pk),neg=np.mean(Ng))
print(f'  {r["name"]}: gain={r["gain"]:+.3f}/step travel={r["travel"]:.0f}/turb peak={r["peak"]:.2f}/s neg={r["neg"]*100:.0f}% ({time.time()-t0:.0f}s)')
with open('../results/j15_deploy_result.json','w') as f: json.dump(r,f,indent=2)
print('Saved.')
