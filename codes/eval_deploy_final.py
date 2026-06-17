#!/usr/bin/env python3
"""Final DRL-Deploy controller evaluation — all baselines, unified benchmark."""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, positions_to_jax, inflow_speeds_jax, power_output_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT="checkpoints_3x3_nnx_jaxenv"; J=3; N=9; AB=10.0; OBS_DIM=144
TL=200; NT=200; T_STEP=10.0
pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

np.random.seed(20260617)
P=np.zeros((NT,TL)); V=np.zeros((NT,TL)); P[:,0]=np.random.uniform(173,353,NT); V[:,0]=np.random.uniform(6,16,NT)
for t in range(1,TL):
    P[:,t]=263+0.95*(P[:,t-1]-263)+2.0*np.random.normal(0,1,NT); V[:,t]=np.clip(11+0.95*(V[:,t-1]-11)+1.0*np.random.normal(0,1,NT),6,16)
    P[:,t]=np.clip(P[:,t],173,353)

def load(tag,odim=144,n=5):
    ms=[]
    for s in range(n):
        cp=os.path.join(CKPT,f'policy_seed{s}_{tag}.pkl')
        if not os.path.exists(cp): continue
        m=ActorCritic(odim,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
        with open(cp,'rb') as f: st=pickle.load(f); ms.append(nnx.merge(gd,st))
    return ms

# DRL-Deploy: static-trained policy + gate + hysteresis + 2° deadband + rate limit + fallback
@jax.jit
def eval_deploy(model,phi,v,gate_in=15.0,gate_out=20.0,deadband=2.0,rate_max=3.0):
    """DRL-Deploy: gate + hysteresis + deadband + rate limit + zero fallback."""
    k=jax.random.key(0)
    s,o=env_reset(k,pj,j=J,specific_wind_dir=phi[0],specific_wind_speed=v[0],randomize_wind=False,max_steps=TL+10)
    ga=jnp.array(False); prev_yaw=jnp.zeros(N)
    def body(c,t):
        st,ob,tg,tt,pr,nc,ga,py=c
        pt=phi[t]; vt=v[t]; dphi=jnp.minimum(jnp.abs(pt-270),360-jnp.abs(pt-270))
        threshold=jnp.where(ga,gate_out,gate_in); in_gate=(dphi<threshold)&(vt<11.4)
        ga_new=jnp.where(True,in_gate,False)  # gate always on evaluation logic
        # DRL action
        mean,_,_=model(ob.reshape(1,-1)); raw_a=jnp.clip(mean.reshape(N),-AB,AB)
        # Gate: zero outside
        a=jnp.where(ga_new,raw_a,jnp.zeros(N))
        # Deadband: suppress if max|a| < deadband
        a=jnp.where((deadband>0.5)&(jnp.max(jnp.abs(a))<deadband),jnp.zeros(N),a)
        # Rate limit
        max_step=rate_max*T_STEP
        a=jnp.clip(a,py-max_step,py+max_step)
        ns,no,_,_=env_step(st,a,pj,max_steps=TL+10)
        # Per-step zero-yaw baseline power
        i0=inflow_speeds_jax(pj,pt,vt,jnp.zeros(N)); p0=jnp.sum(power_output_jax(i0,jnp.zeros(N)))/1e6
        iy=inflow_speeds_jax(pj,pt,vt,ns.gammas); pwr=jnp.sum(power_output_jax(iy,ns.gammas))/1e6
        g=(pwr-p0)/(p0+1e-6)*100; st=jnp.sum(jnp.abs(a-py))
        return (ns,no,tg+g,tt+st,jnp.maximum(pr,st/T_STEP),nc+(g<0).astype(jnp.int32),ga_new,a),None
    (_,_,tg,tt,pr,nc,_,_),_=jax.lax.scan(body,(s,o,jnp.array(0.0),jnp.array(0.0),jnp.array(0.0),jnp.array(0,dtype=jnp.int32),ga,prev_yaw),jnp.arange(TL))
    return tg,tt,pr,nc

# Standard eval (no wrapper)
@jax.jit
def eval_raw(model,phi,v):
    k=jax.random.key(0)
    s,o=env_reset(k,pj,j=J,specific_wind_dir=phi[0],specific_wind_speed=v[0],randomize_wind=False,max_steps=TL+10)
    def body(c,t):
        st,ob,tg,tt,pr,nc=c
        mean,_,_=model(ob.reshape(1,-1)); a=jnp.clip(mean.reshape(N),-AB,AB)
        ns,no,_,_=env_step(st,a,pj,max_steps=TL+10)
        i0=inflow_speeds_jax(pj,phi[t],v[t],jnp.zeros(N)); p0=jnp.sum(power_output_jax(i0,jnp.zeros(N)))/1e6
        iy=inflow_speeds_jax(pj,phi[t],v[t],ns.gammas); pwr=jnp.sum(power_output_jax(iy,ns.gammas))/1e6
        g=(pwr-p0)/(p0+1e-6)*100; st=jnp.sum(jnp.abs(a))
        return (ns,no,tg+g,tt+st,jnp.maximum(pr,st/T_STEP),nc+(g<0).astype(jnp.int32)),None
    (_,_,tg,tt,pr,nc),_=jax.lax.scan(body,(s,o,jnp.array(0.0),jnp.array(0.0),jnp.array(0.0),jnp.array(0,dtype=jnp.int32)),jnp.arange(TL))
    return tg,tt,pr,nc

print('=== DRL-Deploy Final Evaluation ===')
print(f'{"Controller":30s} {"Gain/step":>10s} {"Travel/turb":>10s} {"Peak":>6s} {"Neg%":>5s}')
print('-'*70)

all_results=[]
# Static Config-E raw
ms=load('sens_act10')
_=eval_raw(ms[0],jnp.array(P[0]),jnp.array(V[0])); jax.block_until_ready(True)
G=[];T=[];Pk=[];Ng=[]
for ti in range(NT): mi=ti%len(ms); g,t,p,n=eval_raw(ms[mi],jnp.array(P[ti]),jnp.array(V[ti])); G.append(float(g)/TL);T.append(float(t)/N);Pk.append(float(p));Ng.append(float(n)/TL)
r=dict(name='Static Config-E (raw)',gain=np.mean(G),travel=np.mean(T),peak=np.max(Pk),neg=np.mean(Ng))
all_results.append(r); print(f'{r["name"]:30s} {r["gain"]:>+9.3f} {r["travel"]:>9.0f} {r["peak"]:>5.2f} {r["neg"]*100:>4.0f}%')

# DRL-Deploy (Static + gate + hysteresis + 2° deadband + 3°/s rate limit)
_=eval_deploy(ms[0],jnp.array(P[0]),jnp.array(V[0])); jax.block_until_ready(True)
G=[];T=[];Pk=[];Ng=[]
for ti in range(NT): mi=ti%len(ms); g,t,p,n=eval_deploy(ms[mi],jnp.array(P[ti]),jnp.array(V[ti])); G.append(float(g)/TL);T.append(float(t)/N);Pk.append(float(p));Ng.append(float(n)/TL)
r=dict(name='DRL-Deploy (gate+hyst+2dB+RL)',gain=np.mean(G),travel=np.mean(T),peak=np.max(Pk),neg=np.mean(Ng))
all_results.append(r); print(f'{r["name"]:30s} {r["gain"]:>+9.3f} {r["travel"]:>9.0f} {r["peak"]:>5.2f} {r["neg"]*100:>4.0f}%')

# J=15 + Deploy
ms15=load('dyn_J15_l5e4',720,2)
_=eval_deploy(ms15[0],jnp.array(P[0]),jnp.array(V[0])); jax.block_until_ready(True)
G=[];T=[];Pk=[];Ng=[]
for ti in range(NT): mi=ti%len(ms15); g,t,p,n=eval_deploy(ms15[mi],jnp.array(P[ti]),jnp.array(V[ti])); G.append(float(g)/TL);T.append(float(t)/N);Pk.append(float(p));Ng.append(float(n)/TL)
r=dict(name='J=15 Dyn + Deploy',gain=np.mean(G),travel=np.mean(T),peak=np.max(Pk),neg=np.mean(Ng))
all_results.append(r); print(f'{r["name"]:30s} {r["gain"]:>+9.3f} {r["travel"]:>9.0f} {r["peak"]:>5.2f} {r["neg"]*100:>4.0f}%')

# No-oracle marginal + Deploy
ms_marg=load('marginal_reward',144,3)
_=eval_deploy(ms_marg[0],jnp.array(P[0]),jnp.array(V[0])); jax.block_until_ready(True)
G=[];T=[];Pk=[];Ng=[]
for ti in range(NT): mi=ti%len(ms_marg); g,t,p,n=eval_deploy(ms_marg[mi],jnp.array(P[ti]),jnp.array(V[ti])); G.append(float(g)/TL);T.append(float(t)/N);Pk.append(float(p));Ng.append(float(n)/TL)
r=dict(name='No-oracle + Deploy',gain=np.mean(G),travel=np.mean(T),peak=np.max(Pk),neg=np.mean(Ng))
all_results.append(r); print(f'{r["name"]:30s} {r["gain"]:>+9.3f} {r["travel"]:>9.0f} {r["peak"]:>5.2f} {r["neg"]*100:>4.0f}%')

with open('../results/deploy_final_comparison.json','w') as f: json.dump(all_results,f,indent=2)
print(f'\\n✅ DRL-Deploy evaluation complete. Saved.')
