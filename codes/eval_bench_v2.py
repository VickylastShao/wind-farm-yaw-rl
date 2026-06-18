#!/usr/bin/env python3
"""Final unified benchmark v2 — simplified core metrics only."""
import os, sys, json, time, pickle, numpy as np
import jax, jax.numpy as jnp
from flax import nnx
from scipy.interpolate import NearestNDInterpolator
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, positions_to_jax, inflow_speeds_jax, power_output_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT="checkpoints_3x3_nnx_jaxenv"; J=3; N=9; AB=10.0; TL=200; NT=200; T_STEP=10.0
pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

def gen(rho,sigma,seed):
    np.random.seed(seed); P=np.zeros((NT,TL)); V=np.zeros((NT,TL))
    P[:,0]=np.random.uniform(173,353,NT); V[:,0]=np.random.uniform(6,16,NT)
    for t in range(1,TL):
        P[:,t]=263+rho*(P[:,t-1]-263)+sigma*np.random.normal(0,1,NT)
        V[:,t]=11+rho*(V[:,t-1]-11)+(sigma/2)*np.random.normal(0,1,NT)
        P[:,t]=np.clip(P[:,t],173,353); V[:,t]=np.clip(V[:,t],6,16)
    return P,V

PA,VA=gen(0.99,1.0,42); PB,VB=gen(0.95,2.0,42)
print(f"Trajectories: {NT}x{TL}")

exp=np.load('expert_datasets/slsqp_expert_3x3_seed20260606_n1000.npz')
interp=NearestNDInterpolator(np.column_stack([exp['phi'],exp['v']]),exp['slsqp_yaw'])

def load(tag,od=144,n=5):
    ms=[]
    for s in range(n):
        cp=os.path.join(CKPT,f"policy_seed{s}_{tag}.pkl")
        if not os.path.exists(cp): continue
        m=ActorCritic(od,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
        with open(cp,'rb') as f: st=pickle.load(f); ms.append(nnx.merge(gd,st))
    return ms

m_st=load('sens_act10',144,5); m_j15=load('dyn_J15_l5e4',720,2)
print(f"Loaded Static={len(m_st)}, J15={len(m_j15)}")

# Simple eval: no load proxy, just power+travel+peak+neg
@jax.jit
def ev_raw(model,phi,v):
    k=jax.random.key(0)
    s,o=env_reset(k,pj,j=J,specific_wind_dir=phi[0],specific_wind_speed=v[0],randomize_wind=False,max_steps=TL+10)
    def body(c,t):
        st,ob,tg,tt,pr,nc=c
        mean,_,_=model(ob.reshape(1,-1)); a=jnp.clip(mean.reshape(N),-AB,AB)
        ns,no,_,_=env_step(st,a,pj,max_steps=TL+10)
        i0=inflow_speeds_jax(pj,phi[t],v[t],jnp.zeros(N)); p0=jnp.sum(power_output_jax(i0,jnp.zeros(N)))/1e6
        iy=inflow_speeds_jax(pj,phi[t],v[t],ns.gammas); pw=jnp.sum(power_output_jax(iy,ns.gammas))/1e6
        g=(pw-p0)/(p0+1e-6)*100; st=jnp.sum(jnp.abs(a))
        return (ns,no,tg+g,tt+st,jnp.maximum(pr,st/T_STEP),nc+(g<0).astype(jnp.int32)),None
    (_,_,tg,tt,pr,nc),_=jax.lax.scan(body,(s,o,jnp.array(0.0),jnp.array(0.0),jnp.array(0.0),jnp.array(0,dtype=jnp.int32)),jnp.arange(TL))
    return tg,tt,pr,nc

@jax.jit
def ev_deploy(model,phi,v):
    k=jax.random.key(0); ga=jnp.array(False); py0=jnp.zeros(N)
    s,o=env_reset(k,pj,j=J,specific_wind_dir=phi[0],specific_wind_speed=v[0],randomize_wind=False,max_steps=TL+10)
    def body(c,t):
        st,ob,tg,tt,pr,nc,ga,py=c
        pt=phi[t]; vt=v[t]; dphi=jnp.minimum(jnp.abs(pt-270),360-jnp.abs(pt-270))
        th=jnp.where(ga,20.0,15.0); in_gate=(dphi<th)&(vt<11.4); ga_new=in_gate
        mean,_,_=model(ob.reshape(1,-1)); raw_a=jnp.clip(mean.reshape(N),-AB,AB)
        a=jnp.where(ga_new,raw_a,jnp.zeros(N))
        a=jnp.where(jnp.max(jnp.abs(a))<2.0,jnp.zeros(N),a)
        ms=3.0*T_STEP; a=jnp.clip(a,py-ms,py+ms)
        ns,no,_,_=env_step(st,a,pj,max_steps=TL+10)
        i0=inflow_speeds_jax(pj,pt,vt,jnp.zeros(N)); p0=jnp.sum(power_output_jax(i0,jnp.zeros(N)))/1e6
        iy=inflow_speeds_jax(pj,pt,vt,ns.gammas); pw=jnp.sum(power_output_jax(iy,ns.gammas))/1e6
        g=(pw-p0)/(p0+1e-6)*100; st=jnp.sum(jnp.abs(a-py))
        return (ns,no,tg+g,tt+st,jnp.maximum(pr,st/T_STEP),nc+(g<0).astype(jnp.int32),ga_new,a),None
    (_,_,tg,tt,pr,nc,_,_),_=jax.lax.scan(body,(s,o,jnp.array(0.0),jnp.array(0.0),jnp.array(0.0),jnp.array(0,dtype=jnp.int32),ga,py0),jnp.arange(TL))
    return tg,tt,pr,nc

def ev_slsqp(P,V,rm):
    G=[];T=[];Pk=[];N=[]
    for ti in range(NT):
        py=np.zeros(N);tg=0.0;tt=0.0;pr=0.0;nc=0
        for t in range(TL):
            try:opt=interp(P[ti,t],V[ti,t])
            except:opt=py
            ms=rm*T_STEP if rm else 50.0;dy=np.clip(opt-py,-ms,ms);yw=py+dy
            i0=np.array(inflow_speeds_jax(pj,jnp.array(float(P[ti,t])),jnp.array(float(V[ti,t])),jnp.zeros(N)))
            p0=np.sum(np.array(power_output_jax(jnp.array(i0),jnp.zeros(N))))/1e6
            iy=np.array(inflow_speeds_jax(pj,jnp.array([float(P[ti,t])]),jnp.array([float(V[ti,t])]),jnp.array(yw)))
            pw=np.sum(np.array(power_output_jax(jnp.array(iy),jnp.array(yw))))/1e6
            g=(pw-p0)/(p0+1e-6)*100;tg+=g;tt+=np.sum(np.abs(dy));pr=max(pr,np.sum(np.abs(dy))/T_STEP)
            if g<0:nc+=1; py=yw.copy()
        G.append(tg/TL);T.append(tt/N);Pk.append(pr);N.append(nc/TL)
    return np.mean(G),np.mean(T),np.max(Pk),np.mean(N)

def run(name,models,P,V,efn):
    print(f"  {name}...",end=" ",flush=True);t0=time.time()
    G=[];T=[];Pk=[];Ng=[]
    for ti in range(NT):
        mi=ti%len(models);g,t,p,n=efn(models[mi],jnp.array(P[ti]),jnp.array(V[ti]))
        G.append(float(g)/TL);T.append(float(t)/N);Pk.append(float(p));Ng.append(float(n)/TL)
    r=dict(name=name,gain=np.mean(G),std=np.std(G)/np.sqrt(NT),travel=np.mean(T),peak=np.max(Pk),neg=np.mean(Ng),gain_per_travel=np.mean(G)/(np.mean(T)+1e-6))
    print(f"gain={r['gain']:+.3f} travel={r['travel']:.0f} neg={r['neg']*100:.0f}% ({time.time()-t0:.0f}s)")
    return r

all_r={'protocol_A':[],'protocol_B':[]}
for pname,P,V in [('protocol_A',PA,VA),('protocol_B',PB,VB)]:
    print(f'\n=== {pname} ===')
    res=[]
    res.append(dict(name='Zero yaw',gain=0,std=0,travel=0,peak=0,neg=0,gain_per_travel=0))
    res.append(run('Static (raw)',m_st,P,V,ev_raw))
    res.append(run('Static + Deploy',m_st,P,V,ev_deploy))
    sg,st,sp,sn=ev_slsqp(P,V,None); res.append(dict(name='SLSQP Unlimited',gain=sg,travel=st,peak=sp,neg=sn,gain_per_travel=sg/(st+1e-6)))
    sg,st,sp,sn=ev_slsqp(P,V,0.5); res.append(dict(name='SLSQP RL=0.5/s',gain=sg,travel=st,peak=sp,neg=sn,gain_per_travel=sg/(st+1e-6)))
    res.append(run('J15 Dyn + Deploy',m_j15,P,V,ev_deploy))
    all_r[pname]=res

    print(f'\n{"Controller":25s} {"Gain":>8s} {"Travel":>7s} {"Peak":>6s} {"Neg%":>5s} {"Gain/Tr":>7s}')
    for r in res:
        print(f'{r["name"]:25s} {r["gain"]:>+7.3f} {r["travel"]:>6.0f}° {r["peak"]:>5.2f} {r["neg"]*100:>4.0f}% {r["gain_per_travel"]:>+6.3f}')

with open('../results/unified_bench_v2.json','w') as f: json.dump(all_r,f,indent=2)
print('\nSaved.')
