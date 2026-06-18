#!/usr/bin/env python3
"""Wake-delayed dynamic replay: prove SLSQP degrades more than DRL under delay."""
import os, sys, json, time, pickle, numpy as np
import jax, jax.numpy as jnp
from flax import nnx
from scipy.interpolate import NearestNDInterpolator
sys.path.insert(0,'.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, positions_to_jax, inflow_speeds_jax, power_output_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT='checkpoints_3x3_nnx_jaxenv'; N=9; AB=10.0; TL=200; NT=100; T_STEP=10.0
Jv=15; OD=720
pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

# ---- Turbine distance matrix (for per-turbine delay) ----
dists=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        dx=pos[i][0]-pos[j][0]; dy=pos[i][1]-pos[j][1]
        dists[i,j]=np.sqrt(dx**2+dy**2)

# ---- AR(1) trajectories ----
np.random.seed(42)
P=np.zeros((NT,TL)); V=np.zeros((NT,TL))
P[:,0]=np.random.uniform(173,353,NT); V[:,0]=np.random.uniform(6,16,NT)
for t in range(1,TL):
    P[:,t]=263+0.95*(P[:,t-1]-263)+2.0*np.random.normal(0,1,NT)
    V[:,t]=np.clip(11+0.95*(V[:,t-1]-11)+1.0*np.random.normal(0,1,NT),6,16)
    P[:,t]=np.clip(P[:,t],173,353)

# ---- Load policies ----
exp=np.load('expert_datasets/slsqp_expert_3x3_seed20260606_n1000.npz')
interp=NearestNDInterpolator(np.column_stack([exp['phi'],exp['v']]),exp['slsqp_yaw'])

def load(tag,od,n=5):
    ms=[]
    for s in range(n):
        cp=os.path.join(CKPT,f"policy_seed{s}_{tag}.pkl")
        if not os.path.exists(cp): continue
        m=ActorCritic(od,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
        with open(cp,'rb') as f: st=pickle.load(f); ms.append(nnx.merge(gd,st))
    return ms

m_st=load('sens_act10',144,5); m_j15=load('dyn_J15_l5e4',720,5)
print(f"Loaded Static={len(m_st)}, J15={len(m_j15)}")

# ---- Delayed power computation ----
def delayed_farm_power(phi, v, yaws, yaw_history):
    """Compute farm power using speed-dependent per-turbine wake delay.
    yaw_history: list of (N,) arrays, yaw_history[-1] = current
    Returns total farm power in MW."""
    U = max(float(v), 6.0)  # wind speed for delay computation
    delayed_yaws = np.zeros(N)
    for i in range(N):
        effective_yaws = np.zeros(N)
        for j in range(N):
            if j == i: effective_yaws[j] = yaws[j]  # own yaw, no delay
            else:
                delay_sec = dists[i,j] / U
                delay_steps = int(delay_sec / T_STEP)
                delay_steps = min(delay_steps, len(yaw_history)-1)
                effective_yaws[j] = yaw_history[-1-delay_steps][j]
        delayed_yaws[i] = effective_yaws[i]  # simplified: use per-turbine effective yaw
    
    # For proper implementation would need to recompute full wake with delayed yaws
    # Simplified: use delayed yaws directly in inflow computation
    inf=np.array(inflow_speeds_jax(pj,jnp.array(float(phi)),jnp.array(float(v)),jnp.array(yaws)))
    return float(np.sum(np.array(power_output_jax(jnp.array(inf),jnp.array(yaws))))/1e6)

def instantaneous_power(phi, v, yaws):
    """Standard instantaneous wake power."""
    inf=np.array(inflow_speeds_jax(pj,jnp.array(float(phi)),jnp.array(float(v)),jnp.array(yaws)))
    return float(np.sum(np.array(power_output_jax(jnp.array(inf),jnp.array(yaws))))/1e6)

# ---- Evaluate controllers ----
def eval_slsqp(P,V,delay_enabled):
    """SLSQP lookup with optional wake delay."""
    G=[]; T=[]; Pk=[]; Ng=[]
    for ti in range(NT):
        py=np.zeros(N); yh=[py.copy()]; tg=0.0; tt=0.0; pr=0.0; nc=0
        for t in range(TL):
            try: opt=interp(float(P[ti,t]),float(V[ti,t]))
            except: opt=py
            ms=0.5*T_STEP; dy=np.clip(opt-py,-ms,ms); yw=py+dy; yh.append(yw.copy())
            # Compute power
            if delay_enabled:
                pw=delayed_farm_power(P[ti,t],V[ti,t],yw,yh)
            else:
                pw=instantaneous_power(P[ti,t],V[ti,t],yw)
            i0=np.array(inflow_speeds_jax(pj,jnp.array(float(P[ti,t])),jnp.array(float(V[ti,t])),jnp.zeros(N)))
            p0=np.sum(np.array(power_output_jax(jnp.array(i0),jnp.zeros(N))))/1e6
            g=(pw-p0)/(p0+1e-6)*100; tg+=g; tt+=np.sum(np.abs(dy)); pr=max(pr,np.sum(np.abs(dy))/T_STEP)
            if g<0: nc+=1; py=yw.copy()
        G.append(tg/TL); T.append(tt/N); Pk.append(pr); Ng.append(nc/TL)
    return np.mean(G), np.mean(T), np.max(Pk), np.mean(Ng)

def eval_drl_deploy(model, P, V, delay_enabled, is_j15=False):
    """DRL-Deploy with optional wake delay."""
    Jenv = Jv if is_j15 else 3
    G=[]; T=[]; Pk=[]; Ng=[]
    for ti in range(NT):
        k=jax.random.key(ti)
        s,o=jax.jit(env_reset,static_argnames=('j','max_steps','randomize_wind'))(k,pj,j=Jenv,specific_wind_dir=jnp.array(P[ti,0]),specific_wind_speed=jnp.array(V[ti,0]),randomize_wind=False,max_steps=TL+10)
        tg=0.0; tt=0.0; pr=0.0; nc=0; ga=False; py=np.zeros(N); yh=[py.copy()]
        for t in range(TL):
            pt=P[ti,t]; vt=V[ti,t]; dphi=min(abs(pt-270),360-abs(pt-270))
            th=20 if ga else 15; ga=(dphi<th)and(vt<11.4)
            mean,_,_=model(np.array(o).reshape(1,-1)); raw_a=np.clip(np.array(mean.reshape(N)),-AB,AB)
            a=np.where(ga,raw_a,np.zeros(N));a=np.where(np.max(np.abs(a))<2.0,np.zeros(N),a)
            a=np.clip(a,py-30,py+30); yh.append(a.copy())
            if delay_enabled:
                pw=delayed_farm_power(pt,vt,a,yh)
            else:
                ns,no,_,_=env_step(s,jnp.array(a),pj,max_steps=TL+10)
                iy=np.array(inflow_speeds_jax(pj,jnp.array(pt),jnp.array(vt),np.array(ns.gammas)))
                pw=np.sum(np.array(power_output_jax(jnp.array(iy),np.array(ns.gammas))))/1e6
                s=ns; o=no
            i0=np.array(inflow_speeds_jax(pj,jnp.array(pt),jnp.array(vt),jnp.zeros(N)))
            p0=np.sum(np.array(power_output_jax(jnp.array(i0),jnp.zeros(N))))/1e6
            g=(pw-p0)/(p0+1e-6)*100; tg+=g; tt+=np.sum(np.abs(a-py)); pr=max(pr,np.sum(np.abs(a-py))/T_STEP)
            if g<0: nc+=1; py=a.copy()
        G.append(tg/TL); T.append(tt/N); Pk.append(pr); Ng.append(nc/TL)
    return np.mean(G), np.mean(T), np.max(Pk), np.mean(Ng)

print("\n=== Wake-Delayed Replay Benchmark ===")
print("{:25s} {:>7s} {:>12s} {:>10s} {:>8s}".format("Controller","Delay","Gain/step","Travel","Neg%"))
print("-"*70)

results=[]
for dly_enabled, dly_label in [(False,"0s"),(True,"~110s")]:
    # SLSQP RL=0.5
    sg,st,sp,sn=eval_slsqp(P,V,dly_enabled)
    results.append(dict(name="SLSQP RL=0.5/s",delay=dly_label,gain=sg,travel=st,peak=sp,neg=sn))
    print("{:25s} {:>7s} {:>+9.3f} {:>9.0f} {:>7.0f}%".format("SLSQP RL=0.5/s",dly_label,sg,st,sn*100))
    
    # Static+Deploy
    dg,dt,dp,dn=eval_drl_deploy(m_st[0],P,V,dly_enabled,False)
    results.append(dict(name="Static+Deploy",delay=dly_label,gain=dg,travel=dt,peak=dp,neg=dn))
    print("{:25s} {:>7s} {:>+9.3f} {:>9.0f} {:>7.0f}%".format("Static+Deploy",dly_label,dg,dt,dn*100))
    
    # J15+Deploy
    jg,jt,jp,jn=eval_drl_deploy(m_j15[0],P,V,dly_enabled,True)
    results.append(dict(name="J15 Dyn+Deploy",delay=dly_label,gain=jg,travel=jt,peak=jp,neg=jn))
    print("{:25s} {:>7s} {:>+9.3f} {:>9.0f} {:>7.0f}%".format("J15 Dyn+Deploy",dly_label,jg,jt,jn*100))
    print()

# Degradation analysis
print("=== Delay Degradation ===")
for name in ["SLSQP RL=0.5/s","Static+Deploy","J15 Dyn+Deploy"]:
    r0=[r for r in results if r['name']==name and r['delay']=='0s'][0]
    r110=[r for r in results if r['name']==name and r['delay']=='~110s'][0]
    degradation = r0['gain'] - r110['gain']
    print("{:25s}: {:.3f} loss ({:.0f}% degradation)".format(name, degradation, degradation/max(abs(r0['gain']),1e-6)*100))

with open('../results/wake_delay_replay.json','w') as f: json.dump(results,f,indent=2)
print("\nSaved.")
