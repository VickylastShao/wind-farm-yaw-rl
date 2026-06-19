#!/usr/bin/env python3
"""Delay sensitivity: SLSQP+Deploy vs DRL-Deploy under 0/50/100/150s wake delay."""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['USE_POSITIONS']='1'
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import *
from windfarm_env import create_wind_farm_layout_3x3, U_infinity, calculate_inflow_speeds

CKPT='checkpoints_3x3_nnx_jaxenv'; N=9; AB=10.0; T_STEP=10.0
GATE_IN=15.0; GATE_OUT=20.0; DEADBAND=2.0; RATE_MAX=0.3
N_TRAJ=100; N_STEPS=200  # reduced for speed across 4 delays × 3 configs

pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)
base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base,'latex_draft/figures/lookup_table_baseline.json')) as f: lt=json.load(f)
pg=np.array(lt['phi_grid'],dtype=np.float32); vg=np.array(lt['v_grid'],dtype=np.float32)
yt=np.load(os.path.join(base,'latex_draft/figures/lookup_table_yaw.npy')).astype(np.float32)

# Load DRL models (3 seeds for J=15)
models_j15=[]
for s in range(min(3,5)):
    m=ActorCritic(720,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
    cp=os.path.join(CKPT,f'policy_seed{s}_dyn_J15_l5e4.pkl')
    if os.path.exists(cp):
        with open(cp,'rb') as f: models_j15.append(nnx.merge(gd,pickle.load(f)))
models_s=[]
for s in range(min(3,5)):
    m=ActorCritic(144,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
    cp=os.path.join(CKPT,f'policy_seed{s}_sens_act10.pkl')
    if os.path.exists(cp):
        with open(cp,'rb') as f: models_s.append(nnx.merge(gd,pickle.load(f)))
print(f'Loaded: {len(models_j15)} J=15 seeds, {len(models_s)} Static seeds')

# Generate trajectories (Proto B, fixed seed for reproducibility)
np.random.seed(20260619)
P=np.zeros((N_TRAJ,N_STEPS)); V=np.zeros((N_TRAJ,N_STEPS))
P[:,0]=np.random.normal(270,20,N_TRAJ)%360
V[:,0]=np.clip(np.random.normal(9.4,2,N_TRAJ),6,16)
for t in range(1,N_STEPS):
    dp=(P[:,t-1]-270+180)%360-180
    dp=0.95*dp+2.0*np.random.normal(0,1,N_TRAJ)
    P[:,t]=(270+dp)%360
    V[:,t]=np.clip(9.4+0.95*(V[:,t-1]-9.4)+1.0*np.random.normal(0,1,N_TRAJ),6,16)

def eval_slsqp_deploy_delayed(phi_traj,v_traj,delay_steps):
    cy=np.zeros(N); ig=False; tg=0.0; tt=0.0; ng=0
    buf_len=max(1,delay_steps); yh=np.zeros((buf_len,N))
    for t in range(len(phi_traj)):
        pt=float(phi_traj[t]); vt=float(v_traj[t])
        dp=min(abs(pt-270),360-abs(pt-270)); th=GATE_OUT if ig else GATE_IN
        ig=(dp<th)and(vt<U_infinity)
        i=np.argmin(np.abs(pg-pt)); j=np.argmin(np.abs(vg-vt)); oy=yt[i,j].copy()
        if not ig: oy=np.zeros(N)
        oy=np.clip(oy,cy-RATE_MAX*T_STEP,cy+RATE_MAX*T_STEP)
        if np.max(np.abs(oy-cy))<DEADBAND: oy=cy
        if delay_steps>0:
            yh=np.roll(yh,1,axis=0); yh[0]=oy.copy()
            yaw_use=yh[min(buf_len-1,max(0,buf_len-1))]
        else:
            yaw_use=oy
        inf=calculate_inflow_speeds(pos,pt,0.8,0.065,126.0,U_infinity,yaw_use,2.727630853,0.1,0.53991)
        pw=float(jnp.sum(power_output_jax(jnp.array(inf),jnp.array(yaw_use)))/1e6)
        inf0=calculate_inflow_speeds(pos,pt,0.8,0.065,126.0,U_infinity,np.zeros(N),2.727630853,0.1,0.53991)
        pw0=float(jnp.sum(power_output_jax(jnp.array(inf0),jnp.zeros(N)))/1e6)
        tg+=(pw-pw0)/(pw0+1e-8)*100; tt+=np.sum(np.abs(oy-cy))
        if (pw-pw0)/(pw0+1e-8)*100<0: ng+=1; cy=oy.copy()
    return tg/N_STEPS, tt/N/N_STEPS, ng/N_STEPS

def eval_drl_deploy_delayed(phi_traj,v_traj,model,j_hist,delay_steps):
    k=jax.random.key(0)
    s,o=env_reset(k,pj,j=j_hist,specific_wind_dir=phi_traj[0],specific_wind_speed=v_traj[0],
                  randomize_wind=False,max_steps=N_STEPS+10)
    ig=False; tg=0.0; tt=0.0; ng=0
    buf_len=max(1,delay_steps); yh=np.zeros((buf_len,N))
    for t in range(N_STEPS):
        pt=float(phi_traj[t]); vt=float(v_traj[t])
        dp=min(abs(pt-270),360-abs(pt-270)); th=GATE_OUT if ig else GATE_IN
        ig=(dp<th)and(vt<U_infinity)
        mean,_,_=model(np.array(o).reshape(1,-1))
        raw_a=np.array(jnp.clip(mean.reshape(N),-AB,AB))
        a=raw_a if ig else np.zeros(N)
        if np.max(np.abs(a))<DEADBAND: a=np.zeros(N)
        a=np.clip(a,-RATE_MAX*T_STEP,RATE_MAX*T_STEP)
        ns,no,_,_=env_step(s,jnp.array(a),pj,max_steps=N_STEPS+10); s,o=ns,no
        if delay_steps>0:
            yh=np.roll(yh,1,axis=0); yh[0]=np.array(ns.gammas)
            yaw_use=yh[min(buf_len-1,max(0,buf_len-1))]
        else:
            yaw_use=np.array(ns.gammas)
        inf=calculate_inflow_speeds(pos,pt,0.8,0.065,126.0,U_infinity,yaw_use,2.727630853,0.1,0.53991)
        pw=float(jnp.sum(power_output_jax(jnp.array(inf),jnp.array(yaw_use)))/1e6)
        inf0=calculate_inflow_speeds(pos,pt,0.8,0.065,126.0,U_infinity,np.zeros(N),2.727630853,0.1,0.53991)
        pw0=float(jnp.sum(power_output_jax(jnp.array(inf0),jnp.zeros(N)))/1e6)
        tg+=(pw-pw0)/(pw0+1e-8)*100; tt+=np.sum(np.abs(a))
        if (pw-pw0)/(pw0+1e-8)*100<0: ng+=1
    return tg/N_STEPS, tt/N/N_STEPS, ng/N_STEPS

results={}
delays=[0,5,10,15]  # steps: 0/50/100/150s
configs=[
    ('SLSQP+Deploy', lambda t,d: eval_slsqp_deploy_delayed(P[t],V[t],d), 1),
    ('DRL-Deploy Static', lambda t,d: eval_drl_deploy_delayed(P[t],V[t],models_s[0],3,d), 1),
    ('DRL-Deploy J=15', lambda t,d: eval_drl_deploy_delayed(P[t],V[t],models_j15[0],15,d), 1),
]

for label,fn,n_seeds in configs:
    for d in delays:
        key=f'{label} delay={d*10}s'
        print(f'{key}...',end='',flush=True)
        t0=time.time()
        gains=[]; travels=[]; negs=[]
        for ti in range(N_TRAJ):
            g,t,n=fn(ti,d); gains.append(g); travels.append(t); negs.append(n)
        results[key]=dict(gain=np.mean(gains),gain_std=np.std(gains),
                           travel=np.mean(travels),neg=np.mean(negs))
        print(f' {np.mean(gains):+.4f}% [{time.time()-t0:.0f}s]')

# Multi-seed J=15 evaluation at 100s delay
print('\nMulti-seed J=15 at 100s delay...')
j15_gains=[]; j15_travels=[]
for si,m in enumerate(models_j15):
    gains=[]; travels=[]
    for ti in range(N_TRAJ):
        g,t,_=eval_drl_deploy_delayed(P[ti],V[ti],m,15,10)
        gains.append(g); travels.append(t)
    j15_gains.append(np.mean(gains)); j15_travels.append(np.mean(travels))
    print(f'  Seed {si}: {np.mean(gains):+.4f}% travel={np.mean(travels):.2f}')
results['J=15 multi-seed 100s']=dict(seeds=j15_gains,mean=np.mean(j15_gains),std=np.std(j15_gains))

# Multi-seed Static at 100s delay
print('Multi-seed Static at 100s delay...')
s_gains=[]
for si,m in enumerate(models_s):
    gains=[]
    for ti in range(N_TRAJ):
        g,_,_=eval_drl_deploy_delayed(P[ti],V[ti],m,3,10)
        gains.append(g)
    s_gains.append(np.mean(gains))
    print(f'  Seed {si}: {np.mean(gains):+.4f}%')
results['Static multi-seed 100s']=dict(seeds=s_gains,mean=np.mean(s_gains),std=np.std(s_gains))

out={'n_traj':N_TRAJ,'n_steps':N_STEPS,'delays':[d*10 for d in delays],
     'results':results}
with open('../results/delay_sensitivity.json','w') as f: json.dump(out,f,indent=2)
print(f'\nSaved: ../results/delay_sensitivity.json')
print('Done.')
