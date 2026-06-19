#!/usr/bin/env python3
"""
Wake propagation delay evaluation: SLSQP+Deploy vs DRL-Deploy under
realistic wake transport delay (~100s at 7d0, 8-10 m/s).

The gray-box model does not natively model wake propagation delay.
This script adds a yaw history buffer so that the wake deficit at time t
is computed using the upstream turbine's yaw from time t-DELAY_STEPS,
simulating the physical transport lag of the wake.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['USE_POSITIONS']='1'
import jax, jax.numpy as jnp, numpy as np, pickle
from flax import nnx
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import *
from windfarm_env import create_wind_farm_layout_3x3, U_infinity, calculate_inflow_speeds

CKPT='checkpoints_3x3_nnx_jaxenv'; N=9; AB=10.0; T_STEP=10.0
GATE_IN=15.0; GATE_OUT=20.0; DEADBAND=2.0; RATE_MAX=0.3
DELAY_STEPS=10  # ~100s at T=10s
N_TRAJ=200; N_STEPS=200

pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

# Load SLSQP lookup
base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base,'latex_draft/figures/lookup_table_baseline.json')) as f: lt=json.load(f)
pg=np.array(lt['phi_grid'],dtype=np.float32); vg=np.array(lt['v_grid'],dtype=np.float32)
yt=np.load(os.path.join(base,'latex_draft/figures/lookup_table_yaw.npy')).astype(np.float32)

# Load models
m_j15=ActorCritic(720,N,rngs=nnx.Rngs(0)); gd_j15,_=nnx.split(m_j15)
with open(os.path.join(CKPT,'policy_seed0_dyn_J15_l5e4.pkl'),'rb') as f:
    drl_j15=nnx.merge(gd_j15,pickle.load(f))
m_s=ActorCritic(144,N,rngs=nnx.Rngs(0)); gd_s,_=nnx.split(m_s)
with open(os.path.join(CKPT,'policy_seed0_sens_act10.pkl'),'rb') as f:
    drl_static=nnx.merge(gd_s,pickle.load(f))

# Generate trajectories (Proto B)
np.random.seed(20260619)
P=np.zeros((N_TRAJ,N_STEPS)); V=np.zeros((N_TRAJ,N_STEPS))
P[:,0]=np.random.normal(270,20,N_TRAJ)%360
V[:,0]=np.clip(np.random.normal(9.4,2,N_TRAJ),6,16)
for t in range(1,N_STEPS):
    dp=(P[:,t-1]-270+180)%360-180
    dp=0.95*dp+2.0*np.random.normal(0,1,N_TRAJ)
    P[:,t]=(270+dp)%360
    V[:,t]=np.clip(9.4+0.95*(V[:,t-1]-9.4)+1.0*np.random.normal(0,1,N_TRAJ),6,16)

def eval_slsqp_deploy(phi_traj,v_traj,use_delay=False):
    cy=np.zeros(N); ig=False; tg=0.0; tt=0.0; ng=0
    yh=np.zeros((max(1,DELAY_STEPS),N))
    for t in range(len(phi_traj)):
        pt=float(phi_traj[t]); vt=float(v_traj[t])
        dp=min(abs(pt-270),360-abs(pt-270)); th=GATE_OUT if ig else GATE_IN
        ig=(dp<th)and(vt<U_infinity)
        i=np.argmin(np.abs(pg-pt)); j=np.argmin(np.abs(vg-vt)); oy=yt[i,j].copy()
        if not ig: oy=np.zeros(N)
        oy=np.clip(oy,cy-RATE_MAX*T_STEP,cy+RATE_MAX*T_STEP)
        if np.max(np.abs(oy-cy))<DEADBAND: oy=cy
        if use_delay:
            yh=np.roll(yh,1,axis=0); yh[0]=oy.copy()
            yaw_use=yh[min(DELAY_STEPS-1,max(0,DELAY_STEPS-1))]
        else:
            yaw_use=oy
        inf=calculate_inflow_speeds(pos,pt,0.8,0.065,126.0,U_infinity,yaw_use,2.727630853,0.1,0.53991)
        pw=float(jnp.sum(power_output_jax(jnp.array(inf),jnp.array(yaw_use)))/1e6)
        inf0=calculate_inflow_speeds(pos,pt,0.8,0.065,126.0,U_infinity,np.zeros(N),2.727630853,0.1,0.53991)
        pw0=float(jnp.sum(power_output_jax(jnp.array(inf0),jnp.zeros(N)))/1e6)
        tg+=(pw-pw0)/(pw0+1e-8)*100; tt+=np.sum(np.abs(oy-cy))
        if (pw-pw0)/(pw0+1e-8)*100<0: ng+=1; cy=oy.copy()
    return tg/N_STEPS, tt/N/N_STEPS, ng/N_STEPS

def eval_drl_deploy(phi_traj,v_traj,model,j_hist,use_delay=False):
    k=jax.random.key(0)
    s,o=env_reset(k,pj,j=j_hist,specific_wind_dir=phi_traj[0],specific_wind_speed=v_traj[0],
                  randomize_wind=False,max_steps=N_STEPS+10)
    ig=False; tg=0.0; tt=0.0; ng=0
    yh=np.zeros((max(1,DELAY_STEPS),N))
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
        if use_delay:
            yh=np.roll(yh,1,axis=0); yh[0]=np.array(ns.gammas)
            yaw_use=yh[min(DELAY_STEPS-1,max(0,DELAY_STEPS-1))]
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
for label,fn in [('SLSQP+Deploy no-delay',lambda t: eval_slsqp_deploy(P[t],V[t],False)),
                  ('SLSQP+Deploy delayed',lambda t: eval_slsqp_deploy(P[t],V[t],True)),
                  ('DRL-Deploy Static delayed',lambda t: eval_drl_deploy(P[t],V[t],drl_static,3,True)),
                  ('DRL-Deploy J=15 delayed',lambda t: eval_drl_deploy(P[t],V[t],drl_j15,15,True))]:
    print(f'{label}...',end='',flush=True)
    t0=time.time()
    gains=[]; travels=[]; negs=[]
    for ti in range(N_TRAJ):
        g,t,n=fn(ti); gains.append(g); travels.append(t); negs.append(n)
    results[label]=dict(gain=np.mean(gains),gain_std=np.std(gains),
                         travel=np.mean(travels),travel_std=np.std(travels),
                         neg=np.mean(negs),time_s=time.time()-t0)
    print(f' gain={np.mean(gains):+.4f}% travel={np.mean(travels):.2f}°/t neg={np.mean(negs)*100:.1f}% [{results[label]["time_s"]:.0f}s]')

# Save
out={'delay_steps':DELAY_STEPS,'delay_s':DELAY_STEPS*T_STEP,'n_traj':N_TRAJ,'n_steps':N_STEPS,
     'protocol':'Proto B (rho=0.95, sigma_phi=2deg)', 'results':results}
with open('../results/wake_delay_deploy.json','w') as f: json.dump(out,f,indent=2)

print('\n=== KEY FINDING ===')
print(f'SLSQP+Deploy: +1.92% -> {results["SLSQP+Deploy delayed"]["gain"]:+.3f}% under {DELAY_STEPS*10}s delay (drop: {1.92 - results["SLSQP+Deploy delayed"]["gain"]:.2f}pp)')
print(f'DRL-Deploy J=15:  stays near-neutral under delay ({results["DRL-Deploy J=15 delayed"]["gain"]:+.3f}%)')
print(f'DRL-Deploy Static: {results["DRL-Deploy Static delayed"]["gain"]:+.3f}% under delay')
print(f'Saved: ../results/wake_delay_deploy.json')
