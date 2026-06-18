#!/usr/bin/env python3
"""Final unified dynamic benchmark: all controllers, same trajectories, percentage power gain."""
import os, sys, json, time, pickle, numpy as np
import jax, jax.numpy as jnp
from flax import nnx
from scipy.interpolate import NearestNDInterpolator

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, positions_to_jax, inflow_speeds_jax, power_output_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT="checkpoints_3x3_nnx_jaxenv"; J=3; N=9; AB=10.0; OBS_DIM=144; TL=200; NT=1000; T_STEP=10.0
pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

# ---- Generate trajectories ONCE for both protocols ----
def gen_traj(rho, sigma, seed):
    np.random.seed(seed)
    P=np.zeros((NT,TL)); V=np.zeros((NT,TL))
    P[:,0]=np.random.uniform(173,353,NT); V[:,0]=np.random.uniform(6,16,NT)
    for t in range(1,TL):
        P[:,t]=263+rho*(P[:,t-1]-263)+sigma*np.random.normal(0,1,NT)
        V[:,t]=11+rho*(V[:,t-1]-11)+(sigma/2)*np.random.normal(0,1,NT)
        P[:,t]=np.clip(P[:,t],173,353); V[:,t]=np.clip(V[:,t],6,16)
    return P,V

print("Generating trajectories...")
PA,VA = gen_traj(0.99, 1.0, 42)  # Protocol A: slow
PB,VB = gen_traj(0.95, 2.0, 42)  # Protocol B: fast

# ---- SLSQP interpolator ----
exp=np.load('expert_datasets/slsqp_expert_3x3_seed20260606_n1000.npz')
interp=NearestNDInterpolator(np.column_stack([exp['phi'],exp['v']]),exp['slsqp_yaw'])

# ---- Load DRL policies ----
def load(tag,od=144,n=5):
    ms=[]
    for s in range(n):
        cp=os.path.join(CKPT,f"policy_seed{s}_{tag}.pkl")
        if not os.path.exists(cp): continue
        m=ActorCritic(od,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
        with open(cp,'rb') as f: st=pickle.load(f); ms.append(nnx.merge(gd,st))
    return ms

m_static = load('sens_act10',144,5)
m_dyn_j15 = load('dyn_J15_l5e4',720,2)
print(f"Loaded Static={len(m_static)}, J15={len(m_dyn_j15)} policies")

# ---- Evaluation functions (all return percentage power gain) ----
@jax.jit
def ed_raw(model,phi,v):
    k=jax.random.key(0)
    s,o=env_reset(k,pj,j=J,specific_wind_dir=phi[0],specific_wind_speed=v[0],randomize_wind=False,max_steps=TL+10)
    def body(c,t):
        st,ob,tg,tt,pr,nc,rv,py,lb,xc=c
        mean,_,_=model(ob.reshape(1,-1)); a=jnp.clip(mean.reshape(N),-AB,AB)
        ns,no,_,_=env_step(st,a,pj,max_steps=TL+10)
        i0=inflow_speeds_jax(pj,phi[t],v[t],jnp.zeros(N)); p0=jnp.sum(power_output_jax(i0,jnp.zeros(N)))/1e6
        iy=inflow_speeds_jax(pj,phi[t],v[t],ns.gammas); pw=jnp.sum(power_output_jax(iy,ns.gammas))/1e6
        g=(pw-p0)/(p0+1e-6)*100; st=jnp.sum(jnp.abs(a))
        rv_n=jnp.sum((jnp.sign(a)!=jnp.sign(py)).astype(jnp.int32))
        U2=jnp.square(iy); lb_s=jnp.sum(U2*jnp.square(ns.gammas)); lc_s=jnp.sum(U2*jnp.abs(jnp.sin(jnp.radians(ns.gammas))))
        return (ns,no,tg+g,tt+st,jnp.maximum(pr,st/T_STEP),nc+(g<0).astype(jnp.int32),rv+rv_n,a,lb+lb_s),None
    (_,_,tg,tt,pr,nc,rv,_,lb,_),_=jax.lax.scan(body,(s,o,jnp.array(0.0),jnp.array(0.0),jnp.array(0.0),jnp.array(0,dtype=jnp.int32),jnp.array(0,dtype=jnp.int32),jnp.zeros(N),jnp.array(0.0),jnp.array(0.0),jnp.array(0,dtype=jnp.int32)),jnp.arange(TL))
    return tg,tt,pr,nc,rv,lb

@jax.jit
def ed_deploy(model,phi,v):
    k=jax.random.key(0); ga=jnp.array(False); py0=jnp.zeros(N)
    s,o=env_reset(k,pj,j=J,specific_wind_dir=phi[0],specific_wind_speed=v[0],randomize_wind=False,max_steps=TL+10)
    def body(c,t):
        st,ob,tg,tt,pr,nc,rv,ga,py,lb=c
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
        rv_n=jnp.sum((jnp.sign(a)!=jnp.sign(py)).astype(jnp.int32))
        U2=jnp.square(iy); lb_s=jnp.sum(U2*jnp.square(ns.gammas))
        return (ns,no,tg+g,tt+st,jnp.maximum(pr,st/T_STEP),nc+(g<0).astype(jnp.int32),rv+rv_n,ga_new,a,lb+lb_s),None
    (_,_,tg,tt,pr,nc,rv,_,_,lb),_=jax.lax.scan(body,(s,o,jnp.array(0.0),jnp.array(0.0),jnp.array(0.0),jnp.array(0,dtype=jnp.int32),jnp.array(0,dtype=jnp.int32),ga,py0,jnp.array(0.0)),jnp.arange(TL))
    return tg,tt,pr,nc,rv,lb

def eval_slsqp_rl(P,V,rm):
    G=[];T_=[];Pk=[];Ng=[];Rv=[]
    for ti in range(NT):
        py=np.zeros(N);tg=0.0;tt=0.0;pr=0.0;nc=0;rv=0
        for t in range(TL):
            try:opt=interp(P[ti,t],V[ti,t])
            except:opt=py
            ms=rm*T_STEP if rm else 50.0;dy=np.clip(opt-py,-ms,ms);yw=py+dy
            i0=np.array(inflow_speeds_jax(pj,jnp.array(P[ti,t]),jnp.array(V[ti,t]),jnp.zeros(N)))
            p0=np.sum(np.array(power_output_jax(jnp.array(i0),jnp.zeros(N))))/1e6
            iy=np.array(inflow_speeds_jax(pj,jnp.array(P[ti,t]),jnp.array(V[ti,t]),jnp.array(yw)))
            pw=np.sum(np.array(power_output_jax(jnp.array(iy),jnp.array(yw))))/1e6
            g=(pw-p0)/(p0+1e-6)*100;tg+=g;tt+=np.sum(np.abs(dy));pr=max(pr,np.sum(np.abs(dy))/T_STEP)
            if g<0:nc+=1; rv+=np.sum((np.sign(yw)!=np.sign(py)).astype(int)); py=yw.copy()
        G.append(tg/TL);T_.append(tt/N);Pk.append(pr);Ng.append(nc/TL);Rv.append(rv)
    return np.mean(G),np.mean(T_),np.max(Pk),np.mean(Ng),np.mean(Rv)

def run_controller(name, models, P, V, mode='raw'):
    print(f"  {name}...", end=" ", flush=True); t0=time.time()
    ef = ed_raw if mode=='raw' else ed_deploy
    G=[];T_=[];Pk=[];Ng=[];Rv=[];LB=[]
    for ti in range(NT):
        mi=ti%len(models)
        g,t,p,n,rv,lb=ef(models[mi],jnp.array(P[ti]),jnp.array(V[ti]))
        G.append(float(g)/TL);T_.append(float(t)/N);Pk.append(float(p))
        Ng.append(float(n)/TL);Rv.append(float(rv));LB.append(float(lb))
    r=dict(name=name,gain=np.mean(G),std_gain=np.std(G)/np.sqrt(NT),
           travel=np.mean(T_),peak=np.max(Pk),neg=np.mean(Ng),
           reversals=np.mean(Rv),load_B=np.mean(LB),
           gain_per_travel=np.mean(G)/(np.mean(T_)+1e-6))
    print(f"gain={r['gain']:+.3f} travel={r['travel']:.0f} neg={r['neg']*100:.0f}% ({time.time()-t0:.0f}s)")
    return r

# ---- Run all controllers ----
all_results={'protocol_A':[],'protocol_B':[]}

print("\n=== Protocol A (slow, rho=0.99) ===")
results_A=[]
results_A.append(dict(name='Zero yaw',gain=0,std_gain=0,travel=0,peak=0,neg=0,reversals=0,load_B=0,gain_per_travel=0))
results_A.append(dict(name='Greedy yaw tracking',gain=0,std_gain=0,travel=0,peak=0,neg=0,reversals=0,load_B=0,gain_per_travel=0))
results_A.append(run_controller('Static Config-E (raw)',m_static,PA,VA,'raw'))
results_A.append(run_controller('Static + Deploy',m_static,PA,VA,'deploy'))
results_A.append(run_controller('SLSQP Unlimited',m_static[:1],PA,VA,'raw'))  # placeholder for SLSQP
results_A.append(run_controller('SLSQP RL=0.5',m_static[:1],PA,VA,'raw'))  # placeholder
# Replace with actual SLSQP eval
sg,st,sp,sn,sr=eval_slsqp_rl(PA,VA,None); results_A[4]=dict(name='SLSQP Unlimited',gain=sg,travel=st,peak=sp,neg=sn,reversals=sr,load_B=0,gain_per_travel=sg/(st+1e-6))
sg,st,sp,sn,sr=eval_slsqp_rl(PA,VA,0.5); results_A[5]=dict(name='SLSQP RL=0.5/s',gain=sg,travel=st,peak=sp,neg=sn,reversals=sr,load_B=0,gain_per_travel=sg/(st+1e-6))
results_A.append(run_controller('J=15 Dyn + Deploy',m_dyn_j15,PA,VA,'deploy'))

print("\n=== Protocol B (fast, rho=0.95) ===")
results_B=[]
results_B.append(dict(name='Zero yaw',gain=0,std_gain=0,travel=0,peak=0,neg=0,reversals=0,load_B=0,gain_per_travel=0))
results_B.append(dict(name='Greedy yaw tracking',gain=0,std_gain=0,travel=0,peak=0,neg=0,reversals=0,load_B=0,gain_per_travel=0))
results_B.append(run_controller('Static Config-E (raw)',m_static,PB,VB,'raw'))
results_B.append(run_controller('Static + Deploy',m_static,PB,VB,'deploy'))
sg,st,sp,sn,sr=eval_slsqp_rl(PB,VB,None); results_B.append(dict(name='SLSQP Unlimited',gain=sg,travel=st,peak=sp,neg=sn,reversals=sr,load_B=0,gain_per_travel=sg/(st+1e-6)))
sg,st,sp,sn,sr=eval_slsqp_rl(PB,VB,0.5); results_B.append(dict(name='SLSQP RL=0.5/s',gain=sg,travel=st,peak=sp,neg=sn,reversals=sr,load_B=0,gain_per_travel=sg/(st+1e-6)))
results_B.append(run_controller('J=15 Dyn + Deploy',m_dyn_j15,PB,VB,'deploy'))

all_results['protocol_A']=results_A; all_results['protocol_B']=results_B
with open('../results/unified_final_benchmark.json','w') as f: json.dump(all_results,f,indent=2)

# Print final tables
for pname, res in [('Protocol A (slow)',results_A),('Protocol B (fast)',results_B)]:
    print(f'\n=== {pname} ===')
    print(f'{"Controller":30s} {"Gain":>8s} {"Travel":>7s} {"Peak":>6s} {"Neg%":>5s} {"Rev":>6s} {"Gain/Tr":>7s}')
    for r in res:
        print(f'{r["name"]:30s} {r["gain"]:>+7.3f} {r["travel"]:>6.0f}° {r["peak"]:>5.2f} {r["neg"]*100:>4.0f}% {r["reversals"]:>5.0f} {r["gain_per_travel"]:>+6.3f}')
print('\nSaved to results/unified_final_benchmark.json')
