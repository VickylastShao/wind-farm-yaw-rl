#!/usr/bin/env python3
"""Unified dynamic-wind benchmark: all controllers on same AR(1) trajectories."""
import os, sys, json, time, pickle
import jax, jax.numpy as jnp, numpy as np
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, positions_to_jax,
    inflow_speeds_jax, power_output_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT="checkpoints_3x3_nnx_jaxenv"; RESULTS="../results"
os.makedirs(RESULTS,exist_ok=True)
J=3; N=9; AB=10.0; OBS_DIM=3*(5*N+3); T_STEP=10.0
N_TRAJ=1000; TRAJ_LEN=200
ALPHA_PHI=0.95; SIGMA_PHI=2.0; ALPHA_V=0.95; SIGMA_V=1.0

pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

# Generate fixed AR(1) trajectories ONCE for all controllers
np.random.seed(20260617)
phi_2d=np.zeros((N_TRAJ,TRAJ_LEN)); v_2d=np.zeros((N_TRAJ,TRAJ_LEN))
phi_2d[:,0]=np.random.uniform(173,353,N_TRAJ); v_2d[:,0]=np.random.uniform(6,16,N_TRAJ)
for t in range(1,TRAJ_LEN):
    phi_2d[:,t]=263+ALPHA_PHI*(phi_2d[:,t-1]-263)+SIGMA_PHI*np.random.normal(0,1,N_TRAJ)
    v_2d[:,t]=11+ALPHA_V*(v_2d[:,t-1]-11)+SIGMA_V*np.random.normal(0,1,N_TRAJ)
    phi_2d[:,t]=np.clip(phi_2d[:,t],173,353); v_2d[:,t]=np.clip(v_2d[:,t],6,16)
print(f"Trajectories: {N_TRAJ} × {TRAJ_LEN}")

# Core evaluation: vmap + lax.scan per trajectory (fast)
@jax.jit
def gb_baseline_fn(phi,v):
    inf=inflow_speeds_jax(pj,phi,v,jnp.zeros(N))
    return jnp.sum(power_output_jax(inf,jnp.zeros(N)))/1e6

@jax.jit
def eval_traj_drl(model, phi_arr, v_arr, bl, gate_on, gate_in, gate_out, db):
    """Evaluate DRL policy on one trajectory with optional gate + deadband."""
    k=jax.random.key(0)
    s,o=env_reset(k,pj,j=J,specific_wind_dir=phi_arr[0],specific_wind_speed=v_arr[0],randomize_wind=False,max_steps=TRAJ_LEN+10)
    gate_active=jnp.array(False)
    def body(c,t):
        st,ob,tg,tt,pr,nc,ga,al_g,al_n=c
        phi_t=phi_arr[t]; v_t=v_arr[t]
        dphi=jnp.minimum(jnp.abs(phi_t-270),360-jnp.abs(phi_t-270))
        in_regime=(dphi<gate_in)&(v_t<11.4)
        # Gate logic
        threshold=jnp.where(ga,gate_out,gate_in)
        in_gate=(dphi<threshold)&(v_t<11.4)
        ga_new=jnp.where(gate_on,in_gate,True)
        # Action
        mean,_,_=model(ob.reshape(1,-1)); raw_a=jnp.clip(mean.reshape(N),-AB,AB)
        a=jnp.where(ga_new,raw_a,jnp.zeros(N))
        a=jnp.where((db>0.5)&(jnp.max(jnp.abs(a))<db),jnp.zeros(N),a)
        ns,no,_,_=env_step(st,a,pj,max_steps=TRAJ_LEN+10)
        inf_yaw=inflow_speeds_jax(pj,phi_t,v_t,ns.gammas); inf_zero=inflow_speeds_jax(pj,phi_t,v_t,jnp.zeros(N))
        pwr_yaw=jnp.sum(power_output_jax(inf_yaw,ns.gammas))/1e6
        pwr_zero=jnp.sum(power_output_jax(inf_zero,jnp.zeros(N)))/1e6; gain_pct=(pwr_yaw-pwr_zero)/(pwr_zero+1e-6)*100
        st=jnp.sum(jnp.abs(a))
        return (ns,no,tg+gain_pct,tt+st,jnp.maximum(pr,st/T_STEP),nc+(gain_pct<0).astype(jnp.int32),ga_new,al_g+jnp.where(in_regime,gain_pct,0.0),al_n+in_regime.astype(jnp.int32)),None
    init=(s,o,jnp.array(0.0),jnp.array(0.0),jnp.array(0.0),jnp.array(0,dtype=jnp.int32),gate_active,jnp.array(0.0),jnp.array(0,dtype=jnp.int32))
    (_,_,tg,tt,pr,nc,_,al_g,al_n),_=jax.lax.scan(body,init,jnp.arange(TRAJ_LEN))
    load_proxy=jnp.sum(jnp.abs(tg))  # simplified load proxy = |gain| penalty
    return tg,tt,pr,nc,al_g,al_n,load_proxy

@jax.jit
def eval_traj_zero(phi_arr, v_arr, bl):
    """Zero yaw baseline."""
    tg=0.0
    for t in range(TRAJ_LEN):
        inf=inflow_speeds_jax(pj,phi_arr[t],v_arr[t],jnp.zeros(N))
        pwr=jnp.sum(power_output_jax(inf,jnp.zeros(N)))/1e6
        tg+=(pwr-bl)/bl*100
    return tg,jnp.array(0.0),jnp.array(0.0),jnp.array(0,dtype=jnp.int32),jnp.array(0.0),jnp.array(0,dtype=jnp.int32),jnp.array(0.0)

# Load models
def load_models(tag, n_seeds=5):
    models=[]
    for s in range(n_seeds):
        ckpt=os.path.join(CKPT,f"policy_seed{s}_{tag}.pkl")
        if not os.path.exists(ckpt): continue
        m=ActorCritic(OBS_DIM,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
        with open(ckpt,'rb') as f: st=pickle.load(f)
        models.append(nnx.merge(gd,st))
    return models

controllers = [
    ("Zero yaw", None, None),
    ("Static Config-E", "sens_act10", None),
    ("Static + Gate", "sens_act10", (True,15.0,15.0,0)),
    ("Static + Hyst Gate", "sens_act10", (True,15.0,20.0,0)),
    ("Static + Hyst + 2°DB", "sens_act10", (True,15.0,20.0,2.0)),
    ("Dynamic λ=0", "dyn_lambda0_60M", None),
    ("Dynamic λ=5e-4", "dyn_lambda5_60M", None),
    ("Dynamic λ=5e-4 + Gate", "dyn_lambda5_60M", (True,15.0,15.0,0)),
    ("Dynamic λ=5e-4 + Gate + 2°DB", "dyn_lambda5_60M", (True,15.0,20.0,2.0)),
]

print("\n=== Unified Dynamic-Wind Benchmark ===")
all_results=[]

for name, tag, gate_params in controllers:
    models = load_models(tag) if tag else None
    if tag and not models:
        print(f"  {name}: SKIP (no checkpoints for {tag})")
        continue
    
    print(f"  {name}...", end=" ", flush=True)
    t0=time.time()
    all_g=[]; all_t=[]; all_p=[]; all_n=[]; all_al_g=[]; all_al_n=[]; all_lp=[]
    
    for ti in range(min(N_TRAJ,200)):  # 200 traj for speed
        bl=float(gb_baseline_fn(jnp.array(phi_2d[ti,0]),jnp.array(v_2d[ti,0])))
        
        if name=="Zero yaw":
            g,t,p,n,al_g,al_n,lp=eval_traj_zero(jnp.array(phi_2d[ti]),jnp.array(v_2d[ti]),jnp.array(bl))
        else:
            mi=ti%len(models); model=models[mi]
            gate_on,gi,go,db = gate_params if gate_params else (False,15.0,15.0,0)
            g,t,p,n,al_g,al_n,lp=eval_traj_drl(model,jnp.array(phi_2d[ti]),jnp.array(v_2d[ti]),jnp.array(bl),jnp.array(gate_on),jnp.array(gi),jnp.array(go),jnp.array(float(db)))
        
        all_g.append(float(g)); all_t.append(float(t)); all_p.append(float(p))
        all_n.append(float(n)/TRAJ_LEN); all_al_g.append(float(al_g)); all_al_n.append(float(al_n)); all_lp.append(float(lp))
    
    r=dict(name=name,mean_gain=float(np.mean(all_g)),std_gain=float(np.std(all_g)),
        mean_travel=float(np.mean(all_t)),peak_rate=float(np.max(all_p)),
        aligned_gain=float(np.sum(all_al_g)/max(np.sum(all_al_n),1)),
        neg_frac=float(np.mean(all_n)),load_proxy=float(np.mean(all_lp)),
        gain_per_travel=float(np.mean(all_g)/(np.mean(all_t)+1e-6)))
    all_results.append(r)
    print(f"gain={r['mean_gain']:+.2f}% al={r['aligned_gain']:+.2f}% travel={r['mean_travel']:.0f}° peak={r['peak_rate']:.2f}°/s neg={r['neg_frac']*100:.0f}% ({time.time()-t0:.0f}s)")

# Save
with open(f"{RESULTS}/unified_dynamic_benchmark.json","w") as f: json.dump(all_results,f,indent=2)
print(f"\n=== FINAL TABLE ===")
print(f"{'Controller':30s} {'Gain%':>7s} {'Al%':>6s} {'Travel':>7s} {'Peak':>6s} {'Neg%':>5s} {'G/Tr':>6s}")
for r in all_results:
    print(f"{r['name']:30s} {r['mean_gain']:>+6.2f} {r['aligned_gain']:>+5.2f} {r['mean_travel']:>6.0f}° {r['peak_rate']:>5.2f} {r['neg_frac']*100:>4.0f}% {r['gain_per_travel']:>+5.3f}")
print(f"\nSaved to {RESULTS}/unified_dynamic_benchmark.json")
