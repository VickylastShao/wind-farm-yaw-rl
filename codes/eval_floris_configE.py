#!/usr/bin/env python3
"""Config-E FLORIS cross-validation — uses vmap-batched gray-box rollout + FLORIS eval."""
import os, sys, json, time, pickle
import jax, jax.numpy as jnp, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, inflow_speeds_jax,
    power_output_jax, positions_to_jax)
from windfarm_env import create_wind_farm_layout_3x3
import floris; from floris import FlorisModel

CKPT_DIR = "checkpoints_3x3_nnx_jaxenv"
FIG_DIR = "../latex_draft/figures"; RESULTS_DIR = "../results"
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

J=3; N=9; AB=10.0; OBS_DIM=3*(5*N+3); SETTLE=200
N_COND=int(os.environ.get("N_CONDITIONS","300")); N_SEEDS=5; EVAL_SEED=20260614; TI=0.065

# ---------- FLORIS setup ----------
D=126.0; S=7*D; T=np.radians(7.0)
xs=[i*S+j*S*np.sin(T) for j in range(3) for i in range(3)]
ys=[j*S*np.cos(T) for j in range(3) for i in range(3)]
FM = FlorisModel(os.path.join(os.path.dirname(floris.__file__),"default_inputs.yaml"))
FM.set(layout_x=xs, layout_y=ys)

def floris_power(phi, v, yaw=None):
    ya = np.zeros((1,N)) if yaw is None else np.asarray(yaw).reshape(1,-1)
    FM.set(wind_directions=[float(phi)], wind_speeds=[float(v)],
           turbulence_intensities=[float(TI)], yaw_angles=ya); FM.run()
    return float(np.sum(FM.get_turbine_powers()))/1e6

# ---------- Conditions ----------
np.random.seed(EVAL_SEED)
phis=np.random.uniform(173,353,N_COND); vs=np.random.uniform(6,16,N_COND)
dphi=np.minimum(np.abs(phis-270),360-np.abs(phis-270))
ac_mask=(dphi<15)&(vs<11.4)
print(f"Conditions: {N_COND} total, {ac_mask.sum()} aligned-cube")

# Gray-box baselines
pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)
phis_j=jnp.asarray(phis,jnp.float32); vs_j=jnp.asarray(vs,jnp.float32)
@jax.jit
def gb_baseline(phi,v):
    inf=inflow_speeds_jax(pj,phi,v,jnp.zeros(N)); return jnp.sum(power_output_jax(inf,jnp.zeros(N)))/1e6
gb_bl=np.asarray(jax.vmap(gb_baseline)(phis_j,vs_j))
print(f"GB baseline: {gb_bl.mean():.2f} MW")

# FLORIS baselines  
print("FLORIS baselines...",end="",flush=True)
fl_bl=np.array([floris_power(phis[i],vs[i]) for i in range(N_COND)])
print(f" {fl_bl.mean():.2f} MW")

# ---------- Per-seed eval ----------
@nnx.jit
def rollout_batch(model, phis, vs):
    @jax.vmap
    def r1(phi,v):
        k=jax.random.key(0); s,o=env_reset(k,pj,j=J,specific_wind_dir=phi,specific_wind_speed=v,randomize_wind=False,max_steps=SETTLE+10)
        return s,o
    st,ob=r1(phis,vs)
    @jax.vmap
    def pred(o): mean,_,_=model(o.reshape(1,-1)); return jnp.clip(mean.reshape(N),-AB,AB)
    @jax.vmap
    def step(s,a): return env_step(s,a,pj,max_steps=SETTLE+10)
    def body(c,_):
        st2,ob2=c; a=pred(ob2); ns,no,_,_=step(st2,a); return (ns,no),None
    (fs,_),_=jax.lax.scan(body,(st,ob),None,length=SETTLE)
    # Compute gray-box power at final yaw
    inf_final=jax.vmap(inflow_speeds_jax,in_axes=(None,0,0,0))(pj,phis,vs,fs.gammas)
    gb_pwr=jax.vmap(lambda i,g: jnp.sum(power_output_jax(i,g))/1e6)(inf_final,fs.gammas)
    return gb_pwr, fs.gammas

results=[]
for s in range(N_SEEDS):
    ckpt=os.path.join(CKPT_DIR,f"policy_seed{s}_sens_act10.pkl")
    if not os.path.exists(ckpt): print(f"Seed {s}: MISSING"); continue
    model=ActorCritic(OBS_DIM,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(model)
    with open(ckpt,"rb") as f: st=pickle.load(f); model=nnx.merge(gd,st)
    
    print(f"Seed {s}: rollout...",end="",flush=True)
    gb_pwr,yaws=rollout_batch(model,phis_j,vs_j); gb_pwr=np.asarray(gb_pwr); yaws=np.asarray(yaws)
    gb_gains=(gb_pwr-gb_bl)/gb_bl*100
    print(f" GB marg={gb_gains.mean():+.2f}% ac={gb_gains[ac_mask].mean():+.2f}%",end="",flush=True)
    
    print(" FLORIS...",end="",flush=True)
    fl_pwr=np.array([floris_power(phis[i],vs[i],yaws[i]) for i in range(N_COND)])
    fl_gains=(fl_pwr-fl_bl)/fl_bl*100
    print(f" FL marg={fl_gains.mean():+.2f}% ac={fl_gains[ac_mask].mean():+.2f}%")
    
    results.append(dict(seed=s,
        gb_marginal=float(gb_gains.mean()), gb_ac=float(gb_gains[ac_mask].mean()),
        fl_marginal=float(fl_gains.mean()), fl_ac=float(fl_gains[ac_mask].mean())))

# Summary
gb_m=np.mean([r['gb_marginal'] for r in results])
fl_m=np.mean([r['fl_marginal'] for r in results])
gb_ac=np.mean([r['gb_ac'] for r in results])
fl_ac=np.mean([r['fl_ac'] for r in results])
summary=dict(n_seeds=len(results),n_cond=N_COND,n_ac=int(ac_mask.sum()),
    gb_bl=float(gb_bl.mean()), fl_bl=float(fl_bl.mean()),
    gb_marginal=float(gb_m), fl_marginal=float(fl_m),
    gb_aligned_cube=float(gb_ac), fl_aligned_cube=float(fl_ac),
    erosion=float((1-fl_ac/gb_ac)*100) if gb_ac>0 else None, per_seed=results)
with open(os.path.join(RESULTS_DIR,"configE_floris_cross_validation.json"),"w") as f: json.dump(summary,f,indent=2)
print(f"\n=== FINAL ===")
print(f"GB: marg={gb_m:+.2f}%, ac={gb_ac:+.2f}%")
print(f"FL: marg={fl_m:+.2f}%, ac={fl_ac:+.2f}%")
if gb_ac>0: print(f"Erosion: {(1-fl_ac/gb_ac)*100:.1f}%")
