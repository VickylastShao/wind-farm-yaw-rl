#!/usr/bin/env python3
"""Compute simplified secondary load proxies for Config-E policies."""
import os, sys, json, pickle
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
sys.path.insert(0, '.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, inflow_speeds_jax,
    power_output_jax, positions_to_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT = "checkpoints_3x3_nnx_jaxenv"; J=3; N=9; AB=10.0; OBS_DIM=3*(5*N+3)
SETTLE=200; N_COND=500
pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

np.random.seed(20260614)
phis=np.random.uniform(173,353,N_COND); vs=np.random.uniform(6,16,N_COND)
dphi=np.minimum(np.abs(phis-270),360-np.abs(phis-270)); ac_mask=(dphi<15)&(vs<11.4)

print("=== Secondary Load Proxy Evaluation ===")
print(f"Conditions: {N_COND} ({ac_mask.sum()} aligned-cube)")

# Proxy A: Yaw-activity (cumulative yaw travel, already computed in gated eval)
# Proxy B: Yaw-misalignment aerodynamic load = sum_t U_i(t)^2 * |gamma_i(t)|^p
# Proxy C: Tower-side-load surrogate = sum_t U_i(t)^2 * |sin(gamma_i(t))|

# Load Config-E policy
m=ActorCritic(OBS_DIM,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
with open(f"{CKPT}/policy_seed0_sens_act10.pkl","rb") as f: st=pickle.load(f)
model=nnx.merge(gd,st)

results=[]
for i in range(min(50, N_COND)):  # 50 conditions for speed
    phi=phis[i]; v=vs[i]
    k=jax.random.key(i)
    s,o=env_reset(k,pj,j=J,specific_wind_dir=jnp.array(phi),specific_wind_speed=jnp.array(v),
        randomize_wind=False,max_steps=SETTLE+10)
    
    yaw_travel=0.0; load_proxy_B=0.0; load_proxy_C=0.0
    for _ in range(SETTLE):
        mean,_,_=model(o.reshape(1,-1)); a=jnp.clip(mean.reshape(N),-AB,AB)
        s,o,r,_=env_step(s,a,pj,max_steps=SETTLE+10)
        gammas=np.array(s.gammas); inflow=np.array(s.inflow)
        U_sq = inflow**2  # dynamic pressure proxy
        yaw_travel += np.sum(np.abs(a))
        load_proxy_B += np.sum(U_sq * np.abs(gammas)**2)  # p=2
        load_proxy_C += np.sum(U_sq * np.abs(np.sin(np.radians(gammas))))
    
    results.append(dict(phi=phi,v=v,yaw_travel=yaw_travel,load_B=load_proxy_B,load_C=load_proxy_C,
        in_regime=bool(ac_mask[i])))

# Summary
yaw_all = np.array([r['yaw_travel'] for r in results])
lb_all = np.array([r['load_B'] for r in results])
lc_all = np.array([r['load_C'] for r in results])
in_reg = np.array([r['in_regime'] for r in results])

print(f"\nProxy A — Yaw travel: mean={yaw_all.mean():.1f}°, in-regime only={yaw_all[in_reg].mean():.1f}°")
print(f"Proxy B — |γ|^2 * U^2: mean={lb_all.mean():.1f}, in-regime only={lb_all[in_reg].mean():.1f}")
print(f"Proxy C — |sin(γ)| * U^2: mean={lc_all.mean():.1f}, in-regime only={lc_all[in_reg].mean():.1f}")
print(f"\nFraction of load incurred in aligned-cube regime:")
print(f"  Yaw travel: {yaw_all[in_reg].sum()/yaw_all.sum()*100:.1f}%")
print(f"  Load proxy B: {lb_all[in_reg].sum()/lb_all.sum()*100:.1f}%")
print(f"  Load proxy C: {lc_all[in_reg].sum()/lc_all.sum()*100:.1f}%")
print("\nNote: These are simplified operational proxies, NOT aeroelastic fatigue loads.")
print("Future work should use OpenFAST/FAST.Farm for DEL validation.")
