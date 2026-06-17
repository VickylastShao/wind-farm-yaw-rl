#!/usr/bin/env python3
"""Evaluate Config-E policies under dynamic wind with rate penalty."""
import os, sys, json, time, pickle
import jax, jax.numpy as jnp, numpy as np
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, inflow_speeds_jax,
    power_output_jax, positions_to_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT="checkpoints_3x3_nnx_jaxenv"; RESULTS="../results"
os.makedirs(RESULTS,exist_ok=True)
J=3; N=9; AB=10.0; OBS_DIM=3*(5*N+3); SETTLE=200; T_STEP=10.0
N_TRAJ=200; TRAJ_LEN=200
ALPHA_PHI=0.95; SIGMA_PHI=2.0; ALPHA_V=0.95; SIGMA_V=1.0

pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

# AR(1) trajectories
np.random.seed(20260616)
phi=np.random.uniform(173,353,N_TRAJ); v=np.random.uniform(6,16,N_TRAJ)
phi_hist=[phi.copy()]; v_hist=[v.copy()]
for t in range(SETTLE+TRAJ_LEN-1):
    phi=263+ALPHA_PHI*(phi-263)+SIGMA_PHI*np.random.normal(0,1,N_TRAJ)
    v=11+ALPHA_V*(v-11)+SIGMA_V*np.random.normal(0,1,N_TRAJ)
    phi=np.clip(phi,173,353); v=np.clip(v,6,16)
    phi_hist.append(phi.copy()); v_hist.append(v.copy())
phi_traj=np.column_stack(phi_hist)[:,SETTLE:]; v_traj=np.column_stack(v_hist)[:,SETTLE:]

# Load Config-E policy (seed 0, best balance)
model=ActorCritic(OBS_DIM,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(model)
with open(f"{CKPT}/policy_seed0_sens_act10.pkl","rb") as f: st=pickle.load(f)
model=nnx.merge(gd,st)

# Evaluate for each λrate
lambda_rates = [0, 0.0001, 0.0005, 0.001, 0.003, 0.01, 0.03, 0.1]
results = []

for lr in lambda_rates:
    print(f"λrate={lr:.0e}...", end=" ", flush=True)
    all_gains = []; all_travels = []; all_peaks = []; all_negs = []
    
    for ti in range(N_TRAJ):
        k=jax.random.key(ti)
        s,o=env_reset(k,pj,j=J,specific_wind_dir=jnp.array(phi_traj[ti,0]),
            specific_wind_speed=jnp.array(v_traj[ti,0]),randomize_wind=False,max_steps=TRAJ_LEN+10)
        total_gain=0.0; total_travel=0.0; peak_rate=0.0; neg_count=0
        
        for t in range(TRAJ_LEN):
            mean,_,_=model(o.reshape(1,-1)); a=jnp.clip(mean.reshape(N),-AB,AB)
            # env_step with rate penalty
            s,o,reward,_=env_step(s,a,pj,max_steps=TRAJ_LEN+10,lambda_rate=lr)
            gain_pct = float(reward) / 10.0  # regret-reward → approximate %
            total_gain += gain_pct
            step_travel = float(jnp.sum(jnp.abs(a)))
            total_travel += step_travel
            peak_rate = max(peak_rate, step_travel/T_STEP)
            if gain_pct < 0: neg_count += 1
        
        all_gains.append(total_gain)
        all_travels.append(total_travel)
        all_peaks.append(peak_rate)
        all_negs.append(neg_count/TRAJ_LEN)
    
    results.append(dict(
        lambda_rate=lr,
        mean_gain=float(np.mean(all_gains)),
        std_gain=float(np.std(all_gains)),
        mean_travel=float(np.mean(all_travels)),
        peak_rate=float(np.max(all_peaks)),
        neg_frac=float(np.mean(all_negs)),
    ))
    print(f"gain={np.mean(all_gains):+.2f} travel={np.mean(all_travels):.1f}° peak={np.max(all_peaks):.2f}°/s neg={np.mean(all_negs)*100:.0f}%")

# Save
with open(f"{RESULTS}/configE_dynamic_lambda.json","w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {RESULTS}/configE_dynamic_lambda.json")

# Summary
print("\n=== Config-E Dynamic Wind + λrate ===")
print(f"{'λrate':<10s} {'Gain':>8s} {'Travel':>8s} {'Peak':>8s} {'Neg%':>6s}")
for r in results:
    print(f"{r['lambda_rate']:<10.0e} {r['mean_gain']:>+7.2f} {r['mean_travel']:>7.1f}° {r['peak_rate']:>6.2f}/s {r['neg_frac']*100:>5.0f}%")
