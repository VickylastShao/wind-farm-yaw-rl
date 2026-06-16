#!/usr/bin/env python3
"""Fast batched gated DRL evaluation using vmap + lax.scan."""
import os, sys, json, time, pickle
import jax, jax.numpy as jnp, numpy as np
from flax import nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, inflow_speeds_jax,
    power_output_jax, positions_to_jax, find_downstream_mask_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT_DIR = "checkpoints_3x3_nnx_jaxenv"
RESULTS_DIR = "../results"; os.makedirs(RESULTS_DIR, exist_ok=True)

J=3; N=9; AB=10.0; OBS_DIM=3*(5*N+3)
N_TRAJ=1000; TRAJ_LEN=200; SETTLE=150; T_STEP=10.0
GATE_DPHI_IN=15.0; GATE_DPHI_OUT=20.0; GATE_V=11.4; DEADBAND=float(os.environ.get("DEADBAND","5.0"))
ALPHA_PHI=0.95; SIGMA_PHI=2.0; ALPHA_V=0.95; SIGMA_V=1.0

pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

# AR(1) trajectories
np.random.seed(20260614)
phi=np.random.uniform(173,353,N_TRAJ); v=np.random.uniform(6,16,N_TRAJ)
phi_hist=[phi.copy()]; v_hist=[v.copy()]
for t in range(SETTLE+TRAJ_LEN-1):
    phi=263+ALPHA_PHI*(phi-263)+SIGMA_PHI*np.random.normal(0,1,N_TRAJ)
    v=11+ALPHA_V*(v-11)+SIGMA_V*np.random.normal(0,1,N_TRAJ)
    phi=np.clip(phi,173,353); v=np.clip(v,6,16)
    phi_hist.append(phi.copy()); v_hist.append(v.copy())
phi_traj=np.column_stack(phi_hist)[:,SETTLE:]; v_traj=np.column_stack(v_hist)[:,SETTLE:]
print(f"Trajectories: {phi_traj.shape}")

@jax.jit
def gb_baseline_fn(phi,v):
    inf=inflow_speeds_jax(pj,phi,v,jnp.zeros(N))
    return jnp.sum(power_output_jax(inf,jnp.zeros(N)))/1e6

def in_gate(phi,v,threshold=GATE_DPHI_IN):
    dphi=jnp.minimum(jnp.abs(phi-270),360-jnp.abs(phi-270))
    return (dphi<threshold)&(v<GATE_V)

# Evaluate single trajectory with gating strategy
@jax.jit
def eval_trajectory_gated(model, phi_arr, v_arr, gate_in, gate_out, use_deadband):
    """Evaluate one trajectory. Returns (total_reward, total_yaw_travel, n_negative, n_gate)."""
    key=jax.random.key(0)
    phi0=phi_arr[0]; v0=v_arr[0]
    state,obs=env_reset(key,pj,j=J,specific_wind_dir=phi0,specific_wind_speed=v0,randomize_wind=False,max_steps=TRAJ_LEN+10)
    
    def body(carry, t):
        state, obs, key, cum_yaw, neg_count, gate_count, gate_active = carry
        phi_t=phi_arr[t]; v_t=v_arr[t]
        
        # Gate logic
        threshold = jnp.where(gate_active, gate_out, gate_in)
        dphi=jnp.minimum(jnp.abs(phi_t-270),360-jnp.abs(phi_t-270))
        in_regime = (dphi<threshold)&(v_t<GATE_V)
        
        # Get action
        mean,_,_=model(obs.reshape(1,-1)); raw_action=jnp.clip(mean.reshape(N),-AB,AB)
        
        # Apply gate and deadband
        action=jnp.where(in_regime, raw_action, jnp.zeros(N))
        action=jnp.where(use_deadband.astype(bool) & (jnp.max(jnp.abs(action))<DEADBAND), jnp.zeros(N), action)
        
        # Step
        key,sk=jax.random.split(key)
        new_state,new_obs,reward,_=env_step(state,action,pj,max_steps=TRAJ_LEN+10)
        
        cum_yaw += jnp.sum(jnp.abs(action))
        neg_count += (reward < 0).astype(jnp.int32)
        gate_count += in_regime.astype(jnp.int32)
        
        return (new_state, new_obs, key, cum_yaw, neg_count, gate_count, in_regime), (reward, jnp.sum(jnp.abs(action)))
    
    init = (state, obs, key, jnp.array(0.0), jnp.array(0,dtype=jnp.int32), jnp.array(0,dtype=jnp.int32), jnp.array(False))
    (_,_,_,cum_yaw,neg_count,gate_count,_), (rewards, yaw_steps) = jax.lax.scan(body, init, jnp.arange(TRAJ_LEN))
    
    total_reward = jnp.sum(rewards)
    peak_rate = jnp.max(yaw_steps) / T_STEP
    return total_reward, cum_yaw, peak_rate, neg_count, gate_count

# Load models
models=[]
for s in range(5):
    ckpt=os.path.join(CKPT_DIR,f"policy_seed{s}_sens_act10.pkl")
    if not os.path.exists(ckpt): continue
    m=ActorCritic(OBS_DIM,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
    with open(ckpt,"rb") as f: st=pickle.load(f)
    models.append(nnx.merge(gd,st))
print(f"Loaded {len(models)} models")

# Compile
print("Compiling...",end="",flush=True)
_=eval_trajectory_gated(models[0], jnp.asarray(phi_traj[0]), jnp.asarray(v_traj[0]),
    jnp.array(GATE_DPHI_IN), jnp.array(GATE_DPHI_OUT), jnp.array(False))
jax.block_until_ready(True)
print("done")

# Evaluate
strategies=[
    ("Raw DRL", GATE_DPHI_IN, GATE_DPHI_IN, False),
    ("Gated DRL", GATE_DPHI_IN, GATE_DPHI_IN, False),
    ("Hysteresis Gated", GATE_DPHI_IN, GATE_DPHI_OUT, False),
    ("Gated+Deadband", GATE_DPHI_IN, GATE_DPHI_IN, True),
]

# Raw DRL: always in gate (threshold=999)
for name, gi, go, db in strategies:
    t0=time.time()
    gains=[]; travels=[]; peaks=[]; negs=[]; gates=[]
    
    for traj_idx in range(N_TRAJ):  #  trajectories
        mi=traj_idx%len(models)
        if name=="Raw DRL":
            tr,ty,pr,ng,gt=eval_trajectory_gated(models[mi], jnp.asarray(phi_traj[traj_idx]), jnp.asarray(v_traj[traj_idx]),
                jnp.array(999.0), jnp.array(999.0), jnp.array(False))
        else:
            tr,ty,pr,ng,gt=eval_trajectory_gated(models[mi], jnp.asarray(phi_traj[traj_idx]), jnp.asarray(v_traj[traj_idx]),
                jnp.array(gi), jnp.array(go), jnp.array(db))
        gains.append(float(tr)/10.0)  # regret reward → approximate gain %
        travels.append(float(ty))
        peaks.append(float(pr))
        negs.append(float(ng)/TRAJ_LEN)
        gates.append(float(gt)/TRAJ_LEN)
    
    print(f"{name:20s}: gain={np.mean(gains):+.2f}% travel={np.mean(travels):.1f}° peak={np.max(peaks):.2f}°/s neg={np.mean(negs)*100:.0f}% gate={np.mean(gates)*100:.0f}% time={time.time()-t0:.0f}s")

print("\nDone. Full 1000-trajectory eval pending.")
