import os, sys, json, time, pickle, numpy as np
import jax, jax.numpy as jnp
from flax import nnx
sys.path.insert(0,'.')
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, positions_to_jax, inflow_speeds_jax, power_output_jax)
from windfarm_env import create_wind_farm_layout_3x3

CKPT='checkpoints_3x3_nnx_jaxenv'; N=9; AB=10.0; TL=200; NT=30; Jv=15; OD=720; OPS=48
pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

np.random.seed(42)
P=np.zeros((NT,TL)); V=np.zeros((NT,TL)); P[:,0]=np.random.uniform(173,353,NT); V[:,0]=np.random.uniform(6,16,NT)
for t in range(1,TL):
    P[:,t]=263+0.95*(P[:,t-1]-263)+2.0*np.random.normal(0,1,NT)
    V[:,t]=np.clip(11+0.95*(V[:,t-1]-11)+1.0*np.random.normal(0,1,NT),6,16)
    P[:,t]=np.clip(P[:,t],173,353)

def load(tag,od,n=5):
    ms=[]
    for s in range(n):
        cp=os.path.join(CKPT,f"policy_seed{s}_{tag}.pkl")
        if not os.path.exists(cp): continue
        m=ActorCritic(od,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
        with open(cp,'rb') as f: st=pickle.load(f); ms.append(nnx.merge(gd,st))
    return ms

ms=load('dyn_J15_l5e4',OD,5); model=ms[0]
print("Loaded J15, {} seeds".format(len(ms)))

def mask_obs(obs, mtype):
    o2d=obs.reshape(Jv,OPS).copy()
    if mtype=='full_j15': pass
    elif mtype=='last3': o2d[:12]=0.0
    elif mtype=='mask_10_12': o2d[10:13]=0.0
    elif mtype=='shuffle': o2d=o2d[np.random.permutation(Jv)]
    return o2d.reshape(-1)

def et(model,phi_arr,v_arr,mtype):
    k=jax.random.key(0)
    s,o=jax.jit(env_reset,static_argnames=('j','max_steps','randomize_wind'))(k,pj,j=Jv,specific_wind_dir=jnp.array(phi_arr[0]),specific_wind_speed=jnp.array(v_arr[0]),randomize_wind=False,max_steps=TL+10)
    tg=0.0;tt=0.0;pr=0.0;nc=0;ga=False;py=np.zeros(N)
    for t in range(TL):
        om=mask_obs(np.array(o),mtype); oj=jnp.array(om)
        mean,_,_=model(oj.reshape(1,-1)); raw_a=np.clip(np.array(mean.reshape(N)),-AB,AB)
        pt=phi_arr[t];vt=v_arr[t];dphi=min(abs(pt-270),360-abs(pt-270))
        th=20 if ga else 15;ga=(dphi<th)and(vt<11.4)
        a=np.where(ga,raw_a,np.zeros(N));a=np.where(np.max(np.abs(a))<2.0,np.zeros(N),a)
        a=np.clip(a,py-30,py+30)
        ns,no,_,_=env_step(s,jnp.array(a),pj,max_steps=TL+10)
        i0=np.array(inflow_speeds_jax(pj,jnp.array(phi_arr[t]),jnp.array(v_arr[t]),jnp.zeros(N)))
        p0=np.sum(np.array(power_output_jax(jnp.array(i0),jnp.zeros(N))))/1e6
        iy=np.array(inflow_speeds_jax(pj,jnp.array(phi_arr[t]),jnp.array(v_arr[t]),np.array(ns.gammas)))
        pw=np.sum(np.array(power_output_jax(jnp.array(iy),np.array(ns.gammas))))/1e6
        g=(pw-p0)/(p0+1e-6)*100;tg+=g;tt+=np.sum(np.abs(a-py));pr=max(pr,np.sum(np.abs(a-py))/10.0)
        if g<0:nc+=1;py=a.copy();s=ns;o=no
    return tg/TL,tt/N,pr,nc/TL

print("{:15s} {:>10s} {:>8s} {:>7s}".format("Ablation","Gain","vsFull","Travel"))
results=[]
for m in ['full_j15','last3','mask_10_12','shuffle']:
    print("{:15s}...".format(m),end=' ',flush=True);t0=time.time()
    G=[];T=[];Pk=[];Ng=[]
    for ti in range(NT):
        g,t,p,n=et(model,P[ti],V[ti],m);G.append(g);T.append(t);Pk.append(p);Ng.append(n)
    r=dict(mask=m,gain=float(np.mean(G)),travel=float(np.mean(T)),neg=float(np.mean(Ng)))
    results.append(r)
    base=results[0]['gain'];d=r['gain']-base
    print("{:>+9.3f} {:>+7.3f} {:>6.0f} ({:.0f}s)".format(r['gain'],d,r['travel'],time.time()-t0))

with open('../results/j15_causal_ablation.json','w') as f: json.dump(results,f,indent=2)
print("\nSaved.")
