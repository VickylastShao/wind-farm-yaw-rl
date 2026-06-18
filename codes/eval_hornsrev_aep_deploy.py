#!/usr/bin/env python3
"""
Experiment 5: Horns Rev AEP Annualized Replay — Vmap-Accelerated Version
=========================================================================
Fast version using jax.vmap + lax.scan for parallel evaluation.
All conditions evaluated simultaneously on GPU.
"""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("USE_POSITIONS", "1")

import jax, jax.numpy as jnp
import numpy as np
from scipy.special import gammaln

from flax import nnx
from train_3x3_nnx import ActorCritic
from windfarm_env_jax import (env_reset, env_step, positions_to_jax,
                              inflow_speeds_jax, power_output_jax)
from windfarm_env import create_wind_farm_layout_3x3, U_infinity

# ── Config ──────────────────────────────────────────────────────────────────
CKPT_DIR = "checkpoints_3x3_nnx_jaxenv"
N=9; AB=10.0; T_STEP=10.0
GATE_IN=15.0; GATE_OUT=20.0; DEADBAND=2.0; RATE_MAX=3.0
SLSQP_RATE=0.5
WEIBULL_K=2.1; WEIBULL_A=10.5; VM_MU=270.0; VM_KAPPA=2.0
FARM_MW=500.0; CF=0.45
BASELINE_AEP_MWH = FARM_MW * CF * 8760
N_COND=5000; SETTLE=10; BATCH=500
OBS_J3=144; OBS_J15=720

pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos)

# ── Wind sampling ──────────────────────────────────────────────────────────
def sample_wind(n, seed=42):
    rng=np.random.default_rng(seed)
    v=WEIBULL_A*rng.weibull(WEIBULL_K,n); v=np.clip(v,4.0,25.0)
    phi=rng.vonmises(np.deg2rad(VM_MU),VM_KAPPA,n)
    phi=np.rad2deg(phi)%360
    return phi.astype(np.float32), v.astype(np.float32)

# ── Load models ────────────────────────────────────────────────────────────
def load_models(tag, od, n=5):
    ms=[]
    for s in range(n):
        cp=os.path.join(CKPT_DIR,f'policy_seed{s}_{tag}.pkl')
        if not os.path.exists(cp): continue
        m=ActorCritic(od,N,rngs=nnx.Rngs(0)); gd,_=nnx.split(m)
        with open(cp,'rb') as f: ms.append(nnx.merge(gd,pickle.load(f)))
    return ms

# ── Vmap-batched DRL-Deploy settling evaluation ────────────────────────────
def _make_eval_batch(j_hist):
    """Create a jit'd evaluation function for a specific J (static)."""
    @jax.jit
    def _eval(model_params, gd, phis, vs):
        B = phis.shape[0]
        keys = jax.random.split(jax.random.key(0), B)

        def _single_cond(key, phi, v):
            s, o = env_reset(key, pj, j=j_hist,
                             specific_wind_dir=phi, specific_wind_speed=v,
                             randomize_wind=False, max_steps=SETTLE+5)

            def _step(carry, _):
                st, ob, in_gate, py, cum_travel, peak_rate = carry
                dphi = jnp.minimum(jnp.abs(phi - 270), 360 - jnp.abs(phi - 270))
                threshold = jnp.where(in_gate, GATE_OUT, GATE_IN)
                in_gate_new = (dphi < threshold) & (v < U_infinity)

                model = nnx.merge(gd, model_params)
                mean, _, _ = model(ob.reshape(1, -1))
                raw_a = jnp.clip(mean.reshape(N), -AB, AB)
                a = jnp.where(in_gate_new, raw_a, jnp.zeros(N))
                a = jnp.where((DEADBAND>=0.5)&(jnp.max(jnp.abs(a))<DEADBAND), jnp.zeros(N), a)
                max_step = RATE_MAX * T_STEP
                a = jnp.clip(a, py-st.gammas-max_step, py-st.gammas+max_step)
                ns, no, _, _ = env_step(st, a, pj, max_steps=SETTLE+5)
                travel = jnp.sum(jnp.abs(a))
                return (ns, no, in_gate_new, ns.gammas, cum_travel+travel,
                        jnp.maximum(peak_rate, travel/T_STEP)), None

            init = (s, o, jnp.array(False), jnp.zeros(N), jnp.array(0.0), jnp.array(0.0))
            (final_st, _, _, _, total_travel, peak_rate), _ = jax.lax.scan(
                _step, init, jnp.arange(SETTLE)
            )
            inflow = inflow_speeds_jax(pj, phi, v, final_st.gammas)
            pwr = jnp.sum(power_output_jax(inflow, final_st.gammas)) / 1e6
            inflow0 = inflow_speeds_jax(pj, phi, v, jnp.zeros(N))
            pwr0 = jnp.sum(power_output_jax(inflow0, jnp.zeros(N))) / 1e6
            gain = (pwr - pwr0) / (pwr0 + 1e-8) * 100
            return gain, total_travel, peak_rate

        gains, travels, peaks = jax.vmap(_single_cond)(keys, phis, vs)
        return gains, travels, peaks
    return _eval

# Pre-compile for J=3 and J=15
_eval_batch_j3 = _make_eval_batch(3)
_eval_batch_j15 = _make_eval_batch(15)

def eval_settle_all(model, phis, vs, j_hist):
    """Evaluate all conditions in batches."""
    ef = _eval_batch_j3 if j_hist == 3 else _eval_batch_j15
    n = len(phis)
    all_gains = []; all_travels = []; all_peaks = []
    gd, model_params = nnx.split(model)
    for i in range(0, n, BATCH):
        end = min(i+BATCH, n)
        g, t, p = ef(model_params, gd,
                     jnp.array(phis[i:end]),
                     jnp.array(vs[i:end]))
        all_gains.append(np.array(g))
        all_travels.append(np.array(t))
        all_peaks.append(np.array(p))
    gains = np.concatenate(all_gains)
    travels = np.concatenate(all_travels)
    peaks = np.concatenate(all_peaks)
    return gains, travels, peaks

# ── SLSQP evaluation (numpy, not vmap-able, use subset) ────────────────────
def load_slsqp_lookup():
    base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base,'latex_draft/figures/lookup_table_baseline.json')) as f:
        lt=json.load(f)
    pg=np.array(lt['phi_grid'],dtype=np.float32)
    vg=np.array(lt['v_grid'],dtype=np.float32)
    yt=np.load(os.path.join(base,'latex_draft/figures/lookup_table_yaw.npy')).astype(np.float32)
    return pg,vg,yt

def eval_slsqp_batch(phis, vs, phi_g, v_g, yaw_t, rate_limit=None):
    from windfarm_env import calculate_inflow_speeds
    nc=len(phis); gains=np.zeros(nc,dtype=np.float32); travels=np.zeros(nc,dtype=np.float32)
    peaks=np.zeros(nc,dtype=np.float32); neg=0; cy=np.zeros(N,dtype=np.float32)
    for ci in range(nc):
        pi=float(phis[ci]); vi=float(vs[ci])
        i=np.argmin(np.abs(phi_g-pi)); j=np.argmin(np.abs(v_g-vi))
        oy=yaw_t[i,j].copy()
        ms=(rate_limit if rate_limit else 50.0)*T_STEP
        oy=np.clip(oy,cy-ms,cy+ms)
        inf=calculate_inflow_speeds(pos,pi,0.8,0.065,126.0,U_infinity,oy,2.727630853,0.1,0.53991)
        pw=float(jnp.sum(power_output_jax(jnp.array(inf),jnp.array(oy)))/1e6)
        inf0=calculate_inflow_speeds(pos,pi,0.8,0.065,126.0,U_infinity,np.zeros(N),2.727630853,0.1,0.53991)
        pw0=float(jnp.sum(power_output_jax(jnp.array(inf0),jnp.zeros(N)))/1e6)
        gains[ci]=(pw-pw0)/(pw0+1e-8)*100
        tr=np.sum(np.abs(oy-cy)); travels[ci]=tr; peaks[ci]=tr/T_STEP
        if gains[ci]<0: neg+=1
        cy=oy.copy()
    return gains,travels,peaks,neg

# ── Metrics ────────────────────────────────────────────────────────────────
def metrics(gains, travels, peaks, neg, nc):
    mg=float(np.mean(gains)); mt=float(np.mean(travels))
    cpyr=8760*3600/T_STEP/SETTLE
    at=mt*cpyr/1e6; aep=mg/100*BASELINE_AEP_MWH
    gpt=mg/(mt+1e-8); nf=neg/nc
    return {'aep_gain_pct':mg,'aep_gain_mwh_per_yr':aep,'annual_travel_mdeg':at,
            'mean_travel_per_cond_deg':mt,'peak_rate_deg_per_s':float(np.max(peaks)),
            'neg_frac':nf,'gain_per_travel':gpt,
            'load_proxy_B':float(np.sum(travels**2)),'load_proxy_C':float(np.sum(travels))}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    print("="*85)
    print("  Exp 5: Horns Rev AEP — Vmap-Accelerated")
    print("="*85)

    print(f"\n[1] Sampling {N_COND} i.i.d. conditions...")
    phis,vs=sample_wind(N_COND)
    dphi=np.minimum(np.abs(phis-270),360-np.abs(phis-270))
    ac=float(np.mean(dphi<=5)); na=float(np.mean((dphi>5)&(dphi<=15)))
    print(f"    AC={ac*100:.2f}%, NA={na*100:.2f}%, In-gate={np.mean((dphi<15)&(vs<U_infinity))*100:.1f}%")
    print(f"    Mean ws={np.mean(vs):.2f}, Farm={FARM_MW}MW, CF={CF}, BaseAEP={BASELINE_AEP_MWH:,.0f}MWh/yr")

    print("\n[2] Loading models...")
    ms=load_models('sens_act10',OBS_J3)
    print(f"    Static: {len(ms)} seeds")
    mj15=load_models('dyn_J15_l5e4',OBS_J15,min(5,3))
    print(f"    J=15: {len(mj15)} seeds")

    print(f"\n[3] Vmap-batched settling eval ({SETTLE} steps × batches of {BATCH})...")
    results={}
    results['Greedy yaw']={'aep_gain_pct':0.0,'aep_gain_mwh_per_yr':0.0,
        'annual_travel_mdeg':0.0,'mean_travel_per_cond_deg':0.0,
        'peak_rate_deg_per_s':0.0,'neg_frac':0.0,'gain_per_travel':float('inf'),
        'load_proxy_B':0.0,'load_proxy_C':0.0}

    # Static DRL-Deploy
    print("    Static DRL-Deploy...",end="",flush=True)
    t0=time.time()
    g0,t0_arr,p0=eval_settle_all(ms[0],phis,vs,3)
    jax.block_until_ready((g0,t0_arr,p0))
    sd=metrics(g0,t0_arr,p0,int(np.sum(g0<0)),N_COND)
    # Multi-seed
    sg=[]; st=[]
    for si,m in enumerate(ms):
        g,t,p=eval_settle_all(m,phis,vs,3); sg.append(float(np.mean(g))); st.append(float(np.mean(t)))
    sd['seed_gains']=sg; sd['seed_travels']=st; sd['eval_time_s']=time.time()-t0
    results['Static DRL-Deploy']=sd
    jax.block_until_ready(True)
    print(f"\r    {'Static DRL-Deploy':<28s} {sd['aep_gain_pct']:>+8.4f}% "
          f"{sd['annual_travel_mdeg']:>10.3f}Mdeg G/T={sd['gain_per_travel']:>9.4f} "
          f"pk={sd['peak_rate_deg_per_s']:>5.2f} neg={sd['neg_frac']*100:>4.1f}% "
          f"[{sd['eval_time_s']:.0f}s]")

    # J=15 DRL-Deploy
    if mj15:
        print("    J=15 DRL-Deploy...",end="",flush=True)
        t0=time.time()
        g0j,t0j,p0j=eval_settle_all(mj15[0],phis,vs,15)
        jd=metrics(g0j,t0j,p0j,int(np.sum(g0j<0)),N_COND)
        sgj=[]; stj=[]
        for si,m in enumerate(mj15):
            g,t,p=eval_settle_all(m,phis,vs,15); sgj.append(float(np.mean(g))); stj.append(float(np.mean(t)))
        jd['seed_gains']=sgj; jd['seed_travels']=stj; jd['eval_time_s']=time.time()-t0
        results['J=15 DRL-Deploy']=jd
        jax.block_until_ready(True)
        print(f"\r    {'J=15 DRL-Deploy':<28s} {jd['aep_gain_pct']:>+8.4f}% "
              f"{jd['annual_travel_mdeg']:>10.3f}Mdeg G/T={jd['gain_per_travel']:>9.4f} "
              f"pk={jd['peak_rate_deg_per_s']:>5.2f} neg={jd['neg_frac']*100:>4.1f}% "
              f"[{jd['eval_time_s']:.0f}s]")

    # SLSQP
    pg,vg,yt=load_slsqp_lookup()
    ns=min(N_COND,2000); print(f"    SLSQP on {ns} conds...",end="",flush=True)
    t0=time.time()
    gr,tr,pr,nr=eval_slsqp_batch(phis[:ns],vs[:ns],pg,vg,yt,rate_limit=SLSQP_RATE)
    rm=metrics(gr,tr,pr,nr,ns); rm['eval_time_s']=time.time()-t0; rm['rate_limit']=SLSQP_RATE
    results['SLSQP RL=0.5°/s']=rm
    print(f"\r    {'SLSQP RL=0.5°/s':<28s} {rm['aep_gain_pct']:>+8.4f}% "
          f"{rm['annual_travel_mdeg']:>10.3f}Mdeg G/T={rm['gain_per_travel']:>9.4f} "
          f"pk={rm['peak_rate_deg_per_s']:>5.2f} neg={rm['neg_frac']*100:>4.1f}%")

    t0=time.time()
    gu,tu,pu,nu=eval_slsqp_batch(phis[:ns],vs[:ns],pg,vg,yt,rate_limit=None)
    um=metrics(gu,tu,pu,nu,ns); um['eval_time_s']=time.time()-t0
    results['SLSQP Unlimited']=um
    print(f"    {'SLSQP Unlimited':<28s} {um['aep_gain_pct']:>+8.4f}% "
          f"{um['annual_travel_mdeg']:>10.3f}Mdeg G/T={um['gain_per_travel']:>9.4f} "
          f"pk={um['peak_rate_deg_per_s']:>5.2f} neg={um['neg_frac']*100:>4.1f}%")

    # ── Sensitivity ──────────────────────────────────────────────────────
    print("\n[4] Sensitivity (Static Deploy, seed 0, n=2000 each)...")
    sens={}
    for kappa in [1.0,2.0,4.0,8.0]:
        rng=np.random.default_rng(99); ns2=2000
        vs2=WEIBULL_A*rng.weibull(WEIBULL_K,ns2); vs2=np.clip(vs2,4.0,25.0)
        phi2=rng.vonmises(np.deg2rad(VM_MU),kappa,ns2); phi2=np.rad2deg(phi2)%360
        g2,t2,p2=eval_settle_all(ms[0],phi2,vs2,3)
        dp2=np.minimum(np.abs(phi2-270),360-np.abs(phi2-270))
        m2=metrics(g2,t2,p2,int(np.sum(g2<0)),ns2)
        m2['kappa']=kappa; m2['ac_prob']=float(np.mean(dp2<=5)); sens[f'kappa_{kappa:.1f}']=m2
        print(f"    κ={kappa:.1f}: AC={m2['ac_prob']*100:.1f}% AEP={m2['aep_gain_pct']:+.4f}%")

    for bias in [-5,-2,0,2,5]:
        rng=np.random.default_rng(99); ns2=2000
        vs2=WEIBULL_A*rng.weibull(WEIBULL_K,ns2); vs2=np.clip(vs2,4.0,25.0)
        phi2=rng.vonmises(np.deg2rad(VM_MU+bias),VM_KAPPA,ns2); phi2=np.rad2deg(phi2)%360
        g2,t2,p2=eval_settle_all(ms[0],phi2,vs2,3)
        dp2=np.minimum(np.abs(phi2-(270+bias)),360-np.abs(phi2-(270+bias)))
        m2=metrics(g2,t2,p2,int(np.sum(g2<0)),ns2)
        m2['bias_deg']=bias; m2['ac_prob']=float(np.mean(dp2<=5)); sens[f'bias_{bias:+d}']=m2
        print(f"    bias={bias:+d}°: AC={m2['ac_prob']*100:.1f}% AEP={m2['aep_gain_pct']:+.4f}%")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n"+"="*85)
    print("  FINAL TABLE")
    print("="*85)
    print(f"  Wind: Weibull k={WEIBULL_K}, A={WEIBULL_A}, vonMises μ={VM_MU}°, κ={VM_KAPPA}")
    print(f"  AC={ac*100:.1f}%, Baseline AEP={BASELINE_AEP_MWH:,.0f}MWh/yr")
    print(f"  DRL-Deploy: Gate[{GATE_IN}°/{GATE_OUT}°] DB={DEADBAND}° RL={RATE_MAX}°/s")
    print(f"  {'Controller':<28s} {'AEP gain':>9s} {'Travel/yr':>11s} {'Gain/trav':>10s}")
    print("  "+"-"*65)
    for cn,cd in results.items():
        gpt=cd.get('gain_per_travel',0)
        gs=f'{gpt:>9.4f}' if abs(gpt)<1e6 else '—'
        print(f"  {cn:<28s} {cd['aep_gain_pct']:>+8.4f}% {cd['annual_travel_mdeg']:>10.3f}Mdeg {gs}")

    print(f"\n  Sensitivity κ:")
    for key in sorted(sens.keys()):
        if key.startswith('kappa'):
            v=sens[key]; print(f"    κ={v['kappa']:.1f}: AC={v['ac_prob']*100:.1f}% AEP={v['aep_gain_pct']:+.4f}%")

    print(f"  Sensitivity bias:")
    for key in sorted(sens.keys()):
        if key.startswith('bias'):
            v=sens[key]; print(f"    bias={v['bias_deg']:+d}°: AEP={v['aep_gain_pct']:+.4f}%")

    # ── Save ─────────────────────────────────────────────────────────────
    def _safe(obj):
        if isinstance(obj,(np.floating,np.float32,np.float64)):
            v=float(obj); return None if(np.isnan(v)or np.isinf(v))else v
        if isinstance(obj,(np.integer,np.int32,np.int64)): return int(obj)
        if isinstance(obj,dict): return{str(k):_safe(v)for k,v in obj.items()}
        if isinstance(obj,(list,tuple,np.ndarray)): return[_safe(x)for x in obj]
        return obj

    out=_safe({'description':'Horns Rev AEP replay — vmap-accelerated',
        'wind_rose':{'source':'Horns Rev 1 (Barthelmie 2009)','weibull_k':WEIBULL_K,
            'weibull_A':WEIBULL_A,'vm_mu':VM_MU,'vm_kappa':VM_KAPPA,
            'mean_ws':float(np.mean(vs)),'ac_prob':ac,'na_prob':na,
            'farm_mw':FARM_MW,'cf':CF,'baseline_aep_mwh':BASELINE_AEP_MWH},
        'deploy':{'gate_in':GATE_IN,'gate_out':GATE_OUT,'deadband':DEADBAND,
            'rate_max':RATE_MAX,'slsqp_rate':SLSQP_RATE,'t_step':T_STEP,'settle':SETTLE},
        'controllers':results,'sensitivity':sens})
    with open('../results/hornsrev_aep_deploy_final.json','w') as f: json.dump(out,f,indent=2)
    print(f"\n✅ Saved hornsrev_aep_deploy_final.json")
    print("Done.")
