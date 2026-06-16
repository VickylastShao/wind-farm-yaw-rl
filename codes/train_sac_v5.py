#!/usr/bin/env python3
"""SAC V5 — Inline @jax.jit loss functions (proven pattern)."""
import sys, os, json, time, pickle
import numpy as np; import jax, jax.numpy as jnp, optax
from flax import nnx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_3x3_nnx import MLP, NET_ARCH
from windfarm_env_jax import env_reset_batched, env_step_autoreset, positions_to_jax
from windfarm_env import create_wind_farm_layout_3x3

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_sac_jaxenv_v5")
os.makedirs(CKPT_DIR, exist_ok=True)

N_SEEDS=int(os.environ.get("N_SEEDS",5)); N_ENVS=int(os.environ.get("N_ENVS",4))
TOTAL_STEPS=int(float(os.environ.get("TOTAL_STEPS",60_000_000)))
N_STEPS=int(os.environ.get("N_STEPS",256)); BATCH_SIZE=int(os.environ.get("BATCH_SIZE",256))
N_EPOCHS=int(os.environ.get("N_EPOCHS",1)); MAX_EPISODE_STEPS=int(os.environ.get("MAX_EPISODE_STEPS",200))
GAMMA=float(os.environ.get("GAMMA","0.99")); TAU=float(os.environ.get("TAU","0.005"))
ACTOR_LR=float(os.environ.get("ACTOR_LR","3e-4")); CRITIC_LR=float(os.environ.get("CRITIC_LR","3e-4"))
ALPHA_LR=float(os.environ.get("ALPHA_LR","3e-4")); REPLAY_SIZE=int(os.environ.get("REPLAY_SIZE",500000))
ACT_BOUND=float(os.environ.get("ACT_BOUND","10.0")); J_HIST=int(os.environ.get("J",3))

_WMR=os.environ.get("WIND_MIXTURE",""); WIND_MIXTURE=None
if _WMR:
    p=[float(x.strip()) for x in _WMR.split(",")]
    if len(p)==3: WIND_MIXTURE=tuple(p)

SLSQP_LOOKUP=None
if os.environ.get("USE_REGRET","1")=="1":
    import json as _json
    _lt=os.path.join(os.path.dirname(_SCRIPT_DIR),"latex_draft","figures","lookup_table_baseline.json")
    if os.path.exists(_lt):
        with open(_lt) as _f:
            _d=_json.load(_f)
        SLSQP_LOOKUP=(jnp.asarray(_d["phi_grid"],jnp.float32),jnp.asarray(_d["v_grid"],jnp.float32),jnp.asarray(_d["gain_table"],jnp.float32))

class Actor(nnx.Module):
    def __init__(self,od,ad,rngs): self.net=MLP(od,NET_ARCH,ad,rngs=rngs); self.log_std=nnx.Param(jnp.full((ad,),-0.5))
    def sample(self,obs,key):
        mean=self.net(obs); ls=jnp.broadcast_to(jnp.clip(self.log_std[...],-20.,2.),mean.shape)
        eps=jax.random.normal(key,mean.shape); z=mean+jnp.exp(ls)*eps; u=jnp.tanh(z); a=u*ACT_BOUND
        lp=(-0.5*(eps**2+jnp.log(2*jnp.pi))-ls).sum(-1); return a,lp-jnp.log(1-u**2+1e-6).sum(-1)
class Critic(nnx.Module):
    def __init__(self,od,ad,rngs): self.q1=MLP(od+ad,NET_ARCH,1,rngs=rngs); self.q2=MLP(od+ad,NET_ARCH,1,rngs=rngs)
    def __call__(self,obs,act): x=jnp.concatenate([obs,act],-1); return jnp.stack([self.q1(x).squeeze(-1),self.q2(x).squeeze(-1)],-1)
class RB:
    def __init__(self,c,od,ad):
        self.c=c;self.p=0;self._s=0;self.o=np.zeros((c,od),np.float32);self.a=np.zeros((c,ad),np.float32)
        self.r=np.zeros(c,np.float32);self.no=np.zeros((c,od),np.float32);self.d=np.zeros(c,np.float32)
    def add(self,o,a,r,no,d):
        B=o.shape[0]
        if B>=self.c: o,a,r,no,d=o[-self.c:],a[-self.c:],r[-self.c:],no[-self.c:],d[-self.c:];B=o.shape[0]
        e=self.p+B
        if e<=self.c: self.o[self.p:e]=o;self.a[self.p:e]=a;self.r[self.p:e]=r;self.no[self.p:e]=no;self.d[self.p:e]=d
        else:
            r2=self.c-self.p;self.o[self.p:]=o[:r2];self.a[self.p:]=a[:r2];self.r[self.p:]=r[:r2];self.no[self.p:]=no[:r2];self.d[self.p:]=d[:r2]
            self.o[:B-r2]=o[r2:];self.a[:B-r2]=a[r2:];self.r[:B-r2]=r[r2:];self.no[:B-r2]=no[r2:];self.d[:B-r2]=d[r2:]
        self.p=(self.p+B)%self.c;self._s=min(self.c,self._s+B)
    def sample(self,n,rng): idx=rng.integers(0,self._s,size=n); return self.o[idx],self.a[idx],self.r[idx],self.no[idx],self.d[idx]
    def __len__(self): return self._s

def make_rollout(pj):
    def _r(m,s,o,k,ns):
        def b(c,_):
            st,ob,kk=c;kk,sk=jax.random.split(kk);a,_=m.sample(ob,sk);rk=jax.random.split(sk,a.shape[0])
            ns,no,rw,dn=env_step_autoreset(st,a,rk,pj,j=J_HIST,max_steps=MAX_EPISODE_STEPS,randomize_wind=True,wind_mixture=WIND_MIXTURE,slsqp_lookup=SLSQP_LOOKUP)
            return (ns,no,kk),dict(obs=ob,action=a,reward=rw,next_obs=no,done=dn)
        (fs,fo,fk),t=jax.lax.scan(b,(s,o,k),None,length=ns); return fs,fo,fk,t
    return _r

def train_one_seed(seed):
    print(f'\n# SAC-V5 seed={seed} N_ENVS={N_ENVS}')
    pos,_,_=create_wind_farm_layout_3x3(); pj=positions_to_jax(pos); Nt=pj.shape[0]
    od=J_HIST*(3*Nt+3+(2*Nt if os.environ.get("USE_POSITIONS","1")=="1" else 0)); ad=Nt
    ni=max(1,TOTAL_STEPS//(N_STEPS*N_ENVS))
    if ni*N_STEPS*N_ENVS<TOTAL_STEPS: ni+=1
    print(f'  obs_dim={od} act_dim={ad} iterations={ni}')
    key=jax.random.PRNGKey(seed); mk,rk,rok=jax.random.split(key,3)
    actor=Actor(od,ad,rngs=nnx.Rngs(int(mk[0])))
    critic=Critic(od,ad,rngs=nnx.Rngs(int(mk[0])+1))
    tcritic=Critic(od,ad,rngs=nnx.Rngs(int(mk[0])+2))

    # Inline loss functions (proven pattern: captures AGD/CGD/TGD from enclosing scope)
    AGD,_=nnx.split(actor); CGD,_=nnx.split(critic); TGD,_=nnx.split(tcritic)
    _,ast=nnx.split(actor); _,cst=nnx.split(critic); tst=cst

    @jax.jit
    def cl(cst,ast,tst,alpha,ob,ab,rb,nob,db,key):
        a=nnx.merge(AGD,ast); t=nnx.merge(TGD,tst); c=nnx.merge(CGD,cst)
        qv=c(ob,ab); na,nlp=a.sample(nob,key); tq=t(nob,na)
        y=rb+GAMMA*(1-db)*(tq.min(-1)-alpha*nlp); return ((qv-y[:,None])**2).mean()
    @jax.jit
    def al(ast,cst,alpha,ob,key):
        a=nnx.merge(AGD,ast); c=nnx.merge(CGD,cst)
        acts,lps=a.sample(ob,key); qm=c(ob,acts).min(-1)
        return (alpha*lps-qm).mean(),jnp.mean(lps)

    co=optax.sgd(CRITIC_LR); cos=co.init(cst)
    ao=optax.sgd(ACTOR_LR); aos=ao.init(ast)
    la=jnp.array(jnp.log(0.2)); alo=optax.sgd(ALPHA_LR); alos=alo.init(la)
    te=-ad; rb=RB(REPLAY_SIZE,od,ad); npr=np.random.default_rng(seed)

    rkeys=jax.random.split(rk,N_ENVS)
    s,obs=jax.jit(env_reset_batched,static_argnames=("j","max_steps","randomize_wind","wind_mixture"))(
        rkeys,pj,j=J_HIST,max_steps=MAX_EPISODE_STEPS,randomize_wind=True,wind_mixture=WIND_MIXTURE,slsqp_lookup=SLSQP_LOOKUP)
    rf=make_rollout(pj); rj=nnx.jit(rf,static_argnums=(4,))
    print('  [compile...]',end='',flush=True)
    s,obs,rok,_=rj(actor,s,obs,rok,N_STEPS); jax.block_until_ready(s.gammas)
    _do=jnp.zeros((BATCH_SIZE,od)); _da=jnp.zeros((BATCH_SIZE,ad))
    _dr=jnp.zeros(BATCH_SIZE); _dd=jnp.zeros(BATCH_SIZE); _dk=jax.random.PRNGKey(0)
    cg=jax.grad(cl); ag=jax.grad(lambda *a,**kw: al(*a,**kw)[0])
    _=cg(cst,ast,tst,0.2,_do,_da,_dr,_do,_dd,_dk); _=ag(ast,cst,0.2,_do,_dk)
    jax.block_until_ready(True); print('done')

    ts=0; t0=time.time(); er=[]; rr=np.zeros(N_ENVS,np.float32)
    try:
        for it in range(ni):
            s,obs,rok,tj=rj(actor,s,obs,rok,N_STEPS); jax.block_until_ready(tj["reward"])
            th=jax.tree.map(np.asarray,tj); T=N_STEPS
            for t in range(T):
                rb.add(th["obs"][t],th["action"][t],th["reward"][t],th["next_obs"][t],th["done"][t])
                rr+=th["reward"][t]
                for i,d in enumerate(th["done"][t]):
                    if d: er.append(float(rr[i])); rr[i]=0.0
            ts+=T*N_ENVS
            if len(rb)<BATCH_SIZE: continue
            cls,als=[],[]
            for _ in range(N_EPOCHS):
                nu=max(1,(T*N_ENVS)//BATCH_SIZE)
                for _ in range(nu):
                    ob,ac,rw,no,dn=rb.sample(BATCH_SIZE,npr)
                    obj=jnp.asarray(ob); acj=jnp.asarray(ac); rwj=jnp.asarray(rw)
                    noj=jnp.asarray(no); dnj=jnp.asarray(dn)
                    alpha=jnp.exp(la); key,ck,ak=jax.random.split(key,3)
                    cu,cos=co.update(cg(cst,ast,tst,alpha,obj,acj,rwj,noj,dnj,ck),cos)
                    cst=optax.apply_updates(cst,cu)
                    au,aos=ao.update(ag(ast,cst,alpha,obj,ak),aos)
                    ast=optax.apply_updates(ast,au)
                    _,lp=al(ast,cst,alpha,obj,ak)
                    alu,alos=alo.update(jax.grad(lambda x:(-jnp.exp(x)*(lp+te)).mean())(la),alos)
                    la=optax.apply_updates(la,alu)
                    tst=jax.tree.map(lambda t,s:TAU*s+(1-TAU)*t,tst,cst)
                    cls.append(float(cl(cst,ast,tst,alpha,obj,acj,rwj,noj,dnj,ck)))
                    als.append(float(al(ast,cst,alpha,obj,ak)[0]))
            actor=nnx.merge(AGD,ast)
            nl=max(1,len(cls)-nu) if cls else 0
            ca=float(np.mean(cls[-nl:])) if nl>0 else 0
            aa=float(np.mean(als[-nl:])) if nl>0 else 0
            ea=float(np.mean(er[-10:])) if er else 0
            if (it+1)%max(1,ni//20)==0:
                fps=ts/max(time.time()-t0,1e-6)
                print(f'  iter {it+1:5d}/{ni}  steps {ts/1e6:.1f}M  ep_ret {ea:+.2f}  '
                      f'alpha {alpha:.3f}  c_loss {ca:.4f}  a_loss {aa:.4f}  fps {fps:.0f}  rb {len(rb)}')
    except KeyboardInterrupt: print('\nEarly stop')
    am=nnx.merge(AGD,ast); cm=nnx.merge(CGD,cst)
    _,as2=nnx.split(am); _,cs2=nnx.split(cm)
    with open(os.path.join(CKPT_DIR,f"policy_seed{seed}.pkl"),"wb") as f: pickle.dump(as2,f)
    with open(os.path.join(CKPT_DIR,f"critic_seed{seed}.pkl"),"wb") as f: pickle.dump(cs2,f)
    r=dict(seed=seed,total_steps=int(ts),ep_returns=er,train_time=time.time()-t0)
    with open(os.path.join(CKPT_DIR,f"metrics_seed{seed}.json"),"w") as f: json.dump(r,f,indent=2)
    print(f'# SAC seed={seed} done: {ts/1e6:.1f}M steps  time={r["train_time"]/60:.1f}min')
    return r

if __name__=="__main__":
    ss,sc=0,N_SEEDS
    if len(sys.argv)>1: ss=int(sys.argv[1])
    if len(sys.argv)>2: sc=int(sys.argv[2])
    for s in range(ss,ss+sc): train_one_seed(s)
    print('# All seeds done')
