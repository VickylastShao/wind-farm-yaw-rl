#!/usr/bin/env python3
"""
SAC training — Pure jax.grad + NNX state. 14ms per update step.
"""
import sys, os, json, time, pickle
import numpy as np
import jax, jax.numpy as jnp, optax
from flax import nnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_3x3_nnx import MLP, NET_ARCH
from windfarm_env_jax import env_reset_batched, env_step_autoreset, positions_to_jax
from windfarm_env import create_wind_farm_layout_3x3

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_3x3_sac_jaxenv_v4")
os.makedirs(CKPT_DIR, exist_ok=True)

# ---- Config (env vars) ----
N_SEEDS = int(os.environ.get("N_SEEDS", 5))
N_ENVS = int(os.environ.get("N_ENVS", 8))
TOTAL_STEPS = int(float(os.environ.get("TOTAL_STEPS", 60_000_000)))
N_STEPS = int(os.environ.get("N_STEPS", 256))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 512))
N_EPOCHS = int(os.environ.get("N_EPOCHS", 2))
MAX_EPISODE_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", 200))
GAMMA = float(os.environ.get("GAMMA", "0.99"))
TAU = float(os.environ.get("TAU", "0.005"))
ACTOR_LR = float(os.environ.get("ACTOR_LR", "3e-4"))
CRITIC_LR = float(os.environ.get("CRITIC_LR", "3e-4"))
ALPHA_LR = float(os.environ.get("ALPHA_LR", "3e-4"))
REPLAY_SIZE = int(os.environ.get("REPLAY_SIZE", "500000"))
INITIAL_ALPHA = float(os.environ.get("ALPHA", "0.2"))
ACT_BOUND = float(os.environ.get("ACT_BOUND", "10.0"))
J_HIST = int(os.environ.get("J", 3))
LR_DECAY = os.environ.get("LR_DECAY", "1") == "1"
LR_END = float(os.environ.get("LR_END", "3e-5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))

_WIND_MIX_RAW = os.environ.get("WIND_MIXTURE", "")
WIND_MIXTURE = None
if _WIND_MIX_RAW:
    parts = [float(x.strip()) for x in _WIND_MIX_RAW.split(",")]
    if len(parts) == 3: WIND_MIXTURE = tuple(parts)

USE_REGRET = os.environ.get("USE_REGRET", "1") == "1"
SLSQP_LOOKUP = None
if USE_REGRET:
    import json as _json
    _lt_path = os.path.join(os.path.dirname(_SCRIPT_DIR), "latex_draft", "figures", "lookup_table_baseline.json")
    if os.path.exists(_lt_path):
        with open(_lt_path) as _f:
            _lt = _json.load(_f)
        SLSQP_LOOKUP = (jnp.asarray(_lt["phi_grid"], jnp.float32),
                        jnp.asarray(_lt["v_grid"], jnp.float32),
                        jnp.asarray(_lt["gain_table"], jnp.float32))
        print("# regret reward : SLSQP lookup loaded")


# ---- Networks ----
class Actor(nnx.Module):
    def __init__(self, obs_dim, act_dim, rngs):
        self.net = MLP(obs_dim, NET_ARCH, act_dim, rngs=rngs)
        self.log_std = nnx.Param(jnp.full((act_dim,), -0.5))
    def sample(self, obs, key):
        mean = self.net(obs)
        log_std = jnp.broadcast_to(jnp.clip(self.log_std[...], -20., 2.), mean.shape)
        eps = jax.random.normal(key, mean.shape)
        z = mean + jnp.exp(log_std) * eps
        u = jnp.tanh(z); a = u * ACT_BOUND
        lp_z = (-0.5*(eps**2+jnp.log(2*jnp.pi))-log_std).sum(-1)
        return a, lp_z - jnp.log(1-u**2+1e-6).sum(-1)

class Critic(nnx.Module):
    def __init__(self, obs_dim, act_dim, rngs):
        self.q1 = MLP(obs_dim+act_dim, NET_ARCH, 1, rngs=rngs)
        self.q2 = MLP(obs_dim+act_dim, NET_ARCH, 1, rngs=rngs)
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], -1)
        return jnp.stack([self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)], -1)


# ---- Replay Buffer ----
class ReplayBuffer:
    def __init__(self, cap, od, ad):
        self.cap=cap; self.ptr=0; self._s=0
        self.obs=np.zeros((cap,od),np.float32); self.act=np.zeros((cap,ad),np.float32)
        self.rew=np.zeros(cap,np.float32); self.nobs=np.zeros((cap,od),np.float32)
        self.done=np.zeros(cap,np.float32)
    def add(self,o,a,r,no,d):
        B=o.shape[0]
        if B>=self.cap: o,a,r,no,d=o[-self.cap:],a[-self.cap:],r[-self.cap:],no[-self.cap:],d[-self.cap:]; B=o.shape[0]
        e=self.ptr+B
        if e<=self.cap:
            self.obs[self.ptr:e]=o; self.act[self.ptr:e]=a; self.rew[self.ptr:e]=r
            self.nobs[self.ptr:e]=no; self.done[self.ptr:e]=d
        else:
            rem=self.cap-self.ptr; self.obs[self.ptr:]=o[:rem]; self.act[self.ptr:]=a[:rem]
            self.rew[self.ptr:]=r[:rem]; self.nobs[self.ptr:]=no[:rem]; self.done[self.ptr:]=d[:rem]
            self.obs[:B-rem]=o[rem:]; self.act[:B-rem]=a[rem:]; self.rew[:B-rem]=r[rem:]
            self.nobs[:B-rem]=no[rem:]; self.done[:B-rem]=d[rem:]
        self.ptr=(self.ptr+B)%self.cap; self._s=min(self.cap,self._s+B)
    def sample(self,n,rng):
        idx=rng.integers(0,self._s,size=n)
        return self.obs[idx],self.act[idx],self.rew[idx],self.nobs[idx],self.done[idx]
    def __len__(self): return self._s


# ---- Rollout ----
def make_rollout(positions_j):
    def _rollout(model, state, obs, key, n_steps):
        def body(carry, _):
            st, ob, k = carry
            k, sk = jax.random.split(k)
            a, _ = model.sample(ob, sk)
            rk = jax.random.split(sk, a.shape[0])
            nst, nob, rew, dn = env_step_autoreset(
                st, a, rk, positions_j, j=J_HIST, max_steps=MAX_EPISODE_STEPS,
                randomize_wind=True, wind_mixture=WIND_MIXTURE, slsqp_lookup=SLSQP_LOOKUP)
            return (nst, nob, k), dict(obs=ob, action=a, reward=rew, next_obs=nob, done=dn)
        (fs, fo, fk), traj = jax.lax.scan(body, (state, obs, key), None, length=n_steps)
        return fs, fo, fk, traj
    return _rollout


# ---- Globals for graphdefs (set once per seed) ----
_G_ACTOR_GD = None
_G_CRITIC_GD = None
_G_TCRITIC_GD = None


def _set_graphdefs(actor, critic, tcritic):
    global _G_ACTOR_GD, _G_CRITIC_GD, _G_TCRITIC_GD
    _G_ACTOR_GD, _ = nnx.split(actor)
    _G_CRITIC_GD, _ = nnx.split(critic)
    _G_TCRITIC_GD, _ = nnx.split(tcritic)


@jax.jit
def _critic_loss(critic_st, actor_st, tcritic_st, alpha,
                  obs_b, act_b, rew_b, nobs_b, dones_b, key):
    actor = nnx.merge(_G_ACTOR_GD, actor_st)
    target = nnx.merge(_G_TCRITIC_GD, tcritic_st)
    critic = nnx.merge(_G_CRITIC_GD, critic_st)
    qv = critic(obs_b, act_b)
    na, nlp = actor.sample(nobs_b, key)
    tq = target(nobs_b, na)
    y = rew_b + GAMMA * (1.0 - dones_b) * (tq.min(axis=-1) - alpha * nlp)
    return jnp.mean((qv - y[:, None]) ** 2)


@jax.jit
def _actor_loss(actor_st, critic_st, alpha, obs_b, key):
    actor = nnx.merge(_G_ACTOR_GD, actor_st)
    critic = nnx.merge(_G_CRITIC_GD, critic_st)
    actions, lps = actor.sample(obs_b, key)
    q_min = critic(obs_b, actions).min(axis=-1)
    return jnp.mean(alpha * lps - q_min), jnp.mean(lps)


def train_one_seed(seed: int) -> dict:
    print(f"\n{'='*60}\n# SAC seed={seed}  N_ENVS={N_ENVS}\n{'='*60}")
    pos, _, _ = create_wind_farm_layout_3x3(); pos_j = positions_to_jax(pos)
    Nt = pos_j.shape[0]
    odim = J_HIST * (3*Nt+3 + (2*Nt if os.environ.get("USE_POSITIONS","1")=="1" else 0))
    adim = Nt
    print(f"  obs_dim={odim}  act_dim={adim}")

    n_iter = max(1, TOTAL_STEPS // (N_STEPS * N_ENVS))
    if n_iter * N_STEPS * N_ENVS < TOTAL_STEPS: n_iter += 1
    print(f"  iterations={n_iter}  per_iter={N_STEPS*N_ENVS}  total={n_iter*N_STEPS*N_ENVS}")

    key = jax.random.PRNGKey(seed); mk, rk, rok = jax.random.split(key, 3)
    actor = Actor(odim, adim, rngs=nnx.Rngs(int(mk[0])))
    critic = Critic(odim, adim, rngs=nnx.Rngs(int(mk[0])+1))
    tcritic = Critic(odim, adim, rngs=nnx.Rngs(int(mk[0])+2))
    _set_graphdefs(actor, critic, tcritic)

    _, actor_st = nnx.split(actor)
    _, critic_st = nnx.split(critic)
    tcritic_st = critic_st

    total_updates = n_iter * N_EPOCHS * max(1, (N_STEPS*N_ENVS)//BATCH_SIZE)
    def _mk_opt(lr):
        if LR_DECAY:
            s = optax.cosine_decay_schedule(lr, total_updates, alpha=LR_END/lr)
            a = optax.adamw(s, weight_decay=WEIGHT_DECAY) if WEIGHT_DECAY>0 else optax.adam(s)
        else:
            a = optax.adamw(lr, weight_decay=WEIGHT_DECAY) if WEIGHT_DECAY>0 else optax.adam(lr)
        return a
    c_opt, c_opt_st = _mk_opt(CRITIC_LR), None; c_opt_st = c_opt.init(critic_st)
    a_opt, a_opt_st = _mk_opt(ACTOR_LR), None; a_opt_st = a_opt.init(actor_st)
    log_alpha = jnp.array(jnp.log(INITIAL_ALPHA))
    al_opt, al_opt_st = optax.adam(ALPHA_LR), optax.adam(ALPHA_LR).init(log_alpha)
    target_entropy = -adim

    replay = ReplayBuffer(REPLAY_SIZE, odim, adim)
    np_rng = np.random.default_rng(seed)

    rkeys = jax.random.split(rk, N_ENVS)
    state, obs = jax.jit(env_reset_batched, static_argnames=("j","max_steps","randomize_wind","wind_mixture"))(
        rkeys, pos_j, j=J_HIST, max_steps=MAX_EPISODE_STEPS, randomize_wind=True,
        wind_mixture=WIND_MIXTURE, slsqp_lookup=SLSQP_LOOKUP)

    rollout_fn = make_rollout(pos_j); rollout_jit = nnx.jit(rollout_fn, static_argnums=(4,))
    print("  [compile...]", end="", flush=True)
    state, obs, rok, _ = rollout_jit(actor, state, obs, rok, N_STEPS)
    jax.block_until_ready(state.gammas)
    # Compile grads
    _d_o=jnp.zeros((BATCH_SIZE,odim)); _d_a=jnp.zeros((BATCH_SIZE,adim))
    _d_r=jnp.zeros(BATCH_SIZE); _d_d=jnp.zeros(BATCH_SIZE); _d_k=jax.random.PRNGKey(0)
    _cg = jax.grad(_critic_loss)
    _ag = jax.grad(lambda *a,**kw: _actor_loss(*a,**kw)[0])
    _=_cg(critic_st,actor_st,tcritic_st,0.2,_d_o,_d_a,_d_r,_d_o,_d_d,_d_k)
    _=_ag(actor_st,critic_st,0.2,_d_o,_d_k)
    jax.block_until_ready(True)
    print("done")

    total_steps=0; t0=time.time(); ep_rets=[]; running_ret=np.zeros(N_ENVS,np.float32)
    _critic_grad=_cg; _actor_grad=_ag

    try:
        for it in range(n_iter):
            state,obs,rok,traj=rollout_jit(actor,state,obs,rok,N_STEPS); jax.block_until_ready(traj["reward"])
            traj_h=jax.tree.map(np.asarray,traj); T=N_STEPS
            for t in range(T):
                replay.add(traj_h["obs"][t],traj_h["action"][t],traj_h["reward"][t],
                          traj_h["next_obs"][t],traj_h["done"][t])
                running_ret+=traj_h["reward"][t]
                for i,d in enumerate(traj_h["done"][t]):
                    if d: ep_rets.append(float(running_ret[i])); running_ret[i]=0.0
            total_steps+=T*N_ENVS
            if len(replay)<BATCH_SIZE: continue

            c_losses,a_losses=[],[]
            for _ in range(N_EPOCHS):
                n_up=max(1,(T*N_ENVS)//BATCH_SIZE)
                for _ in range(n_up):
                    ob,ac,rw,no,dn=replay.sample(BATCH_SIZE,np_rng)
                    ob_j=jnp.asarray(ob); ac_j=jnp.asarray(ac); rw_j=jnp.asarray(rw)
                    no_j=jnp.asarray(no); dn_j=jnp.asarray(dn)
                    alpha=jnp.exp(log_alpha); key,ck,ak=jax.random.split(key,3)
                    cg=_critic_grad(critic_st,actor_st,tcritic_st,alpha,ob_j,ac_j,rw_j,no_j,dn_j,ck)
                    c_up,c_opt_st=c_opt.update(cg,c_opt_st,params=critic_st)
                    critic_st=optax.apply_updates(critic_st,c_up)
                    ag=_actor_grad(actor_st,critic_st,alpha,ob_j,ak)
                    a_up,a_opt_st=a_opt.update(ag,a_opt_st,params=actor_st)
                    actor_st=optax.apply_updates(actor_st,a_up)
                    al,lp=_actor_loss(actor_st,critic_st,alpha,ob_j,ak)
                    al_grad=jax.grad(lambda la:(-jnp.exp(la)*(lp+target_entropy)).mean())(log_alpha)
                    al_up,al_opt_st=al_opt.update(al_grad,al_opt_st,params=log_alpha)
                    log_alpha=optax.apply_updates(log_alpha,al_up)
                    tcritic_st=jax.tree.map(lambda t,s:TAU*s+(1-TAU)*t,tcritic_st,critic_st)
                    c_losses.append(float(_critic_loss(critic_st,actor_st,tcritic_st,alpha,ob_j,ac_j,rw_j,no_j,dn_j,ck)))
                    a_losses.append(float(al))
            actor=nnx.merge(_G_ACTOR_GD,actor_st)

            n_l=max(1,len(c_losses)-n_up) if c_losses else 0
            c_avg=float(np.mean(c_losses[-n_l:])) if n_l>0 else 0
            a_avg=float(np.mean(a_losses[-n_l:])) if n_l>0 else 0
            ep_avg=float(np.mean(ep_rets[-10:])) if ep_rets else 0
            if (it+1)%max(1,n_iter//20)==0:
                fps=total_steps/max(time.time()-t0,1e-6)
                print(f"  iter {it+1:5d}/{n_iter}  steps {total_steps/1e6:.1f}M  "
                      f"ep_ret {ep_avg:+.2f}  alpha {alpha:.3f}  "
                      f"c_loss {c_avg:.4f}  a_loss {a_avg:.4f}  fps {fps:.0f}  rb {len(replay)}")
    except KeyboardInterrupt:
        print("\nEarly stop")

    # Save
    actor_m=nnx.merge(_G_ACTOR_GD,actor_st); critic_m=nnx.merge(_G_CRITIC_GD,critic_st)
    _,ast=nnx.split(actor_m); _,cst=nnx.split(critic_m)
    with open(os.path.join(CKPT_DIR,f"policy_seed{seed}.pkl"),"wb") as f: pickle.dump(ast,f)
    with open(os.path.join(CKPT_DIR,f"critic_seed{seed}.pkl"),"wb") as f: pickle.dump(cst,f)
    result=dict(seed=seed,total_steps=int(total_steps),ep_returns=ep_rets,train_time=time.time()-t0)
    with open(os.path.join(CKPT_DIR,f"metrics_seed{seed}.json"),"w") as f: json.dump(result,f,indent=2)
    print(f"# SAC seed={seed} done: {total_steps/1e6:.1f}M steps  time={result['train_time']/60:.1f}min")
    return result


if __name__ == "__main__":
    ss,sc=0,N_SEEDS
    if len(sys.argv)>1: ss=int(sys.argv[1])
    if len(sys.argv)>2: sc=int(sys.argv[2])
    all_r=[train_one_seed(s) for s in range(ss,ss+sc)]
    if len(all_r)>1:
        with open(os.path.join(CKPT_DIR,"summary.json"),"w") as f:
            json.dump(dict(n_seeds=len(all_r),total_steps=[r["total_steps"] for r in all_r],
                          train_times=[r["train_time"] for r in all_r]),f,indent=2)
