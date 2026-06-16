# Expert-Guided DRL for Cooperative Yaw Control

Date: 2026-06-05
Project: JAX-WFCOYAW-RL

## 1. Goal

Reframe the paper from a single PPO-from-scratch controller into an offline-to-online cooperative yaw-control framework:

> Offline SLSQP defines a steady-state optimal yaw-control law under the gray-box wake model. A neural DRL controller learns an amortized, closed-loop approximation of this control law through offline training, while retaining the operational advantages of dynamic incremental yaw control: low latency, smooth yaw trajectories, rate feasibility, and fatigue-aware tuning.

The target is not to claim that the current PPO baseline already matches SLSQP. The target is to demonstrate a credible approximation path:

1. SLSQP provides the per-condition offline optimum and expert target.
2. DRL is trained to approximate the same steady-state optimal-control map.
3. Engineering improvements—JAX acceleration, balanced exploration, regret-normalized rewards, and SLSQP imitation pretraining—progressively close the gap.
4. The resulting policy is a deployable closed-loop controller rather than an online optimizer or static lookup table.

The paper should temporarily avoid discussing action-space variants unrelated to the main comparison. The core story is simply:

> SLSQP is the offline optimizer; DRL is the amortized real-time controller trained to approach it.

## 2. Research Questions

### RQ1: Can DRL approximate the SLSQP optimum map?

For each wind condition \(u=(\phi,v)\), SLSQP defines:

\[
\gamma^\star(u) = \arg\max_{\gamma_i\in[-50^\circ,50^\circ]} P_\mathrm{farm}(\gamma;u).
\]

A closed-loop DRL policy with incremental yaw actions induces, after a fixed settling horizon \(T\), a mapping:

\[
F_\theta(u) = \gamma_T.
\]

The approximation objective is:

\[
P_\mathrm{farm}(F_\theta(u);u) \rightarrow P_\mathrm{farm}(\gamma^\star(u);u).
\]

This frames the problem as policy distillation or amortized optimization: instead of solving a numerical optimization problem online for every wind condition, a neural controller learns the optimizer's input-output structure offline.

### RQ2: Why use a controller instead of online SLSQP or a lookup table?

SLSQP and lookup tables return static yaw targets for each inflow snapshot. Under dynamic wind, the target yaw can change discontinuously, leading to large yaw travel and physically infeasible yaw rates. The DRL controller outputs incremental yaw commands and can include explicit yaw-rate penalties, making it naturally suitable for continuous operation.

### RQ3: What engineering methods help DRL close the SLSQP gap?

The current PPO-from-scratch policy has a measurable steady-state gap relative to SLSQP. The revised framework studies a ladder of increasingly informed training methods:

1. PPO from scratch.
2. PPO with balanced wind-condition sampling.
3. PPO with SLSQP-regret reward shaping.
4. Behavior cloning from SLSQP expert yaw targets.
5. Behavior cloning followed by PPO fine-tuning.
6. Optional yaw-rate regularization for fatigue-aware control.

The expected contribution is not only the final policy, but also a reproducible methodology for turning an offline gray-box optimizer into a real-time DRL controller.

## 3. Experimental Phases

### Phase A — Baseline DRL vs SLSQP Evaluation

Establish a clean baseline comparison between the existing DRL controller and the SLSQP oracle.

Required evaluations:

1. Re-evaluate current PPO checkpoints on the same wind-condition set used for SLSQP comparison.
2. Report marginal mean gain, aligned-cube gain, negative-gain fraction, and recovery relative to SLSQP.
3. Preserve the existing dynamic-wind evaluation as the controller-side comparison.

Outputs:

- `latex_draft/figures/drl_vs_slsqp_regime.json`
- `latex_draft/figures/p0c_eval_randomized.json`
- `latex_draft/figures/dynamic_wind_results.json`

Success criterion:

- Establish the current quantitative gap between PPO-from-scratch and SLSQP.
- Use this gap to motivate expert-guided training rather than to weaken the DRL framing.

### Phase B — SLSQP Expert Dataset

Generate an SLSQP expert dataset for offline training.

Sampling distribution:

- Global: full training range \(\phi\in[173^\circ,353^\circ], v\in[6,16]\).
- Aligned-cube: \(|\phi-270^\circ|<15^\circ, v<11.4\).
- Near-aligned: \(|\phi-270^\circ|<35^\circ\).

Initial dataset size:

- 5k global.
- 5k aligned-cube.
- 5k near-aligned.

This 15k-condition dataset is the first target. Larger datasets can be generated if the training ladder shows benefit.

Stored fields:

```json
{
  "phi": 270.0,
  "v": 10.0,
  "baseline_mw": 0.0,
  "slsqp_opt_mw": 0.0,
  "slsqp_gain_pct": 0.0,
  "opt_gammas": [0.0, 0.0, 0.0]
}
```

Outputs:

- `codes/expert_data/slsqp_3x3_v1.npz`
- `codes/expert_data/slsqp_3x3_v1_summary.json`

Success criterion:

- Dataset generation completes and reproduces existing SLSQP statistics on a held-out subset.
- Expert labels are cached so later DRL training does not repeatedly call SLSQP.

### Phase C — Training Ladder

Train and evaluate the following policies in increasing order of guidance.

#### C0: Current PPO baseline

Existing `p0c` checkpoints.

Purpose:

- Historical baseline.
- Quantifies the gap of PPO-from-scratch relative to SLSQP.

#### C1: PPO from scratch with matched evaluation protocol

Train PPO with the same environment and evaluation protocol used in the SLSQP comparison.

Purpose:

- Ensure the baseline gap is not caused by inconsistent evaluation settings.
- Provide a clean lower bound for later expert-guided methods.

Expected result:

- Similar to the current PPO baseline unless training distribution or reward scaling changes.

#### C2: PPO + balanced wind sampling

Modify wind-condition sampling so rollout resets draw from a mixture:

- 50% aligned-cube.
- 30% near-aligned.
- 20% global.

Purpose:

- Reduce reward dilution from wind conditions where the SLSQP headroom is tiny.
- Increase training signal in regimes where cooperative yaw control has meaningful value.

Expected result:

- Improved aligned-cube gain.
- Possible change in full-range negative-gain fraction; this must be measured.

#### C3: PPO + SLSQP-regret reward

Use SLSQP headroom to normalize the reward:

\[
r = \frac{P(\gamma)-P(0)}{P^\star_\mathrm{SLSQP}-P(0)+\epsilon}.
\]

When SLSQP headroom is below a threshold, use a small neutral reward or zero weight to avoid numerical instability.

Purpose:

- Train directly toward recovery rate relative to the SLSQP optimum rather than absolute marginal gain.

Expected result:

- Better recovery in high-headroom regions.
- More stable comparison to SLSQP across wind regimes.

#### C4: Behavior cloning from SLSQP

Generate supervised training examples from SLSQP yaw targets.

For each expert condition, synthesize intermediate yaw states:

\[
\gamma_t = \alpha\gamma^\star + \xi,
\]

where \(\alpha\in[0,1]\) and \(\xi\) is bounded yaw noise. The target incremental action is:

\[
a_t^\star = \mathrm{clip}(\gamma^\star - \gamma_t, -5^\circ, 5^\circ).
\]

Train the same ActorCritic policy mean head with an MSE behavior-cloning loss:

\[
\mathcal{L}_\mathrm{BC}=\|\pi_\theta(o_t)-a_t^\star\|_2^2.
\]

Purpose:

- Teach the policy where the SLSQP optimum is before reinforcement learning begins.
- Convert SLSQP from a one-time comparator into an offline expert oracle.

Expected result:

- Major improvement in steady-state gain relative to PPO from scratch.
- The policy should recover a much larger fraction of SLSQP headroom.

#### C5: Behavior cloning + PPO fine-tuning

Initialize PPO from the BC-pretrained policy and fine-tune in the JAX environment.

Optional auxiliary loss:

\[
\mathcal{L}=\mathcal{L}_\mathrm{PPO}+\beta\mathcal{L}_\mathrm{BC},
\]

with \(\beta\) annealed to zero.

Optional fatigue-aware fine-tuning:

\[
r_\mathrm{final}=r_\mathrm{power}-\lambda_\mathrm{rate}\sum_i(\Delta\gamma_i)^2.
\]

Purpose:

- Retain expert-level steady-state behavior while allowing closed-loop correction, robustness, and yaw-rate smoothing.

Expected result:

- Best candidate for approaching SLSQP while preserving incremental-control behavior.

## 4. Evaluation Protocol

Every policy in the ladder will be evaluated using the same metrics.

### Steady-state evaluation

Conditions:

- 3000 random full-range conditions.
- 500-condition SLSQP comparison set.
- Aligned-cube subset.

Metrics:

- Marginal mean gain.
- Aligned-cube gain.
- Negative-gain fraction.
- Recovery relative to SLSQP:

\[
\mathrm{Recovery}=\frac{P_\mathrm{DRL}-P_0}{P_\mathrm{SLSQP}-P_0+\epsilon}.
\]

- Mean and max yaw magnitude.
- Inference latency.

### Dynamic wind evaluation

Use the existing AR(1) dynamic-wind protocol:

- 1000 trajectories.
- 200 evaluation steps.
- 100 settle steps.
- Control period: 10 s.

Metrics:

- Mean dynamic gain.
- Cumulative yaw travel.
- Peak yaw rate.
- Gain/travel ratio.

Comparators:

- Current PPO baseline.
- Best expert-guided DRL policy.
- Lookup unlimited.
- Lookup with 0.5°/s, 0.3°/s, and 0.1°/s limits.

Success criterion:

- A strong outcome: expert-guided DRL approaches lookup/SLSQP steady-state gain while retaining lower yaw travel and feasible peak yaw rate.
- A moderate outcome: expert-guided DRL significantly improves recovery over PPO-from-scratch while preserving dynamic smoothness.
- A negative but publishable outcome: even expert guidance fails to close the gap, showing the intrinsic difficulty of amortizing SLSQP into an incremental controller.

## 5. Engineering Components

### New or modified scripts

- `codes/generate_slsqp_expert_dataset.py`
  - Builds the SLSQP expert dataset for behavior cloning and regret rewards.

- `codes/train_3x3_bc.py`
  - Behavior-cloning pretraining for the policy mean.

- `codes/train_3x3_expert_guided_ppo.py`
  - PPO fine-tuning initialized from BC weights; supports balanced sampling, regret reward, and optional yaw-rate penalty.

- `codes/eval_expert_guided_policies.py`
  - Unified evaluator for PPO baseline, balanced PPO, regret PPO, BC, and BC+PPO.

### Reused components

- `windfarm_env.py`
  - Canonical NumPy wake model for SLSQP.

- `windfarm_env_jax.py`
  - JAX implementation used for accelerated DRL training and rollout.

- `eval_drl_vs_slsqp_regime.py`
  - Existing SLSQP and lookup-table comparison logic.

- `eval_dynamic_wind.py`
  - Dynamic wind evaluation, extended to include expert-guided policies.

- `train_3x3_nnx_jaxenv.py`
  - Existing PPO-from-scratch training baseline.

- `train_3x3_nnx_jaxenv_penalty.py`
  - Existing fatigue-aware PPO training baseline.

## 6. Paper Reframing

The revised paper should use the following structure:

1. Gray-box wake model.
2. SLSQP as offline optimal control oracle.
3. DRL as amortized closed-loop approximation of the SLSQP control law.
4. Engineering methods for efficient approximation:
   - JAX on-device rollout.
   - Balanced exploration.
   - SLSQP-regret reward.
   - SLSQP imitation pretraining.
   - PPO fine-tuning with yaw-rate regularization.
5. Steady-state approximation ladder.
6. Dynamic wind control advantage.
7. Limitations and deployment implications.

Suggested contribution statement:

> We formulate cooperative yaw control as offline-to-online policy distillation: a gray-box SLSQP oracle defines the steady-state optimum, and an expert-guided DRL controller amortizes this optimum into a real-time, rate-feasible, fatigue-aware closed-loop policy.

## 7. Risks and Mitigations

### Risk 1: PPO from scratch remains far below SLSQP

Mitigation:

- Treat this as evidence that sparse high-value wind regimes and continuous yaw coordination make pure exploration difficult.
- Use it to motivate SLSQP imitation pretraining.
- Present PPO-from-scratch as the baseline rather than the final method.

### Risk 2: Behavior cloning matches SLSQP in steady state but creates excessive yaw travel in dynamic wind

Mitigation:

- Fine-tune with yaw-rate penalty.
- Report gain/travel Pareto curves.
- Compare with rate-limited lookup tables.

### Risk 3: SLSQP-regret reward is unstable when SLSQP headroom is tiny

Mitigation:

- Use a headroom threshold.
- Weight no-headroom conditions separately.
- Clip normalized rewards.

### Risk 4: Dataset generation is slow

Mitigation:

- Start with 15k conditions.
- Cache every solved condition.
- Use the existing 91×11 lookup table to warm-start nearby SLSQP solves.
- Parallelize CPU SLSQP only if GPU training is not running.

### Risk 5: The work scope becomes too large for the current manuscript

Mitigation:

- Highest-value path: Phase A baseline evaluation → Phase B expert dataset → C4 behavior cloning → C5 BC+PPO → dynamic evaluation.
- C2 and C3 can be reduced to ablations if time is limited.
- The minimum publishable story is: SLSQP oracle, DRL approximation, expert-guided improvement, and dynamic wind feasibility.

## 8. Approval Gate

Before implementation, confirm:

1. Use DRL vs SLSQP as the only main comparison thread.
2. Avoid discussing action-space variants that distract from the main narrative.
3. Use SLSQP imitation pretraining as the primary route to close the steady-state gap.
4. Keep dynamic wind evaluation as the demonstration of controller advantage.
5. Treat balanced sampling and regret reward as engineering ablations rather than mandatory final methods.
