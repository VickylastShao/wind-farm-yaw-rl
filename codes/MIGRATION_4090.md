# Migration & one-shot run guide (RTX 4090 server)

This document explains how to move the project to the 4090 server and reproduce the four supplementary experiments required by the revised paper:

1. **N1.1** — Inference-latency benchmark (CPU first; GPU optional)
2. **N4** — FLORIS cross-validation on Horns Rev I
3. **N2** — Closed-loop tracking under time-varying inflow
4. **S1** — 5x5 closed-loop training (SBX + Numba stage-A path)

All four scripts write outputs directly to `latex_draft/figures/` so re-compiling `main.tex` picks them up automatically.

---

## 0. Move the project

```bash
# on the 4090 host
git clone <your-repo>     # or rsync the directory tree
cd JAX-WFCOYAW-RL
```

If you are using rsync from this WSL2 box:

```bash
rsync -azP --exclude='.git' --exclude='__pycache__' \
      /mnt/c/Users/vicks/MyWork/JAX-WFCOYAW-RL/ \
      user@4090host:/path/to/JAX-WFCOYAW-RL/
```

## 1. Environment

```bash
# Python 3.11 recommended (matches the SB3 / FLORIS support matrix)
python3.11 -m venv .venv
source .venv/bin/activate

# Core stack (CPU pieces): no GPU needed for N1/N2/N4
pip install --upgrade pip
pip install numpy scipy matplotlib gymnasium torch \
            stable-baselines3 floris numba

# Stage-A GPU stack: SBX + JAX (CUDA 12)
pip install sbx-rl
pip install --upgrade "jax[cuda12]"

# Verify GPU is visible
python -c "import jax; print(jax.devices())"
# Expect: [CudaDevice(id=0)]
```

If `jax[cuda12]` fails on a particular driver, fall back to:

```bash
pip install --upgrade "jax[cuda12_local]"
```

## 2. Run the three quick-win experiments (no training, < 1 hour total)

```bash
cd codes

# (N1.1) inference latency on CPU; writes histogram + JSON to figures/
python benchmark_inference_latency.py 2>&1 | tee /tmp/n1.log

# (N4) FLORIS cross-validation; writes 2 figures + JSON
python cross_validate_floris.py 2>&1 | tee /tmp/n4.log

# (N2) closed-loop tracking; uses a random policy by default --
#       point PPO_MODEL_PATH to a trained checkpoint when available
PPO_MODEL_PATH=/path/to/ppo_3x3.zip \
    python closed_loop_tracking.py 2>&1 | tee /tmp/n2.log
```

Outputs after the three scripts finish:

```
latex_draft/figures/
  fig_inference_latency.{pdf,jpg}             # N1.1
  inference_latency_samples.npz
  inference_latency_stats.json
  fig_floris_hornsrev_compare.{pdf,jpg}       # N4
  fig_floris_yaw_sweep.{pdf,jpg}
  floris_validation_stats.json
  fig_tracking_step.{pdf,jpg}                 # N2
  fig_tracking_drift.{pdf,jpg}
  tracking_stats.json
```

## 3. (S1) 5x5 closed-loop training — stage A

The script `train_5x5_sbx.py` performs everything described below; on the
4090 host the only manual step (besides installing dependencies) is:

```bash
cd codes
# default: 3 seeds x 3e7 steps, 16 parallel envs (recommended)
python train_5x5_sbx.py

# quick smoke test (verify the pipeline before committing to a long run):
TOTAL_STEPS=1e6 N_SEEDS=1 N_ENVS=4 python train_5x5_sbx.py
```

What it does, end-to-end:

1. (Optional but recommended) Add `@numba.njit(cache=True)` to
   `calculate_inflow_speeds`, `calculate_y_d`, `calculate_velocity_deficit`
   in `windfarm_env.py`. Run `python -c "from windfarm_env import calculate_inflow_speeds; ..."` once to JIT-compile.
2. Builds the 5x5 layout helper (`create_wind_farm_layout_5x5`, identical to
   the one used in `benchmark_inference_latency.py`).
3. Trains with SBX PPO via `SubprocVecEnv(16) + VecMonitor + VecNormalize`,
   `n_steps=4096`, `batch_size=512`, `lr=3e-4`, `n_epochs=10`, `gamma=0.99`,
   `net_arch=[256, 256]`, with checkpoints every ~500k env-steps under
   `codes/checkpoints_5x5/`.
4. After training, dumps `latex_draft/figures/fig_5x5_training_curve.{pdf,jpg}`
   from the TensorBoard `rollout/ep_rew_mean` scalar of each seed, plus
   `5x5_training_stats.json` recording wall-clock per seed, parallel-env
   count, and final checkpoint paths.

Expected wall-clock on RTX 4090 + Ryzen-class CPU: 12–24 h per seed to reach
breakthrough at ~1.5e7 steps. Watch the TB curve; if reward is still on the
low plateau at 1e7 steps, re-seed.

After training, re-run N1.1 with the 5x5 layout to record real GPU
inference latency, and re-run N2 with
`PPO_MODEL_PATH=codes/checkpoints_5x5/ppo_5x5_seed0_final.zip` on a 5x5
schedule.

## 3b. (S3) Downstream-lock ablation

The script `ablation_downstream_lock.py` answers the round-2 reviewer's
question "is the forced `gamma=0` on the most-downstream turbines actually
needed, or just a hand-engineered prior?". It trains two PPO controllers on
the 3x3 farm — one with the default lock, one with the lock disabled via a
monkey-patch on `find_downstream_turbines` — then evaluates both on a fixed
grid of `(phi, v)` and reports the mean farm-power gain.

```bash
cd codes
# default: 3 seeds x 1e7 steps per condition (6 runs total, ~3-5 h on 4090)
python ablation_downstream_lock.py

# smoke test:
N_SEEDS=2 TOTAL_STEPS=2e6 python ablation_downstream_lock.py
```

Outputs:
```
latex_draft/figures/
  fig_lock_ablation_eval.{pdf,jpg}    bar chart, lock_on vs lock_off
  lock_ablation_stats.json
```

## 4. Folding results back into the paper

After the four scripts finish, you have everything needed to:

- Replace the "engineering-typical" entries in Table 4 (`tab:framework_comparison`) with real measured DRL latency (use the N1.1 JSON).
- Add a new figure `\includegraphics{fig_floris_hornsrev_compare}` to §3.1 alongside the existing Horns Rev validation figure; cite the RMSE-vs-FLORIS number from `floris_validation_stats.json`.
- Add a new subsection §3.2.3 "Tracking under time-varying inflow" that displays `fig_tracking_step` and `fig_tracking_drift`, with settling-time numbers from `tracking_stats.json`.
- Append a new subsection §3.2.4 "Scaling to a 25-turbine farm" with the 5x5 training curve and yaw configuration.

Then re-run:

```bash
cd latex_draft
export PATH=~/.TinyTeX/bin/x86_64-linux:$PATH
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # second pass for cross-refs
```

## 5. Sanity checks before submission

- [ ] `inference_latency_stats.json` has measured values for N=2, 9, 25
- [ ] FLORIS RMSE vs proposed model is reported with one decimal
- [ ] At least one step-change tracking curve shows recovery within the reported settling-time band
- [ ] 5x5 training curve shows breakthrough (not just a low-reward plateau)
- [ ] Table 4 cites the measured latency, not the engineering estimate
- [ ] All limitations identified by the round-2 review (N1.2, N3, S2, S3) are explicitly listed
