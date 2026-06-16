# Summary of SLSQP, Lookup, and DRL Steady State Evaluation Code

## 1. SLSQP Optimizer Implementation and Reuse

### Core Implementation
**Functions**: `optimize_slsqp()` in both `eval_drl_vs_slsqp_regime.py` and `compute_slsqp_optimum.py`
- Uses `scipy.optimize.minimize(method='SLSQP')` with bounds `[-50, 50]` degrees for all turbines
- Multi-start optimization (8 starts by default) with random initial guesses
- High precision settings: `maxiter=2000`, `ftol=1e-13`
- Objective: Maximize total farm power via negative power minimization

### Reuse Patterns
1. **Lookup Table Generation**: `build_lookup_table()` precomputes SLSQP optimal yaw angles on a 91x11 wind condition grid (phi:173-353°, v:6-16m/s)
2. **Direct Import**: `eval_dynamic_wind.py` imports `build_lookup_table()` to generate/load precomputed SLSQP results
3. **Standalone Evaluation**: `compute_slsqp_optimum.py` generates SLSQP baselines for DRL comparison

**Files**: 
- `/home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes/eval_drl_vs_slsqp_regime.py`
- `/home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes/compute_slsqp_optimum.py`
- `/home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes/eval_dynamic_wind.py`


## 2. Expert Dataset Generation Reusable Functions

### Key Reusable Functions
1. **Lookup Table Creation**: `build_lookup_table(positions, N)` - Generates SLSQP optimal yaw angles for all grid conditions
2. **Interpolation**: 
   - `lookup_interpolate(phi_query, v_query, ...)` - Single condition bilinear interpolation
   - `lookup_interpolate_batch(phi_query, v_query, ...)` - Batch vectorized interpolation
3. **Power Calculation**: `total_farm_power_np()`/`total_farm_power()` - Compute total farm power for given yaw angles and wind conditions

**Files**:
- `/home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes/eval_drl_vs_slsqp_regime.py`
- `/home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes/eval_dynamic_wind.py`
- `/home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes/windfarm_env.py`


## 3. DRL Steady State Evaluation

### Workflow
1. **Load Policy**: `load_nnx_policy(path, obs_dim, act_dim)` - Restores NNX ActorCritic policy from pickle checkpoint
2. **Run Rollouts**: 
   - Deterministic rollouts on numpy env: `rollout_policy(predict_fn, env, phi, v, settle)`
   - Vectorized JAX evaluation: `run_policy(model, phis_j, vs_j)` in `eval_drl_vs_slsqp_regime.py`
3. **Statistics**:
   - Gain calculation: `(total_power - baseline) / baseline * 100`%
   - Recovery rate: `drl_gain / slsqp_gain * 100` for conditions where SLSQP provides positive gain

### Key Components
- **Policy Model**: `ActorCritic` from `train_3x3_nnx.py`
- **Environments**: `windfarm_env.WindFarmYawEnv` (numpy) and `windfarm_env_jax` (JAX-accelerated)
- **Utility Functions**: `env_reset()`, `env_step()`, `power_output_jax()` for fast evaluation

**Files**:
- `/home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes/cross_val_jaxenv_vs_numpyenv.py`
- `/home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes/eval_drl_vs_slsqp_regime.py`
- `/home/gpu/sz_workspace/JAX-WFCOYAW-RL/codes/train_3x3_nnx.py`


## 4. Key Paths and Constants

### File Paths
- Checkpoints: `checkpoints_3x3_nnx_jaxenv/policy_seed{seed}_{tag}.pkl`
- Lookup tables: `latex_draft/figures/lookup_table_baseline.json`, `lookup_table_yaw.npy`
- Evaluation outputs: `latex_draft/figures/*.json` and `*.pdf`

### Constants
- Wind range: phi 173°-353°, v 6-16 m/s
- Rated speed: 11.4 m/s (power cap threshold)
- Yaw bounds: [-50°, 50°]
- SLSQP default starts: 8
