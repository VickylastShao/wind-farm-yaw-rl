# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

Research codebase for the paper **"Wind Farm Cooperative Yaw Control Based on Deep Reinforcement Learning"**. The work formulates wind-farm yaw control as a reinforcement learning problem on top of an analytical Gaussian wake model, with the goal of maximizing total farm power output by coordinating per-turbine yaw misalignment.

The repo has two top-level directories:

- `codes/` — Python implementation (Gym environment, physics, debug/visualization scripts).
- `Wind Farm Cooperative Yaw Control Based on Deep Reinforcement Learning/` — paper source assets: `origin.pdf`, `full.md`, extracted layout/content JSON, and `images/`. Edit this only when the user is working on the manuscript.

There is no `requirements.txt`, build system, test framework, lint config, README, or git history. This is a script-based research project, not a packaged library.

## Running Code

All scripts run as plain Python files from the `codes/` directory (Chinese filenames must be quoted on the shell):

```bash
cd codes
python windfarm_env.py            # runs test_baseline_power() (the __main__ guard)
python 版本测试.py                 # prints versions of numpy / torch / stable_baselines3
python debug_env_step.py          # single-step env-vs-physics sanity check
python debug_physics.py           # standalone physics debugging
python "风向风速可视化-测试.py"     # PPO + matplotlib visualization
```

Dependencies (no pinned versions; installed ad-hoc): `numpy`, `gymnasium`, `scipy`, `matplotlib`, `torch`, `stable-baselines3`. Use `版本测试.py` to confirm the active interpreter and SB3/PyTorch versions before training.

## Architecture

### `codes/windfarm_env.py` is the canonical module

Everything else imports from it. Two layers live in this file:

**1. Physics layer (module-level functions + constants).** Implements the Bastankhah–Porté-Agel Gaussian wake model with yaw deflection. Key entry points:

- `calculate_inflow_speeds(positions, wind_dir_meteo, C_T, I, d_0, U_inf, gammas, alpha_star, beta_star, alpha)` — given absolute turbine coordinates and a per-turbine yaw vector, returns the inflow speed at each turbine. Internally rotates coordinates so wind blows along +x (using meteorological convention `θ_math = 270° − φ_meteo`), then sums wake deficits via the `alpha`-weighted RSS scheme.
- `power_output(u_eff, gamma)` — per-turbine power using `cos(γ)^1.88` yaw loss, clipped to `[u_cut_in, u_cut_out]` and `P_rated`.
- `calculate_y_d` / `calculate_velocity_deficit` — wake centerline deflection and Gaussian deficit at a point.
- `find_downstream_turbines(positions, wind_dir, U_inf)` — returns indices of turbines that are *not* significantly waking anyone else (the "most downstream" set). These are masked in the RL env.

Physics constants at the top of the file (`U_infinity=11.4`, `d_0=126.0`, `z_h=87.6`, `P_rated=5.29e6`, `C_T=0.8`, `I=0.065`, `alpha_star=2.7276...`, `beta_star=0.1`, `alpha=0.5399...`) are tuned for the NREL-5MW-style turbine used in the paper. Other scripts (`debug_eval.py`, `模型风向角测试.py`, `风向角测试-优化算法.py`) duplicate these constants with *different* values tuned for the Vestas V-80 (`d_0=80`, `P_rated=2e6`, `I=0.077`, etc.) and ship their own C_P / C_T spline tables. Do **not** unify these without asking — the two parameter sets correspond to different turbine models referenced in the paper.

**2. RL environment: `WindFarmYawEnv(gym.Env)`.**

- Action: `Box(-5, +5, shape=(N,))` — yaw *increments* (degrees), accumulated then clipped to `[-50°, +50°]`.
- Observation: flattened history of length `j` (default 1). Each row has `4N + 3` floats: `[gammas (N), inflow (N), cos(φ), sin(φ), v, locked_mask (N)]`. History is a rolling buffer updated with `np.roll(..., shift=-1, axis=0)`.
- Reward: `((current_total_mw - baseline_mw) / N) * 10` — per-turbine average power gain over the all-zero-yaw baseline, scaled by 10. The baseline is computed once at `reset()` for the current wind condition.
- Downstream masking: turbines in `find_downstream_turbines(...)` have their actions zeroed *and* their yaw forced to 0 every step. This is intentional — the most-downstream turbines should always face the wind directly.
- `reset(options=...)` accepts `specific_wind_dir`, `specific_wind_speed`, `initial_gammas`. With `randomize_wind=True`, wind direction is sampled uniformly from `[173°, 353°]` and speed from `[6, 16] m/s`.

### Debug scripts

- `debug_env_step.py` — **stale**: imports `calculate_inflow_speeds_with_transform`, which does not exist in the current `windfarm_env.py` (renamed to `calculate_inflow_speeds`). Will `ImportError` on run. Fix the import before relying on it.
- `debug_physics.py` — self-contained physics replay; useful for hand-checking the wake model independently of the env.
- `debug_eval.py` — uses the Vestas V-80 parameter set with cubic-spline C_P/C_T tables; **does not** import from `windfarm_env.py`.

### Visualization / training-adjacent scripts

`风向差值图像.py` and `风向风速可视化-测试.py` load a trained `PPO` model from disk and animate wake patterns over wind direction sweeps. There is no training script in the repo — model files are produced externally and loaded by path inside these scripts.

## Conventions & Gotchas

- **Wind direction is meteorological** (`φ_meteo`, where `270°` = wind from the west). Physics internally converts via `θ_math = 270° − φ_meteo`. When adding new code, keep this convention at the boundary and convert once.
- **Positions are 3-tuples `(x, y, z)`** in meters; `z` is hub height (`z_h`).
- **Turbine layouts** are built with `create_wind_farm_layout()` / `create_wind_farm_layout_3x3()` in `windfarm_env.py` (7° tilted rectangular grid, `7·d_0` spacing). The paper's 5×5 reference layout is mentioned in the module docstring but not implemented as a helper.
- **`stable_baselines3`** is the RL library; PPO is the algorithm referenced everywhere.
- **Chinese identifiers and filenames** are used throughout (`风向差值图像.py`, "风机", "偏航角", "下游风机"). Preserve them; do not transliterate.
- Many scripts have large `if __name__ == "__main__":` test blocks. The author leaves notes like `## ...在进行训练时应注释掉` ("comment out during training") — respect those when modifying.
