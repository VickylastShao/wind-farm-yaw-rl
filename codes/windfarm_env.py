# -*- coding: utf-8 -*-
"""
Fixed WindFarmYawEnv with proper state space dimensions for 5x5 layout
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ========== 基础物理和风机参数 ==========
U_infinity = 11.4
u_cut_in = 3.0
u_rated = 11.4
u_cut_out = 25.0
P_rated = 5.29e6
rho = 1.225
d_0 = 126.0
R = d_0 / 2.0
z_h = 87.6

C_T = 0.8


alpha_star = 2.7276529318810914
beta_star = 0.1
I = 0.065
alpha = 0.5399628787810578

S = np.pi * R ** 2
C_P = (2.0 * P_rated) / (rho * S * (u_rated ** 3))


# ========== Physics Functions ==========
def power_output(u_eff, gamma):
    """Calculate power output based on effective wind speed and yaw angle."""
    if (u_eff <= u_cut_in) or (u_eff >= u_cut_out):
        return 0.0
    elif u_eff <= u_rated:
        P = 0.5 * rho * C_P * S * (u_eff ** 3) * np.cos(np.radians(gamma)) ** 1.88
        return min(P, P_rated)
    else:
        return P_rated


def calculate_k_star(I):
    return 0.3837 * I + 0.003678


def calculate_epsilon(C_T):
    beta = (1 / 2) * (1 + np.sqrt(1 - C_T)) / np.sqrt(1 - C_T)
    epsilon = 0.2 * np.sqrt(beta)
    return epsilon


def calculate_x_0(C_T, gamma, alpha_star, beta_star, I):
    numerator = np.cos(np.radians(gamma)) * (1 + np.sqrt(1 - C_T))
    denominator = np.sqrt(2) * (alpha_star * I + beta_star * (1 - np.sqrt(1 - C_T)))
    x_0 = (numerator / denominator) * d_0
    return x_0


def calculate_theta_0(C_T, gamma_rad):
    term = C_T * np.cos(gamma_rad)
    theta_0 = (0.3 * gamma_rad / np.cos(gamma_rad)) * (1 - np.sqrt(1 - term))
    return theta_0


def calculate_y_d(x, C_T, gamma, d_0, alpha_star, beta_star, I):
    gamma_rad = np.radians(gamma)
    x_0 = calculate_x_0(C_T, gamma, alpha_star, beta_star, I)
    k_star = calculate_k_star(I)
    theta_0 = calculate_theta_0(C_T, gamma_rad)

    if x <= x_0:
        y_d = theta_0 * x
    else:
        delta_x = x - x_0
        sigma_y = k_star * delta_x + (np.cos(gamma_rad) / np.sqrt(8)) * d_0
        sigma_z = k_star * delta_x + (1 / np.sqrt(8)) * d_0

        term1 = (2.9 + 1.3 * np.sqrt(1 - C_T) - C_T)
        term2 = np.sqrt(np.cos(gamma_rad) / (k_star ** 2 * C_T))
        term3_numerator = (1.6 + np.sqrt(C_T)) * (
            1.6 * np.sqrt(8 * sigma_y * sigma_z / (d_0 ** 2 * np.cos(gamma_rad))) - np.sqrt(C_T)
        )
        term3_denominator = (1.6 - np.sqrt(C_T)) * (
            1.6 * np.sqrt(8 * sigma_y * sigma_z / (d_0 ** 2 * np.cos(gamma_rad))) + np.sqrt(C_T)
        )
        log_argument = term3_numerator / term3_denominator

        # Ensure log_argument is positive
        log_argument = np.maximum(log_argument, 1e-6)
        y_d = theta_0 * x_0 + (theta_0 / 14.7) * term1 * term2 * np.log(log_argument) * d_0
    return y_d


def calculate_velocity_deficit(x, y, z, z_h_j, C_T, I, d_0, gamma, alpha_star, beta_star):
    gamma_rad = np.radians(gamma)
    k_star = calculate_k_star(I)
    x_0 = calculate_x_0(C_T, gamma, alpha_star, beta_star, I)

    if x <= x_0:
        return 0.0

    sigma_y = k_star * (x - x_0) + (np.cos(gamma_rad) / np.sqrt(8)) * d_0
    sigma_z = k_star * (x - x_0) + (1 / np.sqrt(8)) * d_0

    denominator_term1 = 8 * (sigma_y * sigma_z / d_0 ** 2)
    if denominator_term1 <= 1e-9: return 0.0

    C_T_effective = C_T * np.cos(gamma_rad) / denominator_term1
    if C_T_effective > 1.0: C_T_effective = 1.0

    term1 = 1 - np.sqrt(1 - C_T_effective)

    # 【核心修正】直接使用传入的'y'，它已经是 effective_delta_y
    exp_term_y = (y / sigma_y) ** 2 if sigma_y > 1e-9 else 0
    # 使用传入的 z (观测点高度) 和 z_h_j (尾流源高度)
    exp_term_z = ((z - z_h_j) / sigma_z) ** 2 if sigma_z > 1e-9 else 0
    term2 = np.exp(-0.5 * (exp_term_y + exp_term_z))

    delta_U = term1 * term2
    return delta_U * U_infinity


def calculate_inflow_speeds(original_turbine_positions, wind_direction_meteo, C_T, I, d_0, U_infinity,
                            gamma_values_for_single_scenario,  # <-- 接收一个偏航角数组，如 [25.0, 0.0]
                            alpha_star, beta_star, alpha):
    """
    计算在给定风向和单次偏航场景下，风场中每台风机的入流风速。
    返回: 一个一维Numpy数组，包含N台风机的入流风速。
    """
    angle_rad = np.radians(270.0 - wind_direction_meteo)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    transformed_positions = np.array([
        [pos[0] * cos_a - pos[1] * sin_a, pos[0] * sin_a + pos[1] * cos_a, pos[2]]
        for pos in original_turbine_positions
    ])

    N = len(transformed_positions)
    current_inflow_speeds = np.full(N, U_infinity)

    for i in range(N):
        x_i, y_i, z_i = transformed_positions[i]
        deficit_sq_sum = 0.0
        for j in range(N):
            if i == j: continue
            x_j, y_j, z_j = transformed_positions[j]
            if x_j >= x_i: continue

            delta_x = x_i - x_j
            # 【核心修正】直接从输入的数组中获取每台风机对应的偏航角
            gamma_j = gamma_values_for_single_scenario[j]

            y_d = calculate_y_d(delta_x, C_T, gamma_j, d_0, alpha_star, beta_star, I)
            delta_y_wake = (y_i - y_j) - y_d
            delta_U = calculate_velocity_deficit(
                delta_x, delta_y_wake, z_i, z_j, C_T, I, d_0, gamma_j, alpha_star, beta_star
            )
            deficit_sq_sum += delta_U ** 2

        if deficit_sq_sum > 0:
            total_deficit = alpha * np.sqrt(deficit_sq_sum)
            current_inflow_speeds[i] = U_infinity - total_deficit

    return current_inflow_speeds


def transform_coordinates_to_positive(positions, wind_direction):
    relative_angle = 270 - wind_direction
    angle_rad = np.radians(relative_angle)

    transformed_positions = []
    sin_angle = np.sin(angle_rad)
    cos_angle = np.cos(angle_rad)

    for x_0, y_0, z_0 in positions:
        x_new = x_0 * cos_angle - y_0 * sin_angle
        y_new = x_0 * sin_angle + y_0 * cos_angle
        z_new = z_0
        transformed_positions.append((x_new, y_new, z_new))

    x_min, y_min, z_min = min(transformed_positions, key=lambda pos: pos[0])

    relative_positions = []
    for x_new, y_new, z_new in transformed_positions:
        x_relative = x_new - x_min
        y_relative = y_new - y_min
        z_relative = z_new
        relative_positions.append((x_relative, y_relative, z_relative))

    return relative_positions


def find_downstream_turbines(original_turbine_positions, wind_direction_meteo, U_inf, deficit_threshold_factor=0.01):
    num_turbines = len(original_turbine_positions)
    if num_turbines <= 1: return list(range(num_turbines))

    theta_math_rad = np.radians((270.0 - wind_direction_meteo) % 360.0)
    vx_flow, vy_flow = np.cos(theta_math_rad), np.sin(theta_math_rad)

    downstream_turbine_indices = []
    for i in range(num_turbines):
        is_i_downstream = True
        for j in range(num_turbines):
            if i == j: continue

            # Check if turbine j is downstream of turbine i
            dx_global = original_turbine_positions[j][0] - original_turbine_positions[i][0]
            dy_global = original_turbine_positions[j][1] - original_turbine_positions[i][1]
            delta_x_aligned = dx_global * vx_flow + dy_global * vy_flow

            # If any other turbine j is downstream of i, then i is not a "most downstream" turbine.
            if delta_x_aligned > 0.1 * d_0:
                # To be more precise, we can check for actual wake impingement
                delta_y_aligned = dx_global * (-vy_flow) + dy_global * vx_flow

                # 【修正】调用 calculate_y_d 时移除 k_w
                y_deflection_at_j = calculate_y_d(delta_x_aligned, C_T, 0.0, d_0, alpha_star, beta_star, I)
                effective_delta_y = delta_y_aligned - y_deflection_at_j

                k_star_i = calculate_k_star(I)
                sigma_y_gate = k_star_i * delta_x_aligned + np.cos(0) / np.sqrt(8) * d_0

                if abs(effective_delta_y) <= 3 * sigma_y_gate:
                    # 【修正】调用 calculate_velocity_deficit
                    deficit_at_j = calculate_velocity_deficit(
                        x=delta_x_aligned,
                        y=effective_delta_y,
                        z=original_turbine_positions[j][2],
                        z_h_j=original_turbine_positions[i][2],  # <-- 使用正确的关键字参数
                        C_T=C_T, I=I, d_0=d_0, gamma=0.0,
                        alpha_star=alpha_star, beta_star=beta_star
                        # <-- 移除了多余的 U_inf
                    )
                    if deficit_at_j / U_inf > deficit_threshold_factor:
                        is_i_downstream = False
                        break  # Found a turbine j waked by i, so i is not downstream
        if is_i_downstream:
            downstream_turbine_indices.append(i)

    return sorted(downstream_turbine_indices)


# ========== Main Environment Class ==========
class WindFarmYawEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, turbine_positions, N_rows, N_cols, j=1, max_steps=1000, randomize_wind=False):
        super().__init__()
        self.original_turbine_positions = turbine_positions
        self.N_rows, self.N_cols = N_rows, N_cols
        self.N = len(turbine_positions)
        self.j, self.max_steps = j, max_steps
        self.randomize_wind = randomize_wind
        self.current_total_mw = 0.0

        # MERGE: Action space is standard and correct
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.N,), dtype=np.float32)

        # MERGE: Using the superior state space from Version 1 (cos/sin for wind direction)
        obs_dim_per_step = self.N + self.N + 3 + self.N  # gammas + inflow + (cos,sin,v) + locked
        obs_dim = self.j * obs_dim_per_step
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # MERGE: Using the history buffer initialization from Version 1
        self.history_buffer = np.zeros((self.j, obs_dim_per_step), dtype=np.float32)

        # Internal state variables
        self.current_step = 0
        self.current_gammas = np.zeros(self.N, dtype=np.float32)
        self.current_inflow_speeds = np.zeros(self.N, dtype=np.float32)
        self.downstream_turbines = []
        self.baseline_mw = 0.0
        self.current_phi = 270.0
        self.current_v = 11.4

    def _update_history(self, gammas, inflow, phi, wind_v, locked_turbines):
        """ MERGE: Using the efficient np.roll from Version 1 """
        self.history_buffer = np.roll(self.history_buffer, shift=-1, axis=0)
        phi_rad = np.radians(phi)
        wind_info = [np.cos(phi_rad), np.sin(phi_rad), wind_v]
        new_row = np.concatenate([gammas, inflow, wind_info, locked_turbines])
        self.history_buffer[-1, :] = new_row

    def _get_obs(self):
        """ Return flattened observation from the history buffer. """
        return self.history_buffer.flatten().astype(np.float32)

    def reset(self, seed=None, options=None):
        """ MERGE: Using the robust reset logic from Version 2 """
        super().reset(seed=seed)
        self.current_step = 0

        # --- Process options dictionary for specific conditions ---
        if options is None:
            options = {}
        specific_wind_dir = options.get('specific_wind_dir')
        specific_wind_speed = options.get('specific_wind_speed')
        initial_gammas = options.get('initial_gammas')

        # --- Set initial yaw angles (robust check) ---
        if initial_gammas is not None:
            self.current_gammas = np.array(initial_gammas, dtype=np.float32)
            if self.current_gammas.shape != (self.N,):
                raise ValueError(
                    f"Provided initial_gammas shape {self.current_gammas.shape} does not match ({self.N},)")
        else:
            self.current_gammas = np.zeros(self.N, dtype=np.float32)

        # --- Set wind conditions ---
        if specific_wind_dir is not None:
            self.current_phi = float(specific_wind_dir)
        elif self.randomize_wind:
            self.current_phi = np.random.uniform(173, 353)
        else:
            self.current_phi = 270.0

        if specific_wind_speed is not None:
            self.current_v = float(specific_wind_speed)
        elif self.randomize_wind:
            self.current_v = np.random.uniform(6, 16)
        else:
            self.current_v = 11.4

        # --- Identify downstream turbines ---
        self.downstream_turbines = find_downstream_turbines(
            self.original_turbine_positions, self.current_phi, self.current_v
        )

        # --- Calculate baseline power (all yaws at zero) ---
        # NOTE: We pass original_turbine_positions because the function handles the rotation.
        inflow_0 = calculate_inflow_speeds(
            self.original_turbine_positions, self.current_phi, C_T, I, d_0, self.current_v,
            np.zeros(self.N), alpha_star, beta_star, alpha
        )
        self.baseline_mw = sum(power_output(u, 0.0) for u in inflow_0) / 1e6

        # --- Calculate initial inflow speeds for current (possibly non-zero) yaws ---
        self.current_inflow_speeds = calculate_inflow_speeds(
            self.original_turbine_positions, self.current_phi, C_T, I, d_0, self.current_v,
            self.current_gammas, alpha_star, beta_star, alpha
        )

        # --- Create locked turbines state vector for the observation ---
        locked_turbines = np.zeros(self.N, dtype=np.float32)
        if self.downstream_turbines:
            valid_indices = [idx for idx in self.downstream_turbines if 0 <= idx < self.N]
            if valid_indices:
                locked_turbines[valid_indices] = 1.0

        # --- Initialize history buffer ---
        for _ in range(self.j):
            self._update_history(self.current_gammas, self.current_inflow_speeds, self.current_phi, self.current_v,
                                 locked_turbines)

        # MERGE: Returning a comprehensive info dictionary from Version 2
        info = {
            "wind_direction": self.current_phi,
            "wind_speed": self.current_v,
            "baseline_mw": self.baseline_mw,
            "downstream_turbines": self.downstream_turbines
        }
        return self._get_obs(), info

    def step(self, action):
        # Ensure action is a mutable 1-D numpy array.
        # NB: np.atleast_1d preserves rank for already-2D inputs (e.g. shape
        # (1, N) leaking from a VecEnv); .reshape(-1) flattens those before
        # the downstream-mask integer indexing in this method.
        valid_action = np.asarray(action).reshape(-1).copy()

        # Mask actions for downstream turbines
        if self.downstream_turbines:
            valid_indices = [idx for idx in self.downstream_turbines if 0 <= idx < self.N]
            if valid_indices:
                valid_action[valid_indices] = 0.0

        # Update and clip yaw angles
        self.current_gammas += valid_action
        self.current_gammas = np.clip(self.current_gammas, -50, 50)

        # Force downstream turbine yaws to zero
        if self.downstream_turbines:
            valid_indices_force = [idx for idx in self.downstream_turbines if 0 <= idx < self.N]
            if valid_indices_force:
                self.current_gammas[valid_indices_force] = 0.0

        # Calculate new inflow speeds and total power
        # NOTE: We pass original_turbine_positions because the function handles the rotation.
        inflow = calculate_inflow_speeds(
            self.original_turbine_positions, self.current_phi, C_T, I, d_0, self.current_v,
            self.current_gammas, alpha_star, beta_star, alpha
        )
        self.current_inflow_speeds = inflow
        total_mw = sum(power_output(u, g) for u, g in zip(inflow, self.current_gammas)) / 1e6
        self.current_total_mw = total_mw

        # MERGE: Using the normalized reward from the provided code snippets
        # This reward represents the average power gain per turbine, scaled by 10.
        delta = (self.current_total_mw - self.baseline_mw) / self.N
        reward = delta * 10

        # Create locked turbines state for history update
        locked_turbines = np.zeros(self.N, dtype=np.float32)
        if self.downstream_turbines:
            valid_indices = [idx for idx in self.downstream_turbines if 0 <= idx < self.N]
            if valid_indices:
                locked_turbines[valid_indices] = 1.0

        # Update history buffer with the new state
        self._update_history(self.current_gammas, self.current_inflow_speeds, self.current_phi, self.current_v,
                             locked_turbines)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # MERGE: Returning the more informative info dict from Version 2
        info = {
            "wind_speed": self.current_v,
            "current_total_mw": self.current_total_mw,
            "current_gammas": self.current_gammas.copy()
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        """ Renders the current step's info, matching the normalized reward. """
        avg_gain = (self.current_total_mw - self.baseline_mw) / self.N
        print(f"Step {self.current_step}: φ={self.current_phi:.1f}°, v={self.current_v:.1f} m/s, "
              f"Avg Gain/Turbine={avg_gain:.4f} MW, Total Power={self.current_total_mw:.3f} MW")

    def close(self):
        pass

######################################

## 测试环境模型能够得到正确、准确的传入入流风速及输出功率代码（基线时刻，所有偏航角均为0），在进行训练时应注释掉。

def create_wind_farm_layout():
    """创建带7度偏角的风场布局"""
    N_rows = 1
    N_cols = 2
    turbine_spacing = 7 * d_0  # 最小间距为 7D
    layout_angle_deg = 7.0
    layout_angle_rad = np.radians(layout_angle_deg)

    turbine_positions = []
    for i in range(N_rows):
        for j in range(N_cols):
            x = -i * turbine_spacing * np.sin(layout_angle_rad) + j * turbine_spacing
            y = i * turbine_spacing * np.cos(layout_angle_rad)
            z = z_h
            turbine_positions.append((x, y, z))

    return turbine_positions, N_rows, N_cols


def create_wind_farm_layout_3x3():
    """创建带7度偏角的3x3风场布局"""
    N_rows = 3
    N_cols = 3
    turbine_spacing = 7 * d_0  # 最小间距为 7D
    layout_angle_deg = 7.0
    layout_angle_rad = np.radians(layout_angle_deg)

    turbine_positions = []
    for i in range(N_rows):
        for j in range(N_cols):
            x = -i * turbine_spacing * np.sin(layout_angle_rad) + j * turbine_spacing
            y = i * turbine_spacing * np.cos(layout_angle_rad)
            z = z_h
            turbine_positions.append((x, y, z))

    return turbine_positions, N_rows, N_cols


def create_wind_farm_layout_5x5():
    """创建带7度偏角的5x5风场布局（与 3x3 同构、同间距）。

    返回与 create_wind_farm_layout_3x3 相同形状的元组：
    (positions: List[(x,y,z)], N_rows, N_cols)。25 台 NREL-5MW 机组、
    7D 列距、行向左偏 7°，与 paper Stage II Case B 拓扑一致。
    """
    N_rows = 5
    N_cols = 5
    turbine_spacing = 7 * d_0
    layout_angle_deg = 7.0
    layout_angle_rad = np.radians(layout_angle_deg)

    turbine_positions = []
    for i in range(N_rows):
        for j in range(N_cols):
            x = -i * turbine_spacing * np.sin(layout_angle_rad) + j * turbine_spacing
            y = i * turbine_spacing * np.cos(layout_angle_rad)
            z = z_h
            turbine_positions.append((x, y, z))

    return turbine_positions, N_rows, N_cols


def test_baseline_power():
    """
    测试指定风场布局在风向270°，风速11.4 m/s条件下的基准功率输出（所有偏航角为0）
    """
    # 固定风况：风向270°，风速11.4 m/s
    wind_dir = 270.0
    wind_speed = 11.4

    # 创建1x3布局
    layout_1x3, n_rows_1x3, n_cols_1x3 = create_wind_farm_layout()

    print("===== 测试1×3布局 =====")
    print(f"布局参数: {n_rows_1x3}行 x {n_cols_1x3}列, 间距 = {7 * d_0:.1f}m, 偏角 = 7°")
    print(f"布局坐标:")
    for i, pos in enumerate(layout_1x3):
        print(f"风机 {i}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    env_1x3 = create_and_test_env(layout_1x3, n_rows_1x3, n_cols_1x3, wind_dir, wind_speed)

    # 创建3x3布局
    layout_3x3, n_rows_3x3, n_cols_3x3 = create_wind_farm_layout_3x3()

    print("\n===== 测试3×3布局 =====")
    print(f"布局参数: {n_rows_3x3}行 x {n_cols_3x3}列, 间距 = {7 * d_0:.1f}m, 偏角 = 7°")
    print(f"布局坐标:")
    for i, pos in enumerate(layout_3x3):
        print(f"风机 {i}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    env_3x3 = create_and_test_env(layout_3x3, n_rows_3x3, n_cols_3x3, wind_dir, wind_speed)

    return env_1x3, env_3x3  # 返回环境对象以便进一步检查


def create_and_test_env(turbine_positions, n_rows, n_cols, wind_dir, wind_speed):
    """
    创建环境并测试特定布局在给定风向和风速下的基准功率

    参数:
        turbine_positions: 每个风机的(x, y, z)坐标列表
        n_rows: 布局中的行数
        n_cols: 布局中的列数
        wind_dir: 风向（度）
        wind_speed: 风速（m/s）
    """
    # 创建环境
    env = WindFarmYawEnv(
        turbine_positions=turbine_positions,
        N_rows=n_rows,
        N_cols=n_cols,
        randomize_wind=False
    )

    # 使用特定风向和风速重置环境
    obs, info = env.reset(options={
        'specific_wind_dir': wind_dir,
        'specific_wind_speed': wind_speed
    })

    # 获取基准功率（所有偏航角为0）
    baseline_mw = info["baseline_mw"]
    per_turbine_mw = baseline_mw / (n_rows * n_cols)

    # 显示结果
    print(f"\n风况: 风向 {wind_dir}°, 风速 {wind_speed} m/s")
    print(f"总功率: {baseline_mw:.3f} MW")
    print(f"每台风机平均功率: {per_turbine_mw:.3f} MW")
    print(f"下游风机索引: {info['downstream_turbines']}")

    # 计算并显示各风机的入流速度
    print("\n各风机入流速度和功率:")
    inflow_speeds = env.current_inflow_speeds
    total_power = 0.0
    for i in range(len(inflow_speeds)):
        power = power_output(inflow_speeds[i], 0.0)
        power_mw = power / 1e6
        total_power += power_mw
        print(f"风机 {i}: 入流速度 = {inflow_speeds[i]:.2f} m/s, 功率 = {power_mw:.3f} MW")

    print(f"\n单独计算总功率: {total_power:.3f} MW")

    return env  # 返回环境对象以便进一步检查


#################################

def test_find_downstream_turbines():
    """
    Test the find_downstream_turbines function (wake impingement version)
    on various layouts and wind directions.
    """
    print("===== TESTING FIND_DOWNSTREAM_TURBINES FUNCTION (Wake Impingement Version) =====")

    # Test parameters
    test_U_inf = U_infinity # Use the global U_infinity for testing
    # deficit_threshold_factor will use its default in find_downstream_turbines

    # Test case 1: 1×3 layout (row of 3 turbines)
    layout_1x3, n_rows_1x3, n_cols_1x3 = create_wind_farm_layout()
    gammas_1x3 = np.zeros(len(layout_1x3)) # Assuming zero yaw for testing

    test_wind_dirs = [0, 90, 180, 270, 45, 135, 225, 315, 269.9, 270.1]

    print("\n----- Test Case 1: 1×3 Layout -----")
    print(f"Layout: {n_rows_1x3} rows × {n_cols_1x3} columns")
    print(f"Turbine positions:")
    for i, pos in enumerate(layout_1x3):
        print(f"  Turbine {i}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    for wind_dir in test_wind_dirs:
        downstream = find_downstream_turbines(
            layout_1x3,
            wind_dir,
            gammas_1x3, # Pass current_gammas
            test_U_inf  # Pass U_inf
        )
        print(f"Wind Direction: {wind_dir}° → Downstream turbines: {downstream}")

    # Test case 2: 3×1 layout (column of 3 turbines)
    layout_3x1 = []
    # Create a simple vertical column for testing (original was rotated from 1x3)
    for i in range(3): # 3 turbines
        layout_3x1.append((0.0, i * (7 * d_0), z_h)) # Example: separated by 7D vertically
    gammas_3x1 = np.zeros(len(layout_3x1))

    print("\n----- Test Case 2: 3×1 Layout -----")
    # Note: n_rows and n_cols for this manually created layout are 3 and 1 respectively.
    print(f"Layout: 3 rows × 1 column")
    print(f"Turbine positions:")
    for i, pos in enumerate(layout_3x1):
        print(f"  Turbine {i}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    for wind_dir in test_wind_dirs:
        downstream = find_downstream_turbines(
            layout_3x1,
            wind_dir,
            gammas_3x1, # Pass current_gammas
            test_U_inf  # Pass U_inf
        )
        print(f"Wind Direction: {wind_dir}° → Downstream turbines: {downstream}")

    # Test case 3: 3×3 layout
    layout_3x3, n_rows_3x3, n_cols_3x3 = create_wind_farm_layout_3x3()
    gammas_3x3 = np.zeros(len(layout_3x3))

    print("\n----- Test Case 3: 3×3 Layout -----")
    print(f"Layout: {n_rows_3x3} rows × {n_cols_3x3} columns")
    print(f"Turbine positions:")
    for i, pos in enumerate(layout_3x3):
        print(f"  Turbine {i}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    for wind_dir in [0, 90, 180, 270]: # Test some cardinal directions
        downstream = find_downstream_turbines(
            layout_3x3,
            wind_dir,
            gammas_3x3, # Pass current_gammas
            test_U_inf  # Pass U_inf
        )
        print(f"Wind Direction: {wind_dir}° → Downstream turbines: {downstream}")


# # Add this to the main block to run the tests
# if __name__ == "__main__":
#     test_find_downstream_turbines()


if __name__ == "__main__":
    test_baseline_power()
