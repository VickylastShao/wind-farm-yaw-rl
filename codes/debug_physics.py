# debug_physics.py
import numpy as np
import matplotlib.pyplot as plt
import argparse, os, json, time, math

from scipy.interpolate import interp1d

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
k_w = 0.04
# alpha_star = 3.0
# beta_star = 0.3
# I = 0.10354130989971169
# alpha = 1.034718025799208

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


def find_downstream_turbines(
        original_turbine_positions,  # List of (x,y,z) tuples
        wind_direction_meteo,  # Meteorological wind direction (degrees)
        U_inf,  # Freestream wind speed for deficit calculation
        # Parameters from your global scope used by wake functions:
        # C_T, I, d_0, k_w, alpha_star, beta_star
        deficit_threshold_factor=0.01  # Consider wake significant if >1% velocity deficit
):
    num_turbines = len(original_turbine_positions)
    if num_turbines == 0:
        return []
    if num_turbines == 1:
        return [0]  # A single turbine is always downstream

    # Convert meteorological wind direction to mathematical angle for flow vector
    theta_math_rad = np.radians((270.0 - wind_direction_meteo) % 360.0)
    vx_flow = np.cos(theta_math_rad)
    vy_flow = np.sin(theta_math_rad)
    epsilon_vec = 1e-9  # For numerical stability
    if abs(vx_flow) < epsilon_vec: vx_flow = 0.0
    if abs(vy_flow) < epsilon_vec: vy_flow = 0.0

    downstream_turbine_indices = []

    for i in range(num_turbines):  # Potential wake-generating turbine T_i
        turbine_i_pos = original_turbine_positions[i]
        gamma_i = 0.0  # Yaw of T_i
        hub_height_i = turbine_i_pos[2]  # Assuming z-coordinate is hub height

        is_turbine_i_downstream = True  # Assume T_i is downstream until proven otherwise

        for j in range(num_turbines):  # Potential affected turbine T_j
            if i == j:
                continue

            turbine_j_pos = original_turbine_positions[j]
            hub_height_j = turbine_j_pos[2]

            # Calculate T_j's position relative to T_i in wind-aligned frame
            dx_global = turbine_j_pos[0] - turbine_i_pos[0]
            dy_global = turbine_j_pos[1] - turbine_i_pos[1]

            delta_x_aligned = dx_global * vx_flow + dy_global * vy_flow
            delta_y_aligned = dx_global * (-vy_flow) + dy_global * vx_flow  # Lateral dist of T_j from T_i's x-axis

            # If T_j is not meaningfully downwind of T_i, T_i's wake cannot hit it
            if delta_x_aligned <= 0.1 * d_0:  # Heuristic: T_j must be at least 0.1*D0 downwind
                continue

            # T_j is downwind. Check for wake impingement from T_i.

            # 1. Calculate wake deflection from T_i at T_j's downwind distance
            # (Parameters C_T, gamma_i, k_w, d_0, alpha_star, beta_star, I are used by calculate_y_d)
            y_deflection_at_j = calculate_y_d(
                delta_x_aligned, C_T, gamma_i, k_w, d_0, alpha_star, beta_star, I
            )

            # 2. Effective lateral distance of T_j's hub from T_i's wake centerline
            effective_delta_y_to_wake_center = delta_y_aligned - y_deflection_at_j

            # 3. Calculate sigma_y for the gating condition (as in calculate_inflow_speeds)
            #    This sigma_y is for T_i's wake characteristics.
            k_star_i = calculate_k_star(I)  # I is ambient turbulence intensity
            # This sigma_y is an expansion parameter used for the gate
            sigma_y_for_gate = k_star_i * delta_x_aligned + \
                               np.cos(np.radians(gamma_i)) / np.sqrt(8) * d_0

            # 4. Gating condition: Check if T_j is within the significant lateral extent of T_i's wake
            if abs(effective_delta_y_to_wake_center) <= 3 * sigma_y_for_gate:
                # If within the gate, calculate the actual velocity deficit at T_j's hub center
                # The calculate_velocity_deficit function uses its own internal sigma_y/sigma_z
                # based on x_0 for T_i's wake.
                deficit_at_tj_center = calculate_velocity_deficit(
                    x=delta_x_aligned,  # Downwind distance from T_i to T_j
                    y=effective_delta_y_to_wake_center,  # Lateral distance from T_j to T_i's wake center
                    z=hub_height_j,  # Hub height of T_j (observation point)
                    z_h=hub_height_i,  # Hub height of T_i (wake source)
                    C_T=C_T, I=I, d_0=d_0, gamma=gamma_i,  # Wake properties from T_i
                    alpha_star=alpha_star, beta_star=beta_star,
                    U_inf=U_inf  # Freestream wind to get actual deficit value
                )

                # 5. Check if the deficit is significant
                if deficit_at_tj_center > (deficit_threshold_factor * U_inf):
                    is_turbine_i_downstream = False  # T_i's wake significantly hits T_j
                    break  # T_i is not downstream, no need to check other T_j for this T_i
            # Else (outside 3*sigma_y_for_gate): Assume no significant impingement from T_i on T_j

        if is_turbine_i_downstream:
            downstream_turbine_indices.append(i)

    return sorted(downstream_turbine_indices)




def build_consistent_layout(rows, cols, spacing_D=7.0):
    """
    构建一个与主环境参数完全一致的风场布局。
    使用全局变量 d_0 和 z_h。
    """
    spacing_m = spacing_D * d_0
    turbine_positions = []
    # 创建一个简单的矩形网格，无旋转，便于分析
    for i in range(rows):
        for j in range(cols):
            x = j * spacing_m
            y = i * spacing_m
            # 使用全局一致的轮毂高度
            turbine_positions.append((x, y, z_h))
    return turbine_positions


def calculate_power_vs_yaw(turbine_positions, wind_direction, wind_speed):
    """
    一个辅助函数，用于计算给定风向下，总功率随上游风机偏航角变化的曲线。
    """
    num_turbines = len(turbine_positions)
    # 测试上游风机 (索引0) 从-40度到+40度的效果
    yaw_angles_to_test = np.linspace(-40, 40, 81)
    total_powers_mw = []

    print(f"  测试偏航角范围: {min(yaw_angles_to_test)}° to {max(yaw_angles_to_test)}°...")

    for yaw in yaw_angles_to_test:
        current_gammas = np.zeros(num_turbines)
        # 假设风机0是唯一的上游风机被控制
        # 注意：在更复杂的布局和风向下，需要先识别出所有上游风机
        current_gammas[0] = yaw

        # 【核心修正】: 直接调用我们已经验证过的精确模型函数
        inflow_speeds = calculate_inflow_speeds(
            turbine_positions,
            wind_direction,
            C_T,  # <--- 修正: 显式传入 C_T
            I,
            d_0,
            wind_speed,  # U_infinity
            current_gammas,
            alpha_star,
            beta_star,
            alpha
        )
        total_power_watts = sum(power_output(u, g) for u, g in zip(inflow_speeds, current_gammas))
        total_powers_mw.append(total_power_watts / 1e6)

    return yaw_angles_to_test, total_powers_mw


def main():
    # --- 1. 设置实验参数 ---
    rows, cols = 1, 2  # 测试一个简单的1x2布局
    wind_speed = 11.4  # 使用一个尾流效应明显的风速

    # 我们要对比的两个关键风向
    # 270°: 完全尾流对准
    # 275°: 轻微偏离，尾流部分遮挡，是偏航优化的理想场景
    directions_to_test = [270.0, 267.0, 275]

    # --- 2. 创建风场布局 (使用修正后的、参数一致的函数) ---
    # 布局将是 (0,0,70) 和 (560,0,70)
    turbine_positions = build_consistent_layout(rows, cols, spacing_D=7.0)
    print(f"创建风场布局: {turbine_positions}")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    # --- 3. 为每个风向计算并绘制功率曲线 ---
    for i, wind_dir in enumerate(directions_to_test):
        print(f"\n--- 正在计算风向: {wind_dir}° ---")

        yaws, powers = calculate_power_vs_yaw(turbine_positions, wind_dir, wind_speed)

        # 找到基线（0度偏航）和最优值
        baseline = powers[np.argmin(np.abs(yaws))]
        max_power = max(powers)
        best_yaw = yaws[np.argmax(powers)]
        gain_percent = (max_power - baseline) / baseline * 100 if baseline > 0 else 0

        print(f"  结果:")
        print(f"    基线功率 (0° yaw): {baseline:.4f} MW")
        print(f"    最大功率: {max_power:.4f} MW at {best_yaw:.1f}° yaw")
        print(f"    最大提升率: {gain_percent:.2f}%")

        # 绘图
        plt.plot(yaws, powers, marker='', linestyle='-', lw=2.5,
                 label=f'风向 = {wind_dir}° (最大增益: {gain_percent:.2f}%)')

    # --- 4. 设置图像格式 ---
    plt.title(f'偏航优化效果对比 (风速: {wind_speed} m/s, 间距: 7D)', fontsize=16)
    plt.xlabel('上游风机偏航角 (度)', fontsize=12)
    plt.ylabel('风场总功率 (MW)', fontsize=12)
    plt.legend(fontsize=11)
    plt.axvline(0, color='black', linestyle=':', lw=1, label='0° Yaw (Baseline)')
    plt.grid(True, which='both', linestyle='--')

    # 保存图像
    output_filename = 'physics_yaw_optimization_comparison.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\n[SUCCESS] 对比图像已保存为 '{output_filename}'")

    # 显示图像
    plt.show()


if __name__ == '__main__':
    main()