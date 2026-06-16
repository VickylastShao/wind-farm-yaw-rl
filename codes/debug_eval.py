# debug_physics_whitebox.py
import numpy as np
import math

# ===================================================================
# 关键：请确保下面的所有物理函数和参数，
# 与您最终的、高精度的 windfarm_env.py 中的版本完全一致。
# ===================================================================

# ========== 物理参数和函数 (来自您验证过的测试代码) ==========
rho = 1.225;
d_0 = 80.0;
R = d_0 / 2;
z_h = 70.0;
P_rated = 2.0e6
u_cut_in = 4.0;
u_rated = 15.0;
u_cut_out = 25.0;
S = np.pi * R ** 2
k_w = 0.04;
alpha_star = 3.0;
beta_star = 0.3;
I = 0.077;
alpha = 1.034718025799208

from scipy.interpolate import interp1d

wind_speed_power = np.array(
    [5.05, 5.22, 5.41, 5.69, 5.91, 6.24, 6.48, 6.73, 6.98, 7.25, 7.47, 7.72, 7.97, 8.21, 8.43, 8.68, 8.87, 9.09, 9.23,
     9.42, 9.67, 9.89, 10.11, 10.27, 10.55, 10.82, 10.99, 11.21, 11.43, 11.65, 11.92, 12.2, 12.45, 12.75, 13.19, 13.6,
     13.96, 14.29, 14.62, 14.97, 15.41, 15.82, 16.24, 16.65, 17.06, 17.5, 17.86, 18.24, 18.63, 18.98, 19.4, 19.73]);
power_output_data = np.array(
    [.158, .179, .204, .241, .274, .316, .357, .407, .453, .511, .566, .624, .691, .758, .82, .891, .954, 1.025, 1.075,
     1.142, 1.222, 1.297, 1.368, 1.418, 1.51, 1.577, 1.635, 1.685, 1.735, 1.786, 1.835, 1.873, 1.906, 1.931, 1.947,
     1.955, 1.954, 1.962, 1.97, 1.977, 1.985, 1.98, 1.984, 1.979, 1.978, 1.977, 1.977, 1.976, 1.975, 1.979, 1.97,
     1.974]) * 1e6;
C_P_data = (2 * power_output_data) / (rho * S * wind_speed_power ** 3);
C_P_spline = interp1d(wind_speed_power, C_P_data, kind='cubic', fill_value="extrapolate");
wind_speed_CT = np.array(
    [5.05, 5.71, 6.29, 6.87, 7.39, 7.91, 8.32, 8.68, 9.15, 9.62, 10.11, 10.33, 10.55, 10.99, 11.29, 11.59, 11.87, 12.14,
     12.03, 12.31, 12.39, 12.47, 12.55, 12.64, 12.83, 12.91, 13.05, 13.19, 13.3, 13.43, 13.63, 13.76, 13.96, 14.09,
     14.23, 14.45, 14.62, 14.86, 15.05, 15.27, 15.58, 15.96, 16.32, 16.7, 17.01, 17.47, 17.83, 18.19, 18.6, 18.98,
     1.934e+01, 1.97e+01, 1.995e+01]);
C_T_data = np.array(
    [.807, .805, .803, .803, .805, .805, .805, .805, .803, .798, .786, .769, .754, .746, .739, .729, .702, .655, .683,
     .616, .599, .574, .559, .532, .5, .477, .452, .433, .416, .399, .376, .361, .34, .324, .309, .292, .277, .258,
     .246, .229, .21, .193, .179, .162, .153, .141, .132, .128, .122, .116, .111, .105, .103]);
C_T_spline = interp1d(wind_speed_CT, C_T_data, kind='cubic', fill_value="extrapolate")


def get_C_P(u): return float(C_P_spline(u)) if min(wind_speed_power) <= u <= max(wind_speed_power) else C_P_spline(
    min(max(u, wind_speed_power[0]), wind_speed_power[-1]))


def get_C_T(u): return float(C_T_spline(u)) if min(wind_speed_CT) <= u <= max(wind_speed_CT) else C_T_spline(
    min(max(u, wind_speed_CT[0]), wind_speed_CT[-1]))


def power_output(u, g):
    if u <= u_cut_in or u >= u_cut_out: return 0.0
    if u >= u_rated: return P_rated
    return min(0.5 * rho * get_C_P(u) * S * u ** 3 * np.cos(np.radians(g)) ** 1.88, P_rated)


def calculate_k_star(I): return 0.3837 * I + 0.003678


def calculate_epsilon(C_T):
    if C_T >= 1.0: C_T = 0.9999
    return 0.2 * np.sqrt(0.5 * (1 + np.sqrt(1 - C_T)) / np.sqrt(1 - C_T))


def calculate_x_0(C_T, g, a, b, I):
    if C_T >= 1.0: C_T = 0.9999
    return (np.cos(np.radians(g)) * (1 + np.sqrt(1 - C_T))) / (np.sqrt(2) * (a * I + b * (1 - np.sqrt(1 - C_T)))) * d_0


def calculate_theta_0(C_T, g_rad):
    t = C_T * np.cos(g_rad)
    if t >= 1.0: t = 0.9999
    return (0.3 * g_rad / np.cos(g_rad)) * (1 - np.sqrt(1 - t))


def calculate_y_d(x, C_T, g, k, d, a, b, I):
    if C_T >= 1.0: C_T = 0.9999
    g_rad = np.radians(g);
    x_0 = calculate_x_0(C_T, g, a, b, I);
    th_0 = calculate_theta_0(C_T, g_rad)
    if x <= x_0: return th_0 * x
    k_s = calculate_k_star(I);
    s_y = k_s * (x - x_0) + (np.cos(g_rad) / np.sqrt(8)) * d;
    s_z = k_s * (x - x_0) + (1 / np.sqrt(8)) * d
    t1 = (2.9 + 1.3 * np.sqrt(1 - C_T) - C_T);
    t2 = np.sqrt(np.cos(g_rad) / (k_s ** 2 * C_T))
    n_sqrt = 8 * s_y * s_z / (d ** 2 * np.cos(g_rad));
    num = (1.6 + np.sqrt(C_T)) * (1.6 * np.sqrt(n_sqrt) - np.sqrt(C_T));
    den = (1.6 - np.sqrt(C_T)) * (1.6 * np.sqrt(n_sqrt) + np.sqrt(C_T))
    return th_0 * x_0 + (th_0 / 14.7) * t1 * t2 * np.log(np.maximum(num / den, 1e-9)) * d


def calculate_velocity_deficit(x, y, z, z_h, C_T, I, d, g, a, b, U_j):
    if C_T >= 1.0: C_T = 0.9999
    g_rad = np.radians(g);
    k_s = calculate_k_star(I);
    eps = calculate_epsilon(C_T)
    t1_sqrt = C_T / (8 * (k_s * x / d + eps) ** 2);
    t1 = 1 - np.sqrt(1 - min(t1_sqrt, 0.9999))
    x_0 = calculate_x_0(C_T, g, a, b, I);
    s_y = k_s * (x - x_0) + np.cos(g_rad) / np.sqrt(8) * d;
    s_z = k_s * (x - x_0) + 1 / np.sqrt(8) * d
    t2 = np.exp(-0.5 * ((y / s_y) ** 2 + ((z - z_h) / s_z) ** 2))
    return t1 * t2 * U_j


def build_layout(r, c, s, **kwargs): return [(0.0, 0.0, 70.0), (s, 0.0, 70.0)]


def calculate_inflow_and_debug(original_positions, wind_direction, gamma_values, U_inf, print_debug=False):
    """
    一个自洽的函数，它接收原始坐标和风向，在内部完成坐标变换，计算所有风机的入流风速，并能打印详细的调试信息。
    """
    N = len(original_positions)
    angle_rad = np.radians(270.0 - wind_direction)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    transformed_pos_3d = np.array(
        [[p[0] * cos_a - p[1] * sin_a, p[0] * sin_a + p[1] * cos_a, p[2]] for p in original_positions])

    inflow_speeds = np.full(N, U_inf, dtype=np.float32)
    sorted_indices = np.argsort(transformed_pos_3d[:, 0])

    for i_idx in sorted_indices:
        if i_idx == sorted_indices[0]: continue  # 只计算下游风机的入流速度

        x_i, y_i, z_i = transformed_pos_3d[i_idx]
        deficit_sq_sum = 0.0

        for j_idx in sorted_indices:
            if j_idx >= i_idx: continue
            x_j, y_j, z_j = transformed_pos_3d[j_idx]
            if x_j >= x_i: continue

            u_eff_j = inflow_speeds[j_idx]
            C_T_j = get_C_T(u_eff_j)
            gamma_j = gamma_values[j_idx]

            delta_x = x_i - x_j
            delta_y = y_i - y_j

            y_deflection = calculate_y_d(delta_x, C_T_j, gamma_j, k_w, d_0, alpha_star, beta_star, I)
            effective_delta_y = delta_y - y_deflection

            deficit = calculate_velocity_deficit(delta_x, effective_delta_y, z_i, z_j, C_T_j, I, d_0, gamma_j,
                                                 alpha_star, beta_star, u_eff_j)
            deficit_sq_sum += deficit ** 2

            if print_debug:
                print(f"    - Upstream T{j_idx} (yaw={gamma_j:.1f}°) affecting Downstream T{i_idx}:")
                print(f"      - delta_x={delta_x:.2f}, delta_y={delta_y:.2f}")
                print(f"      - u_eff_j={u_eff_j:.2f} -> C_T_j={C_T_j:.4f}")
                print(f"      - Wake Deflection (y_d) = {y_deflection:.4f} m")
                print(f"      - Effective Lateral Offset = {effective_delta_y:.4f} m")
                print(f"      - Velocity Deficit (ΔU) = {deficit:.4f} m/s")

        total_deficit = alpha * np.sqrt(deficit_sq_sum)
        inflow_speeds[i_idx] = max(U_inf - total_deficit, 0.0)

    return inflow_speeds


def main():
    rows, cols, spacing = 1, 2, 7 * d_0
    wind_speed = 8.0
    turbine_positions = build_layout(rows, cols, spacing)

    # 我们要对比的两个风向和一个固定的最优偏航角
    directions_to_test = [270.0, 267.0]
    optimal_yaw = -20.0

    print("===== 白盒测试: 深入诊断物理模型行为 =====\n")

    for wind_dir in directions_to_test:
        print(f"--- 正在分析风向: {wind_dir}° (上游风机偏航 {optimal_yaw}°) ---")

        # 1. 计算基线情况 (0度偏航)
        gammas_baseline = np.zeros(cols)
        inflows_baseline = calculate_inflow_and_debug(turbine_positions, wind_dir, gammas_baseline, wind_speed)
        power_baseline = sum(power_output(u, g) for u, g in zip(inflows_baseline, gammas_baseline)) / 1e6

        # 2. 计算最优偏航角情况
        gammas_optimal = np.array([optimal_yaw, 0.0])
        inflows_optimal = calculate_inflow_and_debug(turbine_positions, wind_dir, gammas_optimal, wind_speed,
                                                     print_debug=True)
        power_optimal = sum(power_output(u, g) for u, g in zip(inflows_optimal, gammas_optimal)) / 1e6

        gain_percent = (power_optimal - power_baseline) / power_baseline * 100

        print(f"\n  [结果] Wind Dir {wind_dir}°:")
        print(f"    - 基线功率: {power_baseline:.4f} MW (Inflows: {[f'{s:.2f}' for s in inflows_baseline]})")
        print(f"    - 优化功率: {power_optimal:.4f} MW (Inflows: {[f'{s:.2f}' for s in inflows_optimal]})")
        print(f"    - 功率提升率: {gain_percent:.2f} %")
        print("-" * 50)


if __name__ == '__main__':
    main()
