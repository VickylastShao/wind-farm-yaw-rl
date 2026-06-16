import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# 基本参数（保持不变）
# 风场基本参数
U_infinity = 8.0    # 自由流风速 (m/s)
u_cut_in = 4.0      # 切入风速 (m/s)
u_rated = 15.0      # 额定风速 (m/s)
u_cut_out = 25.0    # 切出风速 (m/s)

# 风机基本参数（Vestas V-80）
P_rated = 2.0e6     # 额定功率 (W)
rho = 1.225         # 空气密度 (kg/m³)
d_0 = 80.0          # 风机直径 (m)
R = d_0 / 2
z_h = 70.0          # 轮毂高度 (m)

alpha_star = 2.32
beta_star = 0.154
I = 0.077           # 湍流强度
# alpha = 1.034718025799208         # 尾流叠加系数


alpha = 1.0
# alpha_star = 2.32
# beta_star = 0.154
# I = 0.1
# alpha = 1.0
# Recalculate C_P to match rated power at rated wind speed
S = np.pi * R ** 2  # Rotor swept area (m²)
# C_P = (2 * P_rated) / (rho * S * u_rated ** 3)  # Calculated C_P

# 厂家提供的功率输出数据
wind_speed_power = np.array([
    5.054945054945056, 5.219780219780221, 5.412087912087913, 5.686813186813188,
    5.906593406593408, 6.236263736263737, 6.4835164835164845, 6.730769230769233,
    6.978021978021979, 7.252747252747253, 7.4725274725274735, 7.719780219780221,
    7.967032967032968, 8.214285714285715, 8.434065934065934, 8.681318681318682,
    8.873626373626376, 9.093406593406593, 9.23076923076923, 9.423076923076923,
    9.670329670329672, 9.89010989010989, 10.10989010989011, 10.274725274725276,
    10.54945054945055, 10.824175824175825, 10.989010989010989, 11.208791208791208,
    11.428571428571429, 11.64835164835165, 11.923076923076923, 12.197802197802197,
    12.445054945054945, 12.747252747252748, 13.186813186813186, 13.5989010989011,
    13.956043956043956, 14.285714285714286, 14.615384615384615, 14.97252747252747,
    15.412087912087912, 15.824175824175823, 16.236263736263737, 16.64835164835165,
    17.060439560439562, 17.5, 17.857142857142854, 18.24175824175824,
    18.626373626373628, 18.983516483516482, 19.395604395604394, 19.725274725274723
])

power_output_data = np.array([
    0.1579554898882627, 0.17868685935912776, 0.2035737371871824, 0.24092714008680383,
    0.27417120694431585, 0.31563394588604665, 0.3572352017730165, 0.40723981900452455,
    0.4530427555637637, 0.5114045618247298, 0.5656570320435865, 0.6240650106196324,
    0.6908763505402158, 0.7576876904607996, 0.8203435220241941, 0.8913565426170467,
    0.9540585464955211, 1.0251177394034536, 1.075307045895281, 1.1422107304460243,
    1.2216271123834146, 1.296887985963616, 1.3679471788715485, 1.418090313048296,
    1.5100655646874133, 1.5768307322929171, 1.6353772278142027, 1.6854280173607903,
    1.7354788069073783, 1.7855295964539661, 1.8354880413703942, 1.8728414442700156,
    1.906039338812448, 1.9307415273801827, 1.9468094930279802, 1.95452026964632,
    1.9539200295502814, 1.9617693231138609, 1.96961861667744, 1.9774217379259396,
    1.985086342229199, 1.9801920768307322, 1.9837011727768028, 1.9788069073783356,
    1.9781143226521376, 1.9773755656108596, 1.976775325514821, 1.9761289131037028,
    1.9754825006925845, 1.979083941268815, 1.969987995198079, 1.9736356080893893
]) * 1e6  # 转换为瓦特（W）

# 计算对应风速下的 C_P 值
C_P_data = (2 * power_output_data) / (rho * S * wind_speed_power ** 3)

# 创建 C_P 的插值函数
C_P_spline = interp1d(wind_speed_power, C_P_data, kind='cubic', fill_value="extrapolate")

# 厂家提供的 C_T 数据
wind_speed_CT = np.array([
    5.054945054945056, 5.714285714285715, 6.291208791208792, 6.8681318681318695, 7.390109890109891,
    7.912087912087912, 8.324175824175825, 8.681318681318682, 9.14835164835165, 9.615384615384615,
    10.10989010989011, 10.32967032967033, 10.54945054945055, 10.989010989010989, 11.291208791208792,
    11.593406593406593, 11.868131868131869, 12.142857142857142, 12.032967032967033, 12.307692307692308,
    12.39010989010989, 12.47252747252747, 12.554945054945057, 12.637362637362639, 12.82967032967033,
    12.912087912087912, 13.04945054945055, 13.186813186813186, 13.296703296703297, 13.434065934065934,
    13.626373626373626, 13.763736263736265, 13.956043956043956, 14.093406593406593, 14.23076923076923,
    14.450549450549449, 14.615384615384615, 14.862637362637363, 15.054945054945057, 15.274725274725276,
    15.576923076923078, 15.96153846153846, 16.31868131868132, 16.703296703296704, 17.005494505494504,
    17.472527472527474, 17.82967032967033, 18.186813186813183, 18.598901098901095, 18.983516483516482,
    19.340659340659343, 19.697802197802197, 19.94505494505494
])

C_T_data = np.array([
    0.8067226890756303, 0.8046218487394958, 0.8025210084033614, 0.8025210084033614, 0.8046218487394958,
    0.8046218487394958, 0.8046218487394958, 0.8046218487394958, 0.8025210084033614, 0.7983193277310925,
    0.7857142857142858, 0.7689075630252101, 0.7542016806722689, 0.7457983193277311, 0.7394957983193278,
    0.7289915966386555, 0.7016806722689075, 0.6554621848739496, 0.6827731092436975, 0.615546218487395,
    0.5987394957983194, 0.573529411764706, 0.5588235294117647, 0.5315126050420168, 0.5,
    0.476890756302521, 0.4516806722689075, 0.4327731092436975, 0.4159663865546218, 0.39915966386554624,
    0.3760504201680672, 0.361344537815126, 0.3403361344537815, 0.32352941176470584, 0.3088235294117647,
    0.29201680672268904, 0.2773109243697479, 0.2584033613445379, 0.245798319327731, 0.22899159663865543,
    0.2100840336134453, 0.19327731092436973, 0.1785714285714286, 0.16176470588235303, 0.15336134453781514,
    0.14075630252100835, 0.13235294117647067, 0.12815126050420167, 0.12184873949579833, 0.11554621848739488,
    0.1113445378151261, 0.10504201680672254, 0.10294117647058831
])

# 创建 C_T 的插值函数
C_T_spline = interp1d(wind_speed_CT, C_T_data, kind='cubic', fill_value="extrapolate")

# 获取特定风速下的 C_P 和 C_T 值
def get_C_P(u_eff):
    if u_eff <= u_cut_in or u_eff >= u_cut_out:
        return 0.0
    else:
        return float(C_P_spline(u_eff))


def get_C_T(u_eff):
    if u_eff < min(wind_speed_CT) or u_eff > max(wind_speed_CT):
        return 0.0
    return float(C_T_spline(u_eff))

# 功率输出函数
def power_output(u_eff, gamma):
    if u_eff <= u_cut_in or u_eff >= u_cut_out:
        return 0.0  # 风速在切入风速和切出风速范围外时，输出功率为0
    elif u_eff >= u_rated:
        return P_rated  # 超过额定风速时输出额定功率
    else:
        C_P = get_C_P(u_eff)
        P = 0.5 * rho * C_P * S * u_eff ** 3 * np.cos(np.radians(gamma)) ** 1.88
        return min(P, P_rated)  # 额定功率限制



# -----------------------------------------
# 计算尾流增长率
def calculate_k_star(I):
    # 通过推力系数C_T和湍流强度I计算尾流增长率k*
    return 0.3837 * I + 0.003678  # 近似公式


# 计算epsilon
def calculate_epsilon(C_T):
    # 计算epsilon
    beta = (1 / 2) * (1 + np.sqrt(1 - C_T)) / np.sqrt(1 - C_T)
    epsilon = 0.2 * np.sqrt(beta)
    return epsilon


# 计算theta
def calculate_sigma(C_T, x):
    k_star = calculate_k_star(I)
    epsilon = calculate_epsilon(C_T)
    sigma = k_star * x + epsilon * d_0
    return sigma

def calculate_x_0(C_T, gamma,alpha_star,beta_star,I):
    # Calculate x_0
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
        # Near wake region
        y_d = theta_0 * x
    else:
        delta_x = x - x_0
        sigma_y = k_star * delta_x + (np.cos(gamma_rad) / np.sqrt(8)) * d_0
        sigma_z = k_star * delta_x + (1 / np.sqrt(8)) * d_0

        term1 = (2.9 + 1.3 * np.sqrt(1 - C_T) - C_T)
        term2 = np.sqrt(np.cos(gamma_rad) / (k_star ** 2 * C_T))
        term3_numerator = (1.6 + np.sqrt(C_T)) * (
                    1.6 * np.sqrt(8 * sigma_y * sigma_z / (d_0 ** 2 * np.cos(gamma_rad))) - np.sqrt(C_T))
        term3_denominator = (1.6 - np.sqrt(C_T)) * (
                    1.6 * np.sqrt(8 * sigma_y * sigma_z / (d_0 ** 2 * np.cos(gamma_rad))) + np.sqrt(C_T))
        log_argument = term3_numerator / term3_denominator

        # Ensure log_argument is positive
        log_argument = np.maximum(log_argument, 1e-6)
        y_d = theta_0 * x_0 + (theta_0 / 14.7) * term1 * term2 * np.log(log_argument) * d_0
    return y_d


def calculate_velocity_deficit(x, y, z, z_h, C_T, I, d_0, gamma, alpha_star, beta_star):
    gamma_rad = np.radians(gamma)
    k_star = calculate_k_star(I)
    x_0 = calculate_x_0(C_T, gamma, alpha_star, beta_star, I)

    # 仅在远尾流区有速度亏损
    if x <= x_0:
        return 0.0

    # 【新模型】计算尾流宽度 sigma_y 和 sigma_z
    sigma_y = k_star * (x - x_0) + (np.cos(gamma_rad) / np.sqrt(8)) * d_0
    sigma_z = k_star * (x - x_0) + (1 / np.sqrt(8)) * d_0

    # 【新模型】计算尾流偏移 y_d (调用您原来的函数，但确保其内部逻辑一致)
    # 注意：为了让您原来的y_d函数能用，我们传入一个未使用的k_w=0.04
    y_d = calculate_y_d(x, C_T, gamma,  d_0, alpha_star, beta_star, I)

    # 【核心数学模型替换】
    # 这是旧的公式，我们不再使用它
    # term1_old = 1 - np.sqrt(1 - (C_T / (8 * (k_star * x / d_0 + epsilon) ** 2)))

    # 这是我们最终确定的、新的、正确的公式
    denominator_term1 = 8 * (sigma_y * sigma_z / d_0 ** 2)
    if denominator_term1 <= 1e-9: return 0.0  # 避免除零

    C_T_effective = C_T * np.cos(gamma_rad) / denominator_term1
    if C_T_effective > 1.0: C_T_effective = 1.0  # 避免对负数开方

    term1 = 1 - np.sqrt(1 - C_T_effective)

    # 高斯分布项保持不变
    # 注意输入的y是相对第一个风机的偏移y_i-y_j，在您的代码中是delta_y_wake
    exp_term_y = ((y - y_d) / sigma_y) ** 2 if sigma_y > 1e-9 else 0
    # 注意z_h是绝对高度，而输入的z也是绝对高度，所以是z - z_h
    exp_term_z = ((z - z_h) / sigma_z) ** 2 if sigma_z > 1e-9 else 0
    term2 = np.exp(-0.5 * (exp_term_y + exp_term_z))

    delta_U = term1 * term2

    # 返回与您原始代码一致的单个数字：绝对速度亏损
    return delta_U * U_infinity


def transform_and_sort_coordinates(positions, angle):
    angle_rad = np.radians(270.0 - angle)
    sin_a, cos_a = np.sin(angle_rad), np.cos(angle_rad)
    rotated = [(p[0] * cos_a - p[1] * sin_a, p[0] * sin_a + p[1] * cos_a, p[2]) for p in positions]
    return sorted(rotated, key=lambda pos: pos[0])


# 【修改】这是正确的物理模型，现在让它接收待优化的参数
def calculate_farm_power(turbine_positions, U_inf, gamma, alpha_star, beta_star, alpha):
    N = len(turbine_positions)
    inflow_speeds = np.full(N, U_inf)
    turbine_powers = np.zeros(N)
    for i in range(N):
        x_i, y_i, z_i = turbine_positions[i]
        deficit_sq_sum = 0.0
        for j in range(i):
            x_j, y_j, z_j = turbine_positions[j]
            u_eff_j = inflow_speeds[j]
            C_T_j = get_C_T(u_eff_j)
            if C_T_j <= 0: continue
            absolute_deficit = calculate_velocity_deficit(
                x=(x_i - x_j), y=(y_i - y_j), z=z_i, z_h=z_j, C_T=C_T_j, I=I, d_0=d_0, gamma=gamma,
                alpha_star=alpha_star, beta_star=beta_star  # << 传入优化参数
            )
            deficit_sq_sum += absolute_deficit ** 2

        if deficit_sq_sum > 0:
            total_absolute_deficit = alpha * np.sqrt(deficit_sq_sum)  # << 使用传入的alpha
            inflow_speeds[i] = max(0, U_inf - total_absolute_deficit)
        else:
            inflow_speeds[i] = U_inf

        turbine_powers[i] = power_output(inflow_speeds[i], gamma)
    return np.sum(turbine_powers)


# 【新增】Horns Rev 1 实测数据 (与之前脚本相同)
horns_rev_data = np.array([
    (173.28, 0.618), (176.93, 0.742), (180.57, 0.873), (184.74, 0.909), (187.08, 0.904), (192.03, 0.886),
    (200.89, 0.823), (205.05, 0.889), (208.96, 0.899), (212.86, 0.899), (217.55, 0.813), (219.64, 0.732),
    (221.72, 0.709), (223.80, 0.747), (225.62, 0.818), (230.05, 0.894), (233.96, 0.884), (238.13, 0.863),
    (242.29, 0.810), (246.98, 0.863), (251.15, 0.853), (255.83, 0.878), (258.18, 0.881), (260.00, 0.868),
    (261.30, 0.853), (262.34, 0.841), (263.39, 0.815), (264.43, 0.782), (265.47, 0.742), (266.25, 0.694),
    (267.29, 0.653), (268.07, 0.620), (268.85, 0.608), (269.90, 0.600), (271.20, 0.610), (272.24, 0.633),
    (273.02, 0.676), (274.84, 0.754), (276.67, 0.790), (277.45, 0.813), (277.97, 0.833), (279.27, 0.851),
    (280.31, 0.858), (282.40, 0.868), (284.48, 0.856), (288.12, 0.846), (292.03, 0.838), (294.90, 0.820),
    (300.10, 0.863), (304.53, 0.856), (307.92, 0.792), (309.74, 0.737), (311.82, 0.719), (313.91, 0.759),
    (315.73, 0.833), (319.38, 0.881), (324.06, 0.863), (328.23, 0.830), (331.88, 0.873), (335.26, 0.878),
    (339.43, 0.901), (342.29, 0.904), (345.16, 0.863), (349.58, 0.711), (353.23, 0.610)
])


# 【新增】连接优化器和物理模型的包装器函数
def farm_normalized_power(params, wind_dir, turbine_positions):
    # 从优化器接收参数
    local_alpha_star, local_beta_star, local_alpha = params

    # 旋转和排序坐标
    sorted_positions = transform_and_sort_coordinates(turbine_positions, wind_dir)

    # 使用传入的优化参数计算总功率
    total_waked_power = calculate_farm_power(
        sorted_positions,
        U_infinity,
        gamma=0,  # 假设偏航角为0
        alpha_star=local_alpha_star,
        beta_star=local_beta_star,
        alpha=local_alpha
    )

    # 归一化
    power_standalone = power_output(U_infinity, gamma=0)
    power_total_unwaked = len(turbine_positions) * power_standalone
    if power_total_unwaked < 1e-6: return 0.0

    return total_waked_power / power_total_unwaked


# 【新增】优化目标函数 (RMSE)
def objective(params, turbine_positions, measured_data):
    errors = []
    for wind_dir, norm_pow_meas in measured_data:
        norm_pow_sim = farm_normalized_power(params, wind_dir, turbine_positions)
        errors.append((norm_pow_sim - norm_pow_meas) ** 2)
    return np.sqrt(np.mean(errors))


# ----------------------------------------------------------------------------
# 主程序 (Main Program)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # 定义风场布局
    N_rows, N_cols = 8, 10
    turbine_spacing = 7 * d_0
    layout_angle_deg = 7.0
    layout_angle_rad = np.radians(layout_angle_deg)
    TURBINE_POSITIONS = []
    for i in range(N_rows):
        for j in range(N_cols):
            x_grid = j * turbine_spacing
            y_grid = i * turbine_spacing
            x = x_grid * np.cos(layout_angle_rad) - y_grid * np.sin(layout_angle_rad)
            y = x_grid * np.sin(layout_angle_rad) + y_grid * np.cos(layout_angle_rad)
            TURBINE_POSITIONS.append((x, y, z_h))

    print("--- 优化开始 ---")

    # 【新增】设定优化参数的边界和初始值
    # 根据调试建议，使用更宽的边界
    bounds = [
        (0.4, 3.0),  # alpha_star bounds
        (0.1, 0.3),  # beta_star bounds
        (0.5, 1.5),  # alpha bounds
    ]
    initial_guess = [2.32, 0.154, 1.0]  # 使用文献/默认值作为初始猜测

    # 【新增】调用 minimize 函数
    result = minimize(
        objective,
        x0=initial_guess,
        args=(TURBINE_POSITIONS, horns_rev_data),  # 传递额外参数
        bounds=bounds,
        method='L-BFGS-B',
        options={'disp': True, 'maxiter': 200}
    )

    # --- 结果展示 ---
    print("\n" + "=" * 50)
    print("=== Horns Rev 拟合结果 ===")
    if result.success:
        print(f"优化成功: {result.message}")
    else:
        print(f"优化可能未收敛: {result.message}")

    best_params = result.x
    print(f"最佳参数  alpha_star = {best_params[0]:.4f}, "
          f"beta_star = {best_params[1]:.4f}, alpha = {best_params[2]:.4f}")
    print(f"拟合后的 RMSE = {result.fun:.4f}")
    print("=" * 50 + "\n")

    # 【新增】使用优化后的参数进行最终的对比和绘图
    sim_pwrs = [farm_normalized_power(best_params, wd, TURBINE_POSITIONS) for wd, _ in horns_rev_data]

    print("\n--- 逐风向调试输出（最佳参数）---")
    print(f"{'Dir (°)':>10s} | {'Measured':>10s} | {'Model':>10s} | {'Δabs':>10s}")
    print("-" * 46)
    for i, (wd, meas) in enumerate(horns_rev_data):
        sim = sim_pwrs[i]
        print(f"{wd:10.2f} | {meas:10.3f} | {sim:10.3f} | {abs(sim - meas):10.3f}")
    print("-" * 46, "\n")

    plt.figure(figsize=(12, 6))
    plt.plot(horns_rev_data[:, 0], horns_rev_data[:, 1], 'o', label='Measured Data', alpha=0.7, zorder=2)
    plt.plot(horns_rev_data[:, 0], sim_pwrs, '-s', label=f'Model Fit (RMSE={result.fun:.4f})', markersize=4,
             linewidth=1.5, zorder=3)
    plt.xlabel('Wind Direction (° meteo)')
    plt.ylabel('Normalized Farm Power')
    plt.legend()
    plt.grid(True, linestyle=':', zorder=1)
    plt.title('Horns Rev 1 – Calibrated Model vs Measured Power')
    plt.show()
