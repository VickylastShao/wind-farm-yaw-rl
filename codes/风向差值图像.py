import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 从 windfarm_env 中导入相关定义和工具函数
from windfarm_env import (
    WindFarmYawEnv,
    transform_coordinates_to_positive,
    find_downstream_turbines,
    calculate_inflow_speeds,
    power_output,
    C_T, I, d_0, z_h, alpha_star, beta_star, alpha
)


# --- 1. 修改后的环境创建函数 ---
def make_env_for_test_custom(fixed_phi, fixed_v):
    """
    创建一个用于测试的风场环境，采用用户指定的风向 (fixed_phi) 和风速 (fixed_v)。
    """

    def _init():
        N_rows = 3
        N_cols = 3
        turbine_spacing = 7 * d_0  # 最小间距为 7D

        layout_angle_deg = 7.0
        layout_angle_rad = np.radians(layout_angle_deg)
        positions = []
        for i in range(N_rows):
            for j in range(N_cols):
                x = -i * turbine_spacing * np.sin(layout_angle_rad) + j * turbine_spacing
                y = i * turbine_spacing * np.cos(layout_angle_rad)
                z = z_h
                positions.append((x, y, z))
        # 固定风况时 randomize_wind 设置为 False
        env = WindFarmYawEnv(
            turbine_positions=positions,
            j=1,
            max_steps=1000,
            render_mode=None,
            action_penalty=0.02,
            randomize_wind=False
        )
        # 设置用户指定的风向和风速
        env.fixed_phi = fixed_phi
        env.fixed_v = fixed_v

        # 对 reset 方法进行“猴子补丁”：重置时覆盖默认风况
        original_reset = env.reset

        def custom_reset(*args, **kwargs):
            obs, info = original_reset(*args, **kwargs)
            # 使用用户指定值覆盖默认值
            env.current_phi = env.fixed_phi
            env.current_v = env.fixed_v
            # 重新计算转换后的坐标和下游风机索引
            env.transformed_positions = transform_coordinates_to_positive(env.original_turbine_positions,
                                                                          env.current_phi)
            env.downstream_turbines = find_downstream_turbines(env.original_turbine_positions, env.current_phi,
                                                               env.N_rows, env.N_cols)
            # 根据新风况重新计算入流风速
            env.current_inflow_speeds = calculate_inflow_speeds(
                turbine_positions=env.transformed_positions,
                C_T=C_T, I=I, d_0=d_0,
                U_infinity=env.fixed_v,
                z_h=z_h,
                gamma_values=env.current_gammas,
                alpha_star=alpha_star,
                beta_star=beta_star,
                alpha=alpha
            )
            # 更新历史缓冲区
            locked_turbines = np.zeros(env.N)
            locked_turbines[env.downstream_turbines] = 1
            env._update_history(env.current_gammas, env.current_inflow_speeds, env.current_phi, env.current_v,
                                locked_turbines)
            return env._get_obs(), info

        env.reset = custom_reset  # 替换 reset 方法
        return env

    return _init


# --- 2. 辅助函数：无偏航角情况下的输出功率 ---
def get_no_yaw_output_power(phi, fixed_wind_speed):
    """
    对于指定风向 phi 和风速 fixed_wind_speed，
    计算未引入偏航角优化（所有风机偏航角均为 0）的风场输出功率。
    """
    env = DummyVecEnv([make_env_for_test_custom(phi, fixed_wind_speed)])
    obs = env.reset()
    real_env = env.envs[0]
    # 强制所有风机偏航角为 0
    real_env.current_gammas = np.zeros(real_env.N, dtype=np.float32)
    # 根据当前状态重新计算入流风速（传入偏航角均为 0）
    inflow = calculate_inflow_speeds(
        turbine_positions=real_env.transformed_positions,
        C_T=C_T, I=I, d_0=d_0,
        U_infinity=fixed_wind_speed,
        z_h=z_h,
        gamma_values=real_env.current_gammas,
        alpha_star=alpha_star,
        beta_star=beta_star,
        alpha=alpha
    )
    real_env.current_inflow_speeds = inflow
    total_power = 0.0
    for i in range(real_env.N):
        total_power += power_output(inflow[i], 0.0)
    return total_power


# --- 3. 风向扫面并比较两种策略（输出功率差值图） ---
def test_best_model_sweep():
    """
    让风向从 173° 到 353°（步长1°）自动遍历，
    每个风向下：
      ① 使用最优策略（PPO 得到的偏航角组合）运行一个 episode，记录整个 episode 内的最大总输出功率；
      ② 计算所有风机偏航角均为 0 时的输出功率。
    最后绘制差值图：ΔP = P_with_yaw - P_without_yaw。
    """
    best_model_path = "./best_model/best_model"
    best_model = PPO.load(best_model_path)

    directions = range(173, 354)  # 173° ~ 353°
    fixed_wind_speed = 11.4
    best_opt_powers = []  # 最优偏航角策略情况下的最佳总输出功率（单位 W）
    best_noopt_powers = []  # 无偏航角情况下的总输出功率（单位 W）
    best_yaw_list = []  # 记录每个风向下最优策略对应的偏航角组合（供参考）

    for phi in directions:
        # ① 使用最优偏航角策略
        env_test = DummyVecEnv([make_env_for_test_custom(phi, fixed_wind_speed)])
        obs = env_test.reset()
        done = False
        step_count = 0
        max_steps = env_test.get_attr("max_steps")[0]

        best_total_power = -np.inf
        best_yaw = None

        while not done and step_count < max_steps:
            action, _ = best_model.predict(obs, deterministic=True)
            obs, reward, done_, _ = env_test.step(action)
            done = done_[0]
            step_count += 1

            current_gamma = env_test.get_attr("current_gammas")[0]
            inflows = env_test.get_attr("current_inflow_speeds")[0]

            total_power = 0.0
            for i in range(len(inflows)):
                total_power += power_output(inflows[i], current_gamma[i])

            if total_power > best_total_power:
                best_total_power = total_power
                best_yaw = current_gamma.copy()

        best_opt_powers.append(best_total_power)
        best_yaw_list.append(best_yaw)
        print(
            f"Wind Direction = {phi}°, With Yaw Optimization -> Best Total Power = {best_total_power / 1e6:.2f} MW, Best Yaw = {best_yaw}")

        # ② 无偏航角情况下
        no_yaw_power = get_no_yaw_output_power(phi, fixed_wind_speed)
        best_noopt_powers.append(no_yaw_power)
        print(f"Wind Direction = {phi}°, Without Yaw Optimization -> Total Power = {no_yaw_power / 1e6:.2f} MW\n")

    # 转换为数组并转换单位为 MW
    directions_arr = np.array(list(directions))
    best_opt_powers_mw = np.array(best_opt_powers) / 1e6
    best_noopt_powers_mw = np.array(best_noopt_powers) / 1e6

    # 计算两种策略下的功率差值
    diff_powers_mw = (best_opt_powers_mw - best_noopt_powers_mw)/best_noopt_powers_mw * 100

    # 绘制差值图
    plt.figure(figsize=(8, 5))
    plt.plot(directions_arr, diff_powers_mw, marker='o', color='r', label="ΔP = With Yaw - Without Yaw")
    plt.title("Difference in Total Power vs. Wind Direction")
    plt.xlabel("Wind Direction (°)")
    plt.ylabel("Power Difference (MW)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # 若需要保存图像，可取消下面注释：
    # plt.savefig("power_difference.png", dpi=150)


if __name__ == "__main__":
    test_best_model_sweep()
