# debug_env_step.py (修正版)
import numpy as np
import math

# 关键：确保这里导入的是您本地最新的、修正后的 windfarm_env.py
# 我们将从这个文件中导入环境类和它依赖的核心物理函数
try:
    from windfarm_env import (
        WindFarmYawEnv,
        calculate_inflow_speeds_with_transform,
        power_output,
        # 导入所有需要的全局物理参数，以确保一致性
        I, d_0, z_h, alpha_star, beta_star, alpha
    )
except ImportError as e:
    print(f"[错误] 无法从 windfarm_env.py 导入所需内容: {e}")
    print("请确保 windfarm_env.py 文件存在、无误，并且与此脚本在同一目录下。")
    exit(1)


def build_consistent_layout(rows: int, cols: int, spacing_D: float = 7.0):
    """
    一个自包含的布局创建函数，确保使用与环境一致的参数。
    """
    spacing_m = spacing_D * d_0  # 使用从环境中导入的 d_0
    turbine_positions = []
    # 创建一个简单的、无旋转的1x2布局用于测试
    for j in range(cols):
        x = j * spacing_m
        y = 0.0
        turbine_positions.append((x, y, z_h))  # 使用从环境中导入的 z_h
    return turbine_positions


def calculate_theoretical_power(turbine_positions, wind_direction, wind_speed, gammas):
    """使用核心物理函数直接计算总功率，作为我们的“理论真值”。"""
    inflow_speeds = calculate_inflow_speeds_with_transform(
        turbine_positions,
        wind_direction,
        I, d_0, wind_speed, z_h,
        np.array(gammas),
        alpha_star, beta_star, alpha
    )
    total_power = sum(power_output(u, g) for u, g in zip(inflow_speeds, gammas))
    return total_power / 1e6  # 返回兆瓦 (MW)


def main():
    """
    本脚本验证 WindFarmYawEnv.step() 的正确性。
    1. 使用纯物理函数计算出基线功率和优化后功率的“理论值”。
    2. 创建环境，重置并获取环境计算的基线功率。
    3. 在环境中执行一步动作。
    4. 对比环境返回的最终功率与我们计算的理论值。
    """
    print("--- 正在对 WindFarmYawEnv.step() 进行单步隔离调试 ---")

    # --- 1. 设置与之前测试完全一致的实验参数 ---
    rows, cols = 1, 2
    spacing_D = 7.0  # 【修正】: 使用正确的直径倍数，而不是错误的米数
    wind_direction = 270.0
    wind_speed = 11.4

    # 定义基线动作和优化动作
    baseline_action = np.array([0.0, 0.0], dtype=np.float32)
    # 我们从之前的物理调试中得知，对于267度风，-20度是不错的选择。我们用这个值来测试。
    # 注意：这里的动作是“绝对偏航角”，而环境的action是“增量”，所以我们要小心处理
    target_gammas = np.array([-20.0, 0.0], dtype=np.float32)
    # 因为环境从0偏航角开始，所以第一步的动作就是目标偏航角
    step_action = target_gammas

    # --- 2. 使用纯物理函数计算理论值 ---
    turbine_positions = build_consistent_layout(rows=rows, cols=cols, spacing_D=spacing_D)

    print("\n[理论计算] 正在使用纯物理函数计算期望结果...")
    theoretical_baseline_power = calculate_theoretical_power(turbine_positions, wind_direction, wind_speed,
                                                             baseline_action)
    theoretical_optimized_power = calculate_theoretical_power(turbine_positions, wind_direction, wind_speed,
                                                              target_gammas)
    theoretical_gain_percent = (
                                           theoretical_optimized_power - theoretical_baseline_power) / theoretical_baseline_power * 100

    print(f"    理论基线功率: {theoretical_baseline_power:.6f} MW")
    print(f"    理论优化后功率: {theoretical_optimized_power:.6f} MW")
    print(f"    理论增益率: {theoretical_gain_percent:.4f} %  <--- 这是我们的黄金标准")

    # --- 3. 创建并重置环境，获取环境计算的基线 ---
    env = WindFarmYawEnv(
        turbine_positions=turbine_positions,
        N_rows=rows,
        N_cols=cols,
        randomize_wind=False
    )

    reset_options = {'specific_wind_dir': wind_direction, 'specific_wind_speed': wind_speed}
    _, reset_info = env.reset(seed=42, options=reset_options)
    env_baseline_power = reset_info.get('baseline_mw')

    print(f"\n[环境重置] 环境已重置。")
    print(f"    环境报告的基线功率: {env_baseline_power:.6f} MW")

    # 验证环境的基线计算是否与理论一致
    if abs(env_baseline_power - theoretical_baseline_power) < 1e-5:
        print("    ✅ 基线功率验证成功！env.reset() 计算正确。")
    else:
        print("    ❌ 基线功率验证失败！env.reset() 的计算与理论值不符。")
        env.close()
        return

    # --- 4. 在环境中执行单步动作 ---
    print(f"\n[环境单步] 正在执行动作: action = {step_action.tolist()}")
    _, reward, _, _, step_info = env.step(step_action)
    env_final_power = step_info.get('current_total_mw')

    # --- 5. 打印并分析最终结果 ---
    print(f"\n--- 最终调试结果 ---")
    print(f"环境执行动作后的最终总功率: {env_final_power:.6f} MW")
    print(f"环境返回的奖励值 (reward): {reward:.6f}")

    print("\n--- 最终分析 ---")
    if env_final_power is not None:
        # 核心验证：环境执行一步后的最终状态，是否与我们的理论计算完全一致？
        if abs(env_final_power - theoretical_optimized_power) < 1e-5:
            print("[结论] ✅✅✅ 巨大成功！")
            print("         环境的 step() 方法返回的最终功率与纯物理函数计算的理论值完全匹配。")
            print("         这证明您的 WindFarmYawEnv 环境封装是正确无误的！")
        else:
            print("[结论] ❌❌❌ 失败！")
            print("         环境的 step() 方法返回的结果与理论值严重不符。")
            print(f"         理论值应为 {theoretical_optimized_power:.6f} MW, 但环境返回 {env_final_power:.6f} MW。")
            print("         这意味着问题100%出在 WindFarmYawEnv 的 step() 方法内部的逻辑，")
            print("         请仔细检查其对 `calculate_inflow_speeds_with_transform` 的调用和参数传递。")
    else:
        print("[结论] ❌ 失败！无法从环境的返回信息中获取最终功率。")

    env.close()


if __name__ == '__main__':
    main()