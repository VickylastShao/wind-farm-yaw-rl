import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Import relevant definitions and tool functions from windfarm_env
# For demonstration, we're keeping the function references but they'll need
# to be connected to your actual environment
from windfarm_env import (
    WindFarmYawEnv,
    transform_coordinates_to_positive,
    find_downstream_turbines,
    calculate_inflow_speeds,
    power_output,
    C_T, I, d_0, z_h, alpha_star, beta_star, alpha
)


# --- 1. Modified environment creation function ---
def make_env_for_test_custom(fixed_phi, fixed_v):
    """
    Create a wind farm environment for testing with specified wind direction (fixed_phi) and speed (fixed_v).
    """

    def _init():
        N_rows = 3
        N_cols = 3
        turbine_spacing = 7 * d_0  # Minimum spacing of 7D

        layout_angle_deg = 7.0
        layout_angle_rad = np.radians(layout_angle_deg)
        positions = []
        for i in range(N_rows):
            for j in range(N_cols):
                x = -i * turbine_spacing * np.sin(layout_angle_rad) + j * turbine_spacing
                y = i * turbine_spacing * np.cos(layout_angle_rad)
                z = z_h
                positions.append((x, y, z))
        # Set randomize_wind to False for fixed wind conditions
        env = WindFarmYawEnv(
            turbine_positions=positions,
            j=1,
            max_steps=1000,
            render_mode=None,
            action_penalty=0.02,
            randomize_wind=False
        )
        # Set user specified wind direction and speed
        env.fixed_phi = fixed_phi
        env.fixed_v = fixed_v

        # Monkey patch the reset method to override default wind conditions
        original_reset = env.reset

        def custom_reset(*args, **kwargs):
            obs, info = original_reset(*args, **kwargs)
            # Override defaults with user specified values
            env.current_phi = env.fixed_phi
            env.current_v = env.fixed_v
            # Recalculate transformed coordinates and downstream turbine indices
            env.transformed_positions = transform_coordinates_to_positive(env.original_turbine_positions,
                                                                          env.current_phi)
            env.downstream_turbines = find_downstream_turbines(env.original_turbine_positions, env.current_phi,
                                                               env.N_rows, env.N_cols)
            # Recalculate inflow speeds based on new wind conditions
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
            # Update history buffer
            locked_turbines = np.zeros(env.N)
            locked_turbines[env.downstream_turbines] = 1
            env._update_history(env.current_gammas, env.current_inflow_speeds, env.current_phi, env.current_v,
                                locked_turbines)
            return env._get_obs(), info

        env.reset = custom_reset  # Replace reset method
        return env

    return _init


# --- 2. Helper function: Calculate power without yaw optimization ---
def get_no_yaw_output_power(phi, fixed_wind_speed):
    """
    For the specified wind direction phi and speed fixed_wind_speed,
    calculate wind farm output power without yaw optimization (all turbine yaw angles = 0).
    """
    env = DummyVecEnv([make_env_for_test_custom(phi, fixed_wind_speed)])
    obs = env.reset()
    real_env = env.envs[0]

    # Force all turbine yaw angles to 0
    real_env.current_gammas = np.zeros(real_env.N, dtype=np.float32)

    # Recalculate inflow speeds based on current state (with all yaw angles = 0)
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

    # Calculate total output power
    total_power = 0.0
    for i in range(real_env.N):
        total_power += power_output(inflow[i], 0.0)

    return total_power


# --- 3. Helper function: Get data for a case without yaw optimization ---
def get_no_yaw_data(phi, fixed_wind_speed):
    """
    For the specified wind direction phi and speed fixed_wind_speed,
    calculate wind farm data without yaw optimization (all turbine yaw angles = 0).
    Returns: turbine positions, inflow speeds, output power, etc.
    """
    env = DummyVecEnv([make_env_for_test_custom(phi, fixed_wind_speed)])
    obs = env.reset()
    real_env = env.envs[0]

    # Force all turbine yaw angles to 0
    real_env.current_gammas = np.zeros(real_env.N, dtype=np.float32)

    # Recalculate inflow speeds based on current state (with all yaw angles = 0)
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

    # Calculate output power for each turbine
    powers = np.zeros(real_env.N)
    for i in range(real_env.N):
        powers[i] = power_output(inflow[i], 0.0)

    total_power = powers.sum()

    return {
        'positions': real_env.transformed_positions,
        'original_positions': real_env.original_turbine_positions,
        'inflow_speeds': inflow,
        'powers': powers,
        'total_power': total_power,
        'gamma_values': real_env.current_gammas.copy()
    }


# --- 4. Helper functions for wake calculation ---
def calculate_k_star(I):
    """Calculate wake expansion rate based on turbulence intensity"""
    return 0.3837 * I + 0.003678


def calculate_x_0(C_T, gamma, alpha_star, beta_star, I):
    """Calculate the near wake length"""
    gamma_rad = np.radians(gamma)
    numerator = np.cos(gamma_rad) * (1 + np.sqrt(1 - C_T))
    denominator = np.sqrt(2) * (alpha_star * I + beta_star * (1 - np.sqrt(1 - C_T)))
    x_0 = (numerator / denominator) * d_0
    return x_0


def calculate_theta_0(C_T, gamma_rad):
    """Calculate initial wake deflection angle"""
    term = C_T * np.cos(gamma_rad)
    theta_0 = (0.3 * gamma_rad / np.cos(gamma_rad)) * (1 - np.sqrt(1 - term))
    return theta_0


def calculate_y_d(x, C_T, gamma, k_w, d_0, alpha_star, beta_star, I):
    """Calculate lateral wake deviation based on yaw angle"""
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
        log_argument = np.maximum(log_argument, 1e-6)
        y_d = theta_0 * x_0 + (theta_0 / 14.7) * term1 * term2 * np.log(log_argument) * d_0
    return y_d


def calculate_velocity_deficit(x, y, z, z_h, C_T, I, d_0, gamma, alpha_star, beta_star):
    """Calculate velocity deficit at a point in the wake"""
    gamma_rad = np.radians(gamma)
    k_star = calculate_k_star(I)
    x_0 = calculate_x_0(C_T, gamma, alpha_star, beta_star, I)

    # Only apply deficit for points downstream
    if x < 0:
        return 0

    # Calculate wake widths
    sigma_y = k_star * max(0, (x - x_0)) + np.cos(gamma_rad) / np.sqrt(8) * d_0
    sigma_z = k_star * max(0, (x - x_0)) + 1 / np.sqrt(8) * d_0

    # Calculate deficit
    term1 = 1 - np.sqrt(1 - (C_T / (8 * (sigma_y * sigma_z / (d_0 ** 2 * np.cos(gamma_rad))))))
    term2 = np.exp(-0.5 * ((y / sigma_y) ** 2 + ((z - z_h) / sigma_z) ** 2))

    # Apply a smooth transition that fades out the wake at large distances
    fade_factor = np.exp(-0.001 * x)  # Wake gradually fades at large distances

    delta_U = term1 * term2 * fade_factor
    return delta_U * 11.4  # Scale by reference wind speed


# --- 5. Main function to collect data ---
def collect_wind_farm_data():
    """
    Collect data for different wind directions:
    1. Sweep from 173° to 353°, with finer resolution in interesting ranges
    2. For each wind direction, calculate:
       a) Optimal yaw angles and total power using the PPO model
       b) Total power without yaw optimization (all yaw angles = 0)
    3. Return collected data for animation
    """
    # Load best model
    best_model_path = "./best_model/best_model"
    best_model = PPO.load(best_model_path)

    # Define parameters with variable resolution
    # Create a list for non-uniform wind direction sampling
    directions = []

    # Define the interesting ranges where optimization shows significant effects
    interesting_ranges = [
        (183, 192),  # First interesting range
        (263, 270)  # Second interesting range
    ]

    # Resolution inside and outside interesting ranges
    fine_step = 0.5  # 0.5° step inside interesting ranges
    standard_step = 1.0  # 0.2° step outside interesting ranges

    # Build the list of directions with variable resolution
    current_angle = 173.0
    while current_angle <= 353.0:
        directions.append(current_angle)

        # Check if we're in an interesting range
        in_interesting_range = False
        for start, end in interesting_ranges:
            if start <= current_angle <= end:
                in_interesting_range = True
                break

        # Choose the step size based on whether we're in an interesting range
        step = fine_step if in_interesting_range else standard_step
        current_angle += step

    # Convert to Python's Decimal type for precise floating point operations if needed
    directions = [float(round(dir, 1)) for dir in directions]
    fixed_wind_speed = 11.4

    # Data collections
    all_data = []
    best_opt_powers = []  # Optimized total power (W)
    best_noopt_powers = []  # Non-optimized total power (W)
    best_yaw_list = []  # Optimal yaw angles for each wind direction

    print("Collecting wind farm data for different wind directions...")
    for phi in directions:
        print(f"Processing wind direction: {phi}°")

        # --- 1. With yaw optimization using PPO model ---
        env_test = DummyVecEnv([make_env_for_test_custom(phi, fixed_wind_speed)])
        obs = env_test.reset()
        done = False
        step_count = 0
        max_steps = env_test.get_attr("max_steps")[0]

        best_total_power = -np.inf
        best_yaw = None
        best_data = None

        while not done and step_count < max_steps:
            action, _ = best_model.predict(obs, deterministic=True)
            obs, reward, done_, _ = env_test.step(action)
            done = done_[0]
            step_count += 1

            real_env = env_test.envs[0]
            current_gamma = real_env.current_gammas
            inflows = real_env.current_inflow_speeds

            # Calculate power for each turbine and total power
            powers = np.zeros(real_env.N)
            total_power = 0.0
            for i in range(real_env.N):
                p = power_output(inflows[i], current_gamma[i])
                powers[i] = p
                total_power += p

            # Track best power and corresponding yaw angles
            if total_power > best_total_power:
                best_total_power = total_power
                best_yaw = current_gamma.copy()
                best_data = {
                    'positions': real_env.transformed_positions,
                    'original_positions': real_env.original_turbine_positions,
                    'inflow_speeds': inflows.copy(),
                    'powers': powers.copy(),
                    'total_power': total_power,
                    'gamma_values': current_gamma.copy()
                }

        # Special output for wind direction 272°
        if abs(phi - 272) < 0.1:  # Allow for floating point comparison
            print("\n" + "=" * 50)
            print(f"SPECIAL OUTPUT FOR WIND DIRECTION {phi}°:")
            print(f"Optimal yaw angles: {best_yaw}")
            print(f"Optimized total power: {best_total_power / 1e6:.4f} MW")
            print("=" * 50 + "\n")

        best_opt_powers.append(best_total_power)
        best_yaw_list.append(best_yaw)

        # --- 2. Without yaw optimization ---
        no_yaw_power = get_no_yaw_output_power(phi, fixed_wind_speed)
        best_noopt_powers.append(no_yaw_power)
        no_yaw_data = get_no_yaw_data(phi, fixed_wind_speed)

        # Store both sets of data for this wind direction
        all_data.append({
            'phi': phi,
            'optimized': best_data,
            'no_optimization': no_yaw_data
        })

        # Print summary
        print(f"  Wind Direction = {phi}°")
        print(f"  With Yaw Optimization -> Best Total Power = {best_total_power / 1e6:.4f} MW, Best Yaw = {best_yaw}")
        print(f"  Without Yaw Optimization -> Total Power = {no_yaw_power / 1e6:.4f} MW\n")

    # Convert to arrays and convert units to MW
    directions_arr = np.array(directions)
    best_opt_powers_mw = np.array(best_opt_powers) / 1e6
    best_noopt_powers_mw = np.array(best_noopt_powers) / 1e6

    # Calculate power difference as percentage
    diff_powers_percent = 100 * (best_opt_powers_mw - best_noopt_powers_mw) / best_noopt_powers_mw

    # Store global data
    global_data = {
        'directions': directions_arr,
        'opt_powers': best_opt_powers_mw,
        'noopt_powers': best_noopt_powers_mw,
        'diff_percent': diff_powers_percent,
        'best_yaws': best_yaw_list
    }

    return all_data, global_data


# Modified animation function to create improved visualizations
def create_improved_wind_farm_animation():
    """
    Create an improved wind farm animation with better layout and consistent color scales:
    1. Fixed color scales for wind speed and power output
    2. Better subplot sizing and spacing
    3. Elimination of excess whitespace
    """
    # Define the fixed wind speed
    fixed_wind_speed = 11.4

    # Collect data
    all_data, global_data = collect_wind_farm_data()

    # Create directory for saving frames
    frames_dir = "./wind_frames"
    os.makedirs(frames_dir, exist_ok=True)

    # Define FIXED ranges for consistent color scales across all frames
    # For wind speed, use the fixed environmental wind speed as max
    wind_speed_vmin = 0.0
    wind_speed_vmax = fixed_wind_speed  # Fixed 11.4 m/s environment wind speed

    # For power, use a fixed range that encompasses all possible values
    power_vmin = 0.0  # Minimum power to show
    power_vmax = 5.29  # Maximum possible power

    # Define consistent ranges for difference plots
    speed_diff_vmin = -0.5
    speed_diff_vmax = 0.5
    power_diff_vmin = -0.5
    power_diff_vmax = 1.0

    # Create animation frames
    print("Generating animation frames...")
    frame_info = []  # Will store frame paths and durations

    # Define interesting ranges where frames should appear longer
    interesting_ranges = [
        (183, 192),  # First interesting range
        (263, 270)  # Second interesting range
    ]

    # Base duration and duration multiplier for interesting ranges
    base_duration = 0.2  # seconds
    interesting_multiplier = 1.5  # frames in interesting ranges appear longer

    # Custom colormaps for better visualization
    # Define colors for wind speed colormap (low to high)
    wind_colors = [(0.4, 0.0, 0.6),  # Purple
                   (0.0, 0.0, 0.6),  # Deep blue
                   (0.0, 0.2, 0.8),  # Blue
                   (0.0, 0.6, 0.6),  # Teal-blue
                   (0.4, 0.8, 0.2),  # Light green
                   (0.8, 1.0, 0.0)]  # Yellow-green

    # Create the wind speed colormap
    wind_cmap = LinearSegmentedColormap.from_list('custom_wind', wind_colors, N=256)

    # Define colors for power output colormap (light to dark)
    power_colors = [(1.0, 1.0, 0.8),  # Light yellow
                    (1.0, 0.8, 0.0),  # Yellow
                    (1.0, 0.6, 0.0),  # Orange
                    (1.0, 0.0, 0.0),  # Red
                    (0.8, 0.0, 0.2),  # Dark red
                    (0.5, 0.0, 0.5),  # Purple
                    (0.3, 0.0, 0.5)]  # Deep purple

    # Create the power colormap
    power_cmap = LinearSegmentedColormap.from_list('custom_power', power_colors, N=256)

    for i, data in enumerate(all_data):
        phi = data['phi']
        opt_data = data['optimized']
        no_opt_data = data['no_optimization']

        # Check if this frame is in an interesting range
        in_interesting_range = False
        for start, end in interesting_ranges:
            if start <= phi <= end:
                in_interesting_range = True
                break

        # Set frame duration - longer in interesting ranges
        frame_duration = base_duration * interesting_multiplier if in_interesting_range else base_duration

        # Calculate flow fields in the original coordinate system
        # Define grid size and boundaries based on turbine positions
        pos_opt = np.array(opt_data['original_positions'])
        pos_no = np.array(no_opt_data['original_positions'])

        # Calculate flow fields in the original coordinate system
        x_min = min(pos_opt[:, 0].min(), pos_no[:, 0].min())
        x_max = max(pos_opt[:, 0].max(), pos_no[:, 0].max())
        y_min = min(pos_opt[:, 1].min(), pos_no[:, 1].min())
        y_max = max(pos_opt[:, 1].max(), pos_no[:, 1].max())
        margin = 1000  # Add margin
        grid_size = 100

        # Create grid in original coordinate system
        x_orig = np.linspace(x_min - margin, x_max + margin, grid_size)
        y_orig = np.linspace(y_min - margin, y_max + margin, grid_size)
        X_orig, Y_orig = np.meshgrid(x_orig, y_orig)

        # Initialize velocity fields
        U_opt_orig = np.ones_like(X_orig) * fixed_wind_speed
        U_no_orig = np.ones_like(X_orig) * fixed_wind_speed

        # Calculate wind direction in radians (for rotating vectors)
        wind_dir_rad = np.radians(phi - 270)  # Convert to radians

        # Calculate flow field with optimization
        for j, (x_turb, y_turb, z_turb) in enumerate(opt_data['original_positions']):
            gamma_opt = opt_data['gamma_values'][j]

            for ix in range(grid_size):
                for iy in range(grid_size):
                    grid_x = X_orig[iy, ix]
                    grid_y = Y_orig[iy, ix]

                    # Calculate relative position to turbine
                    rel_x = grid_x - x_turb
                    rel_y = grid_y - y_turb

                    # Rotate to wind direction coordinate system
                    wake_x = rel_x * np.cos(wind_dir_rad) + rel_y * np.sin(wind_dir_rad)
                    wake_y = -rel_x * np.sin(wind_dir_rad) + rel_y * np.cos(wind_dir_rad)

                    # Only calculate for points downstream of turbine in wind direction
                    if wake_x > 0:
                        deficit_opt = calculate_velocity_deficit(
                            wake_x, wake_y, z_turb, z_h, C_T, I, d_0, gamma_opt,
                            alpha_star, beta_star
                        )

                        # Apply deficit
                        U_opt_orig[iy, ix] = max(0, U_opt_orig[iy, ix] - deficit_opt)

        # Calculate flow field without optimization
        for j, (x_turb, y_turb, z_turb) in enumerate(no_opt_data['original_positions']):
            gamma_no = no_opt_data['gamma_values'][j]  # Should be 0

            for ix in range(grid_size):
                for iy in range(grid_size):
                    grid_x = X_orig[iy, ix]
                    grid_y = Y_orig[iy, ix]

                    # Calculate relative position to turbine
                    rel_x = grid_x - x_turb
                    rel_y = grid_y - y_turb

                    # Rotate to wind direction coordinate system
                    wake_x = rel_x * np.cos(wind_dir_rad) + rel_y * np.sin(wind_dir_rad)
                    wake_y = -rel_x * np.sin(wind_dir_rad) + rel_y * np.cos(wind_dir_rad)

                    # Only calculate for points downstream of turbine in wind direction
                    if wake_x > 0:
                        deficit_no = calculate_velocity_deficit(
                            wake_x, wake_y, z_turb, z_h, C_T, I, d_0, gamma_no,
                            alpha_star, beta_star
                        )

                        # Apply deficit
                        U_no_orig[iy, ix] = max(0, U_no_orig[iy, ix] - deficit_no)

        # Calculate differences
        U_diff = U_opt_orig - U_no_orig
        P_diff = (opt_data['powers'] - no_opt_data['powers']) / 1e6  # Convert to MW

        # Create figure with improved layout
        fig = plt.figure(figsize=(20, 14), dpi=100)

        # Divide the figure into separate regions for power curve and subplots
        # First set the overall figure layout
        fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.92, hspace=0.25, wspace=0.3)

        # Create a dedicated GridSpec for the power comparison at the top 20% of the figure
        gs_power = GridSpec(1, 1, figure=fig, top=0.95, bottom=0.75)

        # Create a separate GridSpec for the four subplots in the bottom 75% of the figure
        gs = GridSpec(2, 2, figure=fig, top=0.72, bottom=0.08, hspace=0.3, wspace=0.3)

        # Power comparison plot (top row)
        ax_power = fig.add_subplot(gs_power[0, 0])
        ax_power.plot(global_data['directions'], global_data['opt_powers'], 'b-', label='With Yaw Optimization')
        ax_power.plot(global_data['directions'], global_data['noopt_powers'], 'r-', label='Without Yaw Optimization')
        ax_power.axvline(x=phi, color='k', linestyle='--')
        ax_power.set_xlabel('Wind Direction (°)')
        ax_power.set_ylabel('Total Power (MW)')
        ax_power.set_title(f'Wind Farm Total Power Comparison (Current Direction: {phi}°)')
        ax_power.legend(loc='lower left')
        ax_power.grid(True)

        # Highlight interesting areas in the power comparison plot
        for start, end in interesting_ranges:
            ax_power.axvspan(start, end, alpha=0.2, color='green')

        # Add indicators for interesting ranges
        for start, end in interesting_ranges:
            mid = (start + end) / 2
            ax_power.annotate(f"Significant\nOptimization\n{start}-{end}°",
                              xy=(
                              mid, ax_power.get_ylim()[0] + 0.05 * (ax_power.get_ylim()[1] - ax_power.get_ylim()[0])),
                              xytext=(
                              mid, ax_power.get_ylim()[0] + 0.05 * (ax_power.get_ylim()[1] - ax_power.get_ylim()[0])),
                              ha='center', va='bottom',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.7))

        # Add power gain percentage on secondary y-axis
        ax_gain = ax_power.twinx()
        ax_gain.plot(global_data['directions'], global_data['diff_percent'], 'g-', label='Relative Gain')
        ax_gain.set_ylabel('Relative Gain (%)')
        ax_gain.legend(loc='lower right')

        # VISUALIZATION 1: Combined Wind Speed and Power Output with yaw optimization
        ax_combined_opt = fig.add_subplot(gs[0, 0])
        # Plot wind speed contours with custom colormap
        contour_opt = ax_combined_opt.contourf(
            X_orig, Y_orig, U_opt_orig,
            levels=20, cmap=wind_cmap,
            vmin=wind_speed_vmin, vmax=wind_speed_vmax,
            alpha=0.7
        )
        # Plot turbines colored by power output
        scatter_power_opt = ax_combined_opt.scatter(
            pos_opt[:, 0], pos_opt[:, 1],
            c=opt_data['powers'] / 1e6,  # Convert to MW
            s=300, cmap=power_cmap,
            vmin=power_vmin, vmax=power_vmax,
            edgecolors='k', zorder=10
        )
        ax_combined_opt.set_title('With Yaw Optimization - Power (color) & Wind Speed (background)')
        ax_combined_opt.set_xlabel('X (m)')
        ax_combined_opt.set_ylabel('Y (m)')

        # Add yaw angle arrows
        for j, (x, y, _) in enumerate(opt_data['original_positions']):
            gamma = opt_data['gamma_values'][j]
            if abs(gamma) > 0.1:  # Only show arrows for significant yaw angles
                arrow_angle = phi - 270 + gamma  # Wind direction plus yaw angle
                arrow_rad = np.radians(arrow_angle)
                dx = np.cos(arrow_rad) * 100
                dy = np.sin(arrow_rad) * 100
                ax_combined_opt.arrow(x, y, dx, dy, head_width=50, head_length=50,
                                      fc='r', ec='r', zorder=11)
                # Add text label for yaw angle
                ax_combined_opt.text(x + dx * 1.2, y + dy * 1.2, f"{gamma:.1f}°",
                                     color='red', fontweight='bold', zorder=11)

        # Add wind direction arrow
        ax_combined_opt.arrow(
            -200, -800,
            300 * np.cos(wind_dir_rad), 300 * np.sin(wind_dir_rad),
            head_width=80, head_length=80, fc='blue', ec='blue', zorder=12
        )

        # Add wind direction text label
        ax_combined_opt.text(
            -200, -1000,
            f'Wind: {phi}°',
            fontsize=12, color='blue', zorder=12
        )

        # Set axis limits
        ax_combined_opt.set_aspect('equal')
        ax_combined_opt.set_xlim(x_min - margin, x_max + margin)
        ax_combined_opt.set_ylim(y_min - margin, y_max + margin)

        # VISUALIZATION 2: Combined Wind Speed and Power Output without yaw optimization
        ax_combined_no = fig.add_subplot(gs[0, 1])
        # Use the same custom wind colormap for consistency
        contour_no = ax_combined_no.contourf(
            X_orig, Y_orig, U_no_orig,
            levels=20, cmap=wind_cmap,
            vmin=wind_speed_vmin, vmax=wind_speed_vmax,
            alpha=0.7
        )
        # Use the same custom colormap for consistency
        scatter_power_no = ax_combined_no.scatter(
            pos_no[:, 0], pos_no[:, 1],
            c=no_opt_data['powers'] / 1e6,  # Convert to MW
            s=300, cmap=power_cmap,
            vmin=power_vmin, vmax=power_vmax,
            edgecolors='k', zorder=10
        )
        ax_combined_no.set_title('Without Yaw Optimization - Power (color) & Wind Speed (background)')
        ax_combined_no.set_xlabel('X (m)')
        ax_combined_no.set_ylabel('Y (m)')

        # Add wind direction arrow
        ax_combined_no.arrow(
            -200, -800,
            300 * np.cos(wind_dir_rad), 300 * np.sin(wind_dir_rad),
            head_width=80, head_length=80, fc='blue', ec='blue', zorder=12
        )

        # Add wind direction text label
        ax_combined_no.text(
            -200, -1000,
            f'Wind: {phi}°',
            fontsize=12, color='blue', zorder=12
        )

        # Set axis limits
        ax_combined_no.set_aspect('equal')
        ax_combined_no.set_xlim(x_min - margin, x_max + margin)
        ax_combined_no.set_ylim(y_min - margin, y_max + margin)

        # Add colorbars with distinct color maps and proper labels
        divider = make_axes_locatable(ax_combined_opt)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_wind = fig.colorbar(contour_opt, cax=cax)
        cbar_wind.set_label('Wind Speed (m/s)')

        divider = make_axes_locatable(ax_combined_no)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_power = fig.colorbar(scatter_power_no, cax=cax)
        cbar_power.set_label('Power Output (MW)')

        # Wind speed difference plot
        ax_diff_u = fig.add_subplot(gs[1, 0])
        diff_contour = ax_diff_u.contourf(
            X_orig, Y_orig, U_diff,
            levels=20, cmap='coolwarm',
            vmin=speed_diff_vmin, vmax=speed_diff_vmax  # Consistent fixed range
        )
        ax_diff_u.set_title('Wind Speed Difference (Opt - NoOpt) [m/s]')
        ax_diff_u.set_xlabel('X (m)')
        ax_diff_u.set_ylabel('Y (m)')
        ax_diff_u.set_aspect('equal')
        ax_diff_u.set_xlim(x_min - margin, x_max + margin)
        ax_diff_u.set_ylim(y_min - margin, y_max + margin)

        # Create colorbar with its own separate axis
        divider = make_axes_locatable(ax_diff_u)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_diff_u = fig.colorbar(diff_contour, cax=cax)
        cbar_diff_u.set_label('Δ Wind Speed (m/s)')

        # Power difference plot
        ax_diff_p = fig.add_subplot(gs[1, 1])
        scatter_diff_p = ax_diff_p.scatter(
            pos_opt[:, 0], pos_opt[:, 1],
            c=P_diff, s=300, cmap='coolwarm',
            vmin=power_diff_vmin, vmax=power_diff_vmax,  # Consistent fixed range
            edgecolors='k', zorder=10
        )
        # Add text labels showing power difference values
        for j, (x, y, _) in enumerate(pos_opt):
            label_val = f"{P_diff[j]:.2f}"
            ax_diff_p.text(x + 50, y + 50, label_val, color='black', fontsize=9)

        # Add background contours to show flow field difference
        ax_diff_p.contourf(X_orig, Y_orig, U_diff, levels=20, cmap='coolwarm', alpha=0.3,
                           vmin=speed_diff_vmin, vmax=speed_diff_vmax)
        ax_diff_p.set_title('Power Difference (Opt - NoOpt) [MW]')
        ax_diff_p.set_xlabel('X (m)')
        ax_diff_p.set_ylabel('Y (m)')
        ax_diff_p.set_aspect('equal')
        ax_diff_p.set_xlim(x_min - margin, x_max + margin)
        ax_diff_p.set_ylim(y_min - margin, y_max + margin)

        # Create colorbar with its own separate axis
        divider = make_axes_locatable(ax_diff_p)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_diff_p = fig.colorbar(scatter_diff_p, cax=cax)
        cbar_diff_p.set_label('Δ Power (MW)')

        # Add explanatory text about the visualization
        plt.figtext(0.5, 0.01,
                    "Background color: Wind speed (purple to yellow-green, 0-11.4 m/s)\nCircle color: Power output (light yellow to deep purple, 0-5.29 MW range)",
                    ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

        # Add notice about interesting ranges if we're in one
        if in_interesting_range:
            # Add a text box highlighting this is an interesting range
            plt.figtext(0.5, 0.03,
                        f"SIGNIFICANT OPTIMIZATION RANGE: {phi}°",
                        ha="center", fontsize=14,
                        bbox={"facecolor": "green", "alpha": 0.2, "pad": 5})

        # Save frame
        frame_path = f"{frames_dir}/frame_{i:03d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Store frame info (path and duration)
        frame_info.append({'path': frame_path, 'duration': frame_duration})

        print(f"Generated frame {i + 1}/{len(all_data)}: Wind direction {phi}°")

    # Animation creation code
    print("Creating animation...")
    gif_path = f"{frames_dir}/wind_farm_optimization.gif"

    try:
        # Import imageio.v2 explicitly to avoid deprecation warning
        import imageio.v2 as imageio

        # Create GIF animation with variable frame durations
        print(f"Creating GIF animation: {gif_path}")

        # Read all images first
        images = []
        durations = []
        for frame in frame_info:
            images.append(imageio.imread(frame['path']))
            durations.append(frame['duration'] * 1000)  # Convert to milliseconds

        # Write GIF with durations
        imageio.mimsave(gif_path, images, duration=durations)

        print(f"GIF animation created successfully in the '{frames_dir}' folder")

    except Exception as e:
        print(f"Error creating animation: {e}")
        print(f"Individual frames are still available in the '{frames_dir}' folder")


if __name__ == "__main__":
    # Run the improved animation function
    create_improved_wind_farm_animation()