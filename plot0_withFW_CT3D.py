#!/usr/bin/env python3
"""
plot0_with_FW_CT3D.py: Trajectory visualization for EKF vs DR-EKF comparison results from main0_withFW_CT3D.py
Creates 3D trajectory plots with 1-std uncertainty tubes for 3D CT model with radar measurements.
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_data(results_path, dist):
    """Load saved results from main0.py experiments."""
    optimal_file = os.path.join(results_path, f'optimal_results_{dist}.pkl')
    all_results_file = os.path.join(results_path, f'all_results_{dist}.pkl')
    
    if not os.path.exists(optimal_file):
        raise FileNotFoundError(f"Optimal results file not found: {optimal_file}")
    
    with open(optimal_file, 'rb') as f:
        optimal_results = pickle.load(f)
    
    all_results = None
    if os.path.exists(all_results_file):
        with open(all_results_file, 'rb') as f:
            all_results = pickle.load(f)
    
    return optimal_results, all_results

def load_detailed_results_for_filter(results_path, filter_name, theta_vals, dist):
    """Load detailed trajectory data for a specific filter with its optimal theta values."""

    # Construct filename based on filter type and theta values
    if filter_name == 'EKF':
        filename = f'detailed_results_{filter_name}_{dist}.pkl'
    elif filter_name in ['DR_EKF_CDC', 'DR_EKF_CDC_FW']:
        theta_x, theta_v = theta_vals['theta_x'], theta_vals['theta_v']
        filename = f'detailed_results_{filter_name}_tx{theta_x}_tv{theta_v}_{dist}.pkl'
    elif filter_name == 'DR_EKF_TAC':
        theta_x, theta_v, theta_w = theta_vals['theta_x'], theta_vals['theta_v'], theta_vals['theta_w']
        filename = f'detailed_results_{filter_name}_tx{theta_x}_tv{theta_v}_tw{theta_w}_{dist}.pkl'
    else:
        raise ValueError(f"Unknown filter name: {filter_name}")

    detailed_file = os.path.join(results_path, filename)

    if not os.path.exists(detailed_file):
        raise FileNotFoundError(f"Detailed results file not found: {detailed_file}")

    with open(detailed_file, 'rb') as f:
        detailed_results = pickle.load(f)

    return detailed_results

def generate_desired_trajectory(T_total=10.0, dt=0.2):
    """Generate desired trajectory matching main0_withFW_CT3D.py (3D CT model)."""
    time_steps = int(T_total / dt) + 1
    time = np.linspace(0, T_total, time_steps)
    
    # Curvy trajectory parameters
    Amp = 5.0       # Amplitude of sinusoidal x motion
    slope = 1.0     # Linear slope for y motion  
    omega = 0.5     # Frequency of sinusoidal motion
    
    # Position trajectory
    px_d = Amp * np.sin(omega * time)
    py_d = slope * time
    pz_d = 0.5 * np.sin(0.3 * omega * time)  # Gentle altitude variation
    
    # Velocity trajectory (derivatives of position)
    vx_d = Amp * omega * np.cos(omega * time)
    vy_d = slope * np.ones(time_steps)
    vz_d = 0.5 * 0.3 * omega * np.cos(0.3 * omega * time)  # dz/dt
    
    # Turn rate trajectory (derivative of heading)
    # For this simple trajectory, we'll use a small constant turn rate
    omega_d = 0.1 * np.sin(0.3 * time)  # Small oscillating turn rate
    
    return np.array([px_d, py_d, pz_d, vx_d, vy_d, vz_d, omega_d]), time

def extract_trajectory_data_from_saved(optimal_results, results_path, dist):
    """Extract trajectory data for each filter using saved detailed results."""
    filters_order = ['EKF', 'DR_EKF_TAC', 'DR_EKF_CDC', 'DR_EKF_CDC_FW']
    trajectory_data = {}

    for filt in filters_order:
        if filt not in optimal_results:
            print(f"Warning: Filter '{filt}' not found in optimal results, skipping...")
            continue

        # Get optimal theta values for this filter
        optimal_stats = optimal_results[filt]

        if filt == 'EKF':
            theta_vals = {}
            theta_str = "N/A"
        elif filt in ['DR_EKF_CDC', 'DR_EKF_CDC_FW']:
            theta_vals = {
                'theta_x': optimal_stats['theta_x'],
                'theta_v': optimal_stats['theta_v']
            }
            theta_str = f"θ_x={theta_vals['theta_x']}, θ_v={theta_vals['theta_v']}"
        elif filt == 'DR_EKF_TAC':
            theta_vals = {
                'theta_x': optimal_stats['theta_x'],
                'theta_v': optimal_stats['theta_v'],
                'theta_w': optimal_stats['theta_w']
            }
            theta_str = f"θ_x={theta_vals['theta_x']}, θ_v={theta_vals['theta_v']}, θ_w={theta_vals['theta_w']}"

        print(f"Loading trajectory data for {filt} with {theta_str}")

        try:
            # Load detailed results for this filter
            detailed_results = load_detailed_results_for_filter(results_path, filt, theta_vals, dist)

            if filt not in detailed_results:
                print(f"Warning: Filter {filt} not found in detailed results")
                continue

            # Extract simulation results
            filter_results = detailed_results[filt]
            sim_results = filter_results['results']  # List of simulation results

            est_trajectories = []
            true_trajectories = []

            for result in sim_results:  # Each simulation result
                # Estimated trajectory
                est_traj = result['est_state_traj']  # Shape: (T+1, nx, 1)
                est_trajectories.append(np.squeeze(est_traj, axis=-1))  # Remove last dimension

                # True trajectory (for reference)
                true_traj = result['state_traj']  # Shape: (T+1, nx, 1)
                true_trajectories.append(np.squeeze(true_traj, axis=-1))

            if est_trajectories:
                # Convert to numpy array and compute statistics
                est_trajectories = np.array(est_trajectories)  # Shape: (num_runs, time_steps, state_dim)
                true_trajectories = np.array(true_trajectories)

                # Compute mean and std for estimated trajectories (what we want to plot)
                mean_traj = np.mean(est_trajectories, axis=0)  # (time_steps, state_dim)
                std_traj = np.std(est_trajectories, axis=0)    # (time_steps, state_dim)

                trajectory_data[filt] = {
                    'mean': mean_traj,
                    'std': std_traj,
                    'theta_vals': theta_vals,
                    'theta_str': theta_str,
                    'true_mean': np.mean(true_trajectories, axis=0),  # For reference
                    'num_sims': len(est_trajectories)
                }
                print(f"Successfully loaded {len(est_trajectories)} trajectories for {filt}")
            else:
                print(f"No trajectory data found for {filt}")

        except FileNotFoundError as e:
            print(f"Could not load detailed results for {filt}: {e}")
            continue
        except Exception as e:
            print(f"Error loading data for {filt}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return trajectory_data, filters_order

def plot_trajectory_subplots(trajectory_data, filters_order, desired_traj, time, dist):
    """Create individual 3D plots for each filter showing trajectory visualization"""
    
    # Colors for the four filters (consistent with EKF naming)
    colors = {
        'EKF': '#1f77b4',           # Blue
        'DR_EKF_CDC': '#2ca02c',    # Green 
        'DR_EKF_TAC': '#d62728',    # Red
        'DR_EKF_CDC_FW': '#ff7f0e'  # Orange
    }
    
    # Filter names for titles
    filter_names = {
        'EKF': "Extended Kalman Filter (EKF)",
        'DR_EKF_CDC': "DR-EKF (CDC)",
        'DR_EKF_TAC': "DR-EKF (TAC)",
        'DR_EKF_CDC_FW': "DR-EKF (CDC) with Frank-Wolfe"
    }
    
    # Create results directory - updated for CT data
    results_dir = os.path.join("results", "EKF_comparison_with_FW_CT3D")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    saved_files = []
    
    # Create individual plots for each filter
    for filt in filters_order:
        if filt not in trajectory_data:
            continue
            
        # Get trajectory data
        mean_traj = trajectory_data[filt]['mean']
        std_traj = trajectory_data[filt]['std']
        theta_str = trajectory_data[filt]['theta_str']
        color = colors[filt]
        
        # Create 3D visualization with 3 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract position data
        px_mean = mean_traj[:, 0]  # X position
        py_mean = mean_traj[:, 1]  # Y position  
        pz_mean = mean_traj[:, 2]  # Z position
        px_std = std_traj[:, 0]    # X position std
        py_std = std_traj[:, 1]    # Y position std
        pz_std = std_traj[:, 2]    # Z position std
        
        # Get true trajectory if available
        true_mean = None
        if 'true_mean' in trajectory_data[filt]:
            true_mean = trajectory_data[filt]['true_mean']
        
        # --- Subplot 1: XY plane ---
        ax = ax1
        plot_2d_trajectory_with_tube(ax, px_mean, py_mean, px_std, py_std, 
                                   color, true_mean[:, 0] if true_mean is not None else None, 
                                   true_mean[:, 1] if true_mean is not None else None)
        ax.set_xlabel('X position', fontsize=16)
        ax.set_ylabel('Y position', fontsize=16)
        ax.set_title('XY Plane', fontsize=18)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        # --- Subplot 2: XZ plane ---
        ax = ax2
        plot_2d_trajectory_with_tube(ax, px_mean, pz_mean, px_std, pz_std,
                                   color, true_mean[:, 0] if true_mean is not None else None, 
                                   true_mean[:, 2] if true_mean is not None else None)
        ax.set_xlabel('X position', fontsize=16)
        ax.set_ylabel('Z position (altitude)', fontsize=16)
        ax.set_title('XZ Plane', fontsize=18)
        ax.grid(True, alpha=0.3)
        
        # --- Subplot 3: Altitude vs Time ---
        ax = ax3
        tube_scale = 0.1  # Match the tube scaling from 2D plots
        ax.fill_between(time, pz_mean - tube_scale*pz_std, pz_mean + tube_scale*pz_std, 
                       color=color, alpha=0.3, label='0.1-std band')
        ax.plot(time, pz_mean, '-', color=color, linewidth=2.5, label='Estimated Trajectory')
        if true_mean is not None:
            ax.plot(time, true_mean[:, 2], ':', color='red', linewidth=2.0, 
                   alpha=0.8, label='True Trajectory')
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Z position (altitude)', fontsize=16)
        ax.set_title('Altitude vs Time', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best')
        
        # --- Subplot 4: Leave blank or add YZ plane ---
        ax4.set_visible(False)
        
        # Overall title
        if filt == 'EKF':
            title_text = f"{filter_names[filt]}"
        else:
            title_text = f"{filter_names[filt]}\n({theta_str})"
        fig.suptitle(title_text, fontsize=20, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save 3D plot
        save_path = os.path.join(results_dir, f"traj_3d_EKF_{filt}_{dist}.pdf")
        saved_files.append(f"traj_3d_EKF_{filt}_{dist}.pdf")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
    
    print(f"\nEKF trajectory plots saved to:")
    for filename in saved_files:
        print(f"- ./results/EKF_comparison_with_FW_CT3D/{filename}")

def plot_2d_trajectory_with_tube(ax, x_mean, y_mean, x_std, y_std, color, true_x=None, true_y=None):
    """Helper function to plot 2D trajectory with uncertainty tube"""
    # Create shaded tube for ±0.1 standard deviation
    std_mag = 0.1 * 0.5 * (x_std + y_std)  # Average std as radius, scaled by 0.1
    
    # Create uncertainty tube by offsetting perpendicular to trajectory
    dx = np.gradient(x_mean)
    dy = np.gradient(y_mean)
    norms = np.hypot(dx, dy)
    norms[norms == 0] = 1.0  # Avoid division by zero
    dx /= norms
    dy /= norms
    
    # Perpendicular directions
    perp_x = -dy
    perp_y = dx
    
    # Create upper and lower bounds
    upper_x = x_mean + perp_x * std_mag
    upper_y = y_mean + perp_y * std_mag
    lower_x = x_mean - perp_x * std_mag
    lower_y = y_mean - perp_y * std_mag
    
    # Create polygon for shaded tube
    tube_x = np.concatenate([upper_x, lower_x[::-1]])
    tube_y = np.concatenate([upper_y, lower_y[::-1]])
    
    # Plot shaded tube first
    ax.fill(tube_x, tube_y, color=color, alpha=0.3, label='0.1-std tube')
    
    # Plot mean estimated trajectory
    ax.plot(x_mean, y_mean, '-', color=color, linewidth=2.5, label='Estimated Trajectory')
    
    # Plot true trajectory if provided
    if true_x is not None and true_y is not None:
        ax.plot(true_x, true_y, ':', color='red', linewidth=2.0, alpha=0.8, label='True Trajectory')
        
        # Mark start and end positions
        ax.scatter(true_x[0], true_y[0], marker='X', s=150, color='red', linewidth=3)
        ax.scatter(true_x[-1], true_y[-1], marker='X', s=150, color='red', linewidth=3)

def plot_subplots_all_filters(trajectory_data, filters_order, desired_traj, time, dist):
    """Create a subplot figure with all filters in separate subplots for 3D visualization"""
    
    # Colors for the four filters
    colors = {
        'EKF': '#1f77b4',           # Blue
        'DR_EKF_CDC': '#2ca02c',    # Green 
        'DR_EKF_TAC': '#d62728',    # Red
        'DR_EKF_CDC_FW': '#ff7f0e'  # Orange
    }
    
    # Filter names for subplot titles
    filter_names = {
        'EKF': "Extended Kalman Filter (EKF)",
        'DR_EKF_CDC': "DR-EKF (CDC)",
        'DR_EKF_TAC': "DR-EKF (TAC)",
        'DR_EKF_CDC_FW': "DR-EKF (CDC) with Frank-Wolfe"
    }
    
    # Alphabet mapping for subplot labels
    alphabet_mapping = {filt: chr(ord('A') + i) for i, filt in enumerate(filters_order)}
    
    # Calculate available filters
    available_filters = [filt for filt in filters_order if filt in trajectory_data]
    n_filters = len(available_filters)
    
    # For 3D, create two separate figures: one for XY trajectories, one for altitude vs time
    
    # --- XY Trajectories Figure ---
    fig_xy, axes_xy = plt.subplots(2, 2, figsize=(12, 10))
    axes_xy = axes_xy.flatten()
    
    # --- Altitude vs Time Figure ---
    fig_pz, axes_pz = plt.subplots(2, 2, figsize=(12, 10))
    axes_pz = axes_pz.flatten()
    
    # Plot each filter
    for idx, filt in enumerate(available_filters):
        if idx >= 4:
            break
            
        color = colors[filt]
        
        # Get trajectory data
        mean_traj = trajectory_data[filt]['mean']
        std_traj = trajectory_data[filt]['std']
        theta_str = trajectory_data[filt]['theta_str']

        # Extract position data
        px_mean = mean_traj[:, 0]
        py_mean = mean_traj[:, 1]
        pz_mean = mean_traj[:, 2]
        px_std = std_traj[:, 0]
        py_std = std_traj[:, 1]
        pz_std = std_traj[:, 2]
        
        # Get true trajectory if available
        true_mean = None
        if 'true_mean' in trajectory_data[filt]:
            true_mean = trajectory_data[filt]['true_mean']
        
        # --- XY subplot ---
        ax_xy = axes_xy[idx]
        plot_2d_trajectory_with_tube(ax_xy, px_mean, py_mean, px_std, py_std, color,
                                   true_mean[:, 0] if true_mean is not None else None,
                                   true_mean[:, 1] if true_mean is not None else None)
        
        alphabet_label = alphabet_mapping[filt]
        title_text = f"({alphabet_label}) {filter_names[filt]}"
        ax_xy.set_title(title_text, fontsize=16, pad=8)
        ax_xy.set_xlabel('X position', fontsize=14)
        ax_xy.set_ylabel('Y position', fontsize=14)
        ax_xy.tick_params(axis='both', which='major', labelsize=12)
        ax_xy.grid(True, alpha=0.3)
        ax_xy.set_aspect('equal', adjustable='box')
        
        # --- Altitude vs Time subplot ---
        ax_pz = axes_pz[idx]
        tube_scale = 0.1
        ax_pz.fill_between(time, pz_mean - tube_scale*pz_std, pz_mean + tube_scale*pz_std,
                          color=color, alpha=0.3)
        ax_pz.plot(time, pz_mean, '-', color=color, linewidth=2)
        if true_mean is not None:
            ax_pz.plot(time, true_mean[:, 2], ':', color='red', linewidth=1.5, alpha=0.8)
        
        ax_pz.set_title(title_text, fontsize=16, pad=8)
        ax_pz.set_xlabel('Time (s)', fontsize=14)
        ax_pz.set_ylabel('Z position (altitude)', fontsize=14)
        ax_pz.tick_params(axis='both', which='major', labelsize=12)
        ax_pz.grid(True, alpha=0.3)
    
    # Hide unused subplots in both figures
    for idx in range(len(available_filters), 4):
        if idx < len(axes_xy):
            axes_xy[idx].set_visible(False)
            axes_pz[idx].set_visible(False)
    
    # Create legends for both figures
    legend_elements = [
        plt.Line2D([0], [0], color='red', linestyle=':', linewidth=1.5, label='True Trajectory'),
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3, label='0.1-std tube'),
        plt.Line2D([0], [0], color='gray', linewidth=2, label='Estimated Trajectory')
    ]
    
    # XY figure
    fig_xy.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                 ncol=3, fontsize=16, frameon=True)
    plt.figure(fig_xy.number)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.95, hspace=0.45, wspace=0.3)
    
    # Altitude figure  
    fig_pz.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                 ncol=3, fontsize=16, frameon=True)
    plt.figure(fig_pz.number)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.95, hspace=0.45, wspace=0.3)
    
    # Save both figures
    results_dir = os.path.join("results", "EKF_comparison_with_FW_CT3D")
    save_path_xy = os.path.join(results_dir, f"traj_3d_EKF_subplots_xy_{dist}.pdf")
    save_path_pz = os.path.join(results_dir, f"traj_3d_EKF_subplots_pz_{dist}.pdf")
    
    plt.figure(fig_xy.number)
    plt.savefig(save_path_xy, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig_xy)
    
    plt.figure(fig_pz.number)
    plt.savefig(save_path_pz, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig_pz)
    
    print(f"3D EKF subplots saved to:")
    print(f"- ./results/EKF_comparison_with_FW_CT3D/traj_3d_EKF_subplots_xy_{dist}.pdf")
    print(f"- ./results/EKF_comparison_with_FW_CT3D/traj_3d_EKF_subplots_pz_{dist}.pdf")

def plot_mse_heatmaps(all_results, dist):
    """Create heatmap plots for MSE vs theta parameters (2D parameter space)."""
    if all_results is None:
        print("Warning: No all_results data available. Cannot plot MSE heatmaps.")
        return

    results_dir = os.path.join("results", "EKF_comparison_with_FW_CT3D")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Plot CDC and CDC-FW filters (theta_x vs theta_v)
    cdc_filters = ['DR_EKF_CDC', 'DR_EKF_CDC_FW']
    for filt in cdc_filters:
        if filt not in all_results or not all_results[filt]:
            continue

        # Extract theta_x and theta_v values
        theta_data = {}
        for theta_key, results in all_results[filt].items():
            theta_x, theta_v = theta_key
            if theta_x not in theta_data:
                theta_data[theta_x] = {}
            theta_data[theta_x][theta_v] = results['mse_mean']

        # Create grid
        theta_x_vals = sorted(theta_data.keys())
        theta_v_vals = sorted(set(tv for tx_dict in theta_data.values() for tv in tx_dict.keys()))

        # Create MSE matrix
        mse_matrix = np.zeros((len(theta_v_vals), len(theta_x_vals)))
        for i, theta_v in enumerate(theta_v_vals):
            for j, theta_x in enumerate(theta_x_vals):
                if theta_v in theta_data[theta_x]:
                    mse_matrix[i, j] = theta_data[theta_x][theta_v]
                else:
                    mse_matrix[i, j] = np.nan

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mse_matrix, aspect='auto', cmap='viridis', origin='lower')

        # Set ticks
        ax.set_xticks(np.arange(len(theta_x_vals)))
        ax.set_yticks(np.arange(len(theta_v_vals)))
        ax.set_xticklabels([f'{tx:.2f}' for tx in theta_x_vals], fontsize=14)
        ax.set_yticklabels([f'{tv:.2f}' for tv in theta_v_vals], fontsize=14)

        ax.set_xlabel('θ_x', fontsize=18)
        ax.set_ylabel('θ_v', fontsize=18)
        ax.set_title(f'MSE Heatmap: {filt}', fontsize=20)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('MSE', fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # Add text annotations
        for i in range(len(theta_v_vals)):
            for j in range(len(theta_x_vals)):
                if not np.isnan(mse_matrix[i, j]):
                    text = ax.text(j, i, f'{mse_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="w", fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(results_dir, f"mse_heatmap_{filt}_{dist}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved MSE heatmap: {save_path}")

    # Plot TAC filter (theta_v vs theta_w with fixed theta_x)
    if 'DR_EKF_TAC' in all_results and all_results['DR_EKF_TAC']:
        filt = 'DR_EKF_TAC'

        # Extract theta_v and theta_w values (theta_x is fixed)
        theta_data = {}
        fixed_theta_x = None
        for theta_key, results in all_results[filt].items():
            theta_x, theta_v, theta_w = theta_key
            if fixed_theta_x is None:
                fixed_theta_x = theta_x
            if theta_v not in theta_data:
                theta_data[theta_v] = {}
            theta_data[theta_v][theta_w] = results['mse_mean']

        # Create grid
        theta_v_vals = sorted(theta_data.keys())
        theta_w_vals = sorted(set(tw for tv_dict in theta_data.values() for tw in tv_dict.keys()))

        # Create MSE matrix
        mse_matrix = np.zeros((len(theta_w_vals), len(theta_v_vals)))
        for i, theta_w in enumerate(theta_w_vals):
            for j, theta_v in enumerate(theta_v_vals):
                if theta_w in theta_data[theta_v]:
                    mse_matrix[i, j] = theta_data[theta_v][theta_w]
                else:
                    mse_matrix[i, j] = np.nan

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mse_matrix, aspect='auto', cmap='viridis', origin='lower')

        # Set ticks
        ax.set_xticks(np.arange(len(theta_v_vals)))
        ax.set_yticks(np.arange(len(theta_w_vals)))
        ax.set_xticklabels([f'{tv:.2f}' for tv in theta_v_vals], fontsize=14)
        ax.set_yticklabels([f'{tw:.2f}' for tw in theta_w_vals], fontsize=14)

        ax.set_xlabel('θ_v', fontsize=18)
        ax.set_ylabel('θ_w', fontsize=18)
        ax.set_title(f'MSE Heatmap: {filt} (θ_x={fixed_theta_x} fixed)', fontsize=20)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('MSE', fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # Add text annotations
        for i in range(len(theta_w_vals)):
            for j in range(len(theta_v_vals)):
                if not np.isnan(mse_matrix[i, j]):
                    text = ax.text(j, i, f'{mse_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="w", fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(results_dir, f"mse_heatmap_{filt}_{dist}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved MSE heatmap: {save_path}")



def create_violin_plots(optimal_results, results_path, dist):
    """Create violin plots showing distribution of MSE for optimal parameters only"""

    print(f"Creating violin plots for optimal parameters ({dist} distribution)...")

    # Define filter order and colors
    filters_order = ['EKF', 'DR_EKF_TAC', 'DR_EKF_CDC', 'DR_EKF_CDC_FW']
    colors = {
        'EKF': '#1f77b4',           # Blue
        'DR_EKF_CDC': '#2ca02c',    # Green
        'DR_EKF_TAC': '#d62728',    # Red
        'DR_EKF_CDC_FW': '#ff7f0e'  # Orange
    }

    filter_names = {
        'EKF': "Extended Kalman Filter",
        'DR_EKF_CDC': "DR-EKF (CDC)",
        'DR_EKF_TAC': "DR-EKF (TAC)",
        'DR_EKF_CDC_FW': "DR-EKF (CDC) with Frank-Wolfe"
    }

    # Create violin plot for MSE
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for violin plot (only optimal parameters)
    violin_data = []
    labels = []
    positions = []
    filter_colors = []

    pos_counter = 1
    for filt in filters_order:
        if filt not in optimal_results:
            continue

        optimal_stats = optimal_results[filt]

        # Get optimal theta values for this filter
        if filt == 'EKF':
            theta_vals = {}
            theta_str = "N/A"
        elif filt in ['DR_EKF_CDC', 'DR_EKF_CDC_FW']:
            theta_vals = {
                'theta_x': optimal_stats['theta_x'],
                'theta_v': optimal_stats['theta_v']
            }
            theta_str = f"θ_x={theta_vals['theta_x']}, θ_v={theta_vals['theta_v']}"
        elif filt == 'DR_EKF_TAC':
            theta_vals = {
                'theta_x': optimal_stats['theta_x'],
                'theta_v': optimal_stats['theta_v'],
                'theta_w': optimal_stats['theta_w']
            }
            theta_str = f"θ_x={theta_vals['theta_x']}, θ_v={theta_vals['theta_v']}, θ_w={theta_vals['theta_w']}"

        try:
            # Load detailed results for this filter with optimal theta values
            detailed_results = load_detailed_results_for_filter(results_path, filt, theta_vals, dist)

            if filt in detailed_results:
                filter_results = detailed_results[filt]
                sim_results = filter_results['results']  # List of simulation results

                # Extract raw MSE data
                raw_data = [np.mean(r['mse']) for r in sim_results]

                if len(raw_data) > 0:
                    violin_data.append(raw_data)
                    if filt == 'EKF':
                        labels.append(f"{filter_names[filt]}")
                    else:
                        labels.append(f"{filter_names[filt]}\n({theta_str})")
                    positions.append(pos_counter)
                    filter_colors.append(colors[filt])
                    pos_counter += 1
                    print(f"Loaded {len(raw_data)} MSE samples for {filt}")
        except FileNotFoundError as e:
            print(f"Could not load detailed results for {filt}: {e}")
            continue
        except Exception as e:
            print(f"Error loading data for {filt}: {e}")
            continue

    if not violin_data:
        print(f"Warning: No violin data available - skipping violin plot")
        plt.close()
        return

    # Create violin plot
    parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True, widths=0.8)

    # Color each violin by filter
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(filter_colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)

    # Customize violin plot elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.5)

    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=14)

    ax.set_ylabel('MSE Distribution', fontsize=16, labelpad=15)
    ax.set_title(f'MSE Distribution at Optimal Parameters ({dist.title()})', fontsize=18)
    ax.set_yscale('log')  # Log scale for MSE

    ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.4)
    ax.tick_params(axis='both', which='major', width=1.5, length=6)
    ax.tick_params(axis='y', which='major', labelsize=14)

    plt.tight_layout()

    # Save plot
    results_dir = os.path.join("results", "EKF_comparison_with_FW_CT3D")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_path = os.path.join(results_dir, f'violin_plot_mse_{dist}.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved as: {output_path}")

def print_optimal_results_summary(optimal_results):
    """Print summary of optimal results for each filter."""
    print("\nOptimal Results Summary:")
    print("=" * 80)
    for filter_name, results in optimal_results.items():
        mse_mean = results.get('mse_mean', 'N/A')
        mse_std = results.get('mse_std', 'N/A')

        if filter_name == 'EKF':
            theta_str = "N/A"
        elif filter_name in ['DR_EKF_CDC', 'DR_EKF_CDC_FW']:
            theta_x = results.get('theta_x', 'N/A')
            theta_v = results.get('theta_v', 'N/A')
            theta_str = f"θ_x={theta_x}, θ_v={theta_v}"
        elif filter_name == 'DR_EKF_TAC':
            theta_x = results.get('theta_x', 'N/A')
            theta_v = results.get('theta_v', 'N/A')
            theta_w = results.get('theta_w', 'N/A')
            theta_str = f"θ_x={theta_x}, θ_v={theta_v}, θ_w={theta_w}"
        else:
            theta_str = "Unknown"

        print(f"{filter_name:20s}: {theta_str:40s} MSE={mse_mean:.6f}±{mse_std:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Create 3D trajectory plots for EKF vs DR-EKF comparison from main0_withFW_CT3D.py (3D CT model)')
    parser.add_argument('--dist', default='normal', choices=['normal', 'quadratic'],
                        help='Distribution type to plot trajectories for')
    parser.add_argument('--individual_only', action='store_true',
                        help='Create only individual plots, not subplots')
    parser.add_argument('--subplots_only', action='store_true',
                        help='Create only subplot figure, not individual plots')
    parser.add_argument('--heatmaps_only', action='store_true',
                        help='Create only MSE heatmaps, skip trajectory plots')
    args = parser.parse_args()

    try:
        # Load optimal results and all results
        results_path = "./results/EKF_comparison_with_FW_CT3D/"
        optimal_results, all_results = load_data(results_path, args.dist)

        # Print optimal results summary
        print_optimal_results_summary(optimal_results)

        # Create MSE heatmaps (2D parameter space visualization)
        print(f"\nCreating MSE heatmap plots...")
        plot_mse_heatmaps(all_results, args.dist)

        # Create violin plots for optimal parameters
        print(f"\nCreating violin plots...")
        create_violin_plots(optimal_results, results_path, args.dist)

        # Create trajectory plots unless only heatmaps requested
        if not args.heatmaps_only:
            # Extract trajectory data from saved detailed results first
            trajectory_data, filters_order = extract_trajectory_data_from_saved(optimal_results, results_path, args.dist)
            
            # Determine actual time length from trajectory data
            if trajectory_data:
                first_filter = next(iter(trajectory_data.keys()))
                actual_num_steps = len(trajectory_data[first_filter]['mean'])  # Use mean trajectory length
                dt = 0.2  # Same as in main script
                actual_time = (actual_num_steps - 1) * dt
                print(f"Detected actual simulation time: {actual_time:.1f}s ({actual_num_steps} steps)")
                
                # Generate desired trajectory with correct time length
                desired_traj, time = generate_desired_trajectory(actual_time, dt)
            else:
                print("No trajectory data found, using default time (10.0s)")
                desired_traj, time = generate_desired_trajectory(10.0)
            
            print(f"\nFound trajectory data for {len(trajectory_data)} filters: {list(trajectory_data.keys())}")
            
            if len(trajectory_data) == 0:
                print("No trajectory data found. You need to run main0_withFW_CT3D.py first to generate the detailed trajectory files.")
                print("The detailed trajectory files should be saved as detailed_results_<theta>_<dist>.pkl")
                if not args.performance_plots:
                    return
            else:
                # Create plots based on arguments
                if not args.subplots_only:
                    # Create individual trajectory plots
                    plot_trajectory_subplots(trajectory_data, filters_order, desired_traj, time, args.dist)
                
                if not args.individual_only:
                    # Create subplots figure
                    plot_subplots_all_filters(trajectory_data, filters_order, desired_traj, time, args.dist)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find results files for distribution '{args.dist}'")
        print(f"Make sure you have run main0_withFW_CT3D.py with --dist {args.dist} first")
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"Error creating trajectory plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()