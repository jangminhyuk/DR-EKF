#!/usr/bin/env python3
"""
plot0.py: Trajectory visualization for EKF vs DR-EKF comparison results from main0.py
Creates 2D trajectory plots with 1-std uncertainty tubes similar to plot2_with_MPC.py design.
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

def load_detailed_results(results_path, theta_val, dist):
    """Load detailed trajectory data for a specific theta value."""
    detailed_file = os.path.join(results_path, f'detailed_results_{theta_val}_{dist}.pkl')
    
    if not os.path.exists(detailed_file):
        raise FileNotFoundError(f"Detailed results file not found: {detailed_file}")
    
    with open(detailed_file, 'rb') as f:
        detailed_results = pickle.load(f)
    
    return detailed_results

def generate_desired_trajectory(T_total=10.0, dt=0.2):
    """Generate desired trajectory matching main0.py."""
    time_steps = int(T_total / dt) + 1
    time = np.linspace(0, T_total, time_steps)
    
    # Curvy trajectory parameters
    Amp = 5.0       # Amplitude of sinusoidal x motion
    slope = 1.0     # Linear slope for y motion  
    omega = 0.5     # Frequency of sinusoidal motion
    
    # Position trajectory (for unicycle: px, py, theta)
    x_d = Amp * np.sin(omega * time)
    y_d = slope * time
    
    # Compute orientation from velocity direction
    vx_d = Amp * omega * np.cos(omega * time)
    vy_d = slope * np.ones(time_steps)
    theta_d = np.arctan2(vy_d, vx_d)
    
    return np.array([x_d, y_d, theta_d]), time

def extract_trajectory_data_from_saved(optimal_results, results_path, dist):
    """Extract trajectory data for each filter using saved detailed results."""
    filters_order = ['EKF', 'DR_EKF_CDC', 'DR_EKF_TAC']
    trajectory_data = {}
    
    for filt in filters_order:
        if filt not in optimal_results:
            print(f"Warning: Filter '{filt}' not found in optimal results, skipping...")
            continue
        
        # Get optimal theta for this filter
        optimal_theta = optimal_results[filt]['theta']
        print(f"Loading trajectory data for {filt} with θ* = {optimal_theta}")
        
        try:
            # Load detailed results for this theta value
            detailed_results = load_detailed_results(results_path, optimal_theta, dist)
            
            if filt not in detailed_results:
                print(f"Warning: Filter {filt} not found in detailed results for theta {optimal_theta}")
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
                    'optimal_theta': optimal_theta,
                    'true_mean': np.mean(true_trajectories, axis=0),  # For reference
                    'num_sims': len(est_trajectories)
                }
                print(f"Successfully loaded {len(est_trajectories)} trajectories for {filt}")
            else:
                print(f"No trajectory data found for {filt}")
                
        except FileNotFoundError as e:
            print(f"Could not load detailed results for {filt} (θ={optimal_theta}): {e}")
            continue
        except Exception as e:
            print(f"Error loading data for {filt}: {e}")
            continue
    
    return trajectory_data, filters_order

def plot_trajectory_subplots(trajectory_data, filters_order, desired_traj, time, dist):
    """Create individual plots for each filter showing 2D X-Y position trajectories"""
    
    # Colors for the three filters (consistent with EKF naming)
    colors = {
        'EKF': '#1f77b4',           # Blue
        'DR_EKF_CDC': '#2ca02c',    # Green 
        'DR_EKF_TAC': '#d62728'     # Red
    }
    
    # Filter names for titles
    filter_names = {
        'EKF': "Extended Kalman Filter (EKF)",
        'DR_EKF_CDC': "DR-EKF (CDC)",
        'DR_EKF_TAC': "DR-EKF (TAC)"
    }
    
    # Create results directory
    results_dir = os.path.join("results", "EKF_trajectory_plots")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    saved_files = []
    
    # Create individual plots for each filter
    for filt in filters_order:
        if filt not in trajectory_data:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 8))
        color = colors[filt]
        
        # Get trajectory data
        mean_traj = trajectory_data[filt]['mean']
        std_traj = trajectory_data[filt]['std']
        optimal_theta = trajectory_data[filt]['optimal_theta']
        
        # Extract X and Y positions (unicycle state: [px, py, theta])
        x_mean = mean_traj[:, 0]  # X position
        y_mean = mean_traj[:, 1]  # Y position
        x_std = std_traj[:, 0]    # X position std
        y_std = std_traj[:, 1]    # Y position std
        
        # Create shaded tube for ±1 standard deviation
        std_mag = 0.5 * (x_std + y_std)  # Average std as radius
        
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
        
        # Plot desired trajectory (black dashed line) FIRST
        ax.plot(desired_traj[0, :], desired_traj[1, :], 'k--', linewidth=2.5, label='Desired Trajectory')
        
        # Plot shaded tube for ±1 standard deviation SECOND
        ax.fill(tube_x, tube_y, color=color, alpha=0.3, label='1-std tube')
        
        # Plot mean trajectory (colored curve) THIRD
        ax.plot(x_mean, y_mean, '-', color=color, linewidth=2.5, label='Mean Trajectory')
        
        # Mark start and end positions with X (no legend)
        ax.scatter(desired_traj[0, 0], desired_traj[1, 0], marker='X', s=150, color='black', linewidth=3)
        ax.scatter(desired_traj[0, -1], desired_traj[1, -1], marker='X', s=150, color='black', linewidth=3)
        
        # Formatting
        ax.set_xlabel('X position (m)', fontsize=28)
        ax.set_ylabel('Y position (m)', fontsize=28)
        
        # Create title with optimal theta
        title_text = f"{filter_names[filt]} (θ* = {optimal_theta})"
        ax.set_title(title_text, fontsize=32, pad=15)
        
        # Set specific tick values
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([0, 5, 10])
        
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add legend with proper order and large font
        ax.legend(['Desired Trajectory', '1-std tube', 'Mean Trajectory'], fontsize=22, loc='best')
        
        plt.tight_layout()
        
        # Save individual plot
        save_path = os.path.join(results_dir, f"traj_2d_EKF_{filt}_{dist}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        saved_files.append(f"traj_2d_EKF_{filt}_{dist}.pdf")
        plt.close(fig)
    
    print(f"\nEKF trajectory plots saved to:")
    for filename in saved_files:
        print(f"- ./results/EKF_trajectory_plots/{filename}")

def plot_subplots_all_filters(trajectory_data, filters_order, desired_traj, time, dist):
    """Create a subplot figure with all filters in separate subplots"""
    
    # Colors for the three filters
    colors = {
        'EKF': '#1f77b4',           # Blue
        'DR_EKF_CDC': '#2ca02c',    # Green 
        'DR_EKF_TAC': '#d62728'     # Red
    }
    
    # Filter names for subplot titles
    filter_names = {
        'EKF': "Extended Kalman Filter (EKF)",
        'DR_EKF_CDC': "DR-EKF (CDC)",
        'DR_EKF_TAC': "DR-EKF (TAC)"
    }
    
    # Alphabet mapping for subplot labels
    alphabet_mapping = {filt: chr(ord('A') + i) for i, filt in enumerate(filters_order)}
    
    # Calculate available filters
    available_filters = [filt for filt in filters_order if filt in trajectory_data]
    n_filters = len(available_filters)
    
    # Create subplot layout - horizontal layout for 3 filters
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ensure axes is iterable
    if n_filters == 1:
        axes = [axes]
    
    # Plot each filter in its subplot
    for idx, filt in enumerate(available_filters):
        if idx >= 3:
            break
            
        ax = axes[idx]
        color = colors[filt]
        
        # Get trajectory data
        mean_traj = trajectory_data[filt]['mean']
        std_traj = trajectory_data[filt]['std']
        optimal_theta = trajectory_data[filt]['optimal_theta']
        
        # Extract X and Y positions
        x_mean = mean_traj[:, 0]  # X position
        y_mean = mean_traj[:, 1]  # Y position
        x_std = std_traj[:, 0]    # X position std
        y_std = std_traj[:, 1]    # Y position std
        
        # Create shaded tube for ±1 standard deviation
        std_mag = 0.5 * (x_std + y_std)  # Average std as radius
        
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
        
        # Plot desired trajectory (black dashed line) FIRST
        ax.plot(desired_traj[0, :], desired_traj[1, :], 'k--', linewidth=1.5, alpha=0.8)
        
        # Plot shaded tube for ±1 standard deviation SECOND
        ax.fill(tube_x, tube_y, color=color, alpha=0.3)
        
        # Plot mean trajectory (colored curve) THIRD
        ax.plot(x_mean, y_mean, '-', color=color, linewidth=2)
        
        # Mark start and end positions
        ax.scatter(desired_traj[0, 0], desired_traj[1, 0], marker='X', s=80, color='black', linewidth=2)
        ax.scatter(desired_traj[0, -1], desired_traj[1, -1], marker='X', s=80, color='black', linewidth=2)
        
        # Create title with alphabetical label and optimal theta
        alphabet_label = alphabet_mapping[filt]
        title_text = f"({alphabet_label}) {filter_names[filt]}"
        
        ax.set_title(title_text, fontsize=16, pad=8)
        ax.set_xlabel('X position (m)', fontsize=14)
        ax.set_ylabel('Y position (m)', fontsize=14)
        
        # Set specific tick values
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([0, 5, 10])
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for idx in range(len(available_filters), 3):
        if idx < len(axes):
            axes[idx].set_visible(False)
    
    # Create a custom legend in the figure
    legend_elements = [
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Desired Trajectory'),
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3, label='1-std tube'),
        plt.Line2D([0], [0], color='gray', linewidth=2, label='Mean Trajectory')
    ]
    
    # Add legend to the figure
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=3, fontsize=16, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.95, hspace=0.45, wspace=0.3)
    
    # Save subplot figure
    results_dir = os.path.join("results", "EKF_trajectory_plots")
    save_path = os.path.join(results_dir, f"traj_2d_EKF_subplots_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    
    print(f"EKF subplots trajectory figure saved to: ./results/EKF_trajectory_plots/traj_2d_EKF_subplots_{dist}.pdf")

def plot_performance_vs_robustness(all_results, dist):
    """Plot MSE and tracking cost vs robustness parameters."""
    if all_results is None:
        print("Warning: No all_results data available. Cannot plot performance vs robustness parameters.")
        return
    
    # Extract theta values and performance metrics
    theta_values = sorted(all_results.keys())
    filters_order = ['EKF', 'DR_EKF_CDC', 'DR_EKF_TAC']
    
    # Colors for the three filters
    colors = {
        'EKF': '#1f77b4',           # Blue
        'DR_EKF_CDC': '#2ca02c',    # Green 
        'DR_EKF_TAC': '#d62728'     # Red
    }
    
    # Filter names for legend
    filter_names = {
        'EKF': "Extended Kalman Filter (EKF)",
        'DR_EKF_CDC': "DR-EKF (CDC)",
        'DR_EKF_TAC': "DR-EKF (TAC)"
    }
    
    # Prepare data structures
    performance_data = {filt: {'theta': [], 'mse_mean': [], 'mse_std': [], 'cost_mean': [], 'cost_std': []} 
                       for filt in filters_order}
    
    # Extract data for each theta and filter
    for theta in theta_values:
        results_for_theta = all_results[theta]
        for filt in filters_order:
            if filt in results_for_theta:
                data = results_for_theta[filt]
                performance_data[filt]['theta'].append(theta)
                performance_data[filt]['mse_mean'].append(data['mse_mean'])
                performance_data[filt]['mse_std'].append(data['mse_std'])
                performance_data[filt]['cost_mean'].append(data['cost_mean'])
                performance_data[filt]['cost_std'].append(data['cost_std'])
    
    # Create subplot figure: MSE and Cost vs theta
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot MSE vs theta
    ax1.set_title('Mean Squared Error vs Robustness Parameter', fontsize=16, pad=15)
    ax1.set_xlabel('Robustness Parameter θ', fontsize=14)
    ax1.set_ylabel('MSE', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    for filt in filters_order:
        data = performance_data[filt]
        if len(data['theta']) > 0:
            theta_vals = np.array(data['theta'])
            mse_means = np.array(data['mse_mean'])
            mse_stds = np.array(data['mse_std'])
            
            # Plot mean with error bars
            ax1.errorbar(theta_vals, mse_means, yerr=mse_stds, 
                        marker='o', linewidth=2, markersize=8, capsize=5,
                        color=colors[filt], label=filter_names[filt])
    
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot Tracking Cost vs theta
    ax2.set_title('Tracking Cost vs Robustness Parameter', fontsize=16, pad=15)
    ax2.set_xlabel('Robustness Parameter θ', fontsize=14)
    ax2.set_ylabel('Tracking Cost', fontsize=14)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    for filt in filters_order:
        data = performance_data[filt]
        if len(data['theta']) > 0:
            theta_vals = np.array(data['theta'])
            cost_means = np.array(data['cost_mean'])
            cost_stds = np.array(data['cost_std'])
            
            # Plot mean with error bars
            ax2.errorbar(theta_vals, cost_means, yerr=cost_stds, 
                        marker='s', linewidth=2, markersize=8, capsize=5,
                        color=colors[filt], label=filter_names[filt])
    
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    # Save performance vs robustness plot
    results_dir = os.path.join("results", "EKF_trajectory_plots")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    save_path = os.path.join(results_dir, f"performance_vs_robustness_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    
    print(f"Performance vs robustness plot saved to: ./results/EKF_trajectory_plots/performance_vs_robustness_{dist}.pdf")
    
    # Create separate individual plots for publication
    create_individual_performance_plots(performance_data, filters_order, colors, filter_names, dist)

def create_individual_performance_plots(performance_data, filters_order, colors, filter_names, dist):
    """Create individual MSE and cost plots for publication quality."""
    results_dir = os.path.join("results", "EKF_trajectory_plots")
    
    # MSE plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Mean Squared Error vs Robustness Parameter', fontsize=18, pad=20)
    ax.set_xlabel('Robustness Parameter θ', fontsize=16)
    ax.set_ylabel('MSE', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    for filt in filters_order:
        data = performance_data[filt]
        if len(data['theta']) > 0:
            theta_vals = np.array(data['theta'])
            mse_means = np.array(data['mse_mean'])
            mse_stds = np.array(data['mse_std'])
            
            ax.errorbar(theta_vals, mse_means, yerr=mse_stds, 
                       marker='o', linewidth=3, markersize=10, capsize=6,
                       color=colors[filt], label=filter_names[filt])
    
    ax.legend(fontsize=14, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, f"mse_vs_robustness_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    
    # Tracking Cost plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Tracking Cost vs Robustness Parameter', fontsize=18, pad=20)
    ax.set_xlabel('Robustness Parameter θ', fontsize=16)
    ax.set_ylabel('Tracking Cost', fontsize=16)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    for filt in filters_order:
        data = performance_data[filt]
        if len(data['theta']) > 0:
            theta_vals = np.array(data['theta'])
            cost_means = np.array(data['cost_mean'])
            cost_stds = np.array(data['cost_std'])
            
            ax.errorbar(theta_vals, cost_means, yerr=cost_stds, 
                       marker='s', linewidth=3, markersize=10, capsize=6,
                       color=colors[filt], label=filter_names[filt])
    
    ax.legend(fontsize=14, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, f"cost_vs_robustness_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    
    print(f"Individual MSE plot saved to: ./results/EKF_trajectory_plots/mse_vs_robustness_{dist}.pdf")
    print(f"Individual cost plot saved to: ./results/EKF_trajectory_plots/cost_vs_robustness_{dist}.pdf")

def create_violin_plots(detailed_results, theta_vals, dist, optimal_results):
    """Create violin plots showing distribution of MSE and cost for optimal parameters only"""
    
    print(f"Creating violin plots for optimal parameters ({dist} distribution)...")
    
    # Define filter order and colors
    filters_order = ['EKF', 'DR_EKF_CDC', 'DR_EKF_TAC']
    colors = {
        'EKF': '#1f77b4',           # Blue
        'DR_EKF_CDC': '#ff7f0e',    # Orange  
        'DR_EKF_TAC': '#2ca02c'     # Green
    }
    
    filter_names = {
        'EKF': "Extended Kalman Filter",
        'DR_EKF_CDC': "DR-EKF (CDC)",
        'DR_EKF_TAC': "DR-EKF (TAC)"
    }
    
    # Get optimal theta for each filter
    optimal_thetas = {}
    for filt in filters_order:
        if filt in optimal_results:
            optimal_thetas[filt] = optimal_results[filt]['theta']
    
    print(f"Optimal parameters: {optimal_thetas}")
    
    # Create violin plots for MSE and cost
    for metric in ['mse', 'cost']:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for violin plot (only optimal parameters)
        violin_data = []
        labels = []
        positions = []
        filter_colors = []
        
        pos_counter = 1
        for filt in filters_order:
            if filt in optimal_thetas:
                optimal_theta = optimal_thetas[filt]
                
                # Find closest available theta value
                closest_theta = min(theta_vals, key=lambda x: abs(x - optimal_theta))
                
                if closest_theta in detailed_results and filt in detailed_results[closest_theta]:
                    results = detailed_results[closest_theta][filt]['results']
                    
                    # Extract raw data based on metric
                    if metric == 'mse':
                        raw_data = [np.mean(r['mse']) for r in results]
                    else:  # cost
                        raw_data = [r['tracking_cost'] for r in results]
                    
                    if len(raw_data) > 0:
                        violin_data.append(raw_data)
                        if filt == 'EKF':
                            labels.append(f"{filter_names[filt]}")  # EKF doesn't use theta
                        else:
                            labels.append(f"{filter_names[filt]}\n(θ*={optimal_theta})")
                        positions.append(pos_counter)
                        filter_colors.append(colors[filt])
                        pos_counter += 1
        
        if not violin_data:
            print(f"Warning: No violin data available for {metric} - skipping violin plot")
            plt.close()
            continue
        
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
        
        if metric == 'mse':
            ax.set_ylabel('MSE Distribution', fontsize=16, labelpad=15)
            ax.set_title(f'MSE Distribution at Optimal Parameters ({dist.title()})', fontsize=18)
            ax.set_yscale('log')  # Log scale for MSE
        else:
            ax.set_ylabel('Tracking Cost Distribution', fontsize=16, labelpad=15)
            ax.set_title(f'Tracking Cost Distribution at Optimal Parameters ({dist.title()})', fontsize=18)
        
        ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.4)
        ax.tick_params(axis='both', which='major', width=1.5, length=6)
        ax.tick_params(axis='y', which='major', labelsize=14)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = os.path.join("results", "EKF_trajectory_plots")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        output_path = os.path.join(results_dir, f'violin_plot_{metric}_{dist}.pdf')
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Violin plot for {metric} saved as: {output_path}")

def print_optimal_results_summary(optimal_results):
    """Print summary of optimal results for each filter."""
    print("\nOptimal Results Summary:")
    print("=" * 60)
    for filter_name, results in optimal_results.items():
        theta = results.get('theta', 'N/A')
        mse_mean = results.get('mse_mean', 'N/A')
        mse_std = results.get('mse_std', 'N/A') 
        cost_mean = results.get('cost_mean', 'N/A')
        cost_std = results.get('cost_std', 'N/A')
        
        print(f"{filter_name:12s}: θ*={theta:5.3f}, MSE={mse_mean:.6f}±{mse_std:.6f}, Cost={cost_mean:.3f}±{cost_std:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Create trajectory plots for EKF vs DR-EKF comparison from main0.py')
    parser.add_argument('--dist', default='normal', choices=['normal', 'quadratic'],
                        help='Distribution type to plot trajectories for')
    parser.add_argument('--time', default=10.0, type=float,
                        help='Total simulation time')
    parser.add_argument('--individual_only', action='store_true',
                        help='Create only individual plots, not subplots')
    parser.add_argument('--subplots_only', action='store_true',
                        help='Create only subplot figure, not individual plots')
    parser.add_argument('--performance_plots', action='store_true',
                        help='Create MSE and tracking cost vs robustness parameter plots')
    parser.add_argument('--violin_plots', action='store_true',
                        help='Create violin plots showing distribution of MSE and cost')
    parser.add_argument('--all_plots', action='store_true',
                        help='Create all types of plots (trajectory + performance + violin)')
    args = parser.parse_args()
    
    try:
        # Load optimal results and all results
        results_path = "./results/EKF_comparison/"
        optimal_results, all_results = load_data(results_path, args.dist)
        
        # Print optimal results summary
        print_optimal_results_summary(optimal_results)
        
        # Always create performance vs robustness plots
        print(f"\nCreating performance vs robustness parameter plots...")
        plot_performance_vs_robustness(all_results, args.dist)
        
        # Always create violin plots
        print(f"\nCreating violin plots...")
        # Load detailed results for violin plots
        theta_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]  # Theta values from main0.py
        detailed_results = {}
        
        for theta in theta_vals:
            try:
                detailed_data = load_detailed_results(results_path, theta, args.dist)
                detailed_results[theta] = detailed_data
            except FileNotFoundError:
                print(f"Warning: No detailed results found for θ={theta}")
        
        if detailed_results:
            create_violin_plots(detailed_results, theta_vals, args.dist, optimal_results)
        else:
            print("No detailed results found for violin plots. Run main0.py first.")
        
        # Always create trajectory plots unless only performance plots requested
        if not args.performance_plots:
            # Generate desired trajectory
            desired_traj, time = generate_desired_trajectory(args.time)
            
            # Extract trajectory data from saved detailed results
            trajectory_data, filters_order = extract_trajectory_data_from_saved(optimal_results, results_path, args.dist)
            
            print(f"\nFound trajectory data for {len(trajectory_data)} filters: {list(trajectory_data.keys())}")
            
            if len(trajectory_data) == 0:
                print("No trajectory data found. You need to run main0.py first to generate the detailed trajectory files.")
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
        print(f"Make sure you have run main0.py with --dist {args.dist} first")
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"Error creating trajectory plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()