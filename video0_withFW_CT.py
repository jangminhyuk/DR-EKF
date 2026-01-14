#!/usr/bin/env python3
"""
video0_withFW_CT.py: Generate tracking video for EKF vs DR-EKF comparison
Creates animated video showing true trajectory as airplane and filter estimates as markers.
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

def load_optimal_results(results_path, dist):
    """Load optimal results to get best theta for each filter."""
    optimal_file = os.path.join(results_path, f'optimal_results_{dist}.pkl')
    
    if not os.path.exists(optimal_file):
        raise FileNotFoundError(f"Optimal results file not found: {optimal_file}")
    
    with open(optimal_file, 'rb') as f:
        optimal_results = pickle.load(f)
    
    return optimal_results

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

def extract_mean_trajectories(optimal_results, results_path, dist):
    """Extract mean trajectory for each filter using optimal parameters."""
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
                # Convert to numpy array and compute mean
                est_trajectories = np.array(est_trajectories)  # Shape: (num_runs, time_steps, state_dim)
                true_trajectories = np.array(true_trajectories)

                # Compute mean trajectories
                mean_est_traj = np.mean(est_trajectories, axis=0)  # (time_steps, state_dim)
                mean_true_traj = np.mean(true_trajectories, axis=0)  # (time_steps, state_dim)

                trajectory_data[filt] = {
                    'estimated': mean_est_traj,
                    'true': mean_true_traj,
                    'theta_vals': theta_vals,
                    'theta_str': theta_str
                }
                print(f"Successfully loaded mean trajectory for {filt}")
            else:
                print(f"No trajectory data found for {filt}")

        except FileNotFoundError as e:
            print(f"Could not load detailed results for {filt}: {e}")
            continue
        except Exception as e:
            print(f"Error loading data for {filt}: {e}")
            continue

    return trajectory_data, filters_order

def extract_single_trajectories(optimal_results, results_path, dist, instance_idx=0):
    """Extract single trajectory instance for each filter using optimal parameters."""
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

        print(f"Loading single trajectory data for {filt} with {theta_str}")

        try:
            # Load detailed results for this filter
            detailed_results = load_detailed_results_for_filter(results_path, filt, theta_vals, dist)

            if filt not in detailed_results:
                print(f"Warning: Filter {filt} not found in detailed results")
                continue

            # Extract simulation results
            filter_results = detailed_results[filt]
            sim_results = filter_results['results']  # List of simulation results

            # Check if instance_idx is available
            if instance_idx >= len(sim_results):
                print(f"Warning: Instance {instance_idx} not available for {filt} (only {len(sim_results)} instances)")
                continue

            # Extract single instance trajectory
            single_result = sim_results[instance_idx]

            # Get trajectories
            est_traj = single_result['est_state_traj']  # Shape: (T+1, nx, 1)
            true_traj = single_result['state_traj']  # Shape: (T+1, nx, 1)

            # Remove last dimension
            est_trajectory = np.squeeze(est_traj, axis=-1)  # (time_steps, state_dim)
            true_trajectory = np.squeeze(true_traj, axis=-1)  # (time_steps, state_dim)

            trajectory_data[filt] = {
                'estimated': est_trajectory,
                'true': true_trajectory,
                'theta_vals': theta_vals,
                'theta_str': theta_str
            }
            print(f"Successfully loaded single trajectory for {filt}")

        except FileNotFoundError as e:
            print(f"Could not load detailed results for {filt}: {e}")
            continue
        except Exception as e:
            print(f"Error loading data for {filt}: {e}")
            continue

    return trajectory_data, filters_order

def create_airplane_shape(x, y, heading, scale=1.0):
    """Create simple but clear airplane shape polygon given position and heading."""
    # Simple, clear airplane shape (normalized, pointing right)
    airplane_x = np.array([
        1.5,   # Nose tip
        0.8,   # Nose body
        0.4,   # Front fuselage  
        0.0,   # Wing center
        -0.6,  # Rear fuselage
        -0.8,  # Tail base
        -1.0,  # Tail tip
        -0.8,  # Tail base (return)
        -0.6,  # Rear fuselage (return)
        0.0,   # Wing center (return)
        0.4,   # Front fuselage (return)
        0.8,   # Nose body (return)
        1.5    # Back to nose tip
    ]) * scale
    
    airplane_y = np.array([
        0.0,   # Nose tip
        0.1,   # Nose body
        0.15,  # Front fuselage
        0.6,   # Wing tip
        0.2,   # Rear fuselage
        0.15,  # Tail base
        0.4,   # Tail tip
        -0.15, # Tail base (return)
        -0.2,  # Rear fuselage (return)
        -0.6,  # Wing tip (return)
        -0.15, # Front fuselage (return)
        -0.1,  # Nose body (return)
        0.0    # Back to nose tip
    ]) * scale
    
    # Rotate by heading angle
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    
    rotated_x = cos_h * airplane_x - sin_h * airplane_y
    rotated_y = sin_h * airplane_x + cos_h * airplane_y
    
    # Translate to position
    final_x = rotated_x + x
    final_y = rotated_y + y
    
    return np.column_stack([final_x, final_y])

def compute_heading_from_velocity(vx, vy):
    """Compute heading angle from velocity components."""
    return np.arctan2(vy, vx)

def create_tracking_video(trajectory_data, filters_order, dist, output_filename, fps=10, duration=None, instance_idx=None, output_format='gif'):
    """Create animated video showing tracking with airplane for true state and markers for estimates.

    Args:
        output_format: Output format ('gif' or 'mp4')
    """
    
    # Colors and markers for filters
    colors = {
        'EKF': '#1f77b4',           # Blue
        'DR_EKF_CDC': '#2ca02c',    # Green 
        'DR_EKF_TAC': '#d62728',    # Red
        'DR_EKF_CDC_FW': '#ff7f0e'  # Orange
    }
    
    markers = {
        'EKF': 'o',                 # Circle
        'DR_EKF_CDC': 's',         # Square
        'DR_EKF_TAC': '^',         # Triangle
        'DR_EKF_CDC_FW': 'D'       # Diamond
    }
    
    filter_names = {
        'EKF': "EKF",
        'DR_EKF_TAC': "DR-EKF (TAC) [Ours]",
        'DR_EKF_CDC': "DR-EKF (CDC) [Ours]",
        'DR_EKF_CDC_FW': "DR-EKF (CDC-FW) [Ours]"
    }
    
    # Get the first available filter to extract true trajectory and determine time steps
    first_filter = next(iter(trajectory_data.keys()))
    true_traj = trajectory_data[first_filter]['true']
    num_steps = true_traj.shape[0]
    
    # Determine plot limits from all trajectories
    all_x, all_y = [], []
    for filt_data in trajectory_data.values():
        all_x.extend([filt_data['true'][:, 0], filt_data['estimated'][:, 0]])
        all_y.extend([filt_data['true'][:, 1], filt_data['estimated'][:, 1]])
    
    x_min, x_max = np.min(np.concatenate(all_x)), np.max(np.concatenate(all_x))
    y_min, y_max = np.min(np.concatenate(all_y)), np.max(np.concatenate(all_y))
    
    # Calculate data aspect ratio
    x_range = x_max - x_min
    y_range = y_max - y_min
    data_aspect_ratio = x_range / y_range if y_range > 0 else 1.0

    # Use fixed square figure size
    fig_size = 16
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Add padding to make the plot area square while preserving data aspect ratio
    base_margin = 0.05
    if data_aspect_ratio > 1.0:  # Data is wider than tall
        # Add more vertical padding
        x_margin = x_range * base_margin
        y_margin = y_range * base_margin
        # Expand y limits to match aspect ratio
        y_center = (y_min + y_max) / 2
        y_range_needed = x_range  # To make it square
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_center - y_range_needed/2 - y_margin, y_center + y_range_needed/2 + y_margin)
    else:  # Data is taller than wide (or square)
        # Add more horizontal padding
        x_margin = x_range * base_margin
        y_margin = y_range * base_margin
        # Expand x limits to match aspect ratio
        x_center = (x_min + x_max) / 2
        x_range_needed = y_range  # To make it square
        ax.set_xlim(x_center - x_range_needed/2 - x_margin, x_center + x_range_needed/2 + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Use equal aspect ratio for square appearance
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.set_xlabel('X position [m]', fontsize=42)
    ax.set_ylabel('Y position [m]', fontsize=42)

    # No title - time will be displayed in title position
    
    # Initialize trajectory lines and current position markers
    trajectory_lines = {}
    current_markers = {}
    airplane_text = None
    
    # Create trajectory lines for each filter
    for filt in filters_order:
        if filt in trajectory_data:
            # Estimated trajectory line (will be updated each frame)
            line, = ax.plot([], [], '-', color=colors[filt], linewidth=4, alpha=0.7,
                           label=filter_names[filt])
            trajectory_lines[filt] = line

            # Current position marker - put it in front of airplane
            marker, = ax.plot([], [], markers[filt], color=colors[filt], markersize=16,
                             markeredgecolor='black', markeredgewidth=2, zorder=10)
            current_markers[filt] = marker

    # True trajectory line (will be updated each frame)
    true_line, = ax.plot([], [], ':', color='black', linewidth=4, alpha=0.8, label='True Trajectory')
    
    # Create legend with larger font
    legend_handles = [true_line] + [trajectory_lines[filt] for filt in filters_order if filt in trajectory_data]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=32)

    # Add time display above the instance/mean indicator on the left
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=44,
                       horizontalalignment='left', verticalalignment='top',
                       weight='bold', color='black')

    if instance_idx is not None:
        mode_label = ax.text(0.02, 0.90, f'INSTANCE #{instance_idx+1}', transform=ax.transAxes, fontsize=36,
                            weight='bold', color='darkgreen', verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    else:
        mode_label = ax.text(0.02, 0.90, 'MEAN TRAJECTORIES', transform=ax.transAxes, fontsize=28,
                            weight='bold', color='darkblue', verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def init():
        """Initialize animation."""
        for line in trajectory_lines.values():
            line.set_data([], [])
        for marker in current_markers.values():
            marker.set_data([], [])
        true_line.set_data([], [])
        time_text.set_text('')
        return list(trajectory_lines.values()) + list(current_markers.values()) + [true_line, time_text]
    
    def animate(frame):
        """Animation function called for each frame."""
        nonlocal airplane_text
        
        # Calculate time (assuming dt=0.2 from the original code)
        dt = 0.2
        current_time = frame * dt
        
        # Update time display
        time_text.set_text(f'Time: {current_time:.1f}s')
        
        # Get current step (clamp to available data)
        step = min(frame, num_steps - 1)
        
        # Update true trajectory line up to current step
        true_x = true_traj[:step+1, 0]
        true_y = true_traj[:step+1, 1]
        true_line.set_data(true_x, true_y)
        
        # Update airplane position and orientation
        if airplane_text is not None:
            airplane_text.remove()
        
        if step < num_steps:
            # Get current true position and velocity for heading
            true_pos_x, true_pos_y = true_traj[step, 0], true_traj[step, 1]
            true_vx, true_vy = true_traj[step, 2], true_traj[step, 3]
            
            # Compute heading from velocity
            heading = compute_heading_from_velocity(true_vx, true_vy)
            
            # Use a large text-based airplane symbol
            airplane_symbol = '✈'  # Unicode airplane symbol
            # Convert heading to degrees for rotation
            heading_deg = np.degrees(heading)

            # Create text airplane with rotation - make it bigger and black, put it behind markers
            airplane_text = ax.text(true_pos_x, true_pos_y, airplane_symbol,
                                  fontsize=80, ha='center', va='center',
                                  rotation=heading_deg, color='black',
                                  weight='bold', zorder=5)
        
        # Update estimated trajectories and current positions
        for filt in filters_order:
            if filt in trajectory_data:
                est_traj = trajectory_data[filt]['estimated']
                
                # Update trajectory line up to current step
                est_x = est_traj[:step+1, 0]
                est_y = est_traj[:step+1, 1]
                trajectory_lines[filt].set_data(est_x, est_y)
                
                # Update current position marker
                if step < num_steps:
                    current_pos_x, current_pos_y = est_traj[step, 0], est_traj[step, 1]
                    current_markers[filt].set_data([current_pos_x], [current_pos_y])
                else:
                    current_markers[filt].set_data([], [])
        
        artists = list(trajectory_lines.values()) + list(current_markers.values()) + [true_line, time_text]
        if airplane_text is not None:
            artists.append(airplane_text)
        
        return artists
    
    # Calculate number of frames
    if duration is None:
        # Use all time steps
        frames = num_steps
    else:
        # Use specified duration
        frames = min(int(duration * fps), num_steps)
    
    print(f"Creating animation with {frames} frames at {fps} fps...")
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, 
                                 interval=1000/fps, blit=True, repeat=True)
    
    # Adjust layout to ensure labels aren't trimmed with larger fonts
    plt.subplots_adjust(left=0.14, bottom=0.11, right=0.96, top=0.93)

    # Save animation with higher DPI for better quality
    print(f"Saving video to: {output_filename}")
    if output_format == 'mp4':
        anim.save(output_filename, writer='ffmpeg', fps=fps, dpi=120)
    else:  # gif
        anim.save(output_filename, writer='pillow', fps=fps, dpi=120)
    plt.close(fig)

    print(f"Video saved successfully!")
    return output_filename

def main():
    parser = argparse.ArgumentParser(description='Create tracking video for EKF vs DR-EKF comparison')
    parser.add_argument('--dist', default='normal', choices=['normal', 'quadratic'],
                        help='Distribution type to create video for')
    parser.add_argument('--fps', default=15, type=int,
                        help='Frames per second for video')
    parser.add_argument('--duration', type=float,
                        help='Duration in seconds (if not specified, uses full trajectory)')
    parser.add_argument('--output',
                        help='Output filename (if not specified, auto-generated)')
    parser.add_argument('--format', default='mp4', choices=['gif', 'mp4'],
                        help='Output video format (default: mp4)')
    args = parser.parse_args()
    
    try:
        # Load optimal results
        results_path = "./results/EKF_comparison_with_FW_CT/"
        optimal_results = load_optimal_results(results_path, args.dist)

        print(f"Optimal parameters:")
        for filt, data in optimal_results.items():
            if filt == 'EKF':
                print(f"  {filt}: N/A")
            elif filt in ['DR_EKF_CDC', 'DR_EKF_CDC_FW']:
                theta_x = data.get('theta_x', 'N/A')
                theta_v = data.get('theta_v', 'N/A')
                print(f"  {filt}: θ_x={theta_x}, θ_v={theta_v}")
            elif filt == 'DR_EKF_TAC':
                theta_x = data.get('theta_x', 'N/A')
                theta_v = data.get('theta_v', 'N/A')
                theta_w = data.get('theta_w', 'N/A')
                print(f"  {filt}: θ_x={theta_x}, θ_v={theta_v}, θ_w={theta_w}")
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join("results", "EKF_comparison_with_FW_CT")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 1. Create MEAN trajectory video
        print("\n=== Creating MEAN trajectory video ===")
        trajectory_data_mean, filters_order = extract_mean_trajectories(optimal_results, results_path, args.dist)
        
        print(f"Found mean trajectory data for {len(trajectory_data_mean)} filters: {list(trajectory_data_mean.keys())}")
        
        if len(trajectory_data_mean) > 0:
            if args.output is None:
                output_filename_mean = f"tracking_video_MEAN_{args.dist}_fps{args.fps}.{args.format}"
            else:
                output_filename_mean = f"MEAN_{args.output}"

            output_path_mean = os.path.join(results_dir, output_filename_mean)

            # Create mean video
            create_tracking_video(trajectory_data_mean, filters_order, args.dist, output_path_mean,
                                fps=args.fps, duration=args.duration, instance_idx=None, output_format=args.format)
            
            print(f"Mean trajectory video saved to: {output_path_mean}")
        else:
            print("No mean trajectory data found.")
        
        # 2. Create SINGLE INSTANCE trajectory videos (up to 5 instances)
        print("\n=== Creating SINGLE INSTANCE trajectory videos ===")

        # Define filters_order if not already defined
        if 'filters_order' not in locals():
            filters_order = ['EKF', 'DR_EKF_TAC', 'DR_EKF_CDC', 'DR_EKF_CDC_FW']

        # First check how many instances are available
        max_instances_available = 0
        for filt in filters_order:
            if filt in optimal_results:
                optimal_stats = optimal_results[filt]

                # Get theta values for this filter
                if filt == 'EKF':
                    theta_vals = {}
                elif filt in ['DR_EKF_CDC', 'DR_EKF_CDC_FW']:
                    theta_vals = {
                        'theta_x': optimal_stats['theta_x'],
                        'theta_v': optimal_stats['theta_v']
                    }
                elif filt == 'DR_EKF_TAC':
                    theta_vals = {
                        'theta_x': optimal_stats['theta_x'],
                        'theta_v': optimal_stats['theta_v'],
                        'theta_w': optimal_stats['theta_w']
                    }

                try:
                    detailed_results = load_detailed_results_for_filter(results_path, filt, theta_vals, args.dist)
                    if filt in detailed_results:
                        sim_results = detailed_results[filt]['results']
                        max_instances_available = max(max_instances_available, len(sim_results))
                except:
                    continue
        
        print(f"Maximum instances available: {max_instances_available}")
        
        # Generate up to 5 instance videos
        max_videos = min(5, max_instances_available)
        single_videos_created = 0
        
        for instance_idx in range(max_videos):
            print(f"\nCreating video for instance #{instance_idx+1}...")
            trajectory_data_single, _ = extract_single_trajectories(optimal_results, results_path, args.dist, instance_idx)
            
            if len(trajectory_data_single) > 0:
                if args.output is None:
                    output_filename_single = f"tracking_video_INSTANCE_{instance_idx+1}_{args.dist}_fps{args.fps}.{args.format}"
                else:
                    output_filename_single = f"INSTANCE_{instance_idx+1}_{args.output}"

                output_path_single = os.path.join(results_dir, output_filename_single)

                # Create single instance video
                create_tracking_video(trajectory_data_single, filters_order, args.dist, output_path_single,
                                    fps=args.fps, duration=args.duration, instance_idx=instance_idx, output_format=args.format)
                
                print(f"Instance #{instance_idx+1} video saved to: {output_path_single}")
                single_videos_created += 1
            else:
                print(f"No trajectory data found for instance #{instance_idx+1}")
                break
        
        print(f"\nSummary: Created {single_videos_created} single instance videos")
        
        if len(trajectory_data_mean) == 0 and single_videos_created == 0:
            print("No trajectory data found. You need to run main0_withFW_CT.py first.")
            return
        
    except FileNotFoundError as e:
        print(f"Error: Could not find results files for distribution '{args.dist}'")
        print(f"Make sure you have run main0_withFW_CT.py with --dist {args.dist} first")
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"Error creating tracking video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()