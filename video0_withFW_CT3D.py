#!/usr/bin/env python3
"""
video0_withFW_CT3D.py: Generate 3D tracking video for EKF vs DR-EKF comparison
Creates animated video showing true trajectory as airplane and filter estimates as markers in 3D space.
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
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
    filters_order = ['EKF','DR_EKF_TAC', 'DR_EKF_CDC',  'DR_EKF_CDC_FW']
    trajectory_data = {}

    for filt in filters_order:
        if filt not in optimal_results:
            print(f"Warning: Filter '{filt}' not found in optimal results, skipping...")
            continue

        # Extract optimal theta values based on filter type
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
            # Load detailed results for this filter with its theta values
            detailed_results = load_detailed_results_for_filter(results_path, filt, theta_vals, dist)
            
            if filt not in detailed_results:
                print(f"Warning: Filter {filt} not found in detailed results for {theta_str}")
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
                    'theta_vals': theta_vals
                }
                print(f"Successfully loaded mean trajectory for {filt}")
            else:
                print(f"No trajectory data found for {filt}")

        except FileNotFoundError as e:
            print(f"Could not load detailed results for {filt} ({theta_str}): {e}")
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

        # Extract optimal theta values based on filter type
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
            # Load detailed results for this filter with its theta values
            detailed_results = load_detailed_results_for_filter(results_path, filt, theta_vals, dist)
            
            if filt not in detailed_results:
                print(f"Warning: Filter {filt} not found in detailed results for {theta_str}")
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
                'theta_vals': theta_vals
            }
            print(f"Successfully loaded single trajectory for {filt}")

        except FileNotFoundError as e:
            print(f"Could not load detailed results for {filt} ({theta_str}): {e}")
            continue
        except Exception as e:
            print(f"Error loading data for {filt}: {e}")
            continue
    
    return trajectory_data, filters_order

def compute_heading_from_velocity_3d(vx, vy, vz):
    """Compute heading angles from 3D velocity components.
    Returns (yaw, pitch) where yaw is azimuth in xy-plane, pitch is elevation angle.
    """
    yaw = np.arctan2(vy, vx)  # Heading in xy-plane
    v_horizontal = np.sqrt(vx**2 + vy**2)
    pitch = np.arctan2(vz, v_horizontal)  # Elevation angle
    return yaw, pitch

def create_airplane_arrow(ax, x, y, z, vx, vy, vz, scale=0.3):
    """Create a 3D arrow to represent airplane direction."""
    # Normalize velocity vector
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    if v_mag > 0:
        vx_norm = vx / v_mag * scale
        vy_norm = vy / v_mag * scale
        vz_norm = vz / v_mag * scale
        
        # Create 3D arrow using quiver with low z-order to be behind filter estimates
        arrow = ax.quiver(x, y, z, vx_norm, vy_norm, vz_norm, 
                         color='red', arrow_length_ratio=0.3, linewidth=3, alpha=0.9)
        # Note: quiver doesn't have zorder parameter, but it's generally rendered in order
        return arrow
    return None

def create_3d_airplane(ax, x, y, z, vx, vy, vz, scale=1.2):
    """Create a realistic F-16 style fighter aircraft using proper 3D geometry and filled surfaces."""
    if np.sqrt(vx**2 + vy**2 + vz**2) == 0:
        return []
    
    # Normalize velocity vector to get orientation
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    forward = np.array([vx, vy, vz]) / v_mag  # Forward direction
    
    # Create orthogonal vectors for wing and vertical directions
    if abs(forward[2]) < 0.9:  # If not pointing straight up/down
        up = np.array([0, 0, 1])
    else:
        up = np.array([1, 0, 0])
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Transform to world coordinates
    def to_world(local_vec):
        return np.array([x, y, z]) + local_vec
    
    surfaces = []
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Helper functions for proper triangulation
    def add_tri(v0, v1, v2, facecolor, edgecolor='none', alpha=0.95, lw=0):
        """Add a triangular surface."""
        poly = Poly3DCollection([[v0, v1, v2]])
        poly.set_facecolor(facecolor)
        poly.set_edgecolor(edgecolor)
        poly.set_alpha(alpha)
        poly.set_linewidth(lw)
        poly.set_zsort('average')
        ax.add_collection3d(poly)
        surfaces.append(poly)
    
    def add_quad_as_tris(v0, v1, v2, v3, facecolor, edgecolor='none', alpha=0.95, lw=0):
        """Add a quadrilateral as two triangles."""
        add_tri(v0, v1, v2, facecolor, edgecolor, alpha, lw)
        add_tri(v0, v2, v3, facecolor, edgecolor, alpha, lw)
    
    # F-16 realistic proportions
    fuselage_length = 1.0 * scale
    fuselage_width = 0.12 * scale
    fuselage_height = 0.10 * scale
    
    # Define fuselage stations along forward axis
    c0 = to_world(forward * fuselage_length * 0.45)   # nose
    c1 = to_world(forward * fuselage_length * 0.25)   # cockpit
    c2 = to_world(np.array([0, 0, 0]))                # mid-fuselage  
    c3 = to_world(-forward * fuselage_length * 0.30)  # tail start
    c4 = to_world(-forward * fuselage_length * 0.45)  # tail end
    
    # Nose tip (single point)
    nose_tip = to_world(forward * fuselage_length * 0.48)
    
    # Define cross-section radii for tapering fuselage
    r_up = [0.3, 0.8, 0.7, 0.5, 0.2]     # up radii
    r_right = [0.3, 0.8, 1.0, 0.7, 0.3]  # right radii
    
    # Define fuselage cross-sections (4 points each)
    stations = [c0, c1, c2, c3, c4]
    top_pts = []
    bottom_pts = []
    left_pts = []
    right_pts = []
    
    for i, center in enumerate(stations):
        top_pts.append(center + up * fuselage_height * r_up[i])
        bottom_pts.append(center - up * fuselage_height * r_up[i])
        left_pts.append(center - right * fuselage_width * r_right[i])
        right_pts.append(center + right * fuselage_width * r_right[i])
    
    # Build fuselage as connected cross-sections using triangles
    fuse_color = 'silver'
    
    for i in range(len(stations) - 1):
        # Connect adjacent stations with triangulated quads
        # Top skin
        add_quad_as_tris(top_pts[i], right_pts[i], right_pts[i+1], top_pts[i+1], fuse_color)
        add_quad_as_tris(top_pts[i], top_pts[i+1], left_pts[i+1], left_pts[i], fuse_color)
        
        # Bottom skin
        add_quad_as_tris(bottom_pts[i], bottom_pts[i+1], right_pts[i+1], right_pts[i], fuse_color)
        add_quad_as_tris(bottom_pts[i], left_pts[i], left_pts[i+1], bottom_pts[i+1], fuse_color)
        
        # Side skins
        add_quad_as_tris(left_pts[i], bottom_pts[i], bottom_pts[i+1], left_pts[i+1], fuse_color)
        add_quad_as_tris(right_pts[i], right_pts[i+1], bottom_pts[i+1], bottom_pts[i], fuse_color)
    
    # Close fuselage ends
    # Nose: connect first cross-section to nose tip
    add_tri(nose_tip, right_pts[0], top_pts[0], 'lightsteelblue')
    add_tri(nose_tip, top_pts[0], left_pts[0], 'lightsteelblue')
    add_tri(nose_tip, left_pts[0], bottom_pts[0], 'lightsteelblue')
    add_tri(nose_tip, bottom_pts[0], right_pts[0], 'lightsteelblue')
    
    # Tail: close last cross-section
    tail_center = stations[-1]
    add_tri(tail_center, top_pts[-1], right_pts[-1], fuse_color)
    add_tri(tail_center, right_pts[-1], bottom_pts[-1], fuse_color)
    add_tri(tail_center, bottom_pts[-1], left_pts[-1], fuse_color)
    add_tri(tail_center, left_pts[-1], top_pts[-1], fuse_color)
    
    # Wings - positioned at mid-fuselage (station 2)
    wing_root = stations[2]  # mid-fuselage position
    wing_span = 0.7 * scale
    wing_chord_root = 0.25 * scale
    wing_chord_tip = 0.10 * scale
    wing_sweep = 0.15 * scale
    
    # Wing root positions
    wing_front_root = wing_root + forward * wing_chord_root * 0.5
    wing_back_root = wing_root - forward * wing_chord_root * 0.5
    
    # Wing tip positions (with sweep and taper)
    wing_left_front = to_world(-right * wing_span + forward * (wing_chord_tip * 0.5 - wing_sweep))
    wing_left_back = to_world(-right * wing_span - forward * (wing_chord_tip * 0.5 + wing_sweep))
    wing_right_front = to_world(right * wing_span + forward * (wing_chord_tip * 0.5 - wing_sweep))
    wing_right_back = to_world(right * wing_span - forward * (wing_chord_tip * 0.5 + wing_sweep))
    
    # Wings as triangulated surfaces
    wing_color = 'dimgray'
    add_quad_as_tris(wing_front_root, wing_left_front, wing_left_back, wing_back_root, wing_color)
    add_quad_as_tris(wing_front_root, wing_back_root, wing_right_back, wing_right_front, wing_color)
    
    # Tail surfaces
    tail_base = stations[3]  # tail start position
    
    # Single vertical tail fin
    tail_fin_height = 0.35 * scale
    tail_fin_chord = 0.20 * scale
    
    tail_fin_base_front = tail_base + forward * tail_fin_chord * 0.3
    tail_fin_base_back = tail_base - forward * tail_fin_chord * 0.7
    tail_fin_top = to_world(-forward * fuselage_length * 0.42 + up * tail_fin_height)
    tail_fin_back = to_world(-forward * fuselage_length * 0.48 + up * tail_fin_height * 0.7)
    
    # Vertical tail as triangulated quad
    tail_color = 'darkgray'
    add_quad_as_tris(tail_fin_base_front, tail_fin_top, tail_fin_back, tail_fin_base_back, tail_color)
    
    # Horizontal stabilizers
    htail_span = 0.25 * scale
    htail_chord = 0.12 * scale
    
    htail_front = tail_base + forward * htail_chord * 0.5
    htail_back = tail_base - forward * htail_chord * 0.5
    htail_left_tip = to_world(-forward * fuselage_length * 0.42 - right * htail_span)
    htail_right_tip = to_world(-forward * fuselage_length * 0.42 + right * htail_span)
    
    # Horizontal stabilizers as triangulated quads
    htail_color = 'gray'
    add_quad_as_tris(htail_front, htail_left_tip, htail_back, tail_base, htail_color)
    add_quad_as_tris(htail_front, tail_base, htail_back, htail_right_tip, htail_color)
    
    # Cockpit canopy - positioned at cockpit station (1)
    canopy_base = stations[1]  # cockpit position
    canopy_front = canopy_base + forward * 0.15 * scale
    canopy_back = canopy_base - forward * 0.10 * scale
    canopy_peak = canopy_base + up * fuselage_height * 1.2
    canopy_left = canopy_base - right * fuselage_width * 0.6 + up * fuselage_height * 0.8
    canopy_right = canopy_base + right * fuselage_width * 0.6 + up * fuselage_height * 0.8
    
    # Canopy as triangulated surfaces with no heavy edges
    canopy_color = (0.4, 0.7, 1.0)  # Light blue
    canopy_alpha = 0.5
    
    # Front canopy section
    add_tri(canopy_front, canopy_left, canopy_peak, canopy_color, alpha=canopy_alpha)
    add_tri(canopy_front, canopy_peak, canopy_right, canopy_color, alpha=canopy_alpha)
    
    # Back canopy section  
    add_tri(canopy_peak, canopy_left, canopy_back, canopy_color, alpha=canopy_alpha)
    add_tri(canopy_peak, canopy_back, canopy_right, canopy_color, alpha=canopy_alpha)
    
    return surfaces

def compute_heading_from_velocity(vx, vy):
    """Compute heading angle from velocity components."""
    return np.arctan2(vy, vx)

def create_tracking_video(trajectory_data, filters_order, dist, output_filename, fps=10, duration=None, instance_idx=None, zoomed=False, zoom_radius=2.0, output_format='gif'):
    """Create animated 3D video showing tracking with airplane for true state and markers for estimates.

    Args:
        zoomed: If True, creates a zoomed view centered on the airplane
        zoom_radius: Radius of the zoomed view around the airplane
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
        'DR_EKF_CDC': "DR-EKF (CDC) [Ours]",
        'DR_EKF_TAC': "DR-EKF (TAC) [Ours]",
        'DR_EKF_CDC_FW': "DR-EKF (CDC-FW) [Ours]"
    }
    
    # Get the first available filter to extract true trajectory and determine time steps
    first_filter = next(iter(trajectory_data.keys()))
    true_traj = trajectory_data[first_filter]['true']
    num_steps = true_traj.shape[0]
    
    # Determine plot limits from all trajectories (now including Z dimension)
    all_x, all_y, all_z = [], [], []
    for filt_data in trajectory_data.values():
        all_x.extend([filt_data['true'][:, 0], filt_data['estimated'][:, 0]])
        all_y.extend([filt_data['true'][:, 1], filt_data['estimated'][:, 1]])
        all_z.extend([filt_data['true'][:, 2], filt_data['estimated'][:, 2]])
    
    x_min, x_max = np.min(np.concatenate(all_x)), np.max(np.concatenate(all_x))
    y_min, y_max = np.min(np.concatenate(all_y)), np.max(np.concatenate(all_y))
    z_min, z_max = np.min(np.concatenate(all_z)), np.max(np.concatenate(all_z))
    
    # Set up the 3D figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    if not zoomed:
        # Normal view - show full trajectory
        # Add margins for better visualization
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        z_margin = (z_max - z_min) * 0.1
        
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
    else:
        # Zoomed view - will be updated dynamically in animate function
        ax.set_xlim(-zoom_radius, zoom_radius)
        ax.set_ylim(-zoom_radius, zoom_radius)
        ax.set_zlim(-zoom_radius, zoom_radius)
    
    # Set equal aspect ratio for better 3D visualization
    ax.set_box_aspect([1,1,1])
    
    # Labels and title with extra padding to avoid overlap
    ax.set_xlabel('X position', fontsize=24, labelpad=20)
    ax.set_ylabel('Y position', fontsize=24, labelpad=20)
    ax.set_zlabel('Z position', fontsize=24, labelpad=20)
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Configure grid and ticks based on view mode
    if zoomed:
        # For zoomed view, use fewer ticks and lighter grid
        ax.grid(True, alpha=0.2, linewidth=0.5)
        # Set fewer ticks for cleaner zoomed view
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    else:
        # Normal view with standard grid
        ax.grid(True, alpha=0.3)
    
    # No title - time will be displayed in title position
    
    # Initialize 3D trajectory lines and current position markers
    trajectory_lines = {}
    current_markers = {}
    airplane_marker = None
    
    # Create trajectory lines for each filter
    for filt in filters_order:
        if filt in trajectory_data:
            # Estimated trajectory line (will be updated each frame)
            line, = ax.plot([], [], [], '-', color=colors[filt], linewidth=2, alpha=0.7, 
                           label=filter_names[filt])
            trajectory_lines[filt] = line
            
            # Current position marker with high z-order to appear on top
            marker, = ax.plot([], [], [], markers[filt], color=colors[filt], markersize=15, 
                             markeredgecolor='black', markeredgewidth=2, zorder=10)
            current_markers[filt] = marker
    
    # True trajectory line (will be updated each frame)
    true_line, = ax.plot([], [], [], ':', color='black', linewidth=3, alpha=0.8, label='True Trajectory')
    
    # Create legend
    legend_handles = [true_line] + [trajectory_lines[filt] for filt in filters_order if filt in trajectory_data]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=24)
    
    # Add time display above the instance/mean indicator on the left
    time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=44,
                         horizontalalignment='left', verticalalignment='top',
                         weight='bold', color='black')
    
    if instance_idx is not None:
        mode_label = ax.text2D(0.02, 0.90, f'INSTANCE #{instance_idx+1}', transform=ax.transAxes, fontsize=24,
                              weight='bold', color='darkgreen', verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    else:
        mode_label = ax.text2D(0.02, 0.90, 'MEAN TRAJECTORIES', transform=ax.transAxes, fontsize=24,
                              weight='bold', color='darkblue', verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def init():
        """Initialize animation."""
        for line in trajectory_lines.values():
            line.set_data([], [])
            line.set_3d_properties([])
        for marker in current_markers.values():
            marker.set_data([], [])
            marker.set_3d_properties([])
        true_line.set_data([], [])
        true_line.set_3d_properties([])
        time_text.set_text('')
        return list(trajectory_lines.values()) + list(current_markers.values()) + [true_line, time_text]
    
    def animate(frame):
        """Animation function called for each frame."""
        nonlocal airplane_marker
        
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
        true_z = true_traj[:step+1, 2]
        
        if zoomed and step < num_steps:
            # In zoomed mode, show actual trajectory but update view to center on airplane
            current_pos_x, current_pos_y, current_pos_z = true_traj[step, 0], true_traj[step, 1], true_traj[step, 2]
            true_line.set_data(true_x, true_y)
            true_line.set_3d_properties(true_z)
            
            # Update axis limits to show real position values centered on current position
            ax.set_xlim(current_pos_x - zoom_radius, current_pos_x + zoom_radius)
            ax.set_ylim(current_pos_y - zoom_radius, current_pos_y + zoom_radius)
            ax.set_zlim(current_pos_z - zoom_radius, current_pos_z + zoom_radius)
        else:
            # Normal mode
            true_line.set_data(true_x, true_y)
            true_line.set_3d_properties(true_z)
        
        # Update airplane position and orientation
        if airplane_marker is not None:
            if isinstance(airplane_marker, dict):
                # Remove text, arrow, and 3D airplane surfaces
                if 'text' in airplane_marker and airplane_marker['text'] is not None:
                    airplane_marker['text'].remove()
                if 'arrow' in airplane_marker and airplane_marker['arrow'] is not None:
                    airplane_marker['arrow'].remove()
                if 'airplane_3d' in airplane_marker and airplane_marker['airplane_3d'] is not None:
                    for surface in airplane_marker['airplane_3d']:
                        surface.remove()
            elif isinstance(airplane_marker, list):
                # Remove list of 3D airplane surfaces/lines
                for element in airplane_marker:
                    element.remove()
            else:
                # Remove single element
                airplane_marker.remove()
        
        if step < num_steps:
            # Get current true position
            true_pos_x, true_pos_y, true_pos_z = true_traj[step, 0], true_traj[step, 1], true_traj[step, 2]
            
            # Get velocity for orientation (assuming state: [px, py, pz, vx, vy, vz, ...])
            if true_traj.shape[1] >= 6:  # Check if velocity is available
                true_vx, true_vy, true_vz = true_traj[step, 3], true_traj[step, 4], true_traj[step, 5]
                
                # Create realistic 3D airplane oriented by velocity
                if zoomed:
                    # Smaller airplane for zoomed view to fit better in close-up
                    airplane_3d = create_3d_airplane(ax, true_pos_x, true_pos_y, true_pos_z, 
                                                   true_vx, true_vy, true_vz, scale=0.8)
                else:
                    # Larger airplane for normal view to be more visible
                    airplane_3d = create_3d_airplane(ax, true_pos_x, true_pos_y, true_pos_z, 
                                                   true_vx, true_vy, true_vz, scale=1.8)
                
                # Both views now just show the 3D airplane (no red arrows)
                airplane_marker = airplane_3d
            else:
                # No velocity data - show static airplane symbol
                airplane_symbol = '✈'
                airplane_marker = ax.text(true_pos_x, true_pos_y, true_pos_z, airplane_symbol,
                                        fontsize=100, ha='center', va='center', color='black',
                                        weight='bold', zorder=1)
        
        # Update estimated trajectories and current positions
        for filt in filters_order:
            if filt in trajectory_data:
                est_traj = trajectory_data[filt]['estimated']
                
                # Update trajectory line up to current step
                est_x = est_traj[:step+1, 0]
                est_y = est_traj[:step+1, 1]
                est_z = est_traj[:step+1, 2]
                
                # Always show actual trajectory positions (both normal and zoomed modes)
                trajectory_lines[filt].set_data(est_x, est_y)
                trajectory_lines[filt].set_3d_properties(est_z)
                
                # Update current position marker
                if step < num_steps:
                    current_pos_x, current_pos_y, current_pos_z = est_traj[step, 0], est_traj[step, 1], est_traj[step, 2]
                    current_markers[filt].set_data([current_pos_x], [current_pos_y])
                    current_markers[filt].set_3d_properties([current_pos_z])
                else:
                    current_markers[filt].set_data([], [])
                    current_markers[filt].set_3d_properties([])
        
        artists = list(trajectory_lines.values()) + list(current_markers.values()) + [true_line, time_text]
        if airplane_marker is not None:
            if isinstance(airplane_marker, dict):
                # Add 3D airplane surfaces and other elements to artists
                if 'airplane_3d' in airplane_marker and airplane_marker['airplane_3d'] is not None:
                    artists.extend(airplane_marker['airplane_3d'])
                if 'arrow' in airplane_marker and airplane_marker['arrow'] is not None:
                    artists.append(airplane_marker['arrow'])
                if 'text' in airplane_marker and airplane_marker['text'] is not None:
                    artists.append(airplane_marker['text'])
            elif isinstance(airplane_marker, list):
                # Add list of 3D airplane surfaces
                artists.extend(airplane_marker)
            else:
                # Add single element
                artists.append(airplane_marker)
        return artists
    
    # Calculate number of frames
    if duration is None:
        # Use all time steps
        frames = num_steps
    else:
        # Use specified duration
        frames = min(int(duration * fps), num_steps)
    
    print(f"Creating 3D animation with {frames} frames at {fps} fps...")
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, 
                                 interval=1000/fps, blit=False, repeat=True)  # blit=False for 3D
    
    # Adjust layout
    plt.tight_layout()

    # Save animation with higher DPI for better quality
    print(f"Saving video to: {output_filename}")
    if output_format == 'mp4':
        anim.save(output_filename, writer='ffmpeg', fps=fps, dpi=100)
    else:  # gif
        anim.save(output_filename, writer='pillow', fps=fps, dpi=100)
    plt.close(fig)

    print(f"3D Video saved successfully!")
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
    parser.add_argument('--format', default='gif', choices=['gif', 'mp4'],
                        help='Output video format (default: gif)')
    args = parser.parse_args()
    
    try:
        # Load optimal results
        results_path = "./results/EKF_comparison_with_FW_CT3D/"
        optimal_results = load_optimal_results(results_path, args.dist)
        
        print(f"Optimal parameters:")
        for filt, data in optimal_results.items():
            if filt == 'EKF':
                print(f"  {filt}: N/A")
            elif filt in ['DR_EKF_CDC', 'DR_EKF_CDC_FW']:
                print(f"  {filt}: θ_x* = {data['theta_x']}, θ_v* = {data['theta_v']}")
            elif filt == 'DR_EKF_TAC':
                print(f"  {filt}: θ_x* = {data['theta_x']}, θ_v* = {data['theta_v']}, θ_w* = {data['theta_w']}")
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join("results", "EKF_comparison_with_FW_CT3D")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 1. Create MEAN trajectory videos (normal and zoomed)
        print("\n=== Creating MEAN trajectory videos ===")
        trajectory_data_mean, filters_order = extract_mean_trajectories(optimal_results, results_path, args.dist)
        
        print(f"Found mean trajectory data for {len(trajectory_data_mean)} filters: {list(trajectory_data_mean.keys())}")
        
        if len(trajectory_data_mean) > 0:
            # Create normal view video
            if args.output is None:
                output_filename_mean = f"tracking_video_MEAN_{args.dist}_fps{args.fps}.{args.format}"
                output_filename_mean_zoomed = f"tracking_video_MEAN_ZOOMED_{args.dist}_fps{args.fps}.{args.format}"
            else:
                output_filename_mean = f"MEAN_{args.output}"
                output_filename_mean_zoomed = f"MEAN_ZOOMED_{args.output}"

            output_path_mean = os.path.join(results_dir, output_filename_mean)
            output_path_mean_zoomed = os.path.join(results_dir, output_filename_mean_zoomed)

            # Create normal mean video
            print("Creating normal view mean video...")
            create_tracking_video(trajectory_data_mean, filters_order, args.dist, output_path_mean,
                                fps=args.fps, duration=args.duration, instance_idx=None, zoomed=False, output_format=args.format)

            # Create zoomed mean video
            print("Creating zoomed view mean video...")
            create_tracking_video(trajectory_data_mean, filters_order, args.dist, output_path_mean_zoomed,
                                fps=args.fps, duration=args.duration, instance_idx=None, zoomed=True, output_format=args.format)
            
            print(f"Normal mean trajectory video saved to: {output_path_mean}")
            print(f"Zoomed mean trajectory video saved to: {output_path_mean_zoomed}")
        else:
            print("No mean trajectory data found.")
        
        # 2. Create SINGLE INSTANCE trajectory videos (up to 5 instances)
        print("\n=== Creating SINGLE INSTANCE trajectory videos ===")
        
        # First check how many instances are available
        max_instances_available = 0
        for filt in filters_order:
            if filt in optimal_results:
                optimal_stats = optimal_results[filt]
                # Extract theta_vals based on filter type
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
            print(f"\nCreating videos for instance #{instance_idx+1}...")
            trajectory_data_single, _ = extract_single_trajectories(optimal_results, results_path, args.dist, instance_idx)
            
            if len(trajectory_data_single) > 0:
                if args.output is None:
                    output_filename_single = f"tracking_video_INSTANCE_{instance_idx+1}_{args.dist}_fps{args.fps}.{args.format}"
                    output_filename_single_zoomed = f"tracking_video_INSTANCE_{instance_idx+1}_ZOOMED_{args.dist}_fps{args.fps}.{args.format}"
                else:
                    output_filename_single = f"INSTANCE_{instance_idx+1}_{args.output}"
                    output_filename_single_zoomed = f"INSTANCE_{instance_idx+1}_ZOOMED_{args.output}"

                output_path_single = os.path.join(results_dir, output_filename_single)
                output_path_single_zoomed = os.path.join(results_dir, output_filename_single_zoomed)

                # Create normal single instance video
                print(f"Creating normal view for instance #{instance_idx+1}...")
                create_tracking_video(trajectory_data_single, filters_order, args.dist, output_path_single,
                                    fps=args.fps, duration=args.duration, instance_idx=instance_idx, zoomed=False, output_format=args.format)

                # Create zoomed single instance video
                print(f"Creating zoomed view for instance #{instance_idx+1}...")
                create_tracking_video(trajectory_data_single, filters_order, args.dist, output_path_single_zoomed,
                                    fps=args.fps, duration=args.duration, instance_idx=instance_idx, zoomed=True, output_format=args.format)
                
                print(f"Instance #{instance_idx+1} normal video saved to: {output_path_single}")
                print(f"Instance #{instance_idx+1} zoomed video saved to: {output_path_single_zoomed}")
                single_videos_created += 1
            else:
                print(f"No trajectory data found for instance #{instance_idx+1}")
                break
        
        print(f"\nSummary: Created {single_videos_created} single instance videos")
        
        if len(trajectory_data_mean) == 0 and single_videos_created == 0:
            print("No trajectory data found. You need to run main0_withFW_CT3D.py first.")
            return
        
    except FileNotFoundError as e:
        print(f"Error: Could not find results files for distribution '{args.dist}'")
        print(f"Make sure you have run main0_withFW_CT3D.py with --dist {args.dist} first")
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"Error creating tracking video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()