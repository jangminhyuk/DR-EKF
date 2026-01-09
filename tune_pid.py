#!/usr/bin/env python3
"""
PID Controller Tuning Script for Unicycle Trajectory Tracking
Systematically tests different PID gain combinations to find optimal parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time
import os

# Import required modules from main0.py
from main0 import (
    PIDController, unicycle_dynamics, observation_function,
    generate_reference_trajectory, compute_tracking_cost
)


def simulate_pid_tracking(pid_gains, reference_traj, noise_level=0.01, dt=0.2):
    """
    Simulate PID tracking performance with given gains.
    
    Args:
        pid_gains: Dict with keys 'Kp_pos', 'Ki_pos', 'Kd_pos', 'Kp_theta', 'Ki_theta', 'Kd_theta'
        reference_traj: Reference trajectory [3 x T]
        noise_level: Standard deviation of process/measurement noise
        dt: Time step
    
    Returns:
        Dict with tracking metrics
    """
    T = reference_traj.shape[1] - 1
    nx, nu = 3, 2
    
    # Initialize PID controller with given gains
    pid = PIDController(**pid_gains)
    
    # Allocate arrays
    x = np.zeros((T+1, nx, 1))
    u_traj = np.zeros((T, nu, 1))
    tracking_errors = np.zeros((T+1, 3))  # [pos_x_error, pos_y_error, theta_error]
    
    # Initial state (start near reference with small perturbation)
    x[0] = reference_traj[:, 0].reshape(-1, 1) + noise_level * np.random.randn(nx, 1)
    
    # Simulation loop
    for t in range(T):
        # Reference state at current time
        ref_state = reference_traj[:, t].reshape(-1, 1)
        
        # Compute tracking errors
        pos_error = x[t, :2, 0] - ref_state[:2, 0]
        theta_error = x[t, 2, 0] - ref_state[2, 0]
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))  # Wrap angle
        
        tracking_errors[t] = [pos_error[0], pos_error[1], theta_error]
        
        # PID control
        u = pid.compute_control(x[t], ref_state, dt)
        u_traj[t] = u.copy()
        
        # Unicycle dynamics with process noise
        process_noise = noise_level * np.random.randn(nx, 1)
        x[t+1] = unicycle_dynamics(x[t], u, dt) + process_noise
    
    # Final tracking error
    ref_final = reference_traj[:, -1].reshape(-1, 1)
    pos_error_final = x[-1, :2, 0] - ref_final[:2, 0]
    theta_error_final = x[-1, 2, 0] - ref_final[2, 0]
    theta_error_final = np.arctan2(np.sin(theta_error_final), np.cos(theta_error_final))
    tracking_errors[-1] = [pos_error_final[0], pos_error_final[1], theta_error_final]
    
    # Compute tracking cost (same as main0.py)
    tracking_cost = compute_tracking_cost(x, u_traj, reference_traj)
    
    # Compute various metrics
    pos_rmse = np.sqrt(np.mean(tracking_errors[:, :2]**2))
    theta_rmse = np.sqrt(np.mean(tracking_errors[:, 2]**2))
    max_pos_error = np.max(np.linalg.norm(tracking_errors[:, :2], axis=1))
    max_theta_error = np.max(np.abs(tracking_errors[:, 2]))
    control_effort = np.mean(np.linalg.norm(u_traj, axis=1)**2)
    final_pos_error = np.linalg.norm(pos_error_final)
    final_theta_error = np.abs(theta_error_final)
    
    return {
        'tracking_cost': tracking_cost,
        'pos_rmse': pos_rmse,
        'theta_rmse': theta_rmse,
        'max_pos_error': max_pos_error,
        'max_theta_error': max_theta_error,
        'control_effort': control_effort,
        'final_pos_error': final_pos_error,
        'final_theta_error': final_theta_error,
        'state_traj': x,
        'control_traj': u_traj,
        'tracking_errors': tracking_errors
    }


def grid_search_pid_gains(reference_traj, noise_level=0.01, num_trials=5):
    """
    Grid search over PID gain space to find optimal parameters.
    """
    print("Starting PID parameter grid search...")
    
    # Define search space for PID gains
    Kp_pos_range = [2.0, 5.0, 8.0, 12.0]
    Ki_pos_range = [0.05, 0.1, 0.2, 0.5]
    Kd_pos_range = [0.5, 1.0, 1.5, 2.0]
    
    Kp_theta_range = [1.5, 3.0, 5.0, 8.0]
    Ki_theta_range = [0.05, 0.1, 0.2, 0.4]
    Kd_theta_range = [0.2, 0.5, 0.8, 1.2]
    
    best_cost = np.inf
    best_gains = None
    best_results = None
    
    total_combinations = (len(Kp_pos_range) * len(Ki_pos_range) * len(Kd_pos_range) * 
                         len(Kp_theta_range) * len(Ki_theta_range) * len(Kd_theta_range))
    
    print(f"Testing {total_combinations} gain combinations with {num_trials} trials each...")
    
    results_log = []
    combination_count = 0
    
    for gains in product(Kp_pos_range, Ki_pos_range, Kd_pos_range, 
                        Kp_theta_range, Ki_theta_range, Kd_theta_range):
        combination_count += 1
        
        pid_gains = {
            'Kp_pos': gains[0],
            'Ki_pos': gains[1], 
            'Kd_pos': gains[2],
            'Kp_theta': gains[3],
            'Ki_theta': gains[4],
            'Kd_theta': gains[5]
        }
        
        # Run multiple trials to get robust statistics
        trial_costs = []
        trial_metrics = []
        
        for trial in range(num_trials):
            np.random.seed(1000 + trial * 123 + combination_count)  # Reproducible but varied seeds
            
            try:
                result = simulate_pid_tracking(pid_gains, reference_traj, noise_level)
                trial_costs.append(result['tracking_cost'])
                trial_metrics.append(result)
            except Exception as e:
                print(f"Simulation failed for gains {pid_gains}: {e}")
                trial_costs.append(np.inf)
                continue
        
        if len(trial_costs) > 0:
            mean_cost = np.mean([c for c in trial_costs if c != np.inf])
            std_cost = np.std([c for c in trial_costs if c != np.inf])
            
            # Compute average metrics
            valid_metrics = [m for i, m in enumerate(trial_metrics) if trial_costs[i] != np.inf]
            if len(valid_metrics) > 0:
                avg_metrics = {
                    'pos_rmse': np.mean([m['pos_rmse'] for m in valid_metrics]),
                    'theta_rmse': np.mean([m['theta_rmse'] for m in valid_metrics]),
                    'max_pos_error': np.mean([m['max_pos_error'] for m in valid_metrics]),
                    'control_effort': np.mean([m['control_effort'] for m in valid_metrics]),
                    'final_pos_error': np.mean([m['final_pos_error'] for m in valid_metrics])
                }
            else:
                continue
            
            # Store results
            result_entry = {
                'gains': pid_gains.copy(),
                'mean_cost': mean_cost,
                'std_cost': std_cost,
                **avg_metrics
            }
            results_log.append(result_entry)
            
            # Check if this is the best so far
            if mean_cost < best_cost:
                best_cost = mean_cost
                best_gains = pid_gains.copy()
                best_results = result_entry.copy()
                
                print(f"New best gains found! Cost: {best_cost:.2f}")
                print(f"  Position: Kp={best_gains['Kp_pos']}, Ki={best_gains['Ki_pos']}, Kd={best_gains['Kd_pos']}")
                print(f"  Orientation: Kp={best_gains['Kp_theta']}, Ki={best_gains['Ki_theta']}, Kd={best_gains['Kd_theta']}")
        
        if combination_count % 50 == 0:
            print(f"Progress: {combination_count}/{total_combinations} combinations tested")
    
    print(f"\nPID tuning completed! Tested {len(results_log)} valid combinations.")
    
    return best_gains, best_results, results_log


def fine_tune_around_best(best_gains, reference_traj, noise_level=0.01, num_trials=10):
    """
    Fine-tune around the best gains found in grid search.
    """
    print("\nFine-tuning around best gains...")
    
    # Define fine-tuning ranges (±20% around best values)
    factor = 0.2
    
    def create_fine_range(center_val, factor=0.2, num_points=5):
        min_val = center_val * (1 - factor)
        max_val = center_val * (1 + factor)
        return np.linspace(min_val, max_val, num_points)
    
    Kp_pos_fine = create_fine_range(best_gains['Kp_pos'])
    Ki_pos_fine = create_fine_range(best_gains['Ki_pos'])
    Kd_pos_fine = create_fine_range(best_gains['Kd_pos'])
    
    Kp_theta_fine = create_fine_range(best_gains['Kp_theta'])
    Ki_theta_fine = create_fine_range(best_gains['Ki_theta'])
    Kd_theta_fine = create_fine_range(best_gains['Kd_theta'])
    
    best_fine_cost = np.inf
    best_fine_gains = None
    
    fine_combinations = list(product(Kp_pos_fine, Ki_pos_fine, Kd_pos_fine,
                                   Kp_theta_fine, Ki_theta_fine, Kd_theta_fine))
    
    print(f"Testing {len(fine_combinations)} fine-tuning combinations...")
    
    for i, gains in enumerate(fine_combinations):
        pid_gains = {
            'Kp_pos': gains[0],
            'Ki_pos': gains[1],
            'Kd_pos': gains[2],
            'Kp_theta': gains[3],
            'Ki_theta': gains[4],
            'Kd_theta': gains[5]
        }
        
        trial_costs = []
        for trial in range(num_trials):
            np.random.seed(2000 + trial * 456 + i)
            
            try:
                result = simulate_pid_tracking(pid_gains, reference_traj, noise_level)
                trial_costs.append(result['tracking_cost'])
            except:
                trial_costs.append(np.inf)
        
        mean_cost = np.mean([c for c in trial_costs if c != np.inf])
        
        if mean_cost < best_fine_cost:
            best_fine_cost = mean_cost
            best_fine_gains = pid_gains.copy()
    
    print(f"Fine-tuning completed! Best cost: {best_fine_cost:.2f}")
    return best_fine_gains, best_fine_cost


def visualize_best_performance(best_gains, reference_traj, noise_level=0.01):
    """
    Visualize tracking performance with best gains.
    """
    print("\nVisualizing best PID performance...")
    
    # Run simulation with best gains
    np.random.seed(12345)  # Fixed seed for reproducible visualization
    result = simulate_pid_tracking(best_gains, reference_traj, noise_level)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Optimally Tuned PID Controller Performance', fontsize=14)
    
    # 2D trajectory plot
    ax = axes[0, 0]
    t_array = np.linspace(0, 10, reference_traj.shape[1])
    ax.plot(reference_traj[0], reference_traj[1], 'r--', linewidth=2, label='Reference')
    ax.plot(result['state_traj'][:, 0, 0], result['state_traj'][:, 1, 0], 'b-', linewidth=1.5, label='Actual')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('2D Trajectory Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Position tracking errors
    ax = axes[0, 1]
    ax.plot(t_array, np.linalg.norm(result['tracking_errors'][:, :2], axis=1), 'r-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Tracking Error')
    ax.grid(True, alpha=0.3)
    
    # Orientation tracking
    ax = axes[1, 0]
    ax.plot(t_array, reference_traj[2], 'r--', linewidth=2, label='Reference θ')
    ax.plot(t_array, result['state_traj'][:, 2, 0], 'b-', linewidth=1.5, label='Actual θ')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Orientation (rad)')
    ax.set_title('Orientation Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Control inputs
    ax = axes[1, 1]
    t_control = np.linspace(0, 10, result['control_traj'].shape[0])
    ax.plot(t_control, result['control_traj'][:, 0, 0], 'g-', linewidth=1.5, label='Linear velocity')
    ax.plot(t_control, result['control_traj'][:, 1, 0], 'm-', linewidth=1.5, label='Angular velocity')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Input')
    ax.set_title('Control Signals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if not os.path.exists('./results'):
        os.makedirs('./results')
    plt.savefig('./results/pid_tuning_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print performance summary
    print("\nOptimal PID Performance Summary:")
    print(f"  Tracking cost: {result['tracking_cost']:.2f}")
    print(f"  Position RMSE: {result['pos_rmse']:.4f} m")
    print(f"  Orientation RMSE: {result['theta_rmse']:.4f} rad")
    print(f"  Max position error: {result['max_pos_error']:.4f} m")
    print(f"  Final position error: {result['final_pos_error']:.4f} m")
    print(f"  Control effort: {result['control_effort']:.4f}")


def main():
    """Main PID tuning routine."""
    print("=== PID Controller Tuning for Unicycle Trajectory Tracking ===\n")
    
    # Generate reference trajectory
    reference_traj = generate_reference_trajectory(T_total=10.0, dt=0.2)
    
    # Test with small noise level for clean tuning
    noise_level = 0.005  # Small disturbance as requested
    
    start_time = time.time()
    
    # Step 1: Grid search for coarse tuning
    best_gains, best_results, results_log = grid_search_pid_gains(
        reference_traj, noise_level, num_trials=3
    )
    
    print(f"\nCoarse tuning results:")
    print(f"  Best gains: {best_gains}")
    print(f"  Best cost: {best_results['mean_cost']:.2f}")
    
    # Step 2: Fine-tune around best gains
    fine_tuned_gains, fine_tuned_cost = fine_tune_around_best(
        best_gains, reference_traj, noise_level, num_trials=5
    )
    
    print(f"\nFinal optimized PID gains:")
    print(f"  Kp_pos={fine_tuned_gains['Kp_pos']:.3f}, Ki_pos={fine_tuned_gains['Ki_pos']:.3f}, Kd_pos={fine_tuned_gains['Kd_pos']:.3f}")
    print(f"  Kp_theta={fine_tuned_gains['Kp_theta']:.3f}, Ki_theta={fine_tuned_gains['Ki_theta']:.3f}, Kd_theta={fine_tuned_gains['Kd_theta']:.3f}")
    print(f"  Final cost: {fine_tuned_cost:.2f}")
    
    # Step 3: Visualize performance
    visualize_best_performance(fine_tuned_gains, reference_traj, noise_level)
    
    # Step 4: Test robustness with higher noise
    print(f"\nTesting robustness with higher noise (std=0.02)...")
    np.random.seed(54321)
    robust_result = simulate_pid_tracking(fine_tuned_gains, reference_traj, noise_level=0.02)
    print(f"  Robust tracking cost: {robust_result['tracking_cost']:.2f}")
    print(f"  Robust position RMSE: {robust_result['pos_rmse']:.4f} m")
    
    elapsed_time = time.time() - start_time
    print(f"\nPID tuning completed in {elapsed_time:.1f} seconds.")
    
    return fine_tuned_gains


if __name__ == "__main__":
    optimal_gains = main()