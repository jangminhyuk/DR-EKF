#!/usr/bin/env python3
"""
Computation time comparison between EKF, DR_EKF_CDC, DR_EKF_TAC, and DR_EKF_CDC with Frank-Wolfe.
Sequential execution to measure pure estimator computation time.
"""

import numpy as np
import time
import os

from estimator.EKF import EKF
from estimator.DR_EKF_CDC import DR_EKF_CDC
from estimator.DR_EKF_TAC import DR_EKF_TAC
from common_utils import save_data, enforce_positive_definiteness
from estimator.base_filter import BaseFilter

# Helper for sampling functions
_temp_A, _temp_C = np.eye(3), np.eye(2, 3)
_temp_params = np.zeros((3, 1)), np.eye(3)
_temp_params_v = np.zeros((2, 1)), np.eye(2)
_sampler = BaseFilter(1, 'normal', 'normal', (_temp_A, _temp_C), np.eye(3, 2),
                     *_temp_params, *_temp_params, *_temp_params_v,
                     *_temp_params, *_temp_params, *_temp_params_v)

def sample_from_distribution(mu, Sigma, dist_type, max_val=None, min_val=None, scale=None, N=1):
    """Sample from specified distribution type"""
    if dist_type == "normal":
        return _sampler.normal(mu, Sigma, N)
    elif dist_type == "quadratic":
        return _sampler.quadratic(max_val, min_val, N)
    elif dist_type == "laplace":
        return _sampler.laplace(mu, scale, N)
    else:
        raise ValueError(f"Unsupported distribution: {dist_type}")

# --- Unicycle Dynamics Implementation ---
def unicycle_dynamics(x, u, dt=0.2):
    """Unicycle dynamics: x = [px, py, theta]^T, u = [v, omega]^T"""
    px, py, theta = x[0, 0], x[1, 0], x[2, 0]
    v, omega = u[0, 0], u[1, 0]
    
    px_next = px + v * np.cos(theta) * dt
    py_next = py + v * np.sin(theta) * dt
    theta_next = theta + omega * dt
    
    return np.array([[px_next], [py_next], [theta_next]])

def unicycle_jacobian(x, u, dt=0.2):
    """Jacobian of unicycle dynamics w.r.t. state"""
    theta = x[2, 0]
    v = u[0, 0]
    
    A = np.array([[1, 0, -v * np.sin(theta) * dt],
                  [0, 1,  v * np.cos(theta) * dt],
                  [0, 0,  1]])
    return A

def observation_function(x):
    """Observation function: y = [px, py]^T"""
    return np.array([[x[0, 0]], [x[1, 0]]])

def observation_jacobian(x):
    """Observation Jacobian C_t = ∂h/∂x (constant for this system)"""
    return np.array([[1, 0, 0], [0, 1, 0]])

# --- PID Controller for Unicycle Trajectory Tracking ---
class PIDController:
    def __init__(self, Kp_pos=5.5, Ki_pos=0.12, Kd_pos=1.6, 
                 Kp_theta=4.0, Ki_theta=0.2, Kd_theta=0.22):
        self.Kp_pos = Kp_pos
        self.Ki_pos = Ki_pos  
        self.Kd_pos = Kd_pos
        self.Kp_theta = Kp_theta
        self.Ki_theta = Ki_theta
        self.Kd_theta = Kd_theta
        
        # Error accumulation
        self.integral_pos = np.zeros(2)
        self.integral_theta = 0.0
        self.prev_error_pos = np.zeros(2)
        self.prev_error_theta = 0.0
        self.first_call = True
        
    def reset(self):
        """Reset PID controller state"""
        self.integral_pos = np.zeros(2)
        self.integral_theta = 0.0
        self.prev_error_pos = np.zeros(2)
        self.prev_error_theta = 0.0
        self.first_call = True
    
    def compute_control(self, current_state, reference_state, dt=0.2):
        """
        Compute control input using PID controller for unicycle
        current_state: [px, py, theta]^T
        reference_state: [px_ref, py_ref, theta_ref]^T
        """
        # Position error
        e_pos = np.array([current_state[0, 0] - reference_state[0, 0],
                         current_state[1, 0] - reference_state[1, 0]])
        
        # Orientation error (handle wrap-around)
        e_theta = current_state[2, 0] - reference_state[2, 0]
        e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))
        
        # PID for position
        self.integral_pos += e_pos * dt
        if not self.first_call:
            derivative_pos = (e_pos - self.prev_error_pos) / dt
        else:
            derivative_pos = np.zeros(2)
            self.first_call = False
        
        # PID for orientation
        self.integral_theta += e_theta * dt
        derivative_theta = (e_theta - self.prev_error_theta) / dt
        
        # Unicycle control law: convert position error to velocity commands
        # Linear velocity based on distance to target
        pos_error_magnitude = np.linalg.norm(e_pos)
        v_cmd = self.Kp_pos * pos_error_magnitude + self.Ki_pos * np.linalg.norm(self.integral_pos) + self.Kd_pos * np.linalg.norm(derivative_pos)
        
        # Angular velocity from orientation PID
        omega_cmd = self.Kp_theta * e_theta + self.Ki_theta * self.integral_theta + self.Kd_theta * derivative_theta
        
        # Update previous errors
        self.prev_error_pos = e_pos.copy()
        self.prev_error_theta = e_theta
        
        # Apply control direction: move towards target
        if pos_error_magnitude > 0.01:  # Avoid division by zero
            # Direction towards target
            direction = -e_pos / pos_error_magnitude
            # Adjust velocity direction
            current_theta = current_state[2, 0]
            desired_direction = np.array([np.cos(current_theta), np.sin(current_theta)])
            # If moving away from target, use negative velocity
            if np.dot(direction, desired_direction) < 0:
                v_cmd = -v_cmd
        
        # Limit control inputs (increased saturation limits)
        v_cmd = np.clip(v_cmd, -5.0, 5.0)  # Increased from ±2.0 to ±5.0
        omega_cmd = np.clip(omega_cmd, -2*np.pi, 2*np.pi)  # Increased from ±π to ±2π
        
        return np.array([[v_cmd], [omega_cmd]])

def generate_reference_trajectory(T_total=10.0, dt=0.2):
    """Generate curvy sinusoidal reference trajectory"""
    time_steps = int(T_total / dt) + 1
    time = np.linspace(0, T_total, time_steps)
    
    # Curvy sinusoidal trajectory parameters
    Amp = 5.0       # Amplitude of sinusoidal x motion
    slope = 1.0     # Linear slope for y motion  
    omega = 0.5     # Frequency of sinusoidal motion
    
    # Position trajectory (for unicycle: px, py)
    x_d = Amp * np.sin(omega * time)
    y_d = slope * time
    
    # Compute orientation from velocity direction
    vx_d = Amp * omega * np.cos(omega * time)
    vy_d = slope * np.ones(time_steps)
    theta_d = np.arctan2(vy_d, vx_d)
    
    return np.array([x_d, y_d, theta_d])

def run_timing_simulation(estimator, reference_traj, dt):
    """Run simulation and measure only estimator computation time"""
    T = reference_traj.shape[1] - 1
    nx, ny = estimator.nx, estimator.ny
    nu = 2
    
    pid = PIDController()
    
    # Allocate arrays
    x = np.zeros((T+1, nx, 1))
    y = np.zeros((T+1, ny, 1))
    x_est = np.zeros((T+1, nx, 1))
    u_traj = np.zeros((T, nu, 1))
    
    # Store timing results
    estimator_times = []
    
    # Reset noise index
    estimator._noise_index = 0
    
    # Initialization
    x[0] = estimator.sample_initial_state()
    x_est[0] = estimator.nominal_x0_mean.copy()
    
    # First measurement and update - TIME THIS
    v0 = estimator.sample_measurement_noise()
    y[0] = observation_function(x[0]) + v0
    
    start_time = time.perf_counter()
    x_est[0] = estimator._initial_update(x_est[0], y[0])
    end_time = time.perf_counter()
    estimator_times.append(end_time - start_time)
    
    # Main simulation loop
    for t in range(T):
        # PID control (NOT TIMED - control computation excluded)
        ref_state = reference_traj[:, t].reshape(-1, 1)
        u = pid.compute_control(x_est[t], ref_state, dt)
        u_traj[t] = u.copy()
        
        # True state propagation using unicycle dynamics (NOT TIMED)
        w = estimator.sample_process_noise()
        x[t+1] = unicycle_dynamics(x[t], u) + w
        estimator._noise_index += 1
        
        # Measurement using observation function (NOT TIMED)
        v = estimator.sample_measurement_noise()
        y[t+1] = observation_function(x[t+1]) + v
        
        # State estimation update - TIME ONLY THIS
        start_time = time.perf_counter()
        x_est[t+1] = estimator.update_step(x_est[t], y[t+1], t+1, u)
        end_time = time.perf_counter()
        estimator_times.append(end_time - start_time)
    
    return estimator_times

def estimate_nominal_parameters(true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                              dist, num_samples, x0_max=None, x0_min=None, w_max=None, w_min=None, 
                              v_max=None, v_min=None, x0_scale=None, w_scale=None, v_scale=None):
    """Estimate nominal parameters using samples from true distributions"""
    nx, ny = true_x0_mean.shape[0], true_mu_v.shape[0]
    
    # Sample from true distributions
    if dist == "normal":
        x0_samples = sample_from_distribution(true_x0_mean, true_x0_cov, "normal", N=num_samples)
        w_samples = sample_from_distribution(true_mu_w, true_Sigma_w, "normal", N=num_samples)
        v_samples = sample_from_distribution(true_mu_v, true_Sigma_v, "normal", N=num_samples)
    elif dist == "quadratic":
        x0_samples = sample_from_distribution(None, None, "quadratic", x0_max, x0_min, N=num_samples)
        w_samples = sample_from_distribution(None, None, "quadratic", w_max, w_min, N=num_samples)
        v_samples = sample_from_distribution(None, None, "quadratic", v_max, v_min, N=num_samples)
    elif dist == "laplace":
        x0_samples = sample_from_distribution(true_x0_mean, None, "laplace", scale=x0_scale, N=num_samples)
        w_samples = sample_from_distribution(true_mu_w, None, "laplace", scale=w_scale, N=num_samples)
        v_samples = sample_from_distribution(true_mu_v, None, "laplace", scale=v_scale, N=num_samples)
    
    # Compute sample statistics
    nominal_x0_mean = np.mean(x0_samples, axis=1).reshape(-1, 1)
    nominal_x0_cov = np.cov(x0_samples)
    
    nominal_mu_w = np.mean(w_samples, axis=1).reshape(-1, 1)
    nominal_Sigma_w = np.cov(w_samples)
    
    nominal_mu_v = np.mean(v_samples, axis=1).reshape(-1, 1)
    nominal_Sigma_v = np.cov(v_samples)
    
    # Ensure positive definiteness
    nominal_x0_cov = enforce_positive_definiteness(nominal_x0_cov)
    nominal_Sigma_w = enforce_positive_definiteness(nominal_Sigma_w)
    nominal_Sigma_v = enforce_positive_definiteness(nominal_Sigma_v)
    
    return (nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, 
            nominal_mu_v, nominal_Sigma_v)

def run_timing_experiment(dist='normal', T_total=10.0, num_samples=100, robust_val=0.1, num_trials=10):
    """Run timing comparison for all four filters"""
    
    # Generate reference trajectory
    reference_traj = generate_reference_trajectory(T_total)
    T = reference_traj.shape[1] - 1
    dt = 0.2
    nx, ny, nu = 3, 2, 2
    
    # System matrices for DR-EKF  
    A = np.eye(nx)  # Placeholder - will use jacobians online
    B = np.zeros((nx, nu))  # Placeholder - unicycle dynamics don't use linear B
    C = np.array([[1, 0, 0], [0, 1, 0]])
    system_data = (A, C)
    
    # Set up problem parameters
    x0_mean = reference_traj[:, 0].reshape(-1, 1)
    x0_cov = 0.01 * np.eye(nx)
    
    if dist == "normal":
        mu_w = np.zeros((nx, 1))
        Sigma_w = 0.05 * np.eye(nx)
        mu_v = np.zeros((ny, 1))
        Sigma_v = 0.05 * np.eye(ny)
        v_max = v_min = w_max = w_min = x0_max = x0_min = None
        x0_scale = w_scale = v_scale = None
    else:  # quadratic
        w_max = 0.2 * np.ones(nx)
        w_min = -0.1 * np.ones(nx)
        mu_w = (0.5 * (w_max + w_min))[:, None]
        Sigma_w = 3.0/20.0 * np.diag((w_max - w_min)**2)
        x0_max = 0.1 * np.ones(nx)
        x0_min = -0.1 * np.ones(nx)
        x0_cov = 3.0/20.0 * np.diag((x0_max - x0_min)**2)
        v_min = -0.2 * np.ones(ny)
        v_max = 0.1 * np.ones(ny)
        mu_v = (0.5 * (v_max + v_min))[:, None]
        Sigma_v = 3.0/20.0 * np.diag((v_max - v_min)**2)
        x0_scale = w_scale = v_scale = None
    
    # Generate nominal parameters
    np.random.seed(2024)
    nominal_params = estimate_nominal_parameters(
        x0_mean, x0_cov, mu_w, Sigma_w, mu_v, Sigma_v, dist, num_samples,
        x0_max, x0_min, w_max, w_min, v_max, v_min, x0_scale, w_scale, v_scale)
    
    (nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, 
     nominal_mu_v, nominal_Sigma_v) = nominal_params
    
    # Filters to test
    filters_to_test = ['EKF', 'DR_EKF_CDC', 'DR_EKF_TAC', 'DR_EKF_CDC_FW']
    
    # Store timing results for all trials
    all_timing_results = {filter_name: [] for filter_name in filters_to_test}
    
    print(f"Running timing comparison for {num_trials} trials...")
    print(f"Distribution: {dist}, Trajectory length: {T} steps, Robust parameter: {robust_val}")
    print("-" * 80)
    
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}:")
        
        # Run each filter sequentially (no parallel computation)
        for filter_name in filters_to_test:
            # Set same seed for fair comparison
            np.random.seed(2024 + trial)
            
            if filter_name == 'EKF':
                estimator = EKF(T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
                               true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                               true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                               true_mu_v=mu_v, true_Sigma_v=Sigma_v,
                               nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
                               nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
                               nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
                               nonlinear_dynamics=unicycle_dynamics,
                               dynamics_jacobian=unicycle_jacobian,
                               observation_function=observation_function,
                               observation_jacobian=observation_jacobian,
                               x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min,
                               v_max=v_max, v_min=v_min, x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale)
            
            elif filter_name == 'DR_EKF_CDC':
                estimator = DR_EKF_CDC(T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
                                     true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                                     true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                                     true_mu_v=mu_v, true_Sigma_v=Sigma_v,
                                     nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
                                     nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
                                     nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
                                     nonlinear_dynamics=unicycle_dynamics,
                                     dynamics_jacobian=unicycle_jacobian,
                                     observation_function=observation_function,
                                     observation_jacobian=observation_jacobian,
                                     theta_x=robust_val, theta_v=robust_val,
                                     x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min,
                                     v_max=v_max, v_min=v_min, x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale)
            
            elif filter_name == 'DR_EKF_TAC':
                estimator = DR_EKF_TAC(T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
                                     true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                                     true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                                     true_mu_v=mu_v, true_Sigma_v=Sigma_v,
                                     nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
                                     nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
                                     nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
                                     nonlinear_dynamics=unicycle_dynamics,
                                     dynamics_jacobian=unicycle_jacobian,
                                     observation_function=observation_function,
                                     observation_jacobian=observation_jacobian,
                                     theta_x=robust_val, theta_v=robust_val, theta_w=robust_val,
                                     x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min,
                                     v_max=v_max, v_min=v_min, x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale)
            
            elif filter_name == 'DR_EKF_CDC_FW':
                estimator = DR_EKF_CDC(T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
                                     true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                                     true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                                     true_mu_v=mu_v, true_Sigma_v=Sigma_v,
                                     nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
                                     nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
                                     nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
                                     nonlinear_dynamics=unicycle_dynamics,
                                     dynamics_jacobian=unicycle_jacobian,
                                     observation_function=observation_function,
                                     observation_jacobian=observation_jacobian,
                                     theta_x=robust_val, theta_v=robust_val,
                                     solver="fw",  # Use Frank-Wolfe solver
                                     x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min,
                                     v_max=v_max, v_min=v_min, x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale)
            
            # Run timing simulation
            estimator_times = run_timing_simulation(estimator, reference_traj, dt)
            all_timing_results[filter_name].append(estimator_times)
            
            # Compute statistics for this trial
            avg_time_per_step = np.mean(estimator_times)
            total_time = np.sum(estimator_times)
            frequency_hz = 1.0 / avg_time_per_step if avg_time_per_step > 0 else float('inf')
            
            print(f"  {filter_name:<15}: {avg_time_per_step*1000:.4f} ms/step, {frequency_hz:.2f} Hz, Total: {total_time*1000:.4f} ms")
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS (Averaged over all trials):")
    print("=" * 80)
    
    # Compute final statistics across all trials
    final_results = {}
    
    for filter_name in filters_to_test:
        # Flatten all timing data from all trials
        all_times_flat = [time_step for trial_times in all_timing_results[filter_name] 
                         for time_step in trial_times]
        
        # Compute average time per step across all trials and all steps
        avg_time_per_step = np.mean(all_times_flat)
        std_time_per_step = np.std(all_times_flat)
        frequency_hz = 1.0 / avg_time_per_step if avg_time_per_step > 0 else float('inf')
        
        final_results[filter_name] = {
            'avg_time_per_step_ms': avg_time_per_step * 1000,
            'std_time_per_step_ms': std_time_per_step * 1000,
            'frequency_hz': frequency_hz,
            'all_times': all_times_flat
        }
        
        print(f"{filter_name:<15}: {avg_time_per_step*1000:.4f}±{std_time_per_step*1000:.4f} ms/step, {frequency_hz:.2f} Hz")
    
    # Save results
    results_path = "./results/timing_comparison_with_FW/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    save_data(os.path.join(results_path, f'timing_results_{dist}_{robust_val}.pkl'), final_results)
    
    print(f"\nTiming results saved to {results_path}")
    
    return final_results

def main():
    """Main timing comparison routine"""
    
    # Configuration
    dist = 'normal'  # Can be changed to 'quadratic' for different noise distributions
    T_total = 10.0   # 10 seconds simulation
    num_samples = 100
    robust_val = 0.1  # Medium robustness parameter
    num_trials = 10   # 10 trials as requested
    
    print("EKF vs DR-EKF Computation Time Comparison")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Distribution: {dist}")
    print(f"  Simulation time: {T_total} seconds")
    print(f"  Time step: 0.2 seconds ({int(T_total/0.2)} steps)")
    print(f"  Robustness parameter (θ): {robust_val}")
    print(f"  Number of trials: {num_trials}")
    print(f"  Nominal parameter samples: {num_samples}")
    print("=" * 80)
    
    # Run timing experiments
    results = run_timing_experiment(dist, T_total, num_samples, robust_val, num_trials)
    
    # Print summary comparison
    print("\n" + "=" * 80)
    print("SPEED COMPARISON SUMMARY:")
    print("=" * 80)
    
    ekf_time = results['EKF']['avg_time_per_step_ms']
    
    for filter_name, data in results.items():
        if filter_name != 'EKF':
            speedup_factor = ekf_time / data['avg_time_per_step_ms']
            print(f"{filter_name} is {speedup_factor:.2f}x {'faster' if speedup_factor > 1 else 'slower'} than EKF")
    
    print("=" * 80)

if __name__ == "__main__":
    main()