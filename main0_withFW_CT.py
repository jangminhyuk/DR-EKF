#!/usr/bin/env python3
"""
EKF vs DR-EKF comparison using 2D coordinated-turn (CT) dynamics with radar measurements.
"""

import numpy as np
import argparse
import os
import pickle
from joblib import Parallel, delayed

from estimator.EKF import EKF
from estimator.DR_EKF_CDC import DR_EKF_CDC
from estimator.DR_EKF_TAC import DR_EKF_TAC
from common_utils import save_data, enforce_positive_definiteness, estimate_nominal_parameters_EM, wrap_angle
from estimator.base_filter import BaseFilter

# Helper for sampling functions
_temp_A, _temp_C = np.eye(5), np.eye(2, 5)
_temp_params = np.zeros((5, 1)), np.eye(5)
_temp_params_v = np.zeros((2, 1)), np.eye(2)
_sampler = BaseFilter(1, 'normal', 'normal', (_temp_A, _temp_C), np.eye(5, 2),
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

def sample_measurement_noise_wrapped(mu_v, Sigma_v, dist_type, v_max=None, v_min=None, v_scale=None, N=1):
    """Sample measurement noise with proper bearing angle handling."""
    noise = sample_from_distribution(mu_v, Sigma_v, dist_type, v_max, v_min, v_scale, N)
    
    # For radar measurements, wrap bearing noise to reasonable bounds
    if noise.shape[0] >= 2:  # Check if we have bearing component
        # Wrap bearing noise to [-π, π] to prevent large jumps
        for i in range(N):
            if N == 1:
                noise[1, 0] = wrap_angle(noise[1, 0])
            else:
                noise[1, i] = wrap_angle(noise[1, i])
    
    return noise

# --- Coordinated Turn (CT) Dynamics Implementation ---
def ct_dynamics(x, u, k=None, dt=0.2, omega_eps=1e-4):
    """CT dynamics: x = [px, py, vx, vy, omega]^T (u is unused, kept for compatibility)"""
    px, py, vx, vy, omega = x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0]
    
    phi = omega * dt
    
    if abs(omega) >= omega_eps:
        # Turn rate is significant - use CT model
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        A = sin_phi / omega
        B = (1 - cos_phi) / omega
        C = (cos_phi - 1) / omega
        D = sin_phi / omega
        
        px_next = px + A * vx + B * vy
        py_next = py + C * vx + D * vy
        vx_next = cos_phi * vx - sin_phi * vy
        vy_next = sin_phi * vx + cos_phi * vy
        omega_next = omega
    else:
        # Straight-line approximation (constant velocity)
        px_next = px + vx * dt
        py_next = py + vy * dt
        vx_next = vx
        vy_next = vy
        omega_next = omega
    
    return np.array([[px_next], [py_next], [vx_next], [vy_next], [omega_next]])

def ct_jacobian(x, u, k=None, dt=0.2, omega_eps=1e-4):
    """Jacobian of CT dynamics w.r.t. state (5x5)"""
    px, py, vx, vy, omega = x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0]
    
    phi = omega * dt
    
    if abs(omega) >= omega_eps:
        # Turn rate is significant - use CT Jacobian
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        A = sin_phi / omega
        B = (1 - cos_phi) / omega
        C = (cos_phi - 1) / omega
        D = sin_phi / omega
        
        # Derivatives of A, B, C, D w.r.t. omega
        dA_domega = (dt * cos_phi) / omega - sin_phi / (omega**2)
        dB_domega = (dt * sin_phi) / omega - (1 - cos_phi) / (omega**2)
        dC_domega = (-dt * sin_phi) / omega - (cos_phi - 1) / (omega**2)
        dD_domega = (dt * cos_phi) / omega - sin_phi / (omega**2)
        
        # Derivatives of trigonometric functions w.r.t. omega
        dcos_phi_domega = -dt * sin_phi
        dsin_phi_domega = dt * cos_phi
        
        F = np.array([
            [1, 0, A, B, vx * dA_domega + vy * dB_domega],
            [0, 1, C, D, vx * dC_domega + vy * dD_domega],
            [0, 0, cos_phi, -sin_phi, vx * dcos_phi_domega + vy * (-dsin_phi_domega)],
            [0, 0, sin_phi, cos_phi, vx * dsin_phi_domega + vy * dcos_phi_domega],
            [0, 0, 0, 0, 1]
        ])
    else:
        # Straight-line approximation Jacobian (constant velocity)
        F = np.array([
            [1, 0, dt, 0, 0],
            [0, 1, 0, dt, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
    
    return F

def radar_observation_function(x, sensor_pos=(0, 0)):
    """Radar observation function: y = [range, bearing]^T"""
    px, py = x[0, 0], x[1, 0]
    sx, sy = sensor_pos
    
    dx = px - sx
    dy = py - sy
    
    range_val = np.sqrt(dx**2 + dy**2)
    bearing = np.arctan2(dy, dx)
    
    return np.array([[range_val], [bearing]])

def radar_observation_jacobian(x, sensor_pos=(0, 0), range_eps=1e-6):
    """Radar observation Jacobian H = ∂h/∂x (2x5)"""
    px, py = x[0, 0], x[1, 0]
    sx, sy = sensor_pos
    
    dx = px - sx
    dy = py - sy
    
    range_val = np.sqrt(dx**2 + dy**2)
    r = max(range_val, range_eps)  # Avoid division by zero
    
    # Partial derivatives
    dr_dpx = dx / r
    dr_dpy = dy / r
    db_dpx = -dy / (r**2)
    db_dpy = dx / (r**2)
    
    H = np.array([
        [dr_dpx, dr_dpy, 0, 0, 0],
        [db_dpx, db_dpy, 0, 0, 0]
    ])
    
    return H



def compute_tracking_cost(state_traj, input_traj, reference_traj):
    """Placeholder function - cost calculation removed"""
    return 0.0


def run_single_simulation(estimator, T, dt):
    """Run CT simulation without controller (autonomous dynamics)"""
    nx, ny = estimator.nx, estimator.ny
    nu = 2
    
    # Allocate arrays
    x = np.zeros((T+1, nx, 1))
    y = np.zeros((T+1, ny, 1))
    x_est = np.zeros((T+1, nx, 1))
    u_traj = np.zeros((T, nu, 1))  # Dummy control array (all zeros)
    mse = np.zeros(T+1)
    
    
    # Reset noise index
    estimator._noise_index = 0
    
    # Initialization
    x[0] = estimator.sample_initial_state()
    x_est[0] = estimator.nominal_x0_mean.copy()
    
    # First measurement and update
    v0 = estimator.sample_measurement_noise()
    y_raw0 = radar_observation_function(x[0]) + v0
    
    # For initial measurement, wrap bearing relative to initial predicted measurement
    if hasattr(estimator, 'h'):  # Check if filter has observation function (DR-EKF case)
        y_pred0 = estimator.h(x_est[0]) + estimator.nominal_mu_v
        y[0] = wrap_bearing_measurement(y_raw0, y_pred0)
    else:
        y[0] = y_raw0
    
    x_est[0] = estimator._initial_update(x_est[0], y[0])
    
    mse[0] = np.linalg.norm(x_est[0] - x[0])**2
    
    # Main simulation loop
    for t in range(T):
        # No controller - CT model is autonomous
        u = np.zeros((nu, 1))  # Dummy control input
        u_traj[t] = u.copy()
        
        # True state propagation using autonomous CT dynamics (NOT TIMED)
        w = estimator.sample_process_noise()
        x[t+1] = ct_dynamics(x[t], u, dt=dt) + w
        estimator._noise_index += 1
        
        # Measurement using radar observation function (NOT TIMED)
        v = estimator.sample_measurement_noise()
        y_raw = radar_observation_function(x[t+1]) + v
        
        # CRITICAL FIX: Wrap bearing measurements for consistent innovations (NOT TIMED)
        # Predict what the measurement should be to avoid large bearing jumps
        if hasattr(estimator, 'h'):  # Check if filter has observation function (DR-EKF case)
            # Predict state for bearing wrapping
            x_pred = estimator.f(x_est[t], u) + estimator.nominal_mu_w
            y_pred = estimator.h(x_pred) + estimator.nominal_mu_v
            y[t+1] = wrap_bearing_measurement(y_raw, y_pred)
        else:
            y[t+1] = y_raw
        
        # State estimation update
        x_est[t+1] = estimator.update_step(x_est[t], y[t+1], t+1, u)
        
        mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
    
    return {
        'mse': mse,
        'state_traj': x,
        'est_state_traj': x_est,
        'input_traj': u_traj,
    }

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

def wrap_bearing_innovation(innovation):
    """Wrap bearing innovations to (-pi, pi]. For use in filter updates."""
    wrapped_innovation = innovation.copy()
    wrapped_innovation[1, 0] = wrap_angle(wrapped_innovation[1, 0])  # Wrap bearing component
    return wrapped_innovation

def wrap_bearing_measurement(y_measured, y_predicted):
    """Wrap bearing measurement to be consistent with predicted bearing.
    
    This ensures that bearing innovations stay small and don't cause
    large jumps that can destabilize the DR-EKF optimization.
    """
    y_wrapped = y_measured.copy()
    if y_wrapped.shape[0] >= 2:  # Check if we have bearing measurement
        # Wrap the bearing measurement to be close to predicted bearing
        bearing_diff = y_measured[1, 0] - y_predicted[1, 0]
        wrapped_diff = wrap_angle(bearing_diff)
        y_wrapped[1, 0] = y_predicted[1, 0] + wrapped_diff
        
        # Additional safeguard: ensure innovation magnitude is reasonable
        innovation_magnitude = abs(wrapped_diff)
        if innovation_magnitude > np.pi/2:  # More than 90 degrees - likely still unwrapped
            # Try the opposite wrap direction
            alt_wrapped_diff = wrapped_diff - np.sign(wrapped_diff) * 2 * np.pi
            if abs(alt_wrapped_diff) < innovation_magnitude:
                y_wrapped[1, 0] = y_predicted[1, 0] + alt_wrapped_diff
    
    return y_wrapped

def _as_col(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x

def generate_io_dataset_ct(
    T_em,
    dt,
    num_rollouts,
    dist,
    true_x0_mean, true_x0_cov,
    true_mu_w, true_Sigma_w,
    true_mu_v, true_Sigma_v,
    x0_max=None, x0_min=None,
    w_max=None, w_min=None,
    v_max=None, v_min=None,
    x0_scale=None, w_scale=None, v_scale=None,
    seed=None
):
    """
    Generate an input-output dataset (u,y) for the CT system.
    u_data contains dummy zeros since CT model is autonomous.

    Returns:
      u_data: shape (N, T, 2, 1) - dummy zeros
      y_data: shape (N, T+1, 2, 1) - radar measurements [range, bearing]
    """
    if seed is not None:
        np.random.seed(seed)

    T = int(T_em / dt)
    nx, ny, nu = 5, 2, 2

    u_data = np.zeros((num_rollouts, T, nu, 1))  # Dummy control data (all zeros)
    y_data = np.zeros((num_rollouts, T+1, ny, 1))

    for k in range(num_rollouts):
        # Sample initial state
        if dist == "normal":
            x0 = sample_from_distribution(true_x0_mean, true_x0_cov, "normal", N=1)
        elif dist == "quadratic":
            x0 = sample_from_distribution(None, None, "quadratic", x0_max, x0_min, N=1)
        elif dist == "laplace":
            x0 = sample_from_distribution(true_x0_mean, None, "laplace", scale=x0_scale, N=1)
        else:
            raise ValueError(f"Unsupported dist={dist}")

        x = _as_col(x0[:, 0])  # (5,1)

        # No controller needed for autonomous CT model
    
        # First measurement
        if dist == "normal":
            v0 = sample_from_distribution(true_mu_v, true_Sigma_v, "normal", N=1)
        elif dist == "quadratic":
            v0 = sample_from_distribution(None, None, "quadratic", v_max, v_min, N=1)
        elif dist == "laplace":
            v0 = sample_from_distribution(true_mu_v, None, "laplace", scale=v_scale, N=1)
        
        y_raw_0 = radar_observation_function(x) + _as_col(v0[:, 0])
        y_data[k, 0] = y_raw_0  # First measurement - no previous reference for wrapping
        prev_bearing = y_raw_0[1, 0]  # Store previous bearing for continuity

        # Rollout
        for t in range(T):
            # No control for autonomous CT model
            u = np.zeros((nu, 1))  # Dummy control
            u_data[k, t] = u

            # Process noise
            if dist == "normal":
                w = sample_from_distribution(true_mu_w, true_Sigma_w, "normal", N=1)
            elif dist == "quadratic":
                w = sample_from_distribution(None, None, "quadratic", w_max, w_min, N=1)
            elif dist == "laplace":
                w = sample_from_distribution(true_mu_w, None, "laplace", scale=w_scale, N=1)

            w = _as_col(w[:, 0])

            # Propagate
            x = ct_dynamics(x, u, dt) + w

            # Measurement noise
            if dist == "normal":
                v = sample_from_distribution(true_mu_v, true_Sigma_v, "normal", N=1)
            elif dist == "quadratic":
                v = sample_from_distribution(None, None, "quadratic", v_max, v_min, N=1)
            elif dist == "laplace":
                v = sample_from_distribution(true_mu_v, None, "laplace", scale=v_scale, N=1)

            v = _as_col(v[:, 0])
            y_raw = radar_observation_function(x) + v
            
            # CRITICAL: Ensure bearing continuity in data generation
            # Wrap bearing to be continuous with previous measurement
            current_bearing = y_raw[1, 0]
            bearing_diff = current_bearing - prev_bearing
            wrapped_diff = wrap_angle(bearing_diff)
            continuous_bearing = prev_bearing + wrapped_diff
            
            y_data[k, t+1] = y_raw.copy()
            y_data[k, t+1, 1, 0] = continuous_bearing
            prev_bearing = continuous_bearing  # Update for next iteration

    return u_data, y_data

def run_experiment(exp_idx, dist, num_sim, seed_base, theta_vals, filters_to_execute, T_steps,
                  nominal_params, true_params, num_samples=100):
    """Run single experiment comparing filters with specific theta values

    Args:
        theta_vals: dict with keys 'theta_x', 'theta_v', 'theta_w' containing the theta values for this run
    """
    experiment_seed = seed_base + exp_idx * 12345
    np.random.seed(experiment_seed)
    
    T = T_steps
    dt = 0.2
    nx, ny, nu = 5, 2, 2
    
    # System matrices for DR-EKF  
    A = np.eye(nx)  # Placeholder - will use jacobians online
    B = np.zeros((nx, nu))  # Placeholder - CT dynamics don't use linear B
    C = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])  # Extract position for radar measurements
    system_data = (A, C)
    
    # Unpack true parameters from main function (single source of truth)
    (x0_mean, x0_cov, mu_w, Sigma_w, mu_v, Sigma_v, 
     x0_max, x0_min, w_max, w_min, v_max, v_min, x0_scale, w_scale, v_scale) = true_params
    
    # Extract nominal parameters (shared across all filters)
    (nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w,
     nominal_mu_v, nominal_Sigma_v) = nominal_params

    # Extract theta values
    theta_x = theta_vals['theta_x']
    theta_v = theta_vals['theta_v']
    theta_w = theta_vals.get('theta_w', None)  # Only used by TAC

    results = {filter_name: [] for filter_name in filters_to_execute}
    
    # Run simulations with shared noise realizations
    for sim_idx in range(num_sim):
        # Compute unique seed for this simulation run
        seed_val = (experiment_seed + sim_idx * 10) % (2**32 - 1)
        
        # Store simulation results for all filters with shared noise
        sim_results = {}
        
        for filter_name in filters_to_execute:
            # CRITICAL: Reset to SAME seed before each filter to ensure identical noise sequences
            # This ensures fair comparison - all filters experience the same disturbances
            np.random.seed(seed_val)
            
            if filter_name == 'EKF':
                estimator = EKF(T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
                               true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                               true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                               true_mu_v=mu_v, true_Sigma_v=Sigma_v,
                               nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
                               nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
                               nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
                               nonlinear_dynamics=ct_dynamics,
                               dynamics_jacobian=ct_jacobian,
                               observation_function=radar_observation_function,
                               observation_jacobian=radar_observation_jacobian,
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
                                             nonlinear_dynamics=ct_dynamics,
                                             dynamics_jacobian=ct_jacobian,
                                             observation_function=radar_observation_function,
                                             observation_jacobian=radar_observation_jacobian,
                                             theta_x=theta_x, theta_v=theta_v,
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
                                                nonlinear_dynamics=ct_dynamics,
                                                dynamics_jacobian=ct_jacobian,
                                                observation_function=radar_observation_function,
                                                observation_jacobian=radar_observation_jacobian,
                                                theta_x=theta_x, theta_v=theta_v, theta_w=theta_w,
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
                                             nonlinear_dynamics=ct_dynamics,
                                             dynamics_jacobian=ct_jacobian,
                                             observation_function=radar_observation_function,
                                             observation_jacobian=radar_observation_jacobian,
                                             theta_x=theta_x, theta_v=theta_v,
                                             solver="fw",  # Use Frank-Wolfe solver
                                             x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min,
                                             v_max=v_max, v_min=v_min, x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale)
            else:
                continue
                
            try:
                result = run_single_simulation(estimator, T, dt)
                sim_results[filter_name] = result
                
                    
            except Exception as e:
                print(f"Simulation failed for {filter_name} (sim {sim_idx}): {e}")
                continue
        
        # Append results from this simulation run to each filter's result list
        for filter_name in sim_results:
            results[filter_name].append(sim_results[filter_name])
    
    # Compute aggregated statistics for each filter
    final_results = {}
    for filter_name in filters_to_execute:
        if results[filter_name]:  # If we have results for this filter
            filter_results = results[filter_name]
            result_dict = {
                'mse_mean': np.mean([np.mean(r['mse']) for r in filter_results]),
                'results': filter_results
            }
            
            
            final_results[filter_name] = result_dict
    
    return final_results

def main(dist, num_sim, num_exp, T_total=10.0, T_em=2.0, num_samples=100):
    """Main experiment routine"""
    seed_base = 2024
    
    # Convert total time to number of time steps
    dt = 0.2
    T_steps = int(T_total / dt)
    
    # Separate theta values for different noise sources
    theta_x_vals = [0.05, 0.1, 1.0, 2.0]  
    theta_v_vals = [0.05, 0.1, 1.0]  
    theta_w_vals = [0.05, 0.1, 1.0, 2.0] 

    # Fixed theta_x for TAC filter
    tac_theta_x_fixed = 0.1

    filters_to_execute = ['EKF', 'DR_EKF_CDC', 'DR_EKF_TAC', 'DR_EKF_CDC_FW']
    
    # Set up problem parameters for nominal estimation
    nx, ny = 5, 2
    # Initial mean state (CT benchmark, no input): [px, py, vx, vy, omega]
    x0_mean = np.array([[0.0],   # px0
                        [0.0],   # py0
                        [2.0],   # vx0
                        [0.0],   # vy0
                        [0.10]]) # omega0 (rad/s)
    # Initial state covariance with meaningful scales: [px, py, vx, vy, omega]
    x0_cov = np.diag([0.5**2, 0.5**2, 0.5**2, 0.5**2, 0.05**2])

    
    if dist == "normal":
        # Component-wise physically meaningful noise scales
        mu_w = np.zeros((nx, 1))
        mu_v = np.zeros((ny, 1))
        
        # Process noise: [px, py, vx, vy, omega]
        sigma_px = sigma_py = 0.01  # position noise (m)
        sigma_vx = sigma_vy = 0.05  # velocity noise (m/s)
        sigma_omega = 0.02          # turn rate noise (rad/s)
        Sigma_w = np.diag([sigma_px**2, sigma_py**2, sigma_vx**2, sigma_vy**2, sigma_omega**2])
        
        # Measurement noise: [range, bearing]
        sigma_range = 0.025                    # range noise (m)
        sigma_bearing = np.deg2rad(0.2)      # bearing noise (rad)
        Sigma_v = np.diag([sigma_range**2, sigma_bearing**2])
        
        v_max = v_min = w_max = w_min = x0_max = x0_min = None
        x0_scale = w_scale = v_scale = None
    else:  # U-quadratic (independent definition)
        # For U-quadratic with support [min,max], Var = (3/20)*(max-min)^2.
        # Mean = (max + min) / 2

        # --- Initial state bounds for U-quadratic ---
        # State: [px, py, vx, vy, omega]
        x0_max = np.array([0.5, 0.5, 2.5, 1.5, 0.2])
        x0_min = np.array([-0.5, -0.5, 1.5, 0.5, 0.1])
        x0_mean = (0.5 * (x0_max + x0_min)).reshape(-1, 1)
        x0_cov = 3.0/20.0 * np.diag((x0_max - x0_min)**2)

        # --- Process noise bounds for U-quadratic ---
        # [px, py, vx, vy, omega]
        w_max = np.array([0.01, 0.01, 0.05, 0.05, 0.02])
        w_min = -w_max
        mu_w = np.zeros((nx, 1))
        Sigma_w = 3.0/20.0 * np.diag((w_max - w_min)**2)

        # --- Measurement noise bounds for U-quadratic ---
        # [range, bearing]
        v_max = np.array([0.02, np.deg2rad(0.2)])
        v_min = -v_max
        mu_v = np.zeros((ny, 1))
        Sigma_v = 3.0/20.0 * np.diag((v_max - v_min)**2)

        x0_scale = w_scale = v_scale = None
    
    # Pack true parameters for single source of truth
    true_params = (x0_mean, x0_cov, mu_w, Sigma_w, mu_v, Sigma_v, 
                   x0_max, x0_min, w_max, w_min, v_max, v_min, x0_scale, w_scale, v_scale)
    
    # Generate shared nominal parameters directly using num_samples
    #np.random.seed(seed_base + 999999)  # Fixed seed for nominal parameter estimation
    # nominal_params = estimate_nominal_parameters(
    #     x0_mean, x0_cov, mu_w, Sigma_w, mu_v, Sigma_v, dist, num_samples,
    #     x0_max, x0_min, w_max, w_min, v_max, v_min, x0_scale, w_scale, v_scale)
    
    
    # --- Nominal estimation via EM from input-output dataset ---
    np.random.seed(seed_base + 999999)  # fixed seed for nominal estimation dataset

    # 1) generate input-output dataset
    u_data, y_data = generate_io_dataset_ct(
        T_em=T_em,
        dt=dt,
        num_rollouts=num_samples,     # reinterpret --num_samples as "# rollouts for nominal estimation"
        dist=dist,
        true_x0_mean=x0_mean, true_x0_cov=x0_cov,
        true_mu_w=mu_w, true_Sigma_w=Sigma_w,
        true_mu_v=mu_v, true_Sigma_v=Sigma_v,
        x0_max=x0_max, x0_min=x0_min,
        w_max=w_max, w_min=w_min,
        v_max=v_max, v_min=v_min,
        x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
        seed=seed_base + 999999
    )

    # 2) EM estimation (Gaussian nominal approximation)
    # Define wrapping functions for bearing angle measurements
    def wrap_bearing_innovation(innov):
        """Wrap bearing component of innovation (for filter updates)"""
        innov_wrapped = innov.copy()
        innov_wrapped[1, 0] = wrap_angle(innov_wrapped[1, 0])
        return innov_wrapped

    def wrap_bearing_residual(residual):
        """Wrap bearing component of measurement residual (for EM M-step)"""
        residual_wrapped = residual.copy()
        residual_wrapped[1, 0] = wrap_angle(residual_wrapped[1, 0])
        return residual_wrapped

    nominal_params = estimate_nominal_parameters_EM(
        u_data=u_data,
        y_data=y_data,
        dt=dt,
        x0_mean_init=x0_mean,
        x0_cov_init=x0_cov,
        mu_w_init=np.zeros((nx, 1)),          # or mu_w if you want to start from "true-ish"
        Sigma_w_init=0.01 * np.eye(nx),
        mu_v_init=np.zeros((ny, 1)),
        Sigma_v_init=0.01 * np.eye(ny),
        f=ct_dynamics,
        F_jac=ct_jacobian,
        h=radar_observation_function,
        H_jac=radar_observation_jacobian,
        max_iters=50,
        tol=1e-4,
        estimate_means=False,
        estimate_x0=False,
        cov_structure="full",                 # strong recommendation for identifiability
        reg=1e-6,
        verbose=True,
        wrap_innovation_fn=wrap_bearing_innovation,         # Wrap bearing innovations in filter
        wrap_measurement_residual_fn=wrap_bearing_residual, # Wrap bearing in measurement residuals
        wrap_process_residual_fn=None,                      # No process residual wrapping needed
        wrap_smoothed_state_fn=None                         # No smoothed state wrapping needed
    )
    
    
    print(f"Nominal parameters estimated from {num_samples} samples with T_em={T_em}:")
    print(f"  Nominal x0_mean: {nominal_params[0].flatten()}")
    print(f"  Nominal mu_w: {nominal_params[2].flatten()}")
    print(f"  Nominal mu_v: {nominal_params[4].flatten()}")
    
    # Storage for all results: filter_name -> theta_combination -> results
    all_results = {filter_name: {} for filter_name in filters_to_execute}

    # Generate theta combinations for CDC-based filters (theta_x, theta_v)
    from itertools import product
    cdc_theta_combinations = list(product(theta_x_vals, theta_v_vals))

    # Generate theta combinations for TAC filter (theta_v, theta_w with fixed theta_x)
    tac_theta_combinations = list(product(theta_v_vals, theta_w_vals))

    print(f"Running grid search:")
    print(f"  CDC/CDC_FW: {len(cdc_theta_combinations)} combinations (theta_x × theta_v)")
    print(f"  TAC: {len(tac_theta_combinations)} combinations (theta_v × theta_w, with theta_x={tac_theta_x_fixed} fixed)")
    print(f"  EKF: 1 configuration (no theta parameters)")

    # Run experiments for CDC-based filters (DR_EKF_CDC, DR_EKF_CDC_FW)
    cdc_filters = [f for f in filters_to_execute if f in ['DR_EKF_CDC', 'DR_EKF_CDC_FW']]
    for theta_x, theta_v in cdc_theta_combinations:
        print(f"\nRunning CDC-based filters with θ_x={theta_x}, θ_v={theta_v}")

        theta_vals = {'theta_x': theta_x, 'theta_v': theta_v}

        experiments = Parallel(n_jobs=-1, backend='loky')(
            delayed(run_experiment)(exp_idx, dist, num_sim, seed_base, theta_vals,
                                   cdc_filters, T_steps, nominal_params, true_params, num_samples)
            for exp_idx in range(num_exp)
        )

        # Aggregate results for each CDC filter
        for filter_name in cdc_filters:
            aggregated_mse = []
            aggregated_detailed_results = []

            for exp in experiments:
                if filter_name in exp:
                    aggregated_mse.append(exp[filter_name]['mse_mean'])
                    if 'results' in exp[filter_name]:
                        aggregated_detailed_results.extend(exp[filter_name]['results'])

            if aggregated_mse:
                theta_key = (theta_x, theta_v)
                all_results[filter_name][theta_key] = {
                    'mse_mean': np.mean(aggregated_mse),
                    'mse_std': np.std(aggregated_mse),
                    'theta_x': theta_x,
                    'theta_v': theta_v,
                    'results': aggregated_detailed_results
                }
                print(f"  {filter_name}: MSE={np.mean(aggregated_mse):.4f}±{np.std(aggregated_mse):.4f}")

    # Run experiments for TAC filter (DR_EKF_TAC) with fixed theta_x
    tac_filters = [f for f in filters_to_execute if f == 'DR_EKF_TAC']
    for theta_v, theta_w in tac_theta_combinations:
        theta_x = tac_theta_x_fixed  # Fixed value
        print(f"\nRunning TAC filter with θ_x={theta_x} (fixed), θ_v={theta_v}, θ_w={theta_w}")

        theta_vals = {'theta_x': theta_x, 'theta_v': theta_v, 'theta_w': theta_w}

        experiments = Parallel(n_jobs=-1, backend='loky')(
            delayed(run_experiment)(exp_idx, dist, num_sim, seed_base, theta_vals,
                                   tac_filters, T_steps, nominal_params, true_params, num_samples)
            for exp_idx in range(num_exp)
        )

        # Aggregate results for TAC filter
        for filter_name in tac_filters:
            aggregated_mse = []
            aggregated_detailed_results = []

            for exp in experiments:
                if filter_name in exp:
                    aggregated_mse.append(exp[filter_name]['mse_mean'])
                    if 'results' in exp[filter_name]:
                        aggregated_detailed_results.extend(exp[filter_name]['results'])

            if aggregated_mse:
                theta_key = (theta_x, theta_v, theta_w)
                all_results[filter_name][theta_key] = {
                    'mse_mean': np.mean(aggregated_mse),
                    'mse_std': np.std(aggregated_mse),
                    'theta_x': theta_x,
                    'theta_v': theta_v,
                    'theta_w': theta_w,
                    'results': aggregated_detailed_results
                }
                print(f"  {filter_name}: MSE={np.mean(aggregated_mse):.4f}±{np.std(aggregated_mse):.4f}")

    # Run experiments for EKF (no theta parameters)
    ekf_filters = [f for f in filters_to_execute if f == 'EKF']
    if ekf_filters:
        print(f"\nRunning EKF (no theta parameters)")

        theta_vals = {'theta_x': None, 'theta_v': None}  # Dummy values for EKF

        experiments = Parallel(n_jobs=-1, backend='loky')(
            delayed(run_experiment)(exp_idx, dist, num_sim, seed_base, theta_vals,
                                   ekf_filters, T_steps, nominal_params, true_params, num_samples)
            for exp_idx in range(num_exp)
        )

        # Aggregate results for EKF
        for filter_name in ekf_filters:
            aggregated_mse = []
            aggregated_detailed_results = []

            for exp in experiments:
                if filter_name in exp:
                    aggregated_mse.append(exp[filter_name]['mse_mean'])
                    if 'results' in exp[filter_name]:
                        aggregated_detailed_results.extend(exp[filter_name]['results'])

            if aggregated_mse:
                theta_key = 'no_theta'
                all_results[filter_name][theta_key] = {
                    'mse_mean': np.mean(aggregated_mse),
                    'mse_std': np.std(aggregated_mse),
                    'results': aggregated_detailed_results
                }
                print(f"  {filter_name}: MSE={np.mean(aggregated_mse):.4f}±{np.std(aggregated_mse):.4f}")
    
    # Find optimal theta for each filter based on MSE
    print("\n" + "="*80)
    print("OPTIMAL THETA SELECTION")
    print("="*80)

    optimal_results = {}
    for filter_name in filters_to_execute:
        if filter_name not in all_results or not all_results[filter_name]:
            continue

        best_mse = np.inf
        best_theta_key = None
        best_stats = None

        # Find best theta combination for this filter
        for theta_key, results in all_results[filter_name].items():
            mse = results['mse_mean']
            if mse < best_mse:
                best_mse = mse
                best_theta_key = theta_key
                best_stats = results

        if best_theta_key is not None:
            optimal_results[filter_name] = best_stats

            # Print optimal theta based on filter type
            if filter_name == 'EKF':
                print(f"{filter_name}: MSE={best_mse:.4f} (no theta parameters)")
            elif filter_name in ['DR_EKF_CDC', 'DR_EKF_CDC_FW']:
                print(f"{filter_name}: Optimal θ_x={best_stats['theta_x']}, θ_v={best_stats['theta_v']}, MSE={best_mse:.4f}")
            elif filter_name == 'DR_EKF_TAC':
                print(f"{filter_name}: Optimal θ_x={best_stats['theta_x']} (fixed), θ_v={best_stats['theta_v']}, θ_w={best_stats['theta_w']}, MSE={best_mse:.4f}")
    
    # Save results
    results_path = "./results/EKF_comparison_with_FW_CT/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_data(os.path.join(results_path, f'all_results_{dist}.pkl'), all_results)
    save_data(os.path.join(results_path, f'optimal_results_{dist}.pkl'), optimal_results)

    # Save detailed results for optimal theta combinations
    for filter_name, stats in optimal_results.items():
        if 'results' in stats and stats['results']:
            if filter_name == 'EKF':
                filename = f'detailed_results_{filter_name}_{dist}.pkl'
            elif filter_name in ['DR_EKF_CDC', 'DR_EKF_CDC_FW']:
                theta_x, theta_v = stats['theta_x'], stats['theta_v']
                filename = f'detailed_results_{filter_name}_tx{theta_x}_tv{theta_v}_{dist}.pkl'
            elif filter_name == 'DR_EKF_TAC':
                theta_x, theta_v, theta_w = stats['theta_x'], stats['theta_v'], stats['theta_w']
                filename = f'detailed_results_{filter_name}_tx{theta_x}_tv{theta_v}_tw{theta_w}_{dist}.pkl'
            else:
                filename = f'detailed_results_{filter_name}_{dist}.pkl'

            detailed_path = os.path.join(results_path, filename)
            detailed_data = {
                filter_name: {
                    'mse_mean': stats['mse_mean'],
                    'results': stats['results']
                }
            }
            save_data(detailed_path, detailed_data)
            print(f"Saved detailed results for {filter_name} to {filename}")
    
    print(f"\nEKF vs DR-EKF comparison with Frank-Wolfe completed. Results saved to {results_path}")

    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Filter':<20} {'Optimal Theta':<45} {'MSE':<15}")
    print("-" * 80)
    for filter_name, stats in optimal_results.items():
        if filter_name == 'EKF':
            theta_str = "N/A"
        elif filter_name in ['DR_EKF_CDC', 'DR_EKF_CDC_FW']:
            theta_str = f"θ_x={stats['theta_x']}, θ_v={stats['theta_v']}"
        elif filter_name == 'DR_EKF_TAC':
            theta_str = f"θ_x={stats['theta_x']} (fixed), θ_v={stats['theta_v']}, θ_w={stats['theta_w']}"
        else:
            theta_str = "Unknown"

        print(f"{filter_name:<20} {theta_str:<45} {stats['mse_mean']:<15.4f}")
    print("="*80)
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Uncertainty distribution (normal or quadratic)")
    parser.add_argument('--num_sim', default=1, type=int,
                        help="Number of simulation runs per experiment")
    parser.add_argument('--num_exp', default=5, type=int,
                        help="Number of independent experiments")
    parser.add_argument('--T_total', default=50.0, type=float,
                        help="Total simulation time")
    parser.add_argument('--T_em', default=10.0, type=float,
                        help="Horizon length for EM data generation")
    parser.add_argument('--num_samples', default=100, type=int,
                        help="Number of samples for nominal parameter estimation")
    args = parser.parse_args()
    main(args.dist, args.num_sim, args.num_exp, args.T_total, args.T_em, args.num_samples)