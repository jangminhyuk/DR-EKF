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
from common_utils import save_data, enforce_positive_definiteness
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

def wrap_angle(a):
    """Wrap angle to (-pi, pi]. Works with scalars or 1x1 arrays."""
    return np.arctan2(np.sin(a), np.cos(a))

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


def _ekf_filter_single(
    y_seq, u_seq, dt,
    x0_mean, x0_cov,
    mu_w, Q,
    mu_v, R,
    f, F_jac, h, H_jac
):
    """
    EKF forward pass for one trajectory.
    Shapes:
      y_seq: (T+1, ny, 1)
      u_seq: (T,   nu, 1)
    Returns:
      m_pred, P_pred: predicted at each t (t=0 uses prior)
      m_filt, P_filt: filtered at each t
      F_list: list of F_t (length T) where F_t is Jacobian at (m_filt[t], u_t)
    """
    T = u_seq.shape[0]
    nx = x0_mean.shape[0]
    I = np.eye(nx)

    m_pred = np.zeros((T+1, nx, 1))
    P_pred = np.zeros((T+1, nx, nx))
    m_filt = np.zeros((T+1, nx, 1))
    P_filt = np.zeros((T+1, nx, nx))
    F_list = []

    # prior at t=0
    m_pred[0] = x0_mean.copy()
    P_pred[0] = x0_cov.copy()

    # update with y0
    H0 = H_jac(m_pred[0])
    yhat0 = h(m_pred[0]) + mu_v
    innov0 = y_seq[0] - yhat0
    innov0[1, 0] = wrap_angle(innov0[1, 0])  # Wrap bearing innovation
    S0 = H0 @ P_pred[0] @ H0.T + R
    S0 = 0.5 * (S0 + S0.T)
    K0 = P_pred[0] @ H0.T @ np.linalg.solve(S0, np.eye(S0.shape[0]))
    m_filt[0] = m_pred[0] + K0 @ innov0
    P_filt[0] = (I - K0 @ H0) @ P_pred[0] @ (I - K0 @ H0).T + K0 @ R @ K0.T
    P_filt[0] = 0.5 * (P_filt[0] + P_filt[0].T)

    for t in range(T):
        # predict to t+1
        Ft = F_jac(m_filt[t], u_seq[t], dt)
        F_list.append(Ft)

        m_pred[t+1] = f(m_filt[t], u_seq[t], dt) + mu_w
        P_pred[t+1] = Ft @ P_filt[t] @ Ft.T + Q
        P_pred[t+1] = 0.5 * (P_pred[t+1] + P_pred[t+1].T)

        # update with y_{t+1}
        Ht = H_jac(m_pred[t+1])
        yhat = h(m_pred[t+1]) + mu_v
        innov = y_seq[t+1] - yhat
        innov[1, 0] = wrap_angle(innov[1, 0])  # Wrap bearing innovation
        S = Ht @ P_pred[t+1] @ Ht.T + R
        S = 0.5 * (S + S.T)
        K = P_pred[t+1] @ Ht.T @ np.linalg.solve(S, np.eye(S.shape[0]))

        m_filt[t+1] = m_pred[t+1] + K @ innov
        P_filt[t+1] = (I - K @ Ht) @ P_pred[t+1] @ (I - K @ Ht).T + K @ R @ K.T
        P_filt[t+1] = 0.5 * (P_filt[t+1] + P_filt[t+1].T)

    return m_pred, P_pred, m_filt, P_filt, F_list


def _rts_smoother_single(m_pred, P_pred, m_filt, P_filt, F_list):
    """
    Extended RTS smoother (uses stored Jacobians F_list).
    Returns:
      m_smooth, P_smooth
    """
    T = len(F_list)
    nx = m_filt.shape[1]
    m_smooth = m_filt.copy()
    P_smooth = P_filt.copy()

    for t in range(T-1, -1, -1):
        Ft = F_list[t]
        # smoother gain
        Ppred_next = P_pred[t+1]
        Ppred_next = 0.5 * (Ppred_next + Ppred_next.T)
        Gt = P_filt[t] @ Ft.T @ np.linalg.solve(Ppred_next, np.eye(nx))

        m_smooth[t] = m_filt[t] + Gt @ (m_smooth[t+1] - m_pred[t+1])
        P_smooth[t] = P_filt[t] + Gt @ (P_smooth[t+1] - P_pred[t+1]) @ Gt.T
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

    return m_smooth, P_smooth


def estimate_nominal_parameters_EM(
    u_data, y_data, dt,
    x0_mean_init, x0_cov_init,
    mu_w_init=None, Sigma_w_init=None,
    mu_v_init=None, Sigma_v_init=None,
    f=ct_dynamics,
    F_jac=ct_jacobian,
    h=radar_observation_function,
    H_jac=radar_observation_jacobian,
    max_iters=10,
    tol=1e-4,
    estimate_means=False,
    estimate_x0=False,
    cov_structure="diag",      # "full" | "diag" | "scalar"
    reg=1e-6,
    verbose=True
):
    """
    EM-like nominal parameter estimation from input-output data for CT model.

    Inputs:
      u_data: (N, T, 2, 1) or (T, 2, 1)
      y_data: (N, T+1, 2, 1) or (T+1, 2, 1)

    Returns tuple:
      (nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v)
    """
    # Normalize shapes to (N, ...)
    if u_data.ndim == 3:  # (T,2,1)
        u_data = u_data[None, ...]
    if y_data.ndim == 3:  # (T+1,2,1)
        y_data = y_data[None, ...]

    N = u_data.shape[0]
    T = u_data.shape[1]
    nx = x0_mean_init.shape[0]
    ny = y_data.shape[2]

    x0_mean = x0_mean_init.copy()
    x0_cov = x0_cov_init.copy()

    mu_w = np.zeros((nx, 1)) if mu_w_init is None else mu_w_init.copy()
    Q = 0.1 * np.eye(nx) if Sigma_w_init is None else Sigma_w_init.copy()

    mu_v = np.zeros((ny, 1)) if mu_v_init is None else mu_v_init.copy()
    R = 0.1 * np.eye(ny) if Sigma_v_init is None else Sigma_v_init.copy()

    # enforce SPD at start
    x0_cov = enforce_positive_definiteness(x0_cov)
    Q = enforce_positive_definiteness(Q)
    R = enforce_positive_definiteness(R)

    def _apply_structure(S):
        if cov_structure == "full":
            return S
        if cov_structure == "diag":
            return np.diag(np.diag(S))
        if cov_structure == "scalar":
            s = float(np.trace(S) / S.shape[0])
            return s * np.eye(S.shape[0])
        raise ValueError(f"Unknown cov_structure={cov_structure}")

    prev_obj = None
    for it in range(max_iters):
        # Accumulators for M-step
        w_res_list = []
        v_res_list = []
        x0_list = []
        x0_cov_list = []

        # E-step: smooth each rollout
        for k in range(N):
            y_seq = y_data[k]
            u_seq = u_data[k]

            m_pred, P_pred, m_filt, P_filt, F_list = _ekf_filter_single(
                y_seq, u_seq, dt,
                x0_mean, x0_cov,
                mu_w, Q,
                mu_v, R,
                f, F_jac, h, H_jac
            )
            m_smooth, P_smooth = _rts_smoother_single(m_pred, P_pred, m_filt, P_filt, F_list)
            # No angle wrapping needed for CT state (no theta component)

            if estimate_x0:
                x0_list.append(m_smooth[0])
                x0_cov_list.append(P_smooth[0])

            # residuals
            for t in range(T):
                # w_t approx
                w_hat = m_smooth[t+1] - f(m_smooth[t], u_seq[t], dt)
                w_res_list.append(w_hat)

            for t in range(T+1):
                v_hat = y_seq[t] - h(m_smooth[t])
                # Wrap bearing component of measurement residual
                v_hat[1, 0] = wrap_angle(v_hat[1, 0])
                v_res_list.append(v_hat)

        # Stack residuals: shape (dim, count)
        W = np.hstack(w_res_list)  # (nx, N*T)
        V = np.hstack(v_res_list)  # (ny, N*(T+1))

        # M-step: means
        if estimate_means:
            mu_w_new = np.mean(W, axis=1, keepdims=True)
            mu_v_new = np.mean(V, axis=1, keepdims=True)
        else:
            mu_w_new = mu_w
            mu_v_new = mu_v

        # M-step: covariances (moment-matching)
        Wc = W - mu_w_new
        Vc = V - mu_v_new

        Q_new = (Wc @ Wc.T) / max(Wc.shape[1], 1)
        R_new = (Vc @ Vc.T) / max(Vc.shape[1], 1)

        # Regularize + structure + SPD
        Q_new = _apply_structure(Q_new) + reg * np.eye(nx)
        R_new = _apply_structure(R_new) + reg * np.eye(ny)

        Q_new = enforce_positive_definiteness(Q_new)
        R_new = enforce_positive_definiteness(R_new)

        # x0 updates
        if estimate_x0 and len(x0_list) > 0:
            X0 = np.hstack(x0_list)  # (nx, N)
            x0_mean_new = np.mean(X0, axis=1, keepdims=True)
            # include smoother covariance at t=0
            P0_bar = sum(x0_cov_list) / len(x0_cov_list)
            centered = X0 - x0_mean_new
            x0_cov_new = P0_bar + (centered @ centered.T) / max(centered.shape[1], 1)
            x0_cov_new = _apply_structure(x0_cov_new) + reg * np.eye(nx)
            x0_cov_new = enforce_positive_definiteness(x0_cov_new)
        else:
            x0_mean_new = x0_mean
            x0_cov_new = x0_cov

        # Convergence check (relative change)
        dQ = np.linalg.norm(Q_new - Q, ord="fro") / (np.linalg.norm(Q, ord="fro") + 1e-12)
        dR = np.linalg.norm(R_new - R, ord="fro") / (np.linalg.norm(R, ord="fro") + 1e-12)
        dx0 = np.linalg.norm(x0_mean_new - x0_mean) / (np.linalg.norm(x0_mean) + 1e-12)

        if verbose:
            print(f"[EM] iter={it:02d}  rel_change: dQ={dQ:.3e}, dR={dR:.3e}, dx0={dx0:.3e}")

        # Update params
        mu_w, Q = mu_w_new, Q_new
        mu_v, R = mu_v_new, R_new
        x0_mean, x0_cov = x0_mean_new, x0_cov_new

        if max(dQ, dR, dx0) < tol:
            if verbose:
                print(f"[EM] Converged at iter={it} (tol={tol}).")
            break

    return x0_mean, x0_cov, mu_w, Q, mu_v, R

def run_experiment(exp_idx, dist, num_sim, seed_base, robust_val, filters_to_execute, T_steps, 
                  nominal_params, true_params, num_samples=100):
    """Run single experiment comparing filters"""
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
                                                nonlinear_dynamics=ct_dynamics,
                                                dynamics_jacobian=ct_jacobian,
                                                observation_function=radar_observation_function,
                                                observation_jacobian=radar_observation_jacobian,
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
                                             nonlinear_dynamics=ct_dynamics,
                                             dynamics_jacobian=ct_jacobian,
                                             observation_function=radar_observation_function,
                                             observation_jacobian=radar_observation_jacobian,
                                             theta_x=robust_val, theta_v=robust_val,
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
    
    robust_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
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
        sigma_omega = 0.015          # turn rate noise (rad/s)
        Sigma_w = np.diag([sigma_px**2, sigma_py**2, sigma_vx**2, sigma_vy**2, sigma_omega**2])
        
        # Measurement noise: [range, bearing]
        sigma_range = 0.01                    # range noise (m)
        sigma_bearing = np.deg2rad(0.1)      # bearing noise (rad)
        Sigma_v = np.diag([sigma_range**2, sigma_bearing**2])
        
        v_max = v_min = w_max = w_min = x0_max = x0_min = None
        x0_scale = w_scale = v_scale = None
    else:  # U-quadratic (match Gaussian covariance / scale)
        # For U-quadratic with support [min,max], Var = (3/20)*(max-min)^2.
        # If symmetric bounds ±A, Var = (3/5)*A^2.
        # To match Gaussian std sigma: A = sqrt(5/3)*sigma.

        # --- bounds for U-quadratic ---
        w_max = np.sqrt(5.0 / 3.0) * np.array([0.01, 0.01, 0.05, 0.05, 0.02])
        w_min = -w_max
        mu_w = np.zeros((nx, 1))
        Sigma_w = 3.0/20.0 * np.diag((w_max - w_min)**2)          # == diag(std_w**2)

        v_max = np.sqrt(5.0 / 3.0) * np.array([0.01, np.deg2rad(0.1)])  
        v_min = -v_max
        mu_v = np.zeros((ny, 1))
        Sigma_v = 3.0/20.0 * np.diag((v_max - v_min)**2)          # == diag(std_v**2)

        # initial uncertainty: use same scale as Gaussian x0_cov
        x0_max = np.sqrt(5.0 / 3.0) * np.array([0.5, 0.5, 0.5, 0.5, 0.05])        
        x0_min = -x0_max
        x0_cov = 3.0/20.0 * np.diag((x0_max - x0_min)**2)         # == diag(std_x0**2)

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
        cov_structure="diag",                 # strong recommendation for identifiability
        reg=1e-6,
        verbose=True
    )
    
    
    print(f"Nominal parameters estimated from {num_samples} samples with T_em={T_em}:")
    print(f"  Nominal x0_mean: {nominal_params[0].flatten()}")
    print(f"  Nominal mu_w: {nominal_params[2].flatten()}")
    print(f"  Nominal mu_v: {nominal_params[4].flatten()}")
    
    all_results = {}
    
    for robust_val in robust_vals:
        print(f"Running experiments for robust parameter = {robust_val}")
        
        experiments = Parallel(n_jobs=-1, backend='loky')(
            delayed(run_experiment)(exp_idx, dist, num_sim, seed_base, robust_val, 
                                   filters_to_execute, T_steps, nominal_params, true_params, num_samples)
            for exp_idx in range(num_exp)
        )
        
        # Aggregate results
        aggregated = {filter_name: {'mse': []} for filter_name in filters_to_execute}
        
        for exp in experiments:
            for filter_name in filters_to_execute:
                if filter_name in exp:
                    aggregated[filter_name]['mse'].append(exp[filter_name]['mse_mean'])
                    
        
        # Compute statistics
        final_results = {}
        for filter_name in filters_to_execute:
            if aggregated[filter_name]['mse']:
                result_dict = {
                    'mse_mean': np.mean(aggregated[filter_name]['mse']),
                    'mse_std': np.std(aggregated[filter_name]['mse'])
                }
                
                
                final_results[filter_name] = result_dict
        
        all_results[robust_val] = final_results
        
        # Save detailed experiment data (including trajectories) aggregated across all experiments
        # This will be used by plot0_with_FW.py for visualization
        if robust_val in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:  # Save for all theta values
            # Ensure directory exists
            detailed_results_dir = "./results/EKF_comparison_with_FW_CT/"
            if not os.path.exists(detailed_results_dir):
                os.makedirs(detailed_results_dir)
            
            detailed_results_path = os.path.join(detailed_results_dir, f'detailed_results_{robust_val}_{dist}.pkl')
            
            # Aggregate trajectory data from all experiments
            aggregated_detailed = {}
            for filter_name in filters_to_execute:
                all_sim_results = []
                # Collect simulation results from all experiments
                for exp in experiments:
                    if filter_name in exp and 'results' in exp[filter_name]:
                        all_sim_results.extend(exp[filter_name]['results'])
                
                if all_sim_results:
                    aggregated_detailed[filter_name] = {
                        'mse_mean': np.mean([np.mean(r['mse']) for r in all_sim_results]),
                        'results': all_sim_results  # All num_sim * num_exp trajectories
                    }
            
            save_data(detailed_results_path, aggregated_detailed)
        print(f"Results for θ={robust_val}:")
        for filter_name, stats in final_results.items():
            print(f"  {filter_name}: MSE={stats['mse_mean']:.4f}±{stats['mse_std']:.4f}")
    
    # Find optimal theta for each filter based on MSE
    optimal_results = {}
    for filter_name in filters_to_execute:
        best_mse = np.inf
        best_theta = None
        best_stats = None
        
        for theta, results in all_results.items():
            if filter_name in results:
                mse = results[filter_name]['mse_mean']
                if mse < best_mse:
                    best_mse = mse
                    best_theta = theta
                    best_stats = results[filter_name]
        
        if best_theta is not None:
            optimal_results[filter_name] = {
                'theta': best_theta,
                **best_stats
            }
            print(f"{filter_name}: Optimal θ={best_theta}, MSE={best_mse:.4f}")
    
    # Save results
    results_path = "./results/EKF_comparison_with_FW_CT/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    save_data(os.path.join(results_path, f'all_results_{dist}.pkl'), all_results)
    save_data(os.path.join(results_path, f'optimal_results_{dist}.pkl'), optimal_results)
    
    print(f"\nEKF vs DR-EKF comparison with Frank-Wolfe completed. Results saved to {results_path}")
    
    
    print("\nFinal Results Summary:")
    print("{:<15} {:<15} {:<20}".format("Filter", "Optimal θ", "MSE"))
    print("-" * 50)
    for filter_name, stats in optimal_results.items():
        print("{:<15} {:<15} {:<20.4f}".format(
            filter_name, stats['theta'], stats['mse_mean']))
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Uncertainty distribution (normal or quadratic)")
    parser.add_argument('--num_sim', default=1, type=int,
                        help="Number of simulation runs per experiment")
    parser.add_argument('--num_exp', default=10, type=int,
                        help="Number of independent experiments")
    parser.add_argument('--T_total', default=50.0, type=float,
                        help="Total simulation time")
    parser.add_argument('--T_em', default=10.0, type=float,
                        help="Horizon length for EM data generation")
    parser.add_argument('--num_samples', default=100, type=int,
                        help="Number of samples for nominal parameter estimation")
    args = parser.parse_args()
    main(args.dist, args.num_sim, args.num_exp, args.T_total, args.T_em, args.num_samples)