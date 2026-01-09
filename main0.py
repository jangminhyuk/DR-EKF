#!/usr/bin/env python3
"""
EKF vs DR-EKF comparison using unicycle dynamics.
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


def compute_tracking_cost(state_traj, input_traj, reference_traj):
    """Compute tracking cost: J = sum [10*||e_p||^2 + e_theta^2 + 0.1*||u||^2]"""
    T = input_traj.shape[0]
    total_cost = 0.0
    
    for t in range(T):
        e_p = state_traj[t, :2, 0] - reference_traj[:2, t]
        e_theta = state_traj[t, 2, 0] - reference_traj[2, t]
        e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))
        u = input_traj[t, :, 0]
        
        cost_t = 10 * np.linalg.norm(e_p)**2 + e_theta**2 + 0.1 * np.linalg.norm(u)**2
        total_cost += cost_t
    
    return total_cost

def generate_reference_trajectory(T_total=10.0, dt=0.2):
    """Generate curvy sinusoidal reference trajectory (from main1_with_MPC.py)"""
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

def run_single_simulation(estimator, reference_traj, dt):
    """Run simulation with PID controller"""
    T = reference_traj.shape[1] - 1
    nx, ny = estimator.nx, estimator.ny
    nu = 2
    
    pid = PIDController()
    
    # Allocate arrays
    x = np.zeros((T+1, nx, 1))
    y = np.zeros((T+1, ny, 1))
    x_est = np.zeros((T+1, nx, 1))
    u_traj = np.zeros((T, nu, 1))
    mse = np.zeros(T+1)
    
    # Reset noise index
    estimator._noise_index = 0
    
    # Initialization
    x[0] = estimator.sample_initial_state()
    x_est[0] = estimator.nominal_x0_mean.copy()
    
    # First measurement and update
    v0 = estimator.sample_measurement_noise()
    y[0] = observation_function(x[0]) + v0
    x_est[0] = estimator._initial_update(x_est[0], y[0])
    mse[0] = np.linalg.norm(x_est[0] - x[0])**2
    
    # Main simulation loop
    for t in range(T):
        # PID control
        ref_state = reference_traj[:, t].reshape(-1, 1)
        u = pid.compute_control(x_est[t], ref_state, dt)
        u_traj[t] = u.copy()
        
        # True state propagation using unicycle dynamics
        w = estimator.sample_process_noise()
        x[t+1] = unicycle_dynamics(x[t], u) + w
        estimator._noise_index += 1
        
        # Measurement using observation function
        v = estimator.sample_measurement_noise()
        y[t+1] = observation_function(x[t+1]) + v
        
        # State estimation update using common interface
        x_est[t+1] = estimator.update_step(x_est[t], y[t+1], t+1, u)
        
        mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
    
    tracking_cost = compute_tracking_cost(x, u_traj, reference_traj)
    
    return {
        'mse': mse,
        'tracking_cost': tracking_cost,
        'state_traj': x,
        'est_state_traj': x_est,
        'input_traj': u_traj
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

def _as_col(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x

def generate_io_dataset_unicycle(
    reference_traj,
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
    seed=None,
    control_mode="pid",              # "pid" or "random"
    u_bounds=(( -5.0,  5.0), ( -2*np.pi,  2*np.pi))  # (v bounds, omega bounds)
):
    """
    Generate an input-output dataset (u,y) for the unicycle system.

    Returns:
      u_data: shape (N, T, 2, 1)
      y_data: shape (N, T+1, 2, 1)
    """
    if seed is not None:
        np.random.seed(seed)

    T = reference_traj.shape[1] - 1
    nx, ny, nu = 3, 2, 2

    u_data = np.zeros((num_rollouts, T, nu, 1))
    y_data = np.zeros((num_rollouts, T+1, ny, 1))

    vmin, vmax = u_bounds[0]
    omin, omax = u_bounds[1]

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

        x = _as_col(x0[:, 0])  # (3,1)

        # Controller instance per rollout (for consistent integral state)
        pid = PIDController()

        # First measurement
        if dist == "normal":
            v0 = sample_from_distribution(true_mu_v, true_Sigma_v, "normal", N=1)
        elif dist == "quadratic":
            v0 = sample_from_distribution(None, None, "quadratic", v_max, v_min, N=1)
        elif dist == "laplace":
            v0 = sample_from_distribution(true_mu_v, None, "laplace", scale=v_scale, N=1)
        y_data[k, 0] = observation_function(x) + _as_col(v0[:, 0])

        # Rollout
        for t in range(T):
            # Choose control
            if control_mode == "pid":
                ref_state = reference_traj[:, t].reshape(-1, 1)
                u = pid.compute_control(x, ref_state, dt)
            elif control_mode == "random":
                u = np.array([[np.random.uniform(vmin, vmax)],
                              [np.random.uniform(omin, omax)]])
            else:
                raise ValueError(f"Unknown control_mode={control_mode}")

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
            x = unicycle_dynamics(x, u, dt) + w

            # Measurement noise
            if dist == "normal":
                v = sample_from_distribution(true_mu_v, true_Sigma_v, "normal", N=1)
            elif dist == "quadratic":
                v = sample_from_distribution(None, None, "quadratic", v_max, v_min, N=1)
            elif dist == "laplace":
                v = sample_from_distribution(true_mu_v, None, "laplace", scale=v_scale, N=1)

            v = _as_col(v[:, 0])
            y_data[k, t+1] = observation_function(x) + v

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
    f=unicycle_dynamics,
    F_jac=unicycle_jacobian,
    h=observation_function,
    H_jac=observation_jacobian,
    max_iters=50,
    tol=1e-4,
    estimate_means=True,
    estimate_x0=True,
    cov_structure="diag",      # "full" | "diag" | "scalar"
    reg=1e-6,
    wrap_theta_residual=True,
    verbose=True
):
    """
    EM-like nominal parameter estimation from input-output data for unicycle.

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
            m_smooth[:, 2, 0] = np.arctan2(np.sin(m_smooth[:, 2, 0]), np.cos(m_smooth[:, 2, 0]))

            if estimate_x0:
                x0_list.append(m_smooth[0])
                x0_cov_list.append(P_smooth[0])

            # residuals
            for t in range(T):
                # w_t approx
                w_hat = m_smooth[t+1] - f(m_smooth[t], u_seq[t], dt)
                if wrap_theta_residual:
                    w_hat[2, 0] = wrap_angle(w_hat[2, 0])
                w_res_list.append(w_hat)

            for t in range(T+1):
                v_hat = y_seq[t] - h(m_smooth[t])
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

def run_experiment(exp_idx, dist, num_sim, seed_base, robust_val, filters_to_execute, reference_traj, 
                  nominal_params, true_params, num_samples=100):
    """Run single experiment comparing filters"""
    experiment_seed = seed_base + exp_idx * 12345
    np.random.seed(experiment_seed)
    
    T = reference_traj.shape[1] - 1
    dt = 0.2
    nx, ny, nu = 3, 2, 2
    
    # System matrices for DR-EKF  
    A = np.eye(nx)  # Placeholder - will use jacobians online
    B = np.zeros((nx, nu))  # Placeholder - unicycle dynamics don't use linear B
    C = np.array([[1, 0, 0], [0, 1, 0]])
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
            else:
                continue
                
            try:
                result = run_single_simulation(estimator, reference_traj, dt)
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
            final_results[filter_name] = {
                'mse_mean': np.mean([np.mean(r['mse']) for r in filter_results]),
                'cost_mean': np.mean([r['tracking_cost'] for r in filter_results]),
                'results': filter_results
            }
    
    return final_results

def main(dist, num_sim, num_exp, T_total=10.0, num_samples=100):
    """Main experiment routine"""
    seed_base = 2024
    
    reference_traj = generate_reference_trajectory(T_total)
    robust_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    filters_to_execute = ['EKF', 'DR_EKF_CDC', 'DR_EKF_TAC']
    
    # Set up problem parameters for nominal estimation
    nx, ny = 3, 2
    x0_mean = reference_traj[:, 0].reshape(-1, 1)
    x0_cov = 0.01 * np.eye(nx)
    dt = 0.2

    
    if dist == "normal":
        mu_w = np.zeros((nx, 1))
        Sigma_w = 0.05 * np.eye(nx)
        mu_v = np.zeros((ny, 1))
        Sigma_v = 0.05 * np.eye(ny)
        v_max = v_min = w_max = w_min = x0_max = x0_min = None
        x0_scale = w_scale = v_scale = None
    else:  # quadratic
        w_max = 0.4 * np.ones(nx)
        w_min = -0.4 * np.ones(nx)
        mu_w = (0.5 * (w_max + w_min))[:, None]
        Sigma_w = 3.0/20.0 * np.diag((w_max - w_min)**2)
        x0_max = 0.01 * np.ones(nx)
        x0_min = -0.01 * np.ones(nx)
        x0_cov = 3.0/20.0 * np.diag((x0_max - x0_min)**2)
        v_min = -0.4 * np.ones(ny)
        v_max = 0.4 * np.ones(ny)
        mu_v = (0.5 * (v_max + v_min))[:, None]
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
    u_data, y_data = generate_io_dataset_unicycle(
        reference_traj=reference_traj,
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
        seed=seed_base + 999999,
        control_mode="pid"            # or "random" for persistent excitation
    )

    # 2) EM estimation (Gaussian nominal approximation)
    nominal_params = estimate_nominal_parameters_EM(
        u_data=u_data,
        y_data=y_data,
        dt=dt,
        x0_mean_init=x0_mean,
        x0_cov_init=x0_cov,
        mu_w_init=np.zeros((nx, 1)),          # or mu_w if you want to start from "true-ish"
        Sigma_w_init=0.1 * np.eye(nx),
        mu_v_init=np.zeros((ny, 1)),
        Sigma_v_init=0.1 * np.eye(ny),
        f=unicycle_dynamics,
        F_jac=unicycle_jacobian,
        h=observation_function,
        H_jac=observation_jacobian,
        max_iters=50,
        tol=1e-4,
        estimate_means=True,
        estimate_x0=True,
        cov_structure="diag",                 # strong recommendation for identifiability
        reg=1e-6,
        wrap_theta_residual=True,
        verbose=True
    )

    
    
    print(f"Nominal parameters estimated from {num_samples} samples:")
    print(f"  Nominal x0_mean: {nominal_params[0].flatten()}")
    print(f"  Nominal mu_w: {nominal_params[2].flatten()}")
    print(f"  Nominal mu_v: {nominal_params[4].flatten()}")
    
    all_results = {}
    
    for robust_val in robust_vals:
        print(f"Running experiments for robust parameter = {robust_val}")
        
        experiments = Parallel(n_jobs=-1, backend='loky')(
            delayed(run_experiment)(exp_idx, dist, num_sim, seed_base, robust_val, 
                                   filters_to_execute, reference_traj, nominal_params, true_params, num_samples)
            for exp_idx in range(num_exp)
        )
        
        # Aggregate results
        aggregated = {filter_name: {'mse': [], 'cost': []} for filter_name in filters_to_execute}
        
        for exp in experiments:
            for filter_name in filters_to_execute:
                if filter_name in exp:
                    aggregated[filter_name]['mse'].append(exp[filter_name]['mse_mean'])
                    aggregated[filter_name]['cost'].append(exp[filter_name]['cost_mean'])
        
        # Compute statistics
        final_results = {}
        for filter_name in filters_to_execute:
            if aggregated[filter_name]['mse']:
                final_results[filter_name] = {
                    'mse_mean': np.mean(aggregated[filter_name]['mse']),
                    'mse_std': np.std(aggregated[filter_name]['mse']),
                    'cost_mean': np.mean(aggregated[filter_name]['cost']),
                    'cost_std': np.std(aggregated[filter_name]['cost'])
                }
        
        all_results[robust_val] = final_results
        
        # Save detailed experiment data (including trajectories) aggregated across all experiments
        # This will be used by plot0.py for visualization
        if robust_val in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:  # Save for all theta values
            detailed_results_path = os.path.join("./results/EKF_comparison/", f'detailed_results_{robust_val}_{dist}.pkl')
            
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
                        'cost_mean': np.mean([r['tracking_cost'] for r in all_sim_results]),
                        'results': all_sim_results  # All num_sim * num_exp trajectories
                    }
            
            save_data(detailed_results_path, aggregated_detailed)
        print(f"Results for θ={robust_val}:")
        for filter_name, stats in final_results.items():
            print(f"  {filter_name}: MSE={stats['mse_mean']:.4f}±{stats['mse_std']:.4f}, "
                  f"Cost={stats['cost_mean']:.4f}±{stats['cost_std']:.4f}")
    
    # Find optimal theta for each filter
    optimal_results = {}
    for filter_name in filters_to_execute:
        best_cost = np.inf
        best_theta = None
        best_stats = None
        
        for theta, results in all_results.items():
            if filter_name in results:
                cost = results[filter_name]['cost_mean']
                if cost < best_cost:
                    best_cost = cost
                    best_theta = theta
                    best_stats = results[filter_name]
        
        if best_theta is not None:
            optimal_results[filter_name] = {
                'theta': best_theta,
                **best_stats
            }
            print(f"{filter_name}: Optimal θ={best_theta}, Cost={best_cost:.4f}")
    
    # Save results
    results_path = "./results/EKF_comparison/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    save_data(os.path.join(results_path, f'all_results_{dist}.pkl'), all_results)
    save_data(os.path.join(results_path, f'optimal_results_{dist}.pkl'), optimal_results)
    
    print(f"\nEKF vs DR-EKF comparison completed. Results saved to {results_path}")
    
    print("\nFinal Results Summary:")
    print("{:<15} {:<15} {:<20} {:<20}".format("Filter", "Optimal θ", "MSE", "Tracking Cost"))
    print("-" * 70)
    for filter_name, stats in optimal_results.items():
        print("{:<15} {:<15} {:<20.4f} {:<20.4f}".format(
            filter_name, stats['theta'], stats['mse_mean'], stats['cost_mean']))
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Uncertainty distribution (normal or quadratic)")
    parser.add_argument('--num_sim', default=1, type=int,
                        help="Number of simulation runs per experiment")
    parser.add_argument('--num_exp', default=50, type=int,
                        help="Number of independent experiments")
    parser.add_argument('--T_total', default=10.0, type=float,
                        help="Total simulation time")
    parser.add_argument('--num_samples', default=5, type=int,
                        help="Number of samples for nominal parameter estimation")
    args = parser.parse_args()
    main(args.dist, args.num_sim, args.num_exp, args.T_total, args.num_samples)