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
    px, py, theta = x[0, 0], x[1, 0], x[2, 0]
    v, omega = u[0, 0], u[1, 0]
    
    F = np.array([
        [1, 0, -v * np.sin(theta) * dt],
        [0, 1,  v * np.cos(theta) * dt],
        [0, 0,  1]
    ])
    
    return F

def observation_function(x):
    """Observation function: y = [px, py]^T"""
    return np.array([[x[0, 0]], [x[1, 0]]])

def observation_jacobian(x):
    """Jacobian of observation function w.r.t. state"""
    return np.array([[1, 0, 0], [0, 1, 0]])

# --- PID Controller ---
class PIDController:
    def __init__(self, kp=2.0, ki=0.1, kd=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error_pos = None
        self.integral_pos = np.zeros((2, 1))
        self.prev_error_heading = None
        self.integral_heading = 0.0
        
    def compute_control(self, current_state, desired_state, dt):
        """Compute control input [v, omega] for unicycle"""
        # Current position and heading
        px_curr, py_curr, theta_curr = current_state[0, 0], current_state[1, 0], current_state[2, 0]
        
        # Desired position (heading will be determined by direction to target)
        px_des, py_des = desired_state[0, 0], desired_state[1, 0]
        
        # Position error
        error_pos = np.array([[px_des - px_curr], [py_des - py_curr]])
        
        # PID for position
        if self.prev_error_pos is not None:
            self.integral_pos += error_pos * dt
            derivative_pos = (error_pos - self.prev_error_pos) / dt
        else:
            derivative_pos = np.zeros((2, 1))
        
        self.prev_error_pos = error_pos.copy()
        
        # Desired heading (direction to target)
        if np.linalg.norm(error_pos) > 1e-6:
            theta_des = np.arctan2(error_pos[1, 0], error_pos[0, 0])
        else:
            theta_des = theta_curr
        
        # Heading error (wrapped to [-pi, pi])
        error_heading = theta_des - theta_curr
        error_heading = np.arctan2(np.sin(error_heading), np.cos(error_heading))
        
        # PID for heading
        if self.prev_error_heading is not None:
            self.integral_heading += error_heading * dt
            derivative_heading = (error_heading - self.prev_error_heading) / dt
        else:
            derivative_heading = 0.0
        
        self.prev_error_heading = error_heading
        
        # Control outputs
        # Linear velocity: proportional to distance to target
        v = self.kp * np.linalg.norm(error_pos)
        v = np.clip(v, 0, 3.0)  # Limit max speed
        
        # Angular velocity: PID on heading error
        omega = (self.kp * error_heading + 
                self.ki * self.integral_heading + 
                self.kd * derivative_heading)
        omega = np.clip(omega, -2.0, 2.0)  # Limit max angular velocity
        
        return np.array([[v], [omega]])

def generate_reference_trajectory(T_total=10.0, dt=0.2):
    """Generate curvy reference trajectory for trajectory tracking."""
    time_steps = int(T_total / dt) + 1
    time = np.linspace(0, T_total, time_steps)
    
    # Curvy trajectory parameters
    Amp = 5.0       # Amplitude of sinusoidal x motion
    slope = 1.0     # Linear slope for y motion  
    omega = 0.5     # Frequency of sinusoidal motion
    
    # Position trajectory
    px_d = Amp * np.sin(omega * time)
    py_d = slope * time
    
    # Return as (nx, T+1) array
    reference_traj = np.array([px_d, py_d, np.zeros(time_steps)])  # No reference for heading
    
    return reference_traj

def compute_tracking_cost(x_traj, u_traj, reference_traj, Q_cost=None, R_cost=None):
    """Compute quadratic tracking cost"""
    T = x_traj.shape[0] - 1
    nx = x_traj.shape[1]
    nu = u_traj.shape[1]
    
    if Q_cost is None:
        Q_cost = np.diag([10, 10, 0.1])  # Position tracking more important than heading
    if R_cost is None:
        R_cost = np.diag([0.1, 1.0])  # Penalize angular velocity more
    
    total_cost = 0.0
    
    # State tracking cost
    for t in range(T + 1):
        x_curr = x_traj[t, :, 0]  # Current state
        x_ref = reference_traj[:, min(t, reference_traj.shape[1] - 1)]  # Reference state
        
        error = x_curr - x_ref
        total_cost += error.T @ Q_cost @ error
    
    # Control cost
    for t in range(T):
        u_curr = u_traj[t, :, 0]  # Current control
        total_cost += u_curr.T @ R_cost @ u_curr
    
    return total_cost

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

def generate_io_dataset_unicycle(reference_traj, dt, num_rollouts, dist, 
                               true_x0_mean, true_x0_cov, 
                               true_mu_w, true_Sigma_w, 
                               true_mu_v, true_Sigma_v,
                               x0_max=None, x0_min=None, 
                               w_max=None, w_min=None,
                               v_max=None, v_min=None,
                               x0_scale=None, w_scale=None, v_scale=None,
                               seed=42, control_mode="pid"):
    """
    Generate input-output dataset for unicycle dynamics using PID controller.
    """
    np.random.seed(seed)
    T = reference_traj.shape[1] - 1
    nx, ny = 3, 2  # state and observation dimensions for unicycle
    nu = 2         # control dimension (v, omega)

    # Storage
    u_data = []  # inputs: list of (T, nu, 1) arrays
    y_data = []  # outputs: list of (T+1, ny, 1) arrays

    for rollout in range(num_rollouts):
        # Sample initial state
        if dist == "normal":
            x0 = sample_from_distribution(true_x0_mean, true_x0_cov, "normal", N=1)[:, 0:1]
        elif dist == "quadratic":
            x0 = sample_from_distribution(None, None, "quadratic", x0_max, x0_min, N=1)[:, 0:1]
        elif dist == "laplace":
            x0 = sample_from_distribution(true_x0_mean, None, "laplace", scale=x0_scale, N=1)[:, 0:1]
        
        # Initialize PID controller for this rollout
        pid = PIDController()
        
        # Preallocate
        x_traj = np.zeros((T+1, nx, 1))
        u_traj = np.zeros((T, nu, 1))
        y_traj = np.zeros((T+1, ny, 1))
        
        x_traj[0] = x0
        
        # Initial observation with noise
        if dist == "normal":
            v0 = sample_from_distribution(true_mu_v, true_Sigma_v, "normal", N=1)[:, 0:1]
        elif dist == "quadratic":
            v0 = sample_from_distribution(None, None, "quadratic", v_max, v_min, N=1)[:, 0:1]
        elif dist == "laplace":
            v0 = sample_from_distribution(true_mu_v, None, "laplace", scale=v_scale, N=1)[:, 0:1]
        
        y_traj[0] = observation_function(x_traj[0]) + v0
        
        # Rollout
        for t in range(T):
            # Control policy
            if control_mode == "pid":
                ref_state = reference_traj[:, t].reshape(-1, 1)
                u_traj[t] = pid.compute_control(x_traj[t], ref_state, dt)
            elif control_mode == "random":
                # Random excitation
                u_traj[t] = np.random.randn(nu, 1) * 0.5
            
            # Process noise
            if dist == "normal":
                w = sample_from_distribution(true_mu_w, true_Sigma_w, "normal", N=1)[:, 0:1]
            elif dist == "quadratic":
                w = sample_from_distribution(None, None, "quadratic", w_max, w_min, N=1)[:, 0:1]
            elif dist == "laplace":
                w = sample_from_distribution(true_mu_w, None, "laplace", scale=w_scale, N=1)[:, 0:1]
            
            # Dynamics
            x_traj[t+1] = unicycle_dynamics(x_traj[t], u_traj[t], dt) + w
            
            # Observation noise
            if dist == "normal":
                v = sample_from_distribution(true_mu_v, true_Sigma_v, "normal", N=1)[:, 0:1]
            elif dist == "quadratic":
                v = sample_from_distribution(None, None, "quadratic", v_max, v_min, N=1)[:, 0:1]
            elif dist == "laplace":
                v = sample_from_distribution(true_mu_v, None, "laplace", scale=v_scale, N=1)[:, 0:1]
            
            # Observation
            y_traj[t+1] = observation_function(x_traj[t+1]) + v
        
        u_data.append(u_traj)
        y_data.append(y_traj)
    
    return u_data, y_data

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
    estimate_means=False,
    estimate_x0=False,
    cov_structure="diag",      # "full" | "diag" | "scalar"
    reg=1e-6,
    verbose=True
):
    """
    EM-like nominal parameter estimation from input-output data for unicycle model.
    """
    # Normalize shapes to (N, ...)
    if u_data[0].ndim == 3:  # List of (T,2,1)
        # Convert list to array format compatible with CT implementation
        u_data = [u.squeeze(-1) for u in u_data]  # Convert to (T,2)
        y_data = [y.squeeze(-1) for y in y_data]  # Convert to (T+1,2)
    
    N = len(u_data)
    T = u_data[0].shape[0] if N > 0 else 0
    nx = x0_mean_init.shape[0]
    ny = y_data[0].shape[1] if N > 0 else 2
    
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
    
    for it in range(max_iters):
        # Accumulators for M-step
        w_res_list = []
        v_res_list = []
        x0_list = []
        x0_cov_list = []
        
        # E-step: smooth each rollout
        for k in range(N):
            y_seq = y_data[k]  # (T+1, 2)
            u_seq = u_data[k]  # (T, 2)
            
            # Convert back to expected format for filtering
            y_seq_expanded = y_seq[:, :, np.newaxis]  # (T+1, 2, 1)
            u_seq_expanded = u_seq[:, :, np.newaxis]  # (T, 2, 1)
            
            # Simple EKF filtering and RTS smoothing
            m_smooth, P_smooth = _simple_ekf_smooth(
                y_seq_expanded, u_seq_expanded, dt,
                x0_mean, x0_cov, mu_w, Q, mu_v, R,
                f, F_jac, h, H_jac
            )
            
            if estimate_x0:
                x0_list.append(m_smooth[0])
                x0_cov_list.append(P_smooth[0])
            
            # residuals
            for t in range(T):
                # w_t approximation
                w_hat = m_smooth[t+1] - f(m_smooth[t], u_seq_expanded[t], dt)
                w_res_list.append(w_hat)
            
            for t in range(T+1):
                v_hat = y_seq_expanded[t] - h(m_smooth[t])
                v_res_list.append(v_hat)
        
        # Stack residuals
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
        
        # Convergence test
        if max(dQ, dR, dx0) < tol:
            if verbose:
                print(f"[EM] Converged at iteration {it+1}")
            break
    
    return (x0_mean, x0_cov, mu_w, Q, mu_v, R)

def _simple_ekf_smooth(y_seq, u_seq, dt, x0_mean, x0_cov, mu_w, Q, mu_v, R, f, F_jac, h, H_jac):
    """Simple EKF filtering and RTS smoothing for unicycle dynamics"""
    T = u_seq.shape[0]
    nx = x0_mean.shape[0]
    
    # Forward pass (filtering)
    m_filt = np.zeros((T+1, nx, 1))
    P_filt = np.zeros((T+1, nx, nx))
    m_pred = np.zeros((T+1, nx, 1))
    P_pred = np.zeros((T+1, nx, nx))
    F_list = []
    
    # Initialize
    m_filt[0] = x0_mean.copy()
    P_filt[0] = x0_cov.copy()
    
    for t in range(T):
        # Predict
        F_t = F_jac(m_filt[t], u_seq[t], dt)
        F_list.append(F_t)
        
        m_pred[t+1] = f(m_filt[t], u_seq[t], dt) + mu_w
        P_pred[t+1] = F_t @ P_filt[t] @ F_t.T + Q
        P_pred[t+1] = 0.5 * (P_pred[t+1] + P_pred[t+1].T)
        
        # Update
        H_t = H_jac(m_pred[t+1])
        y_pred = h(m_pred[t+1]) + mu_v
        residual = y_seq[t+1] - y_pred
        
        S = H_t @ P_pred[t+1] @ H_t.T + R
        S = 0.5 * (S + S.T)
        
        try:
            K = P_pred[t+1] @ H_t.T @ np.linalg.inv(S)
        except:
            K = P_pred[t+1] @ H_t.T @ np.linalg.pinv(S)
        
        m_filt[t+1] = m_pred[t+1] + K @ residual
        P_filt[t+1] = P_pred[t+1] - K @ H_t @ P_pred[t+1]
        P_filt[t+1] = 0.5 * (P_filt[t+1] + P_filt[t+1].T)
    
    # Backward pass (smoothing)
    m_smooth = m_filt.copy()
    P_smooth = P_filt.copy()
    
    for t in range(T-1, -1, -1):
        F_t = F_list[t]
        try:
            G_t = P_filt[t] @ F_t.T @ np.linalg.inv(P_pred[t+1])
        except:
            G_t = P_filt[t] @ F_t.T @ np.linalg.pinv(P_pred[t+1])
        
        m_smooth[t] = m_filt[t] + G_t @ (m_smooth[t+1] - m_pred[t+1])
        P_smooth[t] = P_filt[t] + G_t @ (P_smooth[t+1] - P_pred[t+1]) @ G_t.T
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)
    
    return m_smooth, P_smooth

def _ekf_forward_pass(y_data, u_data, params, f, F_jac, h, H_jac, dt, wrap_theta_residual):
    """
    Extended Kalman Filter forward pass.
    Returns:
      m_pred, P_pred, m_filt, P_filt, F_list
    """
    T = u_data.shape[0]
    nx = params['x0_mean'].shape[0]
    ny = y_data.shape[1]
    
    # Storage
    m_pred = np.zeros((T+1, nx, 1))
    P_pred = np.zeros((T+1, nx, nx))
    m_filt = np.zeros((T+1, nx, 1))
    P_filt = np.zeros((T+1, nx, nx))
    F_list = []
    
    # Initial condition
    m_filt[0] = params['x0_mean']
    P_filt[0] = params['x0_cov']
    
    for t in range(T):
        # Predict step
        Ft = F_jac(m_filt[t], u_data[t], dt)
        F_list.append(Ft)
        
        m_pred[t+1] = f(m_filt[t], u_data[t], dt) + params['mu_w']
        P_pred[t+1] = Ft @ P_filt[t] @ Ft.T + params['Sigma_w']
        P_pred[t+1] = 0.5 * (P_pred[t+1] + P_pred[t+1].T)  # Ensure symmetry
        
        # Update step
        Ht = H_jac(m_pred[t+1])
        y_pred = h(m_pred[t+1]) + params['mu_v']
        
        residual = y_data[t+1] - y_pred
        
        S = Ht @ P_pred[t+1] @ Ht.T + params['Sigma_v']
        S = 0.5 * (S + S.T)  # Ensure symmetry
        
        try:
            K = P_pred[t+1] @ Ht.T @ np.linalg.inv(S)
        except:
            K = P_pred[t+1] @ Ht.T @ np.linalg.pinv(S)
        
        m_filt[t+1] = m_pred[t+1] + K @ residual
        P_filt[t+1] = P_pred[t+1] - K @ Ht @ P_pred[t+1]
        P_filt[t+1] = 0.5 * (P_filt[t+1] + P_filt[t+1].T)  # Ensure symmetry
        
        # Wrap theta angle if needed
        if wrap_theta_residual and nx >= 3:
            m_filt[t+1][2, 0] = wrap_angle(m_filt[t+1][2, 0])

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
        
        # smooth
        m_smooth[t] = m_filt[t] + Gt @ (m_smooth[t+1] - m_pred[t+1])
        P_smooth[t] = P_filt[t] + Gt @ (P_smooth[t+1] - Ppred_next) @ Gt.T
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

    return m_smooth, P_smooth

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
    robust_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    filters_to_execute = ['EKF', 'DR_EKF_CDC', 'DR_EKF_TAC', 'DR_EKF_CDC_FW']
    
    # Set up problem parameters for nominal estimation
    nx, ny = 3, 2
    x0_mean = reference_traj[:, 0].reshape(-1, 1)
    x0_cov = 0.01 * np.eye(nx)
    dt = 0.2

    # Define true parameters based on distribution type
    if dist == "normal":
        # Process noise: moderate values
        mu_w = np.zeros((nx, 1))
        Sigma_w = 0.05 * np.eye(nx)
        
        # Measurement noise: position uncertainty  
        mu_v = np.zeros((ny, 1))
        Sigma_v = 0.05 * np.eye(ny)
        
        # Distribution bounds (not used for normal)
        x0_max = x0_min = w_max = w_min = v_max = v_min = None
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
        Sigma_w_init=0.01 * np.eye(nx),
        mu_v_init=np.zeros((ny, 1)),
        Sigma_v_init=0.01 * np.eye(ny),
        f=unicycle_dynamics,
        F_jac=unicycle_jacobian,
        h=observation_function,
        H_jac=observation_jacobian,
        max_iters=50,
        tol=1e-4,
        estimate_means=False,                 # Only estimate covariances, keep true means
        estimate_x0=False,                    # Only estimate covariances, keep true x0_mean
        cov_structure="diag",                 # strong recommendation for identifiability
        reg=1e-6,
        verbose=True
    )

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
        # This will be used by plot0_with_FW.py for visualization
        if robust_val in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:  # Save for all theta values
            # Ensure directory exists
            detailed_results_dir = "./results/EKF_comparison_with_FW/"
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
    results_path = "./results/EKF_comparison_with_FW/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    save_data(os.path.join(results_path, f'all_results_{dist}.pkl'), all_results)
    save_data(os.path.join(results_path, f'optimal_results_{dist}.pkl'), optimal_results)
    
    print(f"\nEKF vs DR-EKF comparison with Frank-Wolfe completed. Results saved to {results_path}")
    
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
    parser.add_argument('--num_exp', default=10, type=int,
                        help="Number of independent experiments")
    parser.add_argument('--T_total', default=10.0, type=float,
                        help="Total simulation time")
    parser.add_argument('--num_samples', default=10, type=int,
                        help="Number of samples for nominal parameter estimation")
    args = parser.parse_args()
    main(args.dist, args.num_sim, args.num_exp, args.T_total, args.num_samples)