#!/usr/bin/env python3
3
# -*- coding: utf-8 -*-
"""
DRKF_neurips.py implements a distributionally robust Kalman filter (DRKF) for state estimation
in a closed-loop LQR experiment. This is the neurips2018 version with simplified SDP formulation.
"""

import numpy as np
import cvxpy as cp
from .base_filter import BaseFilter

class DRKF_neurips(BaseFilter):
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_Sigma_v,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 x0_scale=None, w_scale=None, v_scale=None,
                 theta=None, shared_noise_sequences=None,
                 input_lower_bound=None, input_upper_bound=None):
        """
        Parameters:
          T             : Horizon length.
          dist, noise_dist : Distribution types ('normal' or 'quadratic').
          system_data   : Tuple (A, C).
          B             : Control input matrix.
          
          The following parameters are provided in two sets:
             (i) True parameters (used to simulate the system):
                 - true_x0_mean, true_x0_cov: initial state distribution.
                 - true_mu_w, true_Sigma_w: process noise.
                 - true_mu_v, true_Sigma_v: measurement noise.
             (ii) Nominal parameters (obtained via EM, used in filtering):
                 - Use known means (nominal_x0_mean, nominal_mu_w, nominal_mu_v) and
                   EM–estimated covariances (nominal_x0_cov, nominal_Sigma_w, nominal_Sigma_v).
          x0_max, x0_min, etc.: Bounds for non–normal distributions.
          theta: DRKF robustness parameter.
          shared_noise_sequences: Pre-generated noise sequences for consistent experiments.
        """
        super().__init__(T, dist, noise_dist, system_data, B,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale, shared_noise_sequences,
                        input_lower_bound, input_upper_bound)
        
        self.theta = theta
        
        # Allocate arrays for worst-case covariance matrices
        self.S_xx = np.zeros((T+1, self.nx, self.nx))
        self.S_xy = np.zeros((T+1, self.nx, self.ny))
        self.S_yy = np.zeros((T+1, self.ny, self.ny))
        
        self.V = np.zeros((T+1, self.nx, self.nx))
        
        self.solve_DRSDP_offline()

    # --- Solve DR-SDP to obtain worst-case covariance matrices and DR Kalman gain for each time step (offline)---
    def solve_DRSDP_offline(self):
        self.S_xx[0], self.S_xy[0], self.S_yy[0], self.V[0] = self.solve_initial_sdp()
        for t in range(1, self.T + 1):
            self.S_xx[t], self.S_xy[t], self.S_yy[t], self.V[t] = self.solve_sdp(self.V[t-1])
            

    # --- SDP Formulation and Solver for Worst-Case Measurement Covariance ---
    def create_DR_sdp(self, Sigma_val):
        
        # Construct the SDP problem.
        # Variables
        V = cp.Variable((self.nx, self.nx), symmetric=True, name='V')
        S_xx = cp.Variable((self.nx, self.nx), symmetric=True, name='S_xx')
        S_xy = cp.Variable((self.nx, self.ny), name='S_xy')
        S_yy = cp.Variable((self.ny, self.ny), symmetric=True, name='S_yy')
        S = cp.Variable((self.nx + self.ny, self.nx + self.ny), symmetric=True, name='S')
        Y = cp.Variable((self.nx + self.ny, self.nx + self.ny), name='Y')
        
        # Parameters
        Sigma = cp.Parameter((self.nx + self.ny, self.nx + self.ny), name='Sigma')
        theta = cp.Parameter(nonneg=True, name='theta')
        
        # Compute minimum eigenvalue using the actual Sigma value
        lambda_min_val = np.linalg.eigvalsh(Sigma_val).min()
        
        # Objective: maximize trace(V)
        obj = cp.Maximize(cp.trace(V))
        
        constraints = [
            cp.bmat([[S_xx - V, S_xy],
                     [S_xy.T, S_yy]
                    ]) >> 0,
            S == cp.bmat([[S_xx, S_xy],
                          [S_xy.T, S_yy]]),
            cp.trace(S + Sigma - 2*Y) <= theta**2,
            cp.bmat([[Sigma, Y],
                     [Y.T, S]
                    ]) >> 0,
            S_xx >> 0,
            S_yy >> 0,
            S >> lambda_min_val * np.eye(self.nx + self.ny)
        ]
        
        prob = cp.Problem(obj, constraints)
        return prob
    
    def solve_sdp(self, V_prev):
        X_pred = self.A @ V_prev @ self.A.T + self.nominal_Sigma_w
        
        Sigma = np.block([[X_pred, X_pred @ self.C.T],
                          [self.C @ X_pred, self.C @ X_pred @ self.C.T + self.nominal_Sigma_v]])
        
        prob = self.create_DR_sdp(Sigma)
        params = prob.parameters() 
        
        params[0].value = Sigma
        params[1].value = self.theta
        
        prob.solve(solver=cp.MOSEK)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(prob.status, 'finite DRKF SDP NeurIPS formulation')
        
            
        sol = prob.variables()
        
        V = sol[0].value
        S_xx = sol[1].value
        S_xy = sol[2].value
        S_yy = sol[3].value
        
        
        return S_xx, S_xy, S_yy, V
    
    def solve_initial_sdp(self):
        Sigma = np.block([[self.nominal_x0_cov, self.nominal_x0_cov @ self.C.T],
                          [self.C @ self.nominal_x0_cov, self.C @ self.nominal_x0_cov @ self.C.T + self.nominal_Sigma_v]])
        
        prob = self.create_DR_sdp(Sigma)
        params = prob.parameters() 
        
        params[0].value = Sigma
        params[1].value = self.theta
        
        prob.solve(solver=cp.MOSEK)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(prob.status, 'finite DRKF SDP NeurIPS formulation t=0')
        
            
        sol = prob.variables()
        
        V = sol[0].value
        S_xx = sol[1].value
        S_xy = sol[2].value
        S_yy = sol[3].value
        
        
        return S_xx, S_xy, S_yy, V

    # --- DR-KF Update Step ---
    def DR_kalman_filter(self, v_mean_hat, x_pred, y, t):
        y_pred = self.C @ x_pred + v_mean_hat
        innovation = y - y_pred
        K = self.S_xy[t] @ np.linalg.inv(self.S_yy[t])
        x_new = x_pred + K @ innovation
        return x_new

    def _initial_update(self, x_est_init, y0):
        return self.DR_kalman_filter(self.nominal_mu_v, x_est_init, y0, 0)
    
    def _drkf_finite_neurips_update(self, x_pred, y, t):
        return self.DR_kalman_filter(self.nominal_mu_v, x_pred, y, t)
    
    def forward(self):
        return self._run_simulation_loop(self._drkf_finite_neurips_update)
    
    def forward_track(self, desired_trajectory):
        return self._run_simulation_loop(self._drkf_finite_neurips_update, desired_trajectory)
    
    def forward_track_MPC(self, desired_trajectory):
        return self._run_simulation_loop_MPC(self._drkf_finite_neurips_update, desired_trajectory)
