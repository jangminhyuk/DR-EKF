#!/usr/bin/env python3
"""
DR_EKF_TAC.py implements a distributionally robust Extended Kalman filter (DR-EKF) for state estimation
in nonlinear systems. This is the TAC finite horizon version where SDP must be solved online due to linearization.
"""

import numpy as np
import cvxpy as cp
from .base_filter import BaseFilter

class DR_EKF_TAC(BaseFilter):
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_Sigma_v,
                 nonlinear_dynamics=None,
                 dynamics_jacobian=None,
                 observation_function=None,
                 observation_jacobian=None,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 x0_scale=None, w_scale=None, v_scale=None,
                 theta_x=None, theta_v=None, theta_w=None,
                 input_lower_bound=None, input_upper_bound=None):
        super().__init__(T, dist, noise_dist, system_data, B,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale, None,
                        input_lower_bound, input_upper_bound)
        
        # Store nonlinear dynamics and jacobians (required for DR-EKF)
        if not all([nonlinear_dynamics, dynamics_jacobian, observation_function, observation_jacobian]):
            raise ValueError("DR-EKF requires all nonlinear functions: dynamics, dynamics_jacobian, observation, observation_jacobian")
        
        self.f = nonlinear_dynamics
        self.F_jacobian = dynamics_jacobian
        self.h = observation_function
        self.C_jacobian = observation_jacobian
        
        self.theta_x = theta_x
        self.theta_v = theta_v
        self.theta_w = theta_w
        
        # Initialize posterior covariance for online computation
        self._P = None
        
        # Pre-created SDP problems and parameter references for efficiency
        self._sdp_problem_initial = None
        self._sdp_params_initial = None
        self._sdp_problem_regular = None 
        self._sdp_params_regular = None
        self._warm_start_vars_initial = None
        self._warm_start_vars_regular = None


    def _create_and_cache_sdp_initial(self):
        """Create initial SDP structure once and cache for reuse."""
        if self._sdp_problem_initial is not None:
            return self._sdp_problem_initial, self._sdp_params_initial
            
        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        X_pred = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred')
        Sigma_v = cp.Variable((self.ny, self.ny), symmetric=True, name='Sigma_v')
        Y = cp.Variable((self.nx, self.nx), name='Y')
        Z = cp.Variable((self.ny, self.ny), name='Z')
        
        X_pred_hat = cp.Parameter((self.nx, self.nx), name='X_pred_hat')
        theta_x = cp.Parameter(nonneg=True, name='theta_x')
        Sigma_v_hat = cp.Parameter((self.ny, self.ny), name='Sigma_v_hat')
        theta_v = cp.Parameter(nonneg=True, name='theta_v')
        lam_min_x_nom = cp.Parameter(nonneg=True, name='lam_min_x')
        lam_min_v_nom = cp.Parameter(nonneg=True, name='lam_min_v')
        # Linearized observation matrix as parameter
        C_t = cp.Parameter((self.ny, self.nx), name='C_t')
        
        obj = cp.Maximize(cp.trace(X))
        constraints = [
            cp.bmat([[X_pred - X, X_pred @ C_t.T],
                     [C_t @ X_pred, C_t @ X_pred @ C_t.T + Sigma_v]
                    ]) >> 0,
            cp.trace(X_pred_hat + X_pred - 2*Y) <= theta_x**2,
            cp.bmat([[X_pred_hat, Y],
                     [Y.T, X_pred]
                    ]) >> 0,
            cp.trace(Sigma_v_hat + Sigma_v - 2*Z) <= theta_v**2,
            cp.bmat([[Sigma_v_hat, Z],
                     [Z.T, Sigma_v]
                    ]) >> 0,
            X >> 0,
            #X_pred >> 0,
            #Sigma_v >> 0
            X_pred >> lam_min_x_nom* np.eye(self.nx),
            Sigma_v >> lam_min_v_nom * np.eye(self.ny)
        ]
        
        prob = cp.Problem(obj, constraints)
        
        # Cache problem and parameter references
        self._sdp_problem_initial = prob
        self._sdp_params_initial = {
            'X_pred_hat': X_pred_hat,
            'theta_x': theta_x,
            'Sigma_v_hat': Sigma_v_hat,
            'theta_v': theta_v,
            'lam_min_x_nom': lam_min_x_nom,
            'lam_min_v_nom': lam_min_v_nom,
            'C_t': C_t
        }
        
        # Store variable references for warm starting
        self._warm_start_vars_initial = {
            'X': X,
            'X_pred': X_pred,
            'Sigma_v': Sigma_v,
            'Y': Y,
            'Z': Z
        }
        
        return prob, self._sdp_params_initial

    def solve_sdp_online_initial(self, X_pred_hat, C_t):
        """Solve SDP online for t=0 with linearized observation matrix C_t."""
        # Use cached SDP problem structure
        prob, params = self._create_and_cache_sdp_initial()
        
        # Update parameter values
        params['X_pred_hat'].value = X_pred_hat
        params['theta_x'].value = self.theta_x
        params['Sigma_v_hat'].value = self.nominal_Sigma_v
        params['theta_v'].value = self.theta_v
        params['lam_min_x_nom'].value = np.min(np.real(np.linalg.eigvals(X_pred_hat)))
        params['lam_min_v_nom'].value = np.min(np.real(np.linalg.eigvals(self.nominal_Sigma_v)))
        params['C_t'].value = C_t
        
        # Warm start with previous solution if available
        if self._warm_start_vars_initial is not None:
            for var_name, var in self._warm_start_vars_initial.items():
                if var.value is not None:
                    var.value = var.value
        
        prob.solve(solver=cp.MOSEK, warm_start=True)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(f'DR-EKF TAC SDP initial problem: {prob.status}')
            return None, None, None
        
        sol = prob.variables()
        worst_case_Xpost = sol[0].value
        worst_case_Xprior = sol[1].value  
        worst_case_Sigma_v = sol[2].value
        
        return worst_case_Sigma_v, worst_case_Xprior, worst_case_Xpost
    
    def _create_and_cache_sdp_regular(self):
        """Create regular SDP structure for t>0 once and cache for reuse."""
        if self._sdp_problem_regular is not None:
            return self._sdp_problem_regular, self._sdp_params_regular
            
        # Variables
        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        X_pred = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred')
        Sigma_v = cp.Variable((self.ny, self.ny), symmetric=True, name='Sigma_v')
        Sigma_w = cp.Variable((self.nx, self.nx), symmetric=True, name='Sigma_w')
        Y = cp.Variable((self.nx, self.nx), name='Y')
        Z = cp.Variable((self.ny, self.ny), name='Z')
        W = cp.Variable((self.nx, self.nx), name='W')
        
        # Parameters
        Sigma_w_hat = cp.Parameter((self.nx, self.nx), name='Sigma_w_hat') 
        theta_w = cp.Parameter(nonneg=True, name='theta_w')
        Sigma_v_hat = cp.Parameter((self.ny, self.ny), name='Sigma_v_hat')
        theta_v = cp.Parameter(nonneg=True, name='theta_v')
        X_post_prev = cp.Parameter((self.nx, self.nx), name='X_post_prev')
        lam_min_v_nom = cp.Parameter(nonneg=True, name='lam_min_v')
        lam_min_w_nom = cp.Parameter(nonneg=True, name='lam_min_w')
        # Linearized matrices as parameters  
        A_t = cp.Parameter((self.nx, self.nx), name='A_t')
        C_t = cp.Parameter((self.ny, self.nx), name='C_t')
        
        # Objective: maximize trace(X)
        obj = cp.Maximize(cp.trace(X))
        
        constraints = [
            cp.bmat([[X_pred - X, X_pred @ C_t.T],
                     [C_t @ X_pred, C_t @ X_pred @ C_t.T + Sigma_v]
                    ]) >> 0,
            cp.trace(Sigma_w + Sigma_w_hat - 2*W) <= theta_w**2,
            cp.bmat([[Sigma_w_hat, W],
                     [W.T, Sigma_w]
                    ]) >> 0,
            cp.trace(Sigma_v_hat + Sigma_v - 2*Z) <= theta_v**2,
            cp.bmat([[Sigma_v_hat, Z],
                     [Z.T, Sigma_v]
                    ]) >> 0,
            X_pred == A_t @ X_post_prev @ A_t.T + Sigma_w,
            X >> 0,
            X_pred >> 0,
            #Sigma_v >> 0,
            #Sigma_w >> 0
            Sigma_v >> lam_min_v_nom * np.eye(self.ny),
            Sigma_w >> lam_min_w_nom * np.eye(self.nx)
        ]
        
        prob = cp.Problem(obj, constraints)
        
        # Cache problem and parameter references
        self._sdp_problem_regular = prob
        self._sdp_params_regular = {
            'Sigma_w_hat': Sigma_w_hat,
            'theta_w': theta_w,
            'Sigma_v_hat': Sigma_v_hat,
            'theta_v': theta_v,
            'X_post_prev': X_post_prev,
            'lam_min_v_nom': lam_min_v_nom,
            'lam_min_w_nom': lam_min_w_nom,
            'A_t': A_t,
            'C_t': C_t
        }
        
        # Store variable references for warm starting
        self._warm_start_vars_regular = {
            'X': X,
            'X_pred': X_pred,
            'Sigma_v': Sigma_v,
            'Sigma_w': Sigma_w,
            'Y': Y,
            'Z': Z,
            'W': W
        }
        
        return prob, self._sdp_params_regular

    def solve_sdp_online(self, X_post_prev, A_t, C_t):
        """Solve SDP online for t>0 with linearized matrices A_t, C_t."""
        # Use cached SDP problem structure
        prob, params = self._create_and_cache_sdp_regular()
        
        # Update parameter values
        params['Sigma_w_hat'].value = self.nominal_Sigma_w
        params['theta_w'].value = self.theta_w
        params['Sigma_v_hat'].value = self.nominal_Sigma_v
        params['theta_v'].value = self.theta_v
        params['X_post_prev'].value = X_post_prev
        params['lam_min_v_nom'].value = np.min(np.real(np.linalg.eigvals(self.nominal_Sigma_v)))
        params['lam_min_w_nom'].value = np.min(np.real(np.linalg.eigvals(self.nominal_Sigma_w)))
        params['A_t'].value = A_t
        params['C_t'].value = C_t
        
        # Warm start with previous solution if available
        if self._warm_start_vars_regular is not None:
            for var_name, var in self._warm_start_vars_regular.items():
                if var.value is not None:
                    var.value = var.value
        
        prob.solve(solver=cp.MOSEK, warm_start=True)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(f'DR-EKF TAC SDP problem: {prob.status}')
            return None, None, None, None
        
        sol = prob.variables()
        worst_case_Xpost = sol[0].value
        worst_case_Xprior = sol[1].value
        worst_case_Sigma_v = sol[2].value
        worst_case_Sigma_w = sol[3].value
        
        return worst_case_Sigma_v, worst_case_Sigma_w, worst_case_Xprior, worst_case_Xpost
    

    # --- DR-EKF Update Step ---
    def DR_kalman_filter(self, v_mean_hat, x_prior, y, t, u_prev=None, x_post_prev=None):
        """DR-EKF TAC following theorem equations with online SDP solving.
        
        Measurement Update:
        x_post_t = x_prior_t + K_t^* (y_t - h(x_prior_t) - hat{v}_t)
        X_post = (I - K_t^* C_t) X_prior_opt
        """
        # Linearize observation at prior state: C_t = ∂h/∂x|_{x_prior}
        C_t = self.C_jacobian(x_prior)
        
        # Solve SDP online based on time step
        if t == 0:
            # Initial update: use nominal initial covariance
            X_prior_nom = self.nominal_x0_cov.copy()
            wc_Sigma_v, wc_Xprior, wc_Xpost = self.solve_sdp_online_initial(X_prior_nom, C_t)
        else:
            # Regular update: compute pseudo-nominal prior using linearized A_t
            if x_post_prev is not None and u_prev is not None:
                # A_t = ∂f/∂x|_{x_post_{t-1}}
                A_t = self.F_jacobian(x_post_prev, u_prev)
                # Solve online SDP with linearized matrices
                wc_Sigma_v, wc_Sigma_w, wc_Xprior, wc_Xpost = self.solve_sdp_online(self._P, A_t, C_t)
            else:
                raise RuntimeError(f"DR-EKF TAC requires previous state and control input for t > 0, got: "
                                 f"x_post_prev={x_post_prev is not None}, u_prev={u_prev is not None}")
        
        if wc_Sigma_v is None:
            raise RuntimeError(f"DR-EKF TAC SDP optimization failed at time step {t}. "
                             f"Check theta parameters (theta_x={self.theta_x}, theta_v={self.theta_v}, "
                             f"theta_w={self.theta_w}) and ensure they are feasible for the current problem instance.")
        
        # DR-EKF TAC update using theorem equations
        # K_t^* = X_prior_opt C_t^T (C_t X_prior_opt C_t^T + Sigma_v_opt)^{-1}
        S = C_t @ wc_Xprior @ C_t.T + wc_Sigma_v
        K_star = np.linalg.solve(S, (wc_Xprior @ C_t.T).T).T
        
        # x_post_t = x_prior_t + K_t^* (y_t - h(x_prior_t) - hat{v}_t)
        innovation = y - (self.h(x_prior) + v_mean_hat)
        x_post = x_prior + K_star @ innovation
        
        # X_post = (I - K_t^* C_t) X_prior_opt
        self._P = wc_Xpost  # Use optimized posterior covariance
        
        return x_post

    def _initial_update(self, x_est_init, y0):
        return self.DR_kalman_filter(self.nominal_mu_v, x_est_init, y0, 0, None, None)
    
    def _drkf_finite_update(self, x_prior, y, t, u_prev=None, x_post_prev=None):
        return self.DR_kalman_filter(self.nominal_mu_v, x_prior, y, t, u_prev, x_post_prev)
    
    def forward(self):
        return self._run_simulation_loop(self._drkf_finite_update)
    def forward_track(self, desired_trajectory):
        return self._run_simulation_loop(self._drkf_finite_update, desired_trajectory)
    
    def forward_track_MPC(self, desired_trajectory):
        return self._run_simulation_loop_MPC(self._drkf_finite_update, desired_trajectory)
    
    def update_step(self, x_est_prev, y_curr, t, u_prev):
        """Common interface for filter update step.
        
        Args:
            x_est_prev: Previous state estimate
            y_curr: Current measurement
            t: Time step
            u_prev: Previous control input
            
        Returns:
            x_est_new: Updated state estimate
        """
        # DR-EKF TAC state prediction: x_prior_{t+1} = f(x_post_t, u_t) + hat{w}_t
        x_pred = self.f(x_est_prev, u_prev) + self.nominal_mu_w
        
        # DR-EKF TAC measurement update (uses online solved SDP)
        return self._drkf_finite_update(x_pred, y_curr, t, u_prev, x_est_prev)
