#!/usr/bin/env python3
"""
DR_EKF_joint.py implements a distributionally robust Extended Kalman filter (DR-EKF) with
joint ambiguity sets (joint-ball SDP). Instead of separate Wasserstein/Gelbrich balls for
process and measurement noise, a single joint Bures constraint is used over the stacked
noise vector epsilon = (w, v) (or (x0, v) at t=0).
"""

import numpy as np
import cvxpy as cp
from .base_filter import BaseFilter

class DR_EKF_joint(BaseFilter):
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
                 theta_eps0=None, theta_eps=None,
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

        # Joint radii received directly
        self.theta_eps0 = theta_eps0
        self.theta_eps = theta_eps


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
        """Create initial joint-ball SDP (SDP 5) structure once and cache for reuse.

        Joint ambiguity set at t=0: epsilon_0 = (x_0, v_0) with a single Bures
        constraint on the stacked covariance Sigma_eps0 = blkdiag(Sigma_x0, Sigma_v).
        """
        if self._sdp_problem_initial is not None:
            return self._sdp_problem_initial, self._sdp_params_initial

        n_eps = self.nx + self.ny  # joint dimension

        # Variables
        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        X_pred = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred')
        Sigma_v = cp.Variable((self.ny, self.ny), symmetric=True, name='Sigma_v')
        Sigma_xv = cp.Variable((self.nx, self.ny), name='Sigma_xv')
        Y_joint = cp.Variable((n_eps, n_eps), name='Y_joint')

        # Parameters
        X_pred_hat = cp.Parameter((self.nx, self.nx), name='X_pred_hat')
        Sigma_v_hat = cp.Parameter((self.ny, self.ny), name='Sigma_v_hat')
        theta_eps0 = cp.Parameter(nonneg=True, name='theta_eps0')
        lam_min_eps_nom = cp.Parameter(nonneg=True, name='lam_min_eps')
        lam_min_v_nom = cp.Parameter(nonneg=True, name='lam_min_v')
        C_t = cp.Parameter((self.ny, self.nx), name='C_t')

        # Block covariances
        # Sigma_eps0 = [[X_pred, Sigma_xv], [Sigma_xv.T, Sigma_v]]
        Sigma_eps0 = cp.bmat([[X_pred, Sigma_xv],
                              [Sigma_xv.T, Sigma_v]])
        # Sigma_hat_eps0 = blkdiag(X_pred_hat, Sigma_v_hat)
        # This can also be improved to consider \Sigma_xv,0. 
        # For the current setting, I just let off-diagonal block for nominal as zero.
        Sigma_hat_eps0 = cp.bmat([[X_pred_hat, np.zeros((self.nx, self.ny))],
                                  [np.zeros((self.ny, self.nx)), Sigma_v_hat]])

        # Gain-related expressions
        T0 = X_pred @ C_t.T + Sigma_xv
        S0 = C_t @ X_pred @ C_t.T + Sigma_v + C_t @ Sigma_xv + Sigma_xv.T @ C_t.T

        obj = cp.Maximize(cp.trace(X))
        constraints = [
            # Schur complement PSD for posterior covariance
            cp.bmat([[X_pred - X, T0],
                     [T0.T, S0]
                    ]) >> 0,
            # Joint Bures constraint
            cp.bmat([[Sigma_hat_eps0, Y_joint],
                     [Y_joint.T, Sigma_eps0]
                    ]) >> 0,
            cp.trace(Sigma_eps0 + Sigma_hat_eps0 - 2*Y_joint) <= theta_eps0**2,
            # PSD and lower bound constraints
            X >> 0,
            Sigma_eps0 >> lam_min_eps_nom * np.eye(n_eps),
            Sigma_v >> lam_min_v_nom * np.eye(self.ny)
        ]

        prob = cp.Problem(obj, constraints)

        # Cache problem and parameter references
        self._sdp_problem_initial = prob
        self._sdp_params_initial = {
            'X_pred_hat': X_pred_hat,
            'Sigma_v_hat': Sigma_v_hat,
            'theta_eps0': theta_eps0,
            'lam_min_eps_nom': lam_min_eps_nom,
            'lam_min_v_nom': lam_min_v_nom,
            'C_t': C_t
        }

        # Store variable references for warm starting (avoids prob.variables() ordering)
        self._warm_start_vars_initial = {
            'X': X,
            'X_pred': X_pred,
            'Sigma_v': Sigma_v,
            'Sigma_xv': Sigma_xv,
            'Y_joint': Y_joint
        }

        return prob, self._sdp_params_initial

    def solve_sdp_online_initial(self, X_pred_hat, C_t):
        """Solve joint-ball SDP (5) online for t=0 with linearized observation matrix C_t.

        Returns:
            Sigma_v_star, Sigma_xv_star, Xprior_star, Xpost_star
        """
        prob, params = self._create_and_cache_sdp_initial()

        # Update parameter values
        params['X_pred_hat'].value = X_pred_hat
        params['Sigma_v_hat'].value = self.nominal_Sigma_v
        params['theta_eps0'].value = self.theta_eps0
        Sigma_hat_eps0_val = np.block([[X_pred_hat, np.zeros((self.nx, self.ny))],
                                       [np.zeros((self.ny, self.nx)), self.nominal_Sigma_v]])
        params['lam_min_eps_nom'].value = np.min(np.real(np.linalg.eigvals(Sigma_hat_eps0_val)))
        params['lam_min_v_nom'].value = np.min(np.real(np.linalg.eigvals(self.nominal_Sigma_v)))
        params['C_t'].value = C_t

        # Warm start with previous solution if available
        if self._warm_start_vars_initial is not None:
            for var_name, var in self._warm_start_vars_initial.items():
                if var.value is not None:
                    var.value = var.value

        prob.solve(solver=cp.MOSEK, warm_start=True)

        if prob.status in ["infeasible", "unbounded"]:
            print(f'DR-EKF joint SDP initial problem: {prob.status}')
            return None, None, None, None

        # Read solutions from stored variable references (not prob.variables())
        vars_ = self._warm_start_vars_initial
        worst_case_Xpost = vars_['X'].value
        worst_case_Xprior = vars_['X_pred'].value
        worst_case_Sigma_v = vars_['Sigma_v'].value
        worst_case_Sigma_xv = vars_['Sigma_xv'].value

        return worst_case_Sigma_v, worst_case_Sigma_xv, worst_case_Xprior, worst_case_Xpost

    def _create_and_cache_sdp_regular(self):
        """Create regular joint-ball SDP (SDP 7) structure for t>0 once and cache for reuse.

        Joint ambiguity set at t>=1: epsilon_t = (w_{t-1}, v_t) with a single Bures
        constraint on the stacked covariance Sigma_eps = blkdiag(Sigma_w, Sigma_v).
        """
        if self._sdp_problem_regular is not None:
            return self._sdp_problem_regular, self._sdp_params_regular

        n_eps = self.nx + self.ny  # joint dimension

        # Variables
        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        X_pred = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred')
        Sigma_v = cp.Variable((self.ny, self.ny), symmetric=True, name='Sigma_v')
        Sigma_w = cp.Variable((self.nx, self.nx), symmetric=True, name='Sigma_w')
        Sigma_wv = cp.Variable((self.nx, self.ny), name='Sigma_wv')
        Y_joint = cp.Variable((n_eps, n_eps), name='Y_joint')

        # Parameters
        Sigma_w_hat = cp.Parameter((self.nx, self.nx), name='Sigma_w_hat')
        Sigma_v_hat = cp.Parameter((self.ny, self.ny), name='Sigma_v_hat')
        theta_eps = cp.Parameter(nonneg=True, name='theta_eps')
        X_post_prev = cp.Parameter((self.nx, self.nx), name='X_post_prev')
        lam_min_eps_nom = cp.Parameter(nonneg=True, name='lam_min_eps')
        lam_min_v_nom = cp.Parameter(nonneg=True, name='lam_min_v')
        lam_min_w_nom = cp.Parameter(nonneg=True, name='lam_min_w')
        A_t = cp.Parameter((self.nx, self.nx), name='A_t')
        C_t = cp.Parameter((self.ny, self.nx), name='C_t')

        # Block covariances
        # Sigma_eps = [[Sigma_w, Sigma_wv], [Sigma_wv.T, Sigma_v]]
        Sigma_eps = cp.bmat([[Sigma_w, Sigma_wv],
                             [Sigma_wv.T, Sigma_v]])
        # Sigma_hat_eps = blkdiag(Sigma_w_hat, Sigma_v_hat)
        # The nominal Sigma_hat_eps can also be improved!! We can now consider the \Sigma_wv term!!!
        # For now, I just let off diagonal block as zero, but we can definitely improve this.
        Sigma_hat_eps = cp.bmat([[Sigma_w_hat, np.zeros((self.nx, self.ny))],
                                 [np.zeros((self.ny, self.nx)), Sigma_v_hat]])

        # Gain-related expressions
        T = X_pred @ C_t.T + Sigma_wv
        S = C_t @ X_pred @ C_t.T + Sigma_v + C_t @ Sigma_wv + Sigma_wv.T @ C_t.T

        # Objective: maximize trace(X)
        obj = cp.Maximize(cp.trace(X))

        constraints = [
            # Schur complement PSD for posterior covariance
            cp.bmat([[X_pred - X, T],
                     [T.T, S]
                    ]) >> 0,
            # Prior covariance constraint
            X_pred == A_t @ X_post_prev @ A_t.T + Sigma_w,
            # Joint Bures constraint
            cp.bmat([[Sigma_hat_eps, Y_joint],
                     [Y_joint.T, Sigma_eps]
                    ]) >> 0,
            cp.trace(Sigma_eps + Sigma_hat_eps - 2*Y_joint) <= theta_eps**2,
            # PSD and lower bound constraints
            X >> 0,
            Sigma_eps >> lam_min_eps_nom * np.eye(n_eps),
            #Sigma_v >> lam_min_v_nom * np.eye(self.ny),
            #Sigma_w >> lam_min_w_nom * np.eye(self.nx)
        ]

        prob = cp.Problem(obj, constraints)

        # Cache problem and parameter references
        self._sdp_problem_regular = prob
        self._sdp_params_regular = {
            'Sigma_w_hat': Sigma_w_hat,
            'Sigma_v_hat': Sigma_v_hat,
            'theta_eps': theta_eps,
            'X_post_prev': X_post_prev,
            'lam_min_eps_nom': lam_min_eps_nom,
            'lam_min_v_nom': lam_min_v_nom,
            'lam_min_w_nom': lam_min_w_nom,
            'A_t': A_t,
            'C_t': C_t
        }

        # Store variable references for warm starting (avoids prob.variables() ordering)
        self._warm_start_vars_regular = {
            'X': X,
            'X_pred': X_pred,
            'Sigma_v': Sigma_v,
            'Sigma_w': Sigma_w,
            'Sigma_wv': Sigma_wv,
            'Y_joint': Y_joint
        }

        return prob, self._sdp_params_regular

    def solve_sdp_online(self, X_post_prev, A_t, C_t):
        """Solve joint-ball SDP (7) online for t>0 with linearized matrices A_t, C_t.

        Returns:
            Sigma_v_star, Sigma_w_star, Sigma_wv_star, Xprior_star, Xpost_star
        """
        prob, params = self._create_and_cache_sdp_regular()

        # Update parameter values
        params['Sigma_w_hat'].value = self.nominal_Sigma_w
        params['Sigma_v_hat'].value = self.nominal_Sigma_v
        params['theta_eps'].value = self.theta_eps
        params['X_post_prev'].value = X_post_prev
        Sigma_hat_eps_val = np.block([[self.nominal_Sigma_w, np.zeros((self.nx, self.ny))],
                                      [np.zeros((self.ny, self.nx)), self.nominal_Sigma_v]])
        params['lam_min_eps_nom'].value = np.min(np.real(np.linalg.eigvals(Sigma_hat_eps_val)))
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
            print(f'DR-EKF joint SDP problem: {prob.status}')
            return None, None, None, None, None

        # Read solutions from stored variable references (not prob.variables())
        vars_ = self._warm_start_vars_regular
        worst_case_Xpost = vars_['X'].value
        worst_case_Xprior = vars_['X_pred'].value
        worst_case_Sigma_v = vars_['Sigma_v'].value
        worst_case_Sigma_w = vars_['Sigma_w'].value
        worst_case_Sigma_wv = vars_['Sigma_wv'].value

        return worst_case_Sigma_v, worst_case_Sigma_w, worst_case_Sigma_wv, worst_case_Xprior, worst_case_Xpost


    # --- DR-EKF Update Step ---
    def DR_kalman_filter(self, v_mean_hat, x_prior, y, t, u_prev=None, x_post_prev=None):
        """DR-EKF with joint ambiguity / joint-ball SDP, online SDP solving.

        Measurement Update:
        T = X_prior C^T + Sigma_cross
        S = C X_prior C^T + Sigma_v + C Sigma_cross + Sigma_cross^T C^T
        K = T S^{-1}
        x_post_t = x_prior_t + K (y_t - h(x_prior_t) - hat{v}_t)
        P_post = Xpost_star (SDP output)
        """
        # Linearize observation at prior state: C_t = dh/dx|_{x_prior}
        C_t = self.C_jacobian(x_prior)

        # Solve SDP online based on time step
        if t == 0:
            # Initial update: use nominal initial covariance
            X_prior_nom = self.nominal_x0_cov.copy()
            result = self.solve_sdp_online_initial(X_prior_nom, C_t)
            wc_Sigma_v, wc_Sigma_xv, wc_Xprior, wc_Xpost = result
            wc_Sigma_cross = wc_Sigma_xv  # cross-covariance for gain
        else:
            # Regular update: compute using linearized A_t
            if x_post_prev is not None and u_prev is not None:
                # A_t = df/dx|_{x_post_{t-1}}
                A_t = self.F_jacobian(x_post_prev, u_prev)
                result = self.solve_sdp_online(self._P, A_t, C_t)
                wc_Sigma_v, wc_Sigma_w, wc_Sigma_wv, wc_Xprior, wc_Xpost = result
                wc_Sigma_cross = wc_Sigma_wv  # cross-covariance for gain
            else:
                raise RuntimeError(f"DR-EKF joint requires previous state and control input for t > 0, got: "
                                 f"x_post_prev={x_post_prev is not None}, u_prev={u_prev is not None}")

        if wc_Sigma_v is None:
            raise RuntimeError(f"DR-EKF joint SDP optimization failed at time step {t}. "
                             f"Check theta parameters (theta_eps0={self.theta_eps0}, theta_eps={self.theta_eps}) "
                             f"and ensure they are feasible for the current problem instance.")

        # Joint-ball DR-EKF gain using cross-covariance
        # T = X_prior C^T + Sigma_cross
        T = wc_Xprior @ C_t.T + wc_Sigma_cross
        # S = C X_prior C^T + Sigma_v + C Sigma_cross + Sigma_cross^T C^T
        S = C_t @ wc_Xprior @ C_t.T + wc_Sigma_v + C_t @ wc_Sigma_cross + wc_Sigma_cross.T @ C_t.T
        # K = T S^{-1}  (via solve for numerical stability)
        K_star = np.linalg.solve(S, T.T).T

        # x_post_t = x_prior_t + K (y_t - h(x_prior_t) - hat{v}_t)
        innovation = y - (self.h(x_prior) + v_mean_hat)
        x_post = x_prior + K_star @ innovation

        # Posterior covariance from SDP output
        self._P = wc_Xpost

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
        # DR-EKF joint state prediction: x_prior_{t+1} = f(x_post_t, u_t) + hat{w}_t
        x_pred = self.f(x_est_prev, u_prev) + self.nominal_mu_w

        # DR-EKF joint measurement update (uses online solved SDP)
        return self._drkf_finite_update(x_pred, y_curr, t, u_prev, x_est_prev)
