#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRKF_ours_finite_CDC.py implements a distributionally robust Kalman filter (DRKF) for state estimation
in a closed-loop LQR experiment. This is the CDC version with simplified SDP formulation.
"""

import numpy as np
import cvxpy as cp
from .base_filter import BaseFilter

class DR_EKF_CDC(BaseFilter):
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
                 theta_x=None, theta_v=None, shared_noise_sequences=None,
                 input_lower_bound=None, input_upper_bound=None,
                 solver="cvxpy",
                 fw_beta_minus1=1.0,
                 fw_tau=2.0,
                 fw_zeta=2.0,
                 fw_delta=0.05,
                 fw_max_iters=100,
                 fw_gap_tol=1e-3,
                 fw_bisect_max_iters=30,
                 fw_bisect_tol=1e-6):
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
          theta_x, theta_v: DRKF parameters.
          shared_noise_sequences: Pre-generated noise sequences for consistent experiments.
          
          Frank-Wolfe solver parameters:
          solver: "cvxpy" (default) or "fw" for Frank-Wolfe solver
          fw_beta_minus1: β_{-1} parameter for Frank-Wolfe
          fw_tau: τ parameter (>1) for Frank-Wolfe step size adaptation
          fw_zeta: ζ parameter (>1) for Frank-Wolfe step size adaptation  
          fw_delta: δ parameter (0,1) for oracle precision in Algorithm 2
          fw_max_iters: Maximum iterations for Frank-Wolfe
          fw_gap_tol: Gap tolerance for Frank-Wolfe convergence
          fw_bisect_max_iters: Maximum iterations for bisection oracle
          fw_bisect_tol: Tolerance for bisection oracle
        """
        super().__init__(T, dist, noise_dist, system_data, B,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale, shared_noise_sequences,
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
        
        # Frank-Wolfe solver parameters
        self.solver = solver
        self.fw_beta_minus1 = fw_beta_minus1
        self.fw_tau = fw_tau
        self.fw_zeta = fw_zeta
        self.fw_delta = fw_delta
        self.fw_max_iters = fw_max_iters
        self.fw_gap_tol = fw_gap_tol
        self.fw_bisect_max_iters = fw_bisect_max_iters
        self.fw_bisect_tol = fw_bisect_tol
        
        # Initialize posterior covariance for first step
        self._P = None
        
        # Pre-created SDP problem and parameter references for efficiency
        self._sdp_problem = None
        self._sdp_params = None
        self._warm_start_vars = None
        
        
        # Cache frequently used matrices for performance
        self._I_nx = None  # Identity matrix nx x nx
        self._I_ny = None  # Identity matrix ny x ny
        
    
    def compute_predicted_covariance(self, P_post, x_est, u):
        """Compute predicted covariance using linearized dynamics."""
        F_t = self.F_jacobian(x_est, u)
        P_pred = F_t @ P_post @ F_t.T + self.nominal_Sigma_w
        return P_pred
            
        

    # --- SDP Formulation and Solver for Worst-Case Measurement Covariance ---
    def _create_and_cache_sdp(self):
        """Create DR-SDP structure once and cache for reuse."""
        if self._sdp_problem is not None:
            return self._sdp_problem, self._sdp_params
            
        # Compute lambda_min for nominal measurement noise covariance (Sigma_v_hat)
        lambda_min_val = np.linalg.eigvalsh(self.nominal_Sigma_v).min()
        
        # Construct the SDP problem.
        # Variables
        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        X_pred = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred')
        Sigma_v = cp.Variable((self.ny, self.ny), symmetric=True, name='Sigma_v')
        Y = cp.Variable((self.nx, self.nx), name='Y')
        Z = cp.Variable((self.ny, self.ny), name='Z')
        
        # Parameters that change at each time step
        X_pred_hat = cp.Parameter((self.nx, self.nx), name='X_pred_hat')
        theta_x = cp.Parameter(nonneg=True, name='theta_x')
        Sigma_v_hat = cp.Parameter((self.ny, self.ny), name='Sigma_v_hat')
        theta_v = cp.Parameter(nonneg=True, name='theta_v')
        # Linearized observation matrix as parameter (changes each time step)
        C_t = cp.Parameter((self.ny, self.nx), name='C_t')
        
        # Objective: maximize trace(X)
        obj = cp.Maximize(cp.trace(X))
        
        # Constraints using Schur complements with parameterized C_t matrix
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
            X_pred >> 0,
            # Sigma_v is larger than lambda_min(Sigma_v_hat)*I
            #Sigma_v >> 0
            Sigma_v >> lambda_min_val * np.eye(self.ny)
        ]
        
        prob = cp.Problem(obj, constraints)
        
        # Cache problem and parameter references
        self._sdp_problem = prob
        self._sdp_params = {
            'X_pred_hat': X_pred_hat,
            'theta_x': theta_x, 
            'Sigma_v_hat': Sigma_v_hat,
            'theta_v': theta_v,
            'C_t': C_t
        }
        
        # Store variable references for warm starting
        self._warm_start_vars = {
            'X': X,
            'X_pred': X_pred,
            'Sigma_v': Sigma_v,
            'Y': Y,
            'Z': Z
        }
        
        return prob, self._sdp_params
    
    def solve_sdp_online_cvxpy(self, X_pred_hat, C_t):
        """Solve SDP online with current linearized observation matrix C_t."""
        # Use cached SDP problem structure
        prob, params = self._create_and_cache_sdp()
        
        # Update parameter values for current time step
        params['X_pred_hat'].value = X_pred_hat
        params['theta_x'].value = self.theta_x
        params['Sigma_v_hat'].value = self.nominal_Sigma_v
        params['theta_v'].value = self.theta_v
        params['C_t'].value = C_t
        
        # Warm start with previous solution if available
        if self._warm_start_vars is not None:
            for var_name, var in self._warm_start_vars.items():
                if var.value is not None:
                    var.value = var.value
        
        prob.solve(solver=cp.MOSEK, warm_start=True)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(prob.status, 'DR-EKF SDP CDC formulation')
            return None, None, None
            
        sol = prob.variables()
        
        worst_case_Xpost = sol[0].value
        worst_case_Xprior = sol[1].value
        worst_case_Sigma_v = sol[2].value
        
        return worst_case_Sigma_v, worst_case_Xprior, worst_case_Xpost

    def solve_sdp_online(self, X_pred_hat, C_t):
        """Solver router: dispatch to CVXPY or Frank-Wolfe solver."""
        if self.solver == "cvxpy":
            return self.solve_sdp_online_cvxpy(X_pred_hat, C_t)
        elif self.solver == "fw":
            return self.solve_sdp_online_fw(X_pred_hat, C_t)
        else:
            raise ValueError(f"Unknown solver: {self.solver}. Must be 'cvxpy' or 'fw'.")

    def solve_sdp_online_fw(self, X_pred_hat, C_t):
        """
        Implements Algorithm 1 (Fully Adaptive Frank-Wolfe) exactly.
        Returns (wc_Sigma_v, wc_Xprior, wc_Xpost) with same meaning/signature as CVXPY path.
        """
        # Mapping for THIS code:
        # FW iterate s_t = (X_pred, Sigma_v)
        # Nominals: Sigma_x_hat := X_pred_hat, Sigma_v_hat := self.nominal_Sigma_v
        # Radii: rho_x := self.theta_x, rho_v := self.theta_v
        
        # Alg1-Step0: initialize feasible point s0
        X_pred = X_pred_hat.copy()
        Sigma_v = self.nominal_Sigma_v.copy()
        # Note: these should already be symmetric
        beta_prev = self.fw_beta_minus1
        t = 0
        
        # Alg1-Step1: while stopping criterion not met
        while True:
            # Stopping criterion (implement both):
            # - stop if t >= self.fw_max_iters
            if t >= self.fw_max_iters:
                break
                
            # Alg1-Step2: solve oracle subproblem to find s_tilde = F(s_t)
            # Compute gradients D_x, D_v
            D_x, D_v = self._fw_gradients(X_pred, Sigma_v, C_t)
            
            # Call oracle (Algorithm 2) twice:
            L_x = self._oracle_bisection(Sigma_hat=X_pred_hat, rho=self.theta_x, 
                                       Sigma_ref=X_pred, D=D_x, delta=self.fw_delta)
            L_v = self._oracle_bisection(Sigma_hat=self.nominal_Sigma_v, rho=self.theta_v,
                                       Sigma_ref=Sigma_v, D=D_v, delta=self.fw_delta)
            
            # Alg1-Step3: set d_t = s_tilde - s_t
            d_x = L_x - X_pred
            d_v = L_v - Sigma_v
            
            # Alg1-Step4: set g_t = - d_t^T ∇f(s_t)
            # We are minimizing f_min = -trace(X_post), while D_x,D_v are gradients of f_max
            # This implies ∇f_min = -(D_x, D_v), so:
            # g_t = <d_x, D_x> + <d_v, D_v>
            g_t = self._inner(d_x, D_x) + self._inner(d_v, D_v)
            
            # - stop if surrogate duality gap g_t <= self.fw_gap_tol
            if g_t <= self.fw_gap_tol:
                break
            
            # Alg1-Step5: set beta_t = beta_prev / self.fw_zeta
            beta_t = beta_prev / self.fw_zeta
            
            # Compute ||d_t||^2
            d_norm2 = self._fro_norm2(d_x) + self._fro_norm2(d_v)
            
            # Set eta with guards: if d_norm2 <= 0 or g_t <= 0: break (no descent direction)
            if d_norm2 <= 0 or g_t <= 0:
                break
                
            eta = min(1.0, g_t / (beta_t * d_norm2))
            
            # Alg1-Step6: backtracking line search loop
            # Compute f_curr once outside loop since X_pred, Sigma_v don't change during backtracking
            f_curr = self._fw_objective_f_min(X_pred, Sigma_v, C_t)
            
            max_backtrack = 20  # Safety limit
            for backtrack_iter in range(max_backtrack):
                f_next = self._fw_objective_f_min(X_pred + eta*d_x, Sigma_v + eta*d_v, C_t)
                armijo_rhs = f_curr - eta*g_t + 0.5*(eta**2)*beta_t*d_norm2

                if f_next <= armijo_rhs:
                    break

                beta_t = self.fw_tau * beta_t
                eta = min(1.0, g_t / (beta_t * d_norm2))

            
            # Alg1-Step7: update
            X_pred = X_pred + eta*d_x
            Sigma_v = Sigma_v + eta*d_v
            # Symmetrize only at the end since we're adding symmetric matrices
            
            
            # Alg1-Step8: set beta_prev = beta_t, t += 1 and continue
            beta_prev = beta_t
            t += 1
        
        # After loop: symmetrize final results once
        wc_Xprior = self._sym(X_pred)
        wc_Sigma_v = self._sym(Sigma_v)
        wc_Xpost = self._posterior_cov(wc_Xprior, wc_Sigma_v, C_t)
        
        return wc_Sigma_v, wc_Xprior, wc_Xpost

    # --- Frank-Wolfe Helper Functions ---
    
    def _fw_objective_f_min(self, X_pred, Sigma_v, C_t):
        """
        f_max = trace(X_post) where X_post = X_pred - X_pred C^T (C X_pred C^T + Sigma_v)^{-1} C X_pred
        FW runs minimization on f_min = -f_max
        """
        # Compute posterior covariance
        X_post = self._posterior_cov(X_pred, Sigma_v, C_t)
        
        # f_max = trace(X_post)
        f_max = np.trace(X_post)
        
        # f_min = -f_max (FW minimizes)
        f_min = -f_max
        
        return float(f_min)
    
    def _posterior_cov(self, X_pred, Sigma_v, C_t):
        """Compute posterior covariance X_post."""
        # X_post = X_pred - X_pred C_t^T (C_t X_pred C_t^T + Sigma_v)^{-1} C_t X_pred
        
        # Cache C_t @ X_pred since used twice
        C_X = C_t @ X_pred
        
        # Compute S = C_X @ C_t.T + Sigma_v (already symmetric)
        S = C_X @ C_t.T + Sigma_v

        try:
            # Direct computation of K = X_pred C_t^T S^{-1}
            K = np.linalg.solve(S, C_X).T
        except np.linalg.LinAlgError:
            eps = 1e-12
            S_reg = S + eps * np.eye(S.shape[0])
            K = np.linalg.solve(S_reg, C_X).T

        # Compute X_post (result should be symmetric)
        X_post = X_pred - K @ C_X
        return X_post
    
    def _fw_gradients(self, X_pred, Sigma_v, C_t):
        """
        Return D_x, D_v (symmetric PSD) used in oracle (Algorithm 2).
        Uses analytic formulas from Nguyen et al. Lemma A.5.
        """
        # Map to Nguyen notation: H := C_t, Σx := X_pred, Σw := Sigma_v
        # Define G = C_t @ X_pred @ C_t.T + Sigma_v  (already symmetric)
        G = C_t @ X_pred @ C_t.T + Sigma_v
        
        try:
            # Compute U = solve(G, C_t) => shape (ny, nx)
            U = np.linalg.solve(G, C_t)
            
            # Compute M = C_t.T @ U => shape (nx, nx)
            M = C_t.T @ U
            
            # Compute A = I_nx - X_pred @ M
            if self._I_nx is None:
                self._I_nx = np.eye(self.nx)
            A = self._I_nx - X_pred @ M
            
            # D_x = A.T @ A (automatically symmetric)
            D_x = A.T @ A
            
            # For D_v: compute X2 = X_pred @ X_pred once
            X2 = X_pred @ X_pred
            
            # Compute T = C_t @ X2 @ C_t.T => (ny, ny) (automatically symmetric)
            T = C_t @ X2 @ C_t.T
            
            # D_v = G^{-1} @ T @ G^{-1} (automatically symmetric)
            W1 = np.linalg.solve(G, T)  
            D_v = np.linalg.solve(G, W1.T).T
            
        except np.linalg.LinAlgError:
            # G is singular/ill-conditioned: add jitter
            eps = 1e-12
            G_reg = G + eps * np.eye(G.shape[0])
            
            # Retry computation with regularized G
            U = np.linalg.solve(G_reg, C_t)
            M = C_t.T @ U
            if self._I_nx is None:
                self._I_nx = np.eye(self.nx)
            A = self._I_nx - X_pred @ M
            D_x = A.T @ A
            
            X2 = X_pred @ X_pred
            T = C_t @ X2 @ C_t.T
            W1 = np.linalg.solve(G_reg, T)
            D_v = np.linalg.solve(G_reg, W1.T).T
        
        # Check for numerically zero matrices
        if np.allclose(D_x, 0, atol=1e-15):
            if self._I_nx is None:
                self._I_nx = np.eye(self.nx)
            D_x = 1e-12 * self._I_nx
        if np.allclose(D_v, 0, atol=1e-15):
            if self._I_ny is None:
                self._I_ny = np.eye(self.ny)
            D_v = 1e-12 * self._I_ny
        
        return D_x, D_v
    
    def _oracle_bisection(self, Sigma_hat, rho, Sigma_ref, D, delta):
        """
        Implements Algorithm 2 (Bisection Oracle) exactly; returns L_tilde.
        Inputs:
          Sigma_hat: nominal covariance (Σ_hat)
          rho: radius (ρ)
          Sigma_ref: reference feasible covariance Σ (current iterate in FW)
          D: gradient matrix (PSD, nonzero)
          delta: oracle precision (δ)
        """
        # Alg2-Step0: validate inputs (these should already be symmetric)
        # Only symmetrize input D which comes from gradients calculation
        D = self._sym(D)
        d = D.shape[0]
        I = np.eye(d)
        # If rho <= 0: return Sigma_hat
        if rho <= 0:
            return Sigma_hat
        
        # If D is ~0: return Sigma_hat (or Sigma_ref)
        if np.allclose(D, 0, atol=1e-15):
            return Sigma_hat
        
        # Alg2-Step1: compute λ1 = λmax(D) and eigenvector v1
        eigvals, eigvecs = np.linalg.eigh(D)  # ascending order
        lambda1 = eigvals[-1]  # maximum eigenvalue
        v1 = eigvecs[:, -1]    # corresponding eigenvector
        
        # If lambda1 <= 0: return Sigma_hat (oracle is trivial)
        if lambda1 <= 0:
            return Sigma_hat
        
        # Alg2-Step2: set gamma_low and gamma_high (bracketing)
        v1_Sigma_v1 = v1.T @ Sigma_hat @ v1
        trace_Sigma = np.trace(Sigma_hat)
        
        gamma_low = lambda1 * (1 + np.sqrt(max(v1_Sigma_v1, 0)) / rho)
        gamma_high = lambda1 * (1 + np.sqrt(max(trace_Sigma, 0)) / rho)
        
        # Add small epsilon so gamma_low > lambda1 strictly
        gamma_low = max(gamma_low, lambda1 + 1e-12)
        
        # Alg2-Step3: repeat bisection
        for iteration in range(self.fw_bisect_max_iters):
            gamma_mid = 0.5 * (gamma_low + gamma_high)
            
            denom = gamma_mid - eigvals
            if np.any(denom <= 0):
                # gamma must be strictly larger than lambda_max(D)
                gamma_mid = eigvals[-1] + 1e-12
                denom = gamma_mid - eigvals

            inv_eigvals = 1.0 / denom
            inv_mid = (eigvecs * inv_eigvals) @ eigvecs.T   # faster than diag
            inv_mid = self._sym(inv_mid)
            
            # Compute L_tilde = gamma_mid^2 * inv_mid @ Sigma_hat @ inv_mid (automatically symmetric)
            L_tilde = gamma_mid**2 * inv_mid @ Sigma_hat @ inv_mid
            
            # Alg2-Step4: compute phi(gamma_mid)
            # phi = gamma_mid * ( rho^2 + < gamma_mid*inv_mid - I, Sigma_hat > ) - < Sigma_ref, D >
            gamma_inv = gamma_mid * inv_mid
            gamma_inv_minus_I = gamma_inv - I
            inner1 = self._inner(gamma_inv_minus_I, Sigma_hat)
            inner2 = self._inner(Sigma_ref, D)
            phi = gamma_mid * (rho**2 + inner1) - inner2
            
            # Alg2-Step5: compute dphi = dφ/dγ(gamma_mid)
            # M = I - gamma_mid*inv_mid
            M = I - gamma_inv
            # dphi = rho^2 - < Sigma_hat, M @ M >
            dphi = rho**2 - self._inner(Sigma_hat, M @ M)
            
            # Alg2-Step6: bracket update
            if dphi < 0:
                gamma_low = gamma_mid
            else:
                gamma_high = gamma_mid
            
            # Alg2-Step7: stopping condition
            # Stop when BOTH:
            # - dphi > 0
            # - <L_tilde - Sigma_ref, D> >= delta * phi
            if dphi > 0:
                inner_stopping = self._inner(L_tilde - Sigma_ref, D)
                if inner_stopping >= delta * phi:
                    break
            
            # Safety stops
            if abs(gamma_high - gamma_low) <= self.fw_bisect_tol:
                break
        
        # Return L_tilde (symmetrized)
        return self._sym(L_tilde)
    
    # --- Frank-Wolfe Utility Functions ---
    
    def _inner(self, A, B):
        """Inner product <A,B> = trace(A.T @ B)."""
        return float(np.trace(A.T @ B))
    
    def _sym(self, A):
        """Symmetrize matrix: 0.5*(A + A.T)."""
        return 0.5 * (A + A.T)
    
    def _fro_norm2(self, A):
        """Frobenius norm squared: ||A||_F^2."""
        return np.sum(A * A)
    

    # --- DR-EKF Update Step ---
    def DR_kalman_filter(self, v_mean_hat, x_prior, y, t, u_prev=None, x_post_prev=None):
        """DR-EKF following Theorem equations
        
        Measurement Update:
        x_post_t = x_prior_t + K_t^* (y_t - h(x_prior_t) - hat{v}_t)
        X_post = (I - K_t^* C_t) X_prior_opt
        """
        # Linearize observation at prior state: C_t = ∂h/∂x|_{x_prior}
        C_t = self.C_jacobian(x_prior)
        
        # Compute pseudo-nominal prior covariance: X_prior_nom = A_t X_post A_t^T + hat{Sigma}_w
        if t == 0:
            X_prior_nom = self.nominal_x0_cov.copy()
        else:
            if x_post_prev is not None and u_prev is not None:
                # A_t = ∂f/∂x|_{x_post_{t-1}}
                A_t = self.F_jacobian(x_post_prev, u_prev)
                X_prior_nom = A_t @ self._P @ A_t.T + self.nominal_Sigma_w
            else:
                raise RuntimeError(f"DR-EKF CDC requires previous state and control input for t > 0, got: "
                                 f"x_post_prev={x_post_prev is not None}, u_prev={u_prev is not None}")
        
        # Solve SDP online: get (X_prior_opt, Sigma_v_opt)
        wc_Sigma_v, wc_Xprior, wc_Xpost = self.solve_sdp_online(X_prior_nom, C_t)
        
        if wc_Sigma_v is None:
            raise RuntimeError(f"DR-EKF CDC SDP optimization failed at time step {t}. "
                             f"Check theta parameters (theta_x={self.theta_x}, theta_v={self.theta_v}) "
                             f"and ensure they are feasible for the current problem instance.")
        
        # DR-EKF update using theorem equations
        # K_t^* = X_prior_opt C_t^T (C_t X_prior_opt C_t^T + Sigma_v_opt)^{-1}
        S = C_t @ wc_Xprior @ C_t.T + wc_Sigma_v
        K_star = np.linalg.solve(S, (wc_Xprior @ C_t.T).T).T
        
        # x_post_t = x_prior_t + K_t^* (y_t - h(x_prior_t) - hat{v}_t)
        innovation = y - (self.h(x_prior) + v_mean_hat)
        x_post = x_prior + K_star @ innovation
        
        self._P = wc_Xpost
        return x_post

    def _initial_update(self, x_est_init, y0):
        return self.DR_kalman_filter(self.nominal_mu_v, x_est_init, y0, 0, None)
    
    def _drkf_finite_cdc_update(self, x_prior, y, t, u_prev=None, x_post_prev=None):
        return self.DR_kalman_filter(self.nominal_mu_v, x_prior, y, t, u_prev, x_post_prev)
    
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
        # DR-EKF state prediction: x_prior_{t+1} = f(x_post_t, u_t) + hat{w}_t
        x_prior = self.f(x_est_prev, u_prev) + self.nominal_mu_w
        
        # DR-EKF measurement update following theorem
        return self._drkf_finite_cdc_update(x_prior, y_curr, t, u_prev, x_est_prev)
    
    def forward(self):
        return self._run_simulation_loop(self._drkf_finite_cdc_update)
    
    def forward_track(self, desired_trajectory):
        return self._run_simulation_loop(self._drkf_finite_cdc_update, desired_trajectory)
    
    def forward_track_MPC(self, desired_trajectory):
        return self._run_simulation_loop_MPC(self._drkf_finite_cdc_update, desired_trajectory)
