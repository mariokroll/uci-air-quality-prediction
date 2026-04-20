"""Multivariate Particle Filter for time-series imputation.

State transition (AR(1) + daily periodic forcing):
    x_t = A x_{t-1} + B sin(2π t / 24) + C cos(2π t / 24) + q_t,
    q_t ~ N(0, Q)

Observation model (log-saturation of metal oxide sensors):
    y_t = α ⊙ log(x_t + shift) + β + r_t,
    r_t ~ N(0, diag(R))

All computation runs in z-score space.  The shift vector is chosen per
dimension so that x + shift ≥ 1 everywhere (guaranteeing log > 0).

Missing-data policy
-------------------
- Partial blackout: weight update uses only the subset of active sensors.
- Total blackout  : weight update is skipped entirely; particles drift.
- CO(GT) is ALWAYS treated as unobserved during `run()` to prevent leakage.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.config import NMHC_COL, TARGET_COL


class MultivariateParticleFilter:
    """Fixed-N Multivariate Particle Filter.

    Parameters
    ----------
    n_particles:
        Number of particles N.
    resample_threshold:
        Fraction of N below which ESS triggers systematic resampling.
    process_noise_inflation:
        Multiplier applied to the estimated Q (increase for smoother imputation).
    obs_noise_inflation:
        Multiplier applied to the estimated R (increase for wider uncertainty).
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_particles: int = 1000,
        resample_threshold: float = 0.5,
        process_noise_inflation: float = 1.5,
        obs_noise_inflation: float = 2.0,
        random_state: int = 42,
    ) -> None:
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.process_noise_inflation = process_noise_inflation
        self.obs_noise_inflation = obs_noise_inflation
        self.rng = np.random.default_rng(random_state)

        # Fitted in fit()
        self._cols: list[str] = []
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._shift: Optional[np.ndarray] = None

        self.A_: Optional[np.ndarray] = None   # (D, D)
        self.B_: Optional[np.ndarray] = None   # (D,)
        self.C_: Optional[np.ndarray] = None   # (D,)
        self.Q_: Optional[np.ndarray] = None   # (D, D)
        self._L_Q: Optional[np.ndarray] = None  # Cholesky factor of Q

        self.alpha_: Optional[np.ndarray] = None  # (D,)
        self.beta_: Optional[np.ndarray] = None   # (D,)
        self.R_: Optional[np.ndarray] = None      # (D,) diagonal

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "MultivariateParticleFilter":
        """Estimate A, B, C, Q, α, β, R from the provided data.

        Any NaN values are filled with the column median before fitting
        so that the transition model is defined everywhere.
        """
        self._cols = list(df.columns)
        D = len(self._cols)

        # Fill NaN with column median before fitting
        df_filled = df.copy()
        for col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())

        # Z-score normalisation
        self._mu = df_filled.mean().values.astype(float)
        self._sigma = df_filled.std().replace(0.0, 1.0).values.astype(float)
        data_z: np.ndarray = ((df_filled.values - self._mu) / self._sigma).astype(float)

        # Log-shift: ensures x_z + shift ≥ 1 for all particles
        self._shift = np.abs(data_z.min(axis=0)) + 1.0  # (D,)

        self._fit_transition(data_z, D)
        self._fit_observation(data_z, D)
        return self

    def _fit_transition(self, data_z: np.ndarray, D: int) -> None:
        T = data_z.shape[0]
        X_lag = data_z[:-1]                                         # (T-1, D)
        t_idx = np.arange(1, T, dtype=float)
        sin_t = np.sin(2 * np.pi * t_idx / 24).reshape(-1, 1)
        cos_t = np.cos(2 * np.pi * t_idx / 24).reshape(-1, 1)
        X_design = np.hstack([X_lag, sin_t, cos_t])                # (T-1, D+2)
        Y_target = data_z[1:]                                       # (T-1, D)

        A = np.zeros((D, D))
        B = np.zeros(D)
        C = np.zeros(D)
        residuals = np.zeros_like(Y_target)

        for d in range(D):
            coeffs, _, _, _ = np.linalg.lstsq(
                X_design, Y_target[:, d], rcond=None
            )
            A[d] = coeffs[:D]
            B[d] = coeffs[D]
            C[d] = coeffs[D + 1]
            residuals[:, d] = Y_target[:, d] - X_design @ coeffs

        self.A_ = A
        self.B_ = B
        self.C_ = C
        Q_raw = np.cov(residuals.T) * self.process_noise_inflation
        # Regularise for positive-definiteness
        self.Q_ = Q_raw + 1e-6 * np.eye(D)
        self._L_Q = np.linalg.cholesky(self.Q_)

    def _fit_observation(self, data_z: np.ndarray, D: int) -> None:
        T = data_z.shape[0]
        log_data = np.log(data_z + self._shift)   # safe: shift ≥ 1

        alpha = np.zeros(D)
        beta = np.zeros(D)
        residuals = np.zeros_like(data_z)

        for d in range(D):
            X_obs = np.column_stack([log_data[:, d], np.ones(T)])
            coeffs, _, _, _ = np.linalg.lstsq(
                X_obs, data_z[:, d], rcond=None
            )
            alpha[d] = coeffs[0]
            beta[d] = coeffs[1]
            residuals[:, d] = data_z[:, d] - X_obs @ coeffs

        self.alpha_ = alpha
        self.beta_ = beta
        R_raw = residuals.var(axis=0) * self.obs_noise_inflation
        self.R_ = np.maximum(R_raw, 1e-4)   # floor avoids degenerate weights

    # ------------------------------------------------------------------
    # Running the filter
    # ------------------------------------------------------------------

    def run(
        self,
        df_raw: pd.DataFrame,
        hide_target: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the particle filter over the full time series.

        Parameters
        ----------
        df_raw:
            Original data (NaN where -200 was).
        hide_target:
            Always treat CO(GT) as unobserved (prevents leakage to Phase 3–5).

        Returns
        -------
        df_mean : weighted particle mean (imputed, original units)
        df_std  : weighted particle std  (uncertainty, original units)
        """
        assert self.A_ is not None, "Call fit() before run()."

        D = len(self._cols)
        T = len(df_raw)
        N = self.n_particles

        target_idx = (
            self._cols.index(TARGET_COL) if TARGET_COL in self._cols else None
        )

        # Normalise observations; NaN propagates for missing entries
        obs_z = (
            (df_raw[self._cols].values - self._mu) / self._sigma
        ).astype(float)  # (T, D)

        if hide_target and target_idx is not None:
            obs_z[:, target_idx] = np.nan

        # Initialise particles at the first fully-observed row
        init_state = self._first_valid_row(obs_z)
        noise_init = (self._L_Q @ self.rng.standard_normal((D, N))).T  # (N, D)
        particles = np.tile(init_state, (N, 1)) + noise_init

        log_weights = np.full(N, -np.log(N))  # uniform

        mean_out = np.zeros((T, D))
        std_out = np.zeros((T, D))

        for t in range(T):
            # ---- 1. Transition ----------------------------------------
            hour = t % 24
            sin_h = np.sin(2.0 * np.pi * hour / 24.0)
            cos_h = np.cos(2.0 * np.pi * hour / 24.0)
            process_noise = (
                self._L_Q @ self.rng.standard_normal((D, N))
            ).T  # (N, D)
            particles = (
                particles @ self.A_.T
                + self.B_ * sin_h
                + self.C_ * cos_h
                + process_noise
            )

            # ---- 2. Observation update --------------------------------
            obs_t = obs_z[t]
            obs_mask = ~np.isnan(obs_t)

            if obs_mask.any():
                # Partial or full update: use only active sensors
                log_weights = self._update_weights(
                    particles, obs_t, obs_mask, log_weights
                )
            # else: total blackout → skip update, let particles drift

            # ---- 3. Resample -----------------------------------------
            weights = self._softmax(log_weights)
            ess = 1.0 / float(np.sum(weights ** 2))
            if ess < self.resample_threshold * N:
                particles, log_weights = self._systematic_resample(
                    particles, log_weights
                )
                weights = self._softmax(log_weights)

            # ---- 4. Record weighted statistics -----------------------
            mean_z = weights @ particles                         # (D,)
            var_z = weights @ (particles - mean_z) ** 2         # (D,)
            mean_out[t] = mean_z * self._sigma + self._mu
            std_out[t] = np.sqrt(var_z) * self._sigma

        mean_df = pd.DataFrame(mean_out, index=df_raw.index, columns=self._cols)
        std_df = pd.DataFrame(std_out, index=df_raw.index, columns=self._cols)

        # For non-target observed values, restore exact originals
        observed_mask = df_raw[self._cols].notna()
        if hide_target and target_idx is not None:
            observed_mask.iloc[:, target_idx] = False
        mean_df[observed_mask] = df_raw[self._cols][observed_mask]

        return mean_df, std_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_weights(
        self,
        particles: np.ndarray,
        obs: np.ndarray,
        mask: np.ndarray,
        log_weights: np.ndarray,
    ) -> np.ndarray:
        """Incremental log-weight update for the observed dimensions only."""
        x_shifted = np.maximum(particles[:, mask] + self._shift[mask], 1e-9)
        y_hat = self.alpha_[mask] * np.log(x_shifted) + self.beta_[mask]

        obs_obs = obs[mask]          # (D_obs,)
        r_var = self.R_[mask]        # (D_obs,)
        diff = y_hat - obs_obs       # (N, D_obs)

        log_lik = -0.5 * np.sum(
            diff ** 2 / r_var + np.log(2.0 * np.pi * r_var), axis=1
        )  # (N,)

        new_log_w = log_weights + log_lik
        new_log_w -= self._log_sum_exp(new_log_w)
        return new_log_w

    def _systematic_resample(
        self, particles: np.ndarray, log_weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        N = len(particles)
        weights = self._softmax(log_weights)
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0  # numerical safety

        positions = (self.rng.uniform() + np.arange(N)) / N
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, N - 1)

        return particles[indices], np.full(N, -np.log(N))

    def _first_valid_row(self, obs_z: np.ndarray) -> np.ndarray:
        """Return first fully-observed row; fall back to per-column medians.

        Columns that are entirely NaN (never observed) are initialised at 0,
        which is the prior mean in z-score space.
        """
        for row in obs_z:
            if not np.isnan(row).any():
                return row.copy()
        med = np.nanmedian(obs_z, axis=0)
        med = np.where(np.isnan(med), 0.0, med)   # 0 = N(0,1) prior mean
        return med

    @staticmethod
    def _softmax(log_weights: np.ndarray) -> np.ndarray:
        lw = log_weights - log_weights.max()
        w = np.exp(lw)
        return w / w.sum()

    @staticmethod
    def _log_sum_exp(log_weights: np.ndarray) -> float:
        m = log_weights.max()
        return float(m + np.log(np.exp(log_weights - m).sum()))
