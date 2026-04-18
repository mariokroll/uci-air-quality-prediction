"""Multivariate Bayesian Linear Regression imputer.

Each column is treated as a separate regression target.  For a given target y
and design matrix X (remaining observed columns), the conjugate Normal prior
over weights w gives a closed-form posterior:

    Prior:     w  ~ N(0, alpha * I)
    Likelihood: y | X,w ~ N(X w, sigma² I)

    Posterior precision:  S_N⁻¹ = (1/alpha) I  +  (1/sigma²) Xᵀ X
    Posterior mean:       m_N   = S_N @ [ (1/sigma²) Xᵀ y ]

Imputation uses the posterior predictive mean  ŷ = X_* m_N.

Features are z-scored before fitting and predictions are un-scaled so that
observed values are preserved exactly (up to floating-point round-trip).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.config import NMHC_COL


class BayesianLinearImputer:
    """Column-wise Bayesian Linear Regression imputer with a Normal prior.

    Parameters
    ----------
    alpha:
        Prior variance on each weight (larger → weaker regularisation).
    noise_var:
        Assumed observation noise variance σ².
    exclude_cols:
        Columns never used *as regressors* (default: NMHC_COL, which is 89 %
        missing and would make most regression rows incomplete).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        noise_var: float = 1.0,
        exclude_cols: Optional[list[str]] = None,
    ) -> None:
        self.alpha = alpha
        self.noise_var = noise_var
        self.exclude_cols: list[str] = exclude_cols if exclude_cols is not None else [NMHC_COL]

        # Populated during fit()
        self._posteriors: dict[str, dict] = {}
        self._means: pd.Series | None = None
        self._stds: pd.Series | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _regressor_cols(self, df: pd.DataFrame, target: str) -> list[str]:
        return [c for c in df.columns if c != target and c not in self.exclude_cols]

    def _fit_one(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (m_N, S_N) for a single target column."""
        n, d = X.shape
        # Augment with bias column
        Xb = np.hstack([X, np.ones((n, 1))])
        db = d + 1

        prior_prec = (1.0 / self.alpha) * np.eye(db)
        likelihood_prec = 1.0 / self.noise_var

        S_N_inv = prior_prec + likelihood_prec * (Xb.T @ Xb)
        S_N = np.linalg.inv(S_N_inv)
        m_N = S_N @ (likelihood_prec * (Xb.T @ y))
        return m_N, S_N

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "BayesianLinearImputer":
        """Fit one Bayesian linear model per imputable column using complete cases."""
        self._means = df.mean()
        self._stds = df.std().replace(0.0, 1.0)
        df_z = (df - self._means) / self._stds

        for col in df.columns:
            if col in self.exclude_cols:
                continue
            regressors = self._regressor_cols(df_z, col)
            # Complete cases only (both target and all regressors observed)
            complete = df_z[[col] + regressors].dropna()
            if len(complete) < max(len(regressors) + 2, 10):
                continue  # not enough data to fit reliably
            X = complete[regressors].values.astype(float)
            y = complete[col].values.astype(float)
            m_N, S_N = self._fit_one(X, y)
            self._posteriors[col] = {"m_N": m_N, "S_N": S_N, "regressors": regressors}

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using the posterior predictive mean.

        Observed values are never modified.  Rows where the regressors
        themselves are missing are left as NaN (they will be handled by the
        Particle Filter in Phase 2).
        """
        assert self._means is not None, "Call fit() before transform()."

        df_z = (df - self._means) / self._stds
        df_out_z = df_z.copy()

        for col, post in self._posteriors.items():
            missing = df_z[col].isna()
            if not missing.any():
                continue

            regressors = post["regressors"]
            m_N = post["m_N"]

            # Only impute rows where every regressor is observed
            candidate_rows = df_z.loc[missing, regressors]
            can_impute = candidate_rows.notna().all(axis=1)
            rows = candidate_rows[can_impute]

            if rows.empty:
                continue

            Xb = np.hstack([rows.values.astype(float), np.ones((len(rows), 1))])
            y_hat_z = Xb @ m_N
            df_out_z.loc[rows.index, col] = y_hat_z

        # Un-scale back to original units
        return df_out_z * self._stds + self._means

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
