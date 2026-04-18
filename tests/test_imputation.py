"""Phase 1 — Bayesian Linear Regression imputer tests."""

import numpy as np
import pytest

from src.config import NMHC_COL
from src.data.loader import load_raw
from src.imputation.bayesian_linear import BayesianLinearImputer


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def imputed(raw):
    return BayesianLinearImputer().fit_transform(raw)


# ------------------------------------------------------------------
# Correctness: sentinel absence
# ------------------------------------------------------------------

def test_no_sentinel_after_imputation(imputed):
    assert not (imputed == -200).any().any()


# ------------------------------------------------------------------
# Correctness: missing-value reduction
# ------------------------------------------------------------------

def test_imputation_reduces_missing(raw, imputed):
    before = raw.isna().sum().sum()
    after = imputed.isna().sum().sum()
    assert after < before, "Imputation did not reduce the number of NaN values"


# ------------------------------------------------------------------
# Correctness: observed values are preserved
# ------------------------------------------------------------------

def test_observed_values_unchanged(raw, imputed):
    """Originally observed values must not be modified (tolerance 1e-9)."""
    for col in raw.columns:
        mask = raw[col].notna()
        if not mask.any():
            continue
        np.testing.assert_allclose(
            raw.loc[mask, col].values,
            imputed.loc[mask, col].values,
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"Column '{col}': observed values were altered",
        )


# ------------------------------------------------------------------
# Correctness: NMHC is excluded (not imputed)
# ------------------------------------------------------------------

def test_nmhc_not_imputed(raw, imputed):
    """NMHC(GT) must not be imputed — its NaN count must be unchanged."""
    assert raw[NMHC_COL].isna().sum() == imputed[NMHC_COL].isna().sum()


# ------------------------------------------------------------------
# Model internals: posterior shapes
# ------------------------------------------------------------------

def test_posterior_shapes(raw):
    imputer = BayesianLinearImputer()
    imputer.fit(raw)
    for col, post in imputer._posteriors.items():
        n_reg = len(post["regressors"]) + 1  # +1 bias
        assert post["m_N"].shape == (n_reg,), f"{col}: m_N shape mismatch"
        assert post["S_N"].shape == (n_reg, n_reg), f"{col}: S_N shape mismatch"


# ------------------------------------------------------------------
# Model internals: posterior covariance is positive-definite
# ------------------------------------------------------------------

def test_posterior_covariance_positive_definite(raw):
    imputer = BayesianLinearImputer()
    imputer.fit(raw)
    for col, post in imputer._posteriors.items():
        eigvals = np.linalg.eigvalsh(post["S_N"])
        assert np.all(eigvals > 0), f"{col}: S_N is not positive-definite"


# ------------------------------------------------------------------
# Model internals: synthetic sanity check (known weights)
# ------------------------------------------------------------------

def test_recovers_known_weights():
    """On noise-free synthetic data the posterior mean should be close to w*."""
    rng = np.random.default_rng(42)
    n, d = 500, 3
    w_true = np.array([2.0, -1.5, 0.8, 0.3])  # includes bias

    X = rng.standard_normal((n, d))
    Xb = np.hstack([X, np.ones((n, 1))])
    y = Xb @ w_true + rng.normal(0, 0.01, n)  # very low noise

    imputer = BayesianLinearImputer(alpha=10.0, noise_var=0.01**2)
    m_N, _ = imputer._fit_one(X, y)

    np.testing.assert_allclose(m_N, w_true, atol=0.05,
                               err_msg="Posterior mean far from true weights")
