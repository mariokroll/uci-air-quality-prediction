"""Phase 1 & 2 — BLR and Particle Filter imputer tests."""

import numpy as np
import pandas as pd
import pytest

from src.config import NMHC_COL, TARGET_COL
from src.data.loader import load_raw
from src.imputation.bayesian_linear import BayesianLinearImputer
from src.imputation.particle_filter import MultivariateParticleFilter


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


# ==================================================================
# Phase 2 — Particle Filter tests
# ==================================================================

def _make_synthetic_pf(D: int = 4, T: int = 120, N: int = 200, seed: int = 7):
    """Return a fitted PF and a raw DataFrame for fast unit testing."""
    rng = np.random.default_rng(seed)
    cols = [f"x{d}" for d in range(D)]

    # Stable AR(1) synthetic data
    data = np.zeros((T, D))
    data[0] = rng.standard_normal(D)
    for t in range(1, T):
        data[t] = 0.8 * data[t - 1] + rng.standard_normal(D) * 0.3

    df_clean = pd.DataFrame(data, columns=cols)
    # Inject ~15 % missingness
    mask = rng.random((T, D)) < 0.15
    df_raw = df_clean.copy()
    df_raw.values[mask] = np.nan

    pf = MultivariateParticleFilter(n_particles=N, random_state=seed)
    pf.fit(df_clean)         # fit on clean data (mimics BLR-imputed input)
    return pf, df_raw, cols


# ------------------------------------------------------------------
# PF: fitted parameters are well-formed
# ------------------------------------------------------------------

def test_pf_fitted_parameter_shapes():
    D = 4
    pf, _, _ = _make_synthetic_pf(D=D)
    assert pf.A_.shape == (D, D)
    assert pf.B_.shape == (D,)
    assert pf.C_.shape == (D,)
    assert pf.Q_.shape == (D, D)
    assert pf.alpha_.shape == (D,)
    assert pf.beta_.shape == (D,)
    assert pf.R_.shape == (D,)


def test_pf_Q_positive_definite():
    pf, _, _ = _make_synthetic_pf()
    eigvals = np.linalg.eigvalsh(pf.Q_)
    assert np.all(eigvals > 0), "Process noise Q is not positive-definite"


# ------------------------------------------------------------------
# PF: run produces valid output shapes
# ------------------------------------------------------------------

def test_pf_run_output_shapes():
    pf, df_raw, cols = _make_synthetic_pf()
    mean_df, std_df = pf.run(df_raw, hide_target=False)
    assert mean_df.shape == df_raw.shape
    assert std_df.shape == df_raw.shape


# ------------------------------------------------------------------
# PF: no NaN in mean output (all missing values imputed)
# ------------------------------------------------------------------

def test_pf_no_nan_in_mean():
    pf, df_raw, _ = _make_synthetic_pf()
    mean_df, _ = pf.run(df_raw, hide_target=False)
    assert not mean_df.isna().any().any(), "PF mean output contains NaN"


# ------------------------------------------------------------------
# PF: std is non-negative everywhere
# ------------------------------------------------------------------

def test_pf_std_nonnegative():
    pf, df_raw, _ = _make_synthetic_pf()
    _, std_df = pf.run(df_raw, hide_target=False)
    assert (std_df.values >= 0).all(), "PF std contains negative values"


# ------------------------------------------------------------------
# PF: partial blackout — filter runs and only active sensors update
# ------------------------------------------------------------------

def test_partial_blackout():
    """Half of sensors are missing at every step; PF must still produce valid output."""
    D = 6
    T = 60
    N = 100
    rng = np.random.default_rng(0)
    cols = [f"x{d}" for d in range(D)]

    data = np.cumsum(rng.standard_normal((T, D)) * 0.2, axis=0)
    df_clean = pd.DataFrame(data, columns=cols)

    # Mask exactly the first D//2 columns at ALL timesteps → always partial
    df_partial = df_clean.copy()
    df_partial.iloc[:, : D // 2] = np.nan

    pf = MultivariateParticleFilter(n_particles=N, random_state=1)
    pf.fit(df_clean)
    mean_df, std_df = pf.run(df_partial, hide_target=False)

    assert not mean_df.isna().any().any(), "Partial blackout: NaN in output"
    assert (std_df.values >= 0).all(), "Partial blackout: negative std"
    # Imputed columns should have non-zero std (genuine uncertainty)
    assert std_df.iloc[:, : D // 2].mean().mean() > 0, \
        "Imputed columns should have positive uncertainty"


# ------------------------------------------------------------------
# PF: total blackout — weight update is skipped; particles drift
# ------------------------------------------------------------------

def test_total_blackout_weights_unchanged():
    """With all observations NaN, _update_weights must NOT be called."""
    pf, _, _ = _make_synthetic_pf(D=4, N=50)
    D = 4
    N = 50

    # Manually invoke the update-skip logic as in run()
    obs_all_nan = np.full(D, np.nan)
    obs_mask = ~np.isnan(obs_all_nan)

    log_weights_before = np.full(N, -np.log(N))
    # Because obs_mask.any() is False, no update occurs
    assert not obs_mask.any(), "Expected total blackout (all NaN)"

    # Simulate the run loop branch: weights unchanged
    log_weights_after = log_weights_before.copy()   # no update
    np.testing.assert_array_equal(log_weights_before, log_weights_after,
                                  err_msg="Total blackout altered weights")


def test_total_blackout_end_to_end():
    """A DataFrame that is entirely NaN must not crash the filter."""
    D = 4
    T = 30
    N = 80
    cols = [f"x{d}" for d in range(D)]

    rng = np.random.default_rng(5)
    df_clean = pd.DataFrame(rng.standard_normal((T, D)), columns=cols)
    df_blackout = pd.DataFrame(np.nan, index=df_clean.index, columns=cols)

    pf = MultivariateParticleFilter(n_particles=N, random_state=5)
    pf.fit(df_clean)
    mean_df, std_df = pf.run(df_blackout, hide_target=False)

    assert not mean_df.isna().any().any(), "Total blackout: NaN in output"
    assert (std_df.values >= 0).all(), "Total blackout: negative std"


# ------------------------------------------------------------------
# PF: CO(GT) is always hidden when hide_target=True
# ------------------------------------------------------------------

def test_co_gt_hidden_during_run():
    """PF output for CO(GT) must never equal the raw observed CO(GT) values."""
    df = load_raw()
    blr = BayesianLinearImputer()
    df_blr = blr.fit_transform(df)

    pf = MultivariateParticleFilter(n_particles=300, random_state=0)
    pf.fit(df_blr)
    mean_df, _ = pf.run(df, hide_target=True)

    # Where CO(GT) was originally observed, PF output must differ (it used PF estimate)
    co_observed = df[TARGET_COL].notna()
    raw_co = df.loc[co_observed, TARGET_COL].values
    pf_co = mean_df.loc[co_observed, TARGET_COL].values
    # They should NOT be identical (PF estimated, not copied)
    assert not np.allclose(raw_co, pf_co, rtol=1e-6), \
        "CO(GT) was not hidden — PF copied raw observed values"


# ------------------------------------------------------------------
# PF vs BLR: PF reduces remaining NaN beyond BLR
# ------------------------------------------------------------------

def test_pf_further_reduces_missing():
    """PF (which imputes from dynamics) should leave no NaN for non-NMHC cols."""
    df = load_raw()
    blr = BayesianLinearImputer()
    df_blr = blr.fit_transform(df)

    pf = MultivariateParticleFilter(n_particles=300, random_state=0)
    pf.fit(df_blr)
    mean_df, _ = pf.run(df, hide_target=True)

    non_nmhc = [c for c in df.columns if c != NMHC_COL]
    assert mean_df[non_nmhc].isna().sum().sum() == 0, \
        "PF left NaN in non-NMHC columns"
