"""Phase 5 — Evaluation metrics and decision policy tests."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    empirical_coverage,
    gaussian_nll,
    rmse,
)
from src.evaluation.decision_policy import UncertaintyPolicy


# ── Shared fixtures ──────────────────────────────────────────────────────────

RNG = np.random.default_rng(0)
N = 200

Y_TRUE = RNG.standard_normal(N).astype(np.float32)
MEAN_GOOD = Y_TRUE + RNG.standard_normal(N) * 0.1   # low-error predictions
MEAN_BAD = RNG.standard_normal(N)                    # uncorrelated predictions
STD_CONST = np.full(N, 0.5, dtype=np.float32)
STD_VARY = (np.abs(RNG.standard_normal(N)) + 0.1).astype(np.float32)


# ------------------------------------------------------------------
# rmse
# ------------------------------------------------------------------

def test_rmse_perfect():
    assert rmse(Y_TRUE, Y_TRUE) == pytest.approx(0.0, abs=1e-6)


def test_rmse_known_value():
    y = np.array([0.0, 1.0, 2.0])
    pred = np.array([1.0, 1.0, 1.0])
    # errors: -1, 0, 1 → mean sq = 2/3 → rmse = sqrt(2/3)
    assert rmse(y, pred) == pytest.approx(np.sqrt(2 / 3), abs=1e-6)


def test_rmse_good_less_than_bad():
    assert rmse(Y_TRUE, MEAN_GOOD) < rmse(Y_TRUE, MEAN_BAD)


# ------------------------------------------------------------------
# gaussian_nll
# ------------------------------------------------------------------

def test_nll_finite():
    assert np.isfinite(gaussian_nll(Y_TRUE, MEAN_GOOD, STD_CONST))


def test_nll_lower_for_better_mean():
    nll_good = gaussian_nll(Y_TRUE, MEAN_GOOD, STD_CONST)
    nll_bad = gaussian_nll(Y_TRUE, MEAN_BAD, STD_CONST)
    assert nll_good < nll_bad


def test_nll_lower_for_calibrated_std():
    """A std matching residual magnitude should beat a large constant std."""
    errors = Y_TRUE - MEAN_GOOD
    calibrated_std = np.abs(errors) + 1e-3
    nll_calib = gaussian_nll(Y_TRUE, MEAN_GOOD, calibrated_std)
    nll_large = gaussian_nll(Y_TRUE, MEAN_GOOD, np.full(N, 10.0))
    assert nll_calib < nll_large


def test_nll_known_value():
    """NLL for a single point N(0,1) evaluated at 0: 0.5*log(2π) ≈ 0.9189."""
    y = np.array([0.0])
    m = np.array([0.0])
    s = np.array([1.0])
    expected = 0.5 * np.log(2 * np.pi)
    assert gaussian_nll(y, m, s) == pytest.approx(expected, abs=1e-5)


# ------------------------------------------------------------------
# empirical_coverage
# ------------------------------------------------------------------

def test_coverage_at_k0_is_zero():
    assert empirical_coverage(Y_TRUE, MEAN_GOOD, STD_CONST, k=0.0) == 0.0


def test_coverage_at_large_k_is_one():
    assert empirical_coverage(Y_TRUE, MEAN_GOOD, STD_CONST, k=1000.0) == 1.0


def test_coverage_increases_with_k():
    cov1 = empirical_coverage(Y_TRUE, MEAN_GOOD, STD_CONST, k=1.0)
    cov2 = empirical_coverage(Y_TRUE, MEAN_GOOD, STD_CONST, k=2.0)
    assert cov2 >= cov1


def test_coverage_standard_normal():
    """For a N(0,1) model at k=1 the expected coverage is ≈68%."""
    rng = np.random.default_rng(42)
    n = 10_000
    y = rng.standard_normal(n)
    m = np.zeros(n)
    s = np.ones(n)
    cov = empirical_coverage(y, m, s, k=1.0)
    assert abs(cov - 0.6827) < 0.02, f"Coverage at k=1 for N(0,1): {cov:.4f}"


def test_coverage_standard_normal_k2():
    """For a N(0,1) model at k=2 the expected coverage is ≈95%."""
    rng = np.random.default_rng(42)
    n = 10_000
    y = rng.standard_normal(n)
    m = np.zeros(n)
    s = np.ones(n)
    cov = empirical_coverage(y, m, s, k=2.0)
    assert abs(cov - 0.9545) < 0.02, f"Coverage at k=2 for N(0,1): {cov:.4f}"


# ------------------------------------------------------------------
# compute_all_metrics
# ------------------------------------------------------------------

def test_compute_all_metrics_keys():
    m = compute_all_metrics(Y_TRUE, MEAN_GOOD, STD_CONST)
    assert set(m.keys()) == {"RMSE", "NLL", "Coverage@1σ", "Coverage@2σ"}


def test_compute_all_metrics_values_in_range():
    m = compute_all_metrics(Y_TRUE, MEAN_GOOD, STD_CONST)
    assert m["RMSE"] >= 0
    assert 0.0 <= m["Coverage@1σ"] <= 1.0
    assert 0.0 <= m["Coverage@2σ"] <= 1.0


# ------------------------------------------------------------------
# UncertaintyPolicy: construction
# ------------------------------------------------------------------

def test_policy_fixed_threshold():
    p = UncertaintyPolicy(threshold=0.5)
    assert p.threshold == pytest.approx(0.5)


def test_policy_rejects_nonpositive_threshold():
    with pytest.raises(ValueError):
        UncertaintyPolicy(threshold=0.0)
    with pytest.raises(ValueError):
        UncertaintyPolicy(threshold=-1.0)


def test_policy_from_percentile():
    std = np.linspace(0.1, 1.0, 100)
    p = UncertaintyPolicy.from_percentile(std, percentile=90.0)
    assert p.threshold == pytest.approx(np.percentile(std, 90.0), abs=1e-5)


def test_policy_from_percentile_rejects_bad_value():
    std = np.ones(10)
    with pytest.raises(ValueError):
        UncertaintyPolicy.from_percentile(std, percentile=0.0)
    with pytest.raises(ValueError):
        UncertaintyPolicy.from_percentile(std, percentile=100.0)


# ------------------------------------------------------------------
# UncertaintyPolicy: flag
# ------------------------------------------------------------------

def test_flag_shape():
    p = UncertaintyPolicy(threshold=0.5)
    flags = p.flag(STD_VARY)
    assert flags.shape == (N,)
    assert flags.dtype == bool


def test_flag_correctness():
    std = np.array([0.1, 0.5, 0.9, 1.1])
    p = UncertaintyPolicy(threshold=0.5)
    flags = p.flag(std)
    # Only values strictly > 0.5 are flagged
    np.testing.assert_array_equal(flags, [False, False, True, True])


def test_flag_none():
    p = UncertaintyPolicy(threshold=100.0)
    assert p.flag(STD_VARY).sum() == 0


def test_flag_all():
    p = UncertaintyPolicy(threshold=0.0001)
    assert p.flag(STD_VARY).all()


def test_flagged_fraction_range():
    p = UncertaintyPolicy(threshold=np.median(STD_VARY))
    frac = p.flagged_fraction(STD_VARY)
    assert 0.0 <= frac <= 1.0


# ------------------------------------------------------------------
# UncertaintyPolicy: summary
# ------------------------------------------------------------------

def test_summary_contains_threshold():
    p = UncertaintyPolicy(threshold=0.5)
    report = p.summary(Y_TRUE, MEAN_GOOD, STD_VARY)
    assert "threshold" in report
    assert report["threshold"] == pytest.approx(0.5)


def test_summary_accepted_rmse_lower_than_flagged():
    """Flagged (high-std) predictions should have worse RMSE than accepted ones
    when errors correlate with uncertainty — verified on synthetic data."""
    rng = np.random.default_rng(7)
    n = 500
    y_true = rng.standard_normal(n)
    std = (rng.uniform(0.1, 2.0, n)).astype(np.float32)
    # Larger std → larger error (artificial correlation)
    mean = y_true + rng.standard_normal(n) * std

    p = UncertaintyPolicy(threshold=float(np.median(std)))
    report = p.summary(y_true, mean, std)

    assert "accepted_RMSE" in report and "flagged_RMSE" in report
    assert report["flagged_RMSE"] > report["accepted_RMSE"], (
        f"Expected flagged RMSE > accepted RMSE, got "
        f"flagged={report['flagged_RMSE']:.4f}  accepted={report['accepted_RMSE']:.4f}"
    )


def test_summary_all_accepted():
    """With threshold above max std, everything is accepted."""
    p = UncertaintyPolicy(threshold=float(STD_VARY.max()) + 1.0)
    report = p.summary(Y_TRUE, MEAN_GOOD, STD_VARY)
    assert report["flagged_fraction"] == 0.0
    assert "accepted_RMSE" in report
    assert "flagged_RMSE" not in report


def test_summary_all_flagged():
    """With threshold below min std, everything is flagged."""
    p = UncertaintyPolicy(threshold=float(STD_VARY.min()) - 0.001)
    report = p.summary(Y_TRUE, MEAN_GOOD, STD_VARY)
    assert report["flagged_fraction"] == 1.0
    assert "flagged_RMSE" in report
    assert "accepted_RMSE" not in report
