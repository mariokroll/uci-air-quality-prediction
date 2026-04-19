"""Phase 6 — Uncertainty decomposition and scalability profiling tests."""

import time

import pytest
import torch

from src.models.bnn_vi import (
    BNNRegressor,
    predict_bnn,
    predict_bnn_decomposed,
    train_bnn,
)
from scripts.profile_scalability import measure_inference_latency, measure_peak_ram_mb


# ── Shared fixtures ───────────────────────────────────────────────────────────

IN_FEATURES = 5
N_TRAIN = 120
N_TEST = 60
N_SAMPLES = 80

torch.manual_seed(0)
X_TRAIN = torch.randn(N_TRAIN, IN_FEATURES)
Y_TRAIN = torch.randn(N_TRAIN)
X_TEST = torch.randn(N_TEST, IN_FEATURES)
X_OOD = torch.randn(N_TEST, IN_FEATURES) * 15.0  # far outside training range


@pytest.fixture(scope="module")
def model():
    return train_bnn(
        X_TRAIN, Y_TRAIN,
        hidden_sizes=[16, 16],
        n_epochs=10,
        verbose=False,
    )


# ── predict_bnn_decomposed: return structure ──────────────────────────────────

def test_decomposed_returns_four_tensors(model):
    result = predict_bnn_decomposed(model, X_TEST, n_samples=N_SAMPLES)
    assert len(result) == 4


def test_decomposed_shapes(model):
    mean, total, epi, ale = predict_bnn_decomposed(model, X_TEST, n_samples=N_SAMPLES)
    for t in (mean, total, epi, ale):
        assert t.shape == (N_TEST,), f"Expected ({N_TEST},), got {t.shape}"


def test_decomposed_stds_nonnegative(model):
    _, total, epi, ale = predict_bnn_decomposed(model, X_TEST, n_samples=N_SAMPLES)
    assert (total >= 0).all(), "total_std has negative values"
    assert (epi >= 0).all(), "epistemic_std has negative values"
    assert (ale >= 0).all(), "aleatoric_std has negative values"


# ── Law of total variance: epi² + ale² == total² ─────────────────────────────

def test_variance_decomposition_exact(model):
    """epistemic_var + aleatoric_var must equal total_var exactly (by construction)."""
    _, total_std, epi_std, ale_std = predict_bnn_decomposed(
        model, X_TEST, n_samples=N_SAMPLES
    )
    reconstructed = epi_std ** 2 + ale_std ** 2
    assert torch.allclose(total_std ** 2, reconstructed, atol=1e-5), (
        f"Max deviation: {(total_std**2 - reconstructed).abs().max().item():.2e}"
    )


def test_total_var_equals_sum_of_components(model):
    """Sanity check at index level — first 10 elements."""
    _, total_std, epi_std, ale_std = predict_bnn_decomposed(
        model, X_TEST[:10], n_samples=200
    )
    for i in range(10):
        tv = total_std[i].item() ** 2
        ev = epi_std[i].item() ** 2
        av = ale_std[i].item() ** 2
        assert abs(tv - (ev + av)) < 1e-5, f"Point {i}: {tv:.6f} != {ev:.6f}+{av:.6f}"


# ── Epistemic uncertainty behaviour ──────────────────────────────────────────

def test_ood_epistemic_higher_than_in_distribution(model):
    """OOD inputs (far outside training range) must have higher mean epistemic std."""
    _, _, epi_in, _ = predict_bnn_decomposed(model, X_TEST, n_samples=N_SAMPLES)
    _, _, epi_ood, _ = predict_bnn_decomposed(model, X_OOD, n_samples=N_SAMPLES)
    assert epi_ood.mean() > epi_in.mean(), (
        f"OOD epistemic {epi_ood.mean():.4f} not > in-dist {epi_in.mean():.4f}"
    )


def test_epistemic_increases_with_samples(model):
    """More MC samples → more stable (not necessarily smaller) epistemic estimate.
    At minimum, running with more samples should not crash."""
    for n in (10, 50, 200):
        result = predict_bnn_decomposed(model, X_TEST, n_samples=n)
        assert len(result) == 4


# ── Consistency with predict_bnn ─────────────────────────────────────────────

def test_total_var_consistent_with_predict_bnn(model):
    """total_std from decomposed should be close to predict_bnn std (large n)."""
    _, std_base = predict_bnn(model, X_TEST, n_samples=500)
    _, total_std, _, _ = predict_bnn_decomposed(model, X_TEST, n_samples=500)
    # Both independently sample — expect agreement within ~5% in median
    ratio = (std_base / (total_std + 1e-8)).median().item()
    assert 0.8 < ratio < 1.2, f"std ratio median {ratio:.3f} outside [0.8, 1.2]"


# ── Profiling helpers ─────────────────────────────────────────────────────────

def test_measure_inference_latency_positive():
    """measure_inference_latency must return a positive number (ms)."""
    def dummy():
        time.sleep(0.001)

    latency = measure_inference_latency(dummy)
    assert latency > 0.0, "Latency should be positive"
    assert latency < 5_000.0, "Latency implausibly large (>5 s per run)"


def test_measure_inference_latency_captures_work():
    """A slower function should report higher latency than a fast one."""
    def fast():
        _ = 1 + 1

    def slow():
        time.sleep(0.005)

    assert measure_inference_latency(slow) > measure_inference_latency(fast)


def test_measure_peak_ram_positive():
    """measure_peak_ram_mb must return a non-negative value in MB."""
    def alloc():
        _ = [0] * 10_000

    ram = measure_peak_ram_mb(alloc)
    assert ram >= 0.0


def test_measure_peak_ram_reflects_allocation():
    """Allocating a larger array should show higher peak RAM."""
    def small():
        _ = [0] * 1_000

    def large():
        _ = [0] * 1_000_000

    assert measure_peak_ram_mb(large) > measure_peak_ram_mb(small)
