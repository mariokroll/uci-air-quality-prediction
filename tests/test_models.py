"""Phase 3 — Sparse GP tests (multi-dimensional input with active_dims kernels)."""

import torch
import gpytorch
import pytest

from src.models.sparse_gp import (
    SVGPModel,
    build_kernel,
    init_inducing_points,
    predict_svgp,
    train_svgp,
    _init_inducing_for_kernel,
)

# ── Shared test constants ───────────────────────────────────────────────────
N_FEATS = 5           # sensor feature columns (F)
N_DIM = 1 + N_FEATS   # total input width: [time | features]
PERIOD = 0.05         # arbitrary scaled period for tests
KERNEL_TYPES = ["rbf", "periodic", "locally_periodic"]


def _dummy_X(N: int = 20) -> torch.Tensor:
    """(N, N_DIM) tensor: col-0 is time, cols 1..F are features."""
    torch.manual_seed(0)
    t = torch.linspace(0.0, 3 * PERIOD, N).unsqueeze(-1)   # scaled time col
    feats = torch.randn(N, N_FEATS)
    return torch.cat([t, feats], dim=1)


# ------------------------------------------------------------------
# build_kernel: forward pass through dummy tensors
# ------------------------------------------------------------------

@pytest.mark.parametrize("kernel_type", KERNEL_TYPES)
def test_kernel_forward_pass(kernel_type):
    """Each kernel must produce a finite (N, N) covariance matrix."""
    X = _dummy_X(20)
    kern = build_kernel(kernel_type, n_sensor_feats=N_FEATS, period_scaled=PERIOD)
    K = kern(X, X).to_dense()
    assert K.shape == (20, 20)
    assert torch.isfinite(K).all(), f"{kernel_type}: non-finite values in K"


@pytest.mark.parametrize("kernel_type", KERNEL_TYPES)
def test_kernel_is_symmetric(kernel_type):
    X = _dummy_X(15)
    kern = build_kernel(kernel_type, n_sensor_feats=N_FEATS, period_scaled=PERIOD)
    K = kern(X, X).to_dense()
    assert torch.allclose(K, K.T, atol=1e-5), f"{kernel_type}: K is not symmetric"


@pytest.mark.parametrize("kernel_type", KERNEL_TYPES)
def test_kernel_positive_semidefinite(kernel_type):
    X = _dummy_X(12)
    kern = build_kernel(kernel_type, n_sensor_feats=N_FEATS, period_scaled=PERIOD)
    K = kern(X, X).to_dense()
    eigvals = torch.linalg.eigvalsh(K)
    assert (eigvals >= -1e-4).all(), (
        f"{kernel_type}: min eigenvalue = {eigvals.min().item():.2e}"
    )


# ------------------------------------------------------------------
# active_dims: each kernel sees only its designated columns
# ------------------------------------------------------------------

def test_rbf_ignores_time_column():
    """RBF kernel must produce identical rows when only the time column changes."""
    X1 = _dummy_X(10)
    X2 = X1.clone()
    X2[:, 0] = X2[:, 0] + 99.0      # perturb time only

    kern = build_kernel("rbf", n_sensor_feats=N_FEATS, period_scaled=PERIOD)
    K1 = kern(X1, X1).to_dense()
    K2 = kern(X2, X2).to_dense()
    assert torch.allclose(K1, K2, atol=1e-5), \
        "RBF kernel changed when only the time column was modified"


def test_periodic_ignores_feature_columns():
    """Periodic kernel must produce identical K when only sensor features change."""
    X1 = _dummy_X(10)
    X2 = X1.clone()
    X2[:, 1:] = torch.randn_like(X2[:, 1:]) * 100  # perturb features only

    kern = build_kernel("periodic", n_sensor_feats=N_FEATS, period_scaled=PERIOD)
    K1 = kern(X1, X1).to_dense()
    K2 = kern(X2, X2).to_dense()
    assert torch.allclose(K1, K2, atol=1e-5), \
        "Periodic kernel changed when only sensor features were modified"


# ------------------------------------------------------------------
# Composite kernel: locally_periodic == k_Periodic × k_RBF
# ------------------------------------------------------------------

def test_locally_periodic_equals_periodic_times_rbf():
    """locally_periodic must equal ScaleKernel(Periodic × RBF) with same params."""
    X = _dummy_X(10)
    sensor_dims = list(range(1, 1 + N_FEATS))

    composite = build_kernel("locally_periodic", n_sensor_feats=N_FEATS,
                             period_scaled=PERIOD)

    # Build manually with same structural parameters
    per_ref = composite.base_kernel.kernels[0]
    rbf_ref = composite.base_kernel.kernels[1]

    per = gpytorch.kernels.PeriodicKernel(active_dims=[0])
    per.period_length = per_ref.period_length.detach()
    per.lengthscale = per_ref.lengthscale.detach()

    rbf = gpytorch.kernels.RBFKernel(ard_num_dims=N_FEATS, active_dims=sensor_dims)
    rbf.lengthscale = rbf_ref.lengthscale.detach()

    manual = gpytorch.kernels.ScaleKernel(per * rbf)
    manual.outputscale = composite.outputscale.detach()

    K_comp = composite(X, X).to_dense()
    K_man = manual(X, X).to_dense()
    assert torch.allclose(K_comp, K_man, atol=1e-5), \
        "locally_periodic does not equal ScaleKernel(Periodic * RBF)"


# ------------------------------------------------------------------
# Period preservation
# ------------------------------------------------------------------

def test_periodic_period_is_set_correctly():
    kern = build_kernel("periodic", n_sensor_feats=N_FEATS, period_scaled=PERIOD)
    assert abs(kern.base_kernel.period_length.item() - PERIOD) < 1e-5


def test_locally_periodic_period_is_set_correctly():
    kern = build_kernel("locally_periodic", n_sensor_feats=N_FEATS, period_scaled=PERIOD)
    per_sub = kern.base_kernel.kernels[0]
    assert abs(per_sub.period_length.item() - PERIOD) < 1e-5


# ------------------------------------------------------------------
# RBF ARD: lengthscale vector has one entry per sensor feature
# ------------------------------------------------------------------

def test_rbf_ard_lengthscale_shape():
    kern = build_kernel("rbf", n_sensor_feats=N_FEATS, period_scaled=PERIOD)
    ls = kern.base_kernel.lengthscale
    assert ls.shape[-1] == N_FEATS, \
        f"Expected ARD lengthscale of size {N_FEATS}, got {ls.shape[-1]}"


def test_locally_periodic_rbf_ard_lengthscale_shape():
    kern = build_kernel("locally_periodic", n_sensor_feats=N_FEATS, period_scaled=PERIOD)
    rbf_sub = kern.base_kernel.kernels[1]
    ls = rbf_sub.lengthscale
    assert ls.shape[-1] == N_FEATS


# ------------------------------------------------------------------
# Inducing-point initialisation
# ------------------------------------------------------------------

def test_init_inducing_points_shape():
    X = _dummy_X(200)
    M = 30
    Z = init_inducing_points(X, M)
    assert Z.shape == (M, N_DIM)


def test_periodic_inducing_points_cover_one_period():
    """For the periodic kernel, all inducing points must lie within [0, period)."""
    X = _dummy_X(500)
    Z = _init_inducing_for_kernel(X, M=24, kernel_type="periodic",
                                  period_scaled=PERIOD)
    t_vals = Z[:, 0]
    assert t_vals.min() >= 0.0 - 1e-6
    assert t_vals.max() < PERIOD + 1e-6, \
        f"Periodic inducing points exceed one period: max={t_vals.max():.4f}"


def test_periodic_inducing_points_max_24():
    """Periodic kernel must never use more than 24 inducing points."""
    X = _dummy_X(500)
    Z = _init_inducing_for_kernel(X, M=200, kernel_type="periodic",
                                  period_scaled=PERIOD)
    assert Z.shape[0] <= 24


# ------------------------------------------------------------------
# SVGPModel: forward pass returns correct distribution type
# ------------------------------------------------------------------

@pytest.mark.parametrize("kernel_type", KERNEL_TYPES)
def test_svgp_forward_returns_mvn(kernel_type):
    X = _dummy_X(15)
    Z = _init_inducing_for_kernel(X, M=8, kernel_type=kernel_type,
                                  period_scaled=PERIOD)
    model = SVGPModel(Z, kernel_type=kernel_type,
                      n_sensor_feats=N_FEATS, period_scaled=PERIOD)
    model.train()
    dist = model(X)
    assert isinstance(dist, gpytorch.distributions.MultivariateNormal)
    assert dist.mean.shape == (15,)


# ------------------------------------------------------------------
# Training: smoke test on synthetic sinusoidal data
# ------------------------------------------------------------------

@pytest.mark.parametrize("kernel_type", KERNEL_TYPES)
def test_svgp_trains_without_error(kernel_type):
    torch.manual_seed(0)
    N = 200
    t = torch.linspace(0.0, 3 * PERIOD, N)
    feats = torch.randn(N, N_FEATS)
    X = torch.cat([t.unsqueeze(-1), feats], dim=1)
    y = torch.sin(2 * torch.pi * t / PERIOD) + 0.1 * torch.randn(N)

    model, likelihood = train_svgp(
        X, y, M=20, n_epochs=5, lr=0.05, batch_size=64,
        kernel_type=kernel_type, n_sensor_feats=N_FEATS,
        period_scaled=PERIOD, verbose=False,
    )
    assert model is not None
    assert likelihood is not None


# ------------------------------------------------------------------
# Prediction: output shapes and non-negative std
# ------------------------------------------------------------------

@pytest.mark.parametrize("kernel_type", KERNEL_TYPES)
def test_svgp_prediction_shapes(kernel_type):
    torch.manual_seed(1)
    N_tr, N_te = 120, 30
    t_tr = torch.linspace(0.0, 2 * PERIOD, N_tr)
    t_te = torch.linspace(2 * PERIOD, 3 * PERIOD, N_te)

    X_tr = torch.cat([t_tr.unsqueeze(-1), torch.randn(N_tr, N_FEATS)], dim=1)
    X_te = torch.cat([t_te.unsqueeze(-1), torch.randn(N_te, N_FEATS)], dim=1)
    y_tr = torch.sin(2 * torch.pi * t_tr / PERIOD)

    model, likelihood = train_svgp(
        X_tr, y_tr, M=15, n_epochs=3, lr=0.05, batch_size=50,
        kernel_type=kernel_type, n_sensor_feats=N_FEATS,
        period_scaled=PERIOD, verbose=False,
    )
    mean, std = predict_svgp(model, likelihood, X_te)

    assert mean.shape == (N_te,)
    assert std.shape == (N_te,)
    assert (std >= 0).all(), f"{kernel_type}: negative std in predictions"


# ------------------------------------------------------------------
# Uncertainty: extrapolation region should be more uncertain than interpolation
# ------------------------------------------------------------------

def test_uncertainty_reflects_data_density():
    """Out-of-distribution features must produce higher std than in-distribution.

    With active_dims=[1..F] the RBF kernel is blind to the time column — only
    feature-space distance drives uncertainty.  We therefore compare:
      * X_ood : features shifted 8 sigma away from the training distribution
      * X_ind : features drawn from the same N(0,1) as training data
    """
    torch.manual_seed(2)
    N = 300
    t_tr = torch.linspace(0.0, 2 * PERIOD, N)
    feats_tr = torch.randn(N, N_FEATS)                            # N(0,1) features
    X_tr = torch.cat([t_tr.unsqueeze(-1), feats_tr], dim=1)
    y_tr = torch.sin(2 * torch.pi * t_tr / PERIOD) + 0.05 * torch.randn(N)

    torch.manual_seed(3)
    t_test = torch.linspace(0.0, 2 * PERIOD, 20)
    # In-distribution: same feature range as training
    X_ind = torch.cat([t_test.unsqueeze(-1), torch.randn(20, N_FEATS)], dim=1)
    # Out-of-distribution: features shifted far from training region
    X_ood = torch.cat([t_test.unsqueeze(-1),
                       torch.randn(20, N_FEATS) + 8.0], dim=1)

    model, likelihood = train_svgp(
        X_tr, y_tr, M=30, n_epochs=20, lr=0.05, batch_size=64,
        kernel_type="rbf", n_sensor_feats=N_FEATS,
        period_scaled=PERIOD, verbose=False,
    )
    _, std_ood = predict_svgp(model, likelihood, X_ood)
    _, std_ind = predict_svgp(model, likelihood, X_ind)

    assert std_ood.mean() > std_ind.mean(), (
        f"Expected higher uncertainty for OOD features "
        f"(std_ood={std_ood.mean():.4f}, std_ind={std_ind.mean():.4f})"
    )
