"""Phase 4 — BNN (Bayes by Backprop) tests."""

import torch
import pytest

from src.models.bnn_vi import (
    BayesianLinear,
    BNNRegressor,
    bnn_elbo_loss,
    predict_bnn,
    train_bnn,
)

# ── Shared constants ─────────────────────────────────────────────────────────
IN_FEATS = 6       # 1 time + 5 sensor features
HIDDEN = [32, 32]
N_TR = 200
N_TE = 40


def _dummy_data(N: int = N_TR, seed: int = 0):
    torch.manual_seed(seed)
    X = torch.randn(N, IN_FEATS)
    y = torch.sin(X[:, 0]) + 0.1 * torch.randn(N)
    return X, y


# ------------------------------------------------------------------
# BayesianLinear: basic properties
# ------------------------------------------------------------------

def test_bayesian_linear_output_shape():
    layer = BayesianLinear(IN_FEATS, 16)
    x = torch.randn(10, IN_FEATS)
    out = layer(x)
    assert out.shape == (10, 16)


def test_bayesian_linear_kl_is_positive():
    layer = BayesianLinear(IN_FEATS, 16)
    kl = layer.kl_divergence()
    assert kl.item() > 0, "KL divergence must be strictly positive"


def test_bayesian_linear_stochastic():
    """Two forward passes must differ (weight sampling is active)."""
    torch.manual_seed(0)
    layer = BayesianLinear(IN_FEATS, 8)
    x = torch.randn(5, IN_FEATS)
    out1 = layer(x)
    out2 = layer(x)
    assert not torch.allclose(out1, out2), \
        "BayesianLinear outputs identical results — sampling not active"


# ------------------------------------------------------------------
# BNNRegressor: architecture and stochasticity
# ------------------------------------------------------------------

def test_bnn_output_shapes():
    model = BNNRegressor(IN_FEATS, hidden_sizes=HIDDEN)
    model.train()
    x = torch.randn(12, IN_FEATS)
    mean, log_var = model(x)
    assert mean.shape == (12,)
    assert log_var.shape == (12,)


def test_bnn_kl_positive():
    model = BNNRegressor(IN_FEATS, hidden_sizes=HIDDEN)
    assert model.kl_divergence().item() > 0


def test_bnn_stochastic_forward():
    """Different forward passes must yield different outputs."""
    torch.manual_seed(1)
    model = BNNRegressor(IN_FEATS, hidden_sizes=HIDDEN)
    model.train()
    x = torch.randn(20, IN_FEATS)
    mean1, _ = model(x)
    mean2, _ = model(x)
    assert not torch.allclose(mean1, mean2), \
        "BNNRegressor is deterministic — weight sampling not working"


# ------------------------------------------------------------------
# ELBO loss: value and gradients
# ------------------------------------------------------------------

def test_elbo_loss_is_finite():
    model = BNNRegressor(IN_FEATS, hidden_sizes=HIDDEN)
    model.train()
    x, y = _dummy_data(32)
    mean, log_var = model(x)
    kl = model.kl_divergence()
    loss = bnn_elbo_loss(mean, log_var, y, kl, n_data=N_TR)
    assert torch.isfinite(loss), f"ELBO loss is not finite: {loss.item()}"


def test_elbo_loss_gradients_flow():
    model = BNNRegressor(IN_FEATS, hidden_sizes=HIDDEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = _dummy_data(32)

    model.train()
    optimizer.zero_grad()
    mean, log_var = model(x)
    kl = model.kl_divergence()
    loss = bnn_elbo_loss(mean, log_var, y, kl, n_data=N_TR)
    loss.backward()

    any_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert any_grad, "No gradients flowed through the ELBO loss"


# ------------------------------------------------------------------
# Training: smoke test
# ------------------------------------------------------------------

def test_train_bnn_runs():
    X, y = _dummy_data(N_TR)
    model = train_bnn(
        X, y,
        hidden_sizes=HIDDEN,
        n_epochs=5,
        lr=1e-3,
        batch_size=64,
        verbose=False,
    )
    assert model is not None


def test_train_bnn_loss_decreases():
    """ELBO loss should decrease over sufficient training on synthetic data."""
    torch.manual_seed(42)
    N = 400
    X = torch.randn(N, IN_FEATS)
    y = 2.0 * X[:, 0] + 0.05 * torch.randn(N)  # near-linear signal

    losses = []

    class _TracingBNN(BNNRegressor):
        pass

    import src.models.bnn_vi as bnn_mod
    original_train = bnn_mod.train_bnn

    # Train short and long; compare final batch losses
    model_short = train_bnn(X, y, hidden_sizes=HIDDEN, n_epochs=2,
                            lr=5e-3, batch_size=128, verbose=False)
    model_long = train_bnn(X, y, hidden_sizes=HIDDEN, n_epochs=40,
                           lr=5e-3, batch_size=128, verbose=False)

    # After more training the ELBO NLL component should be lower
    model_short.train()
    model_long.train()
    with torch.no_grad():
        mean_s, lv_s = model_short(X)
        mean_l, lv_l = model_long(X)

    nll_short = (0.5 * (lv_s + (y - mean_s) ** 2 / lv_s.exp())).mean().item()
    nll_long = (0.5 * (lv_l + (y - mean_l) ** 2 / lv_l.exp())).mean().item()
    assert nll_long < nll_short, (
        f"NLL did not improve: short={nll_short:.4f}  long={nll_long:.4f}"
    )


# ------------------------------------------------------------------
# Prediction: shapes, non-negative std, stochastic variance
# ------------------------------------------------------------------

def test_predict_bnn_shapes():
    X_tr, y_tr = _dummy_data(N_TR)
    X_te, _ = _dummy_data(N_TE, seed=1)
    model = train_bnn(X_tr, y_tr, hidden_sizes=HIDDEN, n_epochs=3,
                      lr=1e-3, batch_size=64, verbose=False)
    mean, std = predict_bnn(model, X_te, n_samples=10)
    assert mean.shape == (N_TE,)
    assert std.shape == (N_TE,)


def test_predict_bnn_std_nonnegative():
    X_tr, y_tr = _dummy_data(N_TR)
    X_te, _ = _dummy_data(N_TE, seed=2)
    model = train_bnn(X_tr, y_tr, hidden_sizes=HIDDEN, n_epochs=3,
                      lr=1e-3, batch_size=64, verbose=False)
    _, std = predict_bnn(model, X_te, n_samples=20)
    assert (std >= 0).all(), "predict_bnn returned negative std values"


def test_predict_bnn_std_varies_with_samples():
    """std with n_samples=1 vs n_samples=50 should differ (MC variance)."""
    X_tr, y_tr = _dummy_data(N_TR)
    X_te, _ = _dummy_data(N_TE, seed=3)
    model = train_bnn(X_tr, y_tr, hidden_sizes=HIDDEN, n_epochs=3,
                      lr=1e-3, batch_size=64, verbose=False)
    _, std1 = predict_bnn(model, X_te, n_samples=1)
    _, std50 = predict_bnn(model, X_te, n_samples=50)
    # std with 1 sample has no epistemic component; they should differ
    assert not torch.allclose(std1, std50, atol=1e-6), \
        "std is identical for n_samples=1 and n_samples=50"


# ------------------------------------------------------------------
# Uncertainty: OOD inputs should yield higher std than in-distribution
# ------------------------------------------------------------------

def test_ood_uncertainty_higher():
    """Features shifted 8 sigma away should produce higher predictive std."""
    torch.manual_seed(5)
    N = 300
    X_tr = torch.randn(N, IN_FEATS)
    y_tr = torch.sin(X_tr[:, 0]) + 0.05 * torch.randn(N)

    model = train_bnn(X_tr, y_tr, hidden_sizes=HIDDEN, n_epochs=20,
                      lr=5e-3, batch_size=64, verbose=False)

    torch.manual_seed(6)
    X_ind = torch.randn(50, IN_FEATS)               # same distribution as train
    X_ood = torch.randn(50, IN_FEATS) + 8.0         # shifted 8 sigma

    _, std_ind = predict_bnn(model, X_ind, n_samples=50)
    _, std_ood = predict_bnn(model, X_ood, n_samples=50)

    assert std_ood.mean() > std_ind.mean(), (
        f"OOD std ({std_ood.mean():.4f}) not greater than in-dist std "
        f"({std_ind.mean():.4f})"
    )
