"""Bayesian Neural Network with Variational Inference (Bayes by Backprop).

Architecture
------------
Each linear layer keeps a mean (mu) and log-variance (log_var) for every
weight and bias.  During the forward pass weights are sampled via the local
reparametrisation trick:

    w ~ N(mu, softplus(rho)^2)   where rho = log(exp(sigma) - 1)

The ELBO loss is:

    L = E_q[log p(y|x,w)] - KL[q(w) || p(w)]

with a standard Normal prior p(w) = N(0, prior_std^2).

Input convention
----------------
Same as the SVGP: X = [scaled_time | scaled_sensor_features], (N, 1+F).
y = standardised CO(GT).

Outputs
-------
`predict_bnn` returns the predictive (mean, std) averaged over `n_samples`
stochastic forward passes; std captures both epistemic and aleatoric
uncertainty because the network outputs a heteroscedastic log-variance.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Bayesian linear layer
# ------------------------------------------------------------------

class BayesianLinear(nn.Module):
    """Linear layer with Gaussian variational posterior over weights and biases.

    Parameters
    ----------
    in_features, out_features : layer dimensions
    prior_std                 : std of the isotropic Gaussian weight prior
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Variational parameters — posterior q(w) = N(mu, softplus(rho)^2)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # Initialise rho so sigma ≈ 0.1 — small but non-negligible uncertainty
        nn.init.constant_(self.weight_rho, -3.0)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -3.0)

    @staticmethod
    def _softplus(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_sigma = self._softplus(self.weight_rho)
        bias_sigma = self._softplus(self.bias_rho)

        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """KL[q(w) || p(w)] for weights and biases (closed form)."""
        prior_var = self.prior_std ** 2

        def _kl(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            # KL between two Gaussians: N(mu, sigma^2) || N(0, prior_std^2)
            return 0.5 * (
                (sigma ** 2 + mu ** 2) / prior_var
                - 1.0
                + math.log(prior_var)
                - 2.0 * torch.log(sigma)
            ).sum()

        weight_sigma = self._softplus(self.weight_rho)
        bias_sigma = self._softplus(self.bias_rho)
        return _kl(self.weight_mu, weight_sigma) + _kl(self.bias_mu, bias_sigma)


# ------------------------------------------------------------------
# BNN regressor
# ------------------------------------------------------------------

class BNNRegressor(nn.Module):
    """Heteroscedastic BNN: outputs (mean, log_var) for each input.

    Parameters
    ----------
    in_features   : input dimensionality (1 + F)
    hidden_sizes  : list of hidden layer widths
    prior_std     : weight prior std (shared across all layers)
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: list[int] | None = None,
        prior_std: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        self.prior_std = prior_std
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(BayesianLinear(prev, h, prior_std=prior_std))
            prev = h

        self.hidden_layers = nn.ModuleList(layers)
        self.out_mean = BayesianLinear(prev, 1, prior_std=prior_std)
        self.out_log_var = BayesianLinear(prev, 1, prior_std=prior_std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_var) — both shape (N,)."""
        h = x
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        mean = self.out_mean(h).squeeze(-1)
        log_var = self.out_log_var(h).squeeze(-1)
        return mean, log_var

    def kl_divergence(self) -> torch.Tensor:
        kl = sum(layer.kl_divergence() for layer in self.hidden_layers)
        kl = kl + self.out_mean.kl_divergence() + self.out_log_var.kl_divergence()
        return kl


# ------------------------------------------------------------------
# ELBO loss
# ------------------------------------------------------------------

def bnn_elbo_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    y: torch.Tensor,
    kl: torch.Tensor,
    n_data: int,
) -> torch.Tensor:
    """Negative ELBO = heteroscedastic NLL + KL / n_data.

    NLL = 0.5 * mean(log_var + (y - mean)^2 / exp(log_var))
    """
    nll = 0.5 * (log_var + (y - mean) ** 2 / (log_var.exp() + 1e-8)).mean()
    return nll + kl / n_data


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_bnn(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    hidden_sizes: list[int] | None = None,
    prior_std: float = 1.0,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose: bool = True,
) -> BNNRegressor:
    """Train a heteroscedastic BNN via mini-batch ELBO optimisation.

    Parameters
    ----------
    X_train      : (N, 1+F) scaled input
    y_train      : (N,) standardised target
    hidden_sizes : widths of hidden layers
    prior_std    : Gaussian weight prior std
    n_epochs     : training epochs
    lr           : Adam learning rate
    batch_size   : mini-batch size

    Returns
    -------
    BNNRegressor in eval mode
    """
    if hidden_sizes is None:
        hidden_sizes = [64, 64]

    in_features = X_train.shape[1]
    model = BNNRegressor(in_features, hidden_sizes=hidden_sizes, prior_std=prior_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_data = len(y_train)

    model.train()
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            mean, log_var = model(X_batch)
            kl = model.kl_divergence()
            loss = bnn_elbo_loss(mean, log_var, y_batch, kl, n_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if verbose and epoch % 10 == 0:
            n_batches = len(loader)
            print(
                f"  [BNN] Epoch {epoch:4d}/{n_epochs}"
                f"  avg-ELBO={-epoch_loss / n_batches:.4f}"
            )

    model.eval()
    return model


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

@torch.no_grad()
def predict_bnn(
    model: BNNRegressor,
    X_test: torch.Tensor,
    n_samples: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (predictive_mean, predictive_std) via MC sampling.

    Each forward pass draws different weights; the final variance combines
    epistemic uncertainty (variance of sample means) and aleatoric uncertainty
    (mean of sample variances from the network's log_var output).

    Parameters
    ----------
    model     : trained BNNRegressor (eval mode)
    X_test    : (N, 1+F) scaled input
    n_samples : number of stochastic forward passes

    Returns
    -------
    mean : (N,) predictive mean
    std  : (N,) predictive standard deviation (>= 0)
    """
    model.train()  # enable weight sampling
    means = []
    vars_ = []
    for _ in range(n_samples):
        m, lv = model(X_test)
        means.append(m)
        vars_.append(lv.exp())
    model.eval()

    means_stack = torch.stack(means, dim=0)   # (S, N)
    vars_stack = torch.stack(vars_, dim=0)    # (S, N)

    pred_mean = means_stack.mean(dim=0)
    # Law of total variance: Var = E[Var] + Var[E]
    # correction=0 avoids the degenerate-dof warning when n_samples=1
    pred_var = vars_stack.mean(dim=0) + means_stack.var(dim=0, correction=0)
    pred_std = pred_var.sqrt()

    return pred_mean, pred_std


@torch.no_grad()
def predict_bnn_decomposed(
    model: BNNRegressor,
    X_test: torch.Tensor,
    n_samples: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose predictive uncertainty into epistemic and aleatoric components.

    Uses the law of total variance:
        Var[Y] = E[Var[Y|w]]  +  Var[E[Y|w]]
                  aleatoric        epistemic

    Parameters
    ----------
    model     : trained BNNRegressor
    X_test    : (N, 1+F) scaled input
    n_samples : MC forward passes

    Returns
    -------
    mean          : (N,) predictive mean
    total_std     : (N,) sqrt(epistemic_var + aleatoric_var)
    epistemic_std : (N,) sqrt(Var[E[Y|w]])  — model / parameter uncertainty
    aleatoric_std : (N,) sqrt(E[Var[Y|w]])  — irreducible / data noise
    """
    model.train()
    means = []
    vars_ = []
    for _ in range(n_samples):
        m, lv = model(X_test)
        means.append(m)
        vars_.append(lv.exp())
    model.eval()

    means_stack = torch.stack(means, dim=0)   # (S, N)
    vars_stack = torch.stack(vars_, dim=0)    # (S, N)

    pred_mean = means_stack.mean(dim=0)
    aleatoric_var = vars_stack.mean(dim=0)                      # E[σ²_aleatoric]
    epistemic_var = means_stack.var(dim=0, correction=0)        # Var[μ]
    total_var = aleatoric_var + epistemic_var

    return (
        pred_mean,
        total_var.sqrt(),
        epistemic_var.sqrt(),
        aleatoric_var.sqrt(),
    )
