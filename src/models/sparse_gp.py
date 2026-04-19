"""Sparse Variational Gaussian Process (SVGP) for CO(GT) prediction.

Input convention
----------------
X : (N, 1 + F) float tensor standardised by a sklearn StandardScaler.
    Column 0 : continuous time index (raw hours, then scaled).
    Columns 1..F : Phase-2 Particle-Filter imputed sensor features (scaled).

y : (N,) float tensor — standardised CO(GT) from the Phase-2 PF output.

Kernel progression (all wrapped in a ScaleKernel)
--------------------------------------------------
1. "rbf"
       k_RBF(active_dims=[1..F], ard_num_dims=F)
       Temporal structure ignored; captures proximity in feature space.

2. "periodic"
       k_Periodic(active_dims=[0], period = 24 / time_std)
       Ignores sensor features; models the strict daily cycle in time.

3. "locally_periodic"
       k_Periodic(active_dims=[0]) × k_RBF(active_dims=[1..F])
       Daily envelope modulated by feature-space similarity.

Inducing-point initialisation
------------------------------
RBF / locally_periodic : K-Means in the full (1+F)-dimensional scaled space.
Periodic               : M ≤ 24 points evenly spaced within one period of
                         the scaled time axis so that every point occupies a
                         distinct phase — prevents K(Z,Z) from becoming
                         near-singular.  Feature dimensions are set to the
                         training-set median.
"""

from __future__ import annotations

from typing import Literal

import torch
import gpytorch
import linear_operator.settings as li_settings
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
from sklearn.cluster import KMeans

KernelType = Literal["rbf", "periodic", "locally_periodic"]


# ------------------------------------------------------------------
# Kernel factory
# ------------------------------------------------------------------

def build_kernel(
    kernel_type: KernelType,
    n_sensor_feats: int,
    period_scaled: float = 24.0,
) -> gpytorch.kernels.Kernel:
    """Construct a ScaleKernel with the correct active_dims for each variant.

    Parameters
    ----------
    kernel_type     : "rbf" | "periodic" | "locally_periodic"
    n_sensor_feats  : F — number of sensor feature columns (X[:, 1:].shape[1])
    period_scaled   : 24-h period expressed in the *scaled* time column units
                      = 24.0 / StandardScaler.scale_[0]
    """
    sensor_dims = list(range(1, 1 + n_sensor_feats))

    if kernel_type == "rbf":
        # ARD RBF over sensor features only
        base = gpytorch.kernels.RBFKernel(
            ard_num_dims=n_sensor_feats,
            active_dims=sensor_dims,
        )

    elif kernel_type == "periodic":
        # Strict periodicity on the time column only
        base = gpytorch.kernels.PeriodicKernel(active_dims=[0])
        base.period_length = torch.tensor(period_scaled)

    elif kernel_type == "locally_periodic":
        # k_Periodic(time) × k_RBF(features) — locally periodic composite
        per = gpytorch.kernels.PeriodicKernel(active_dims=[0])
        per.period_length = torch.tensor(period_scaled)
        rbf = gpytorch.kernels.RBFKernel(
            ard_num_dims=n_sensor_feats,
            active_dims=sensor_dims,
        )
        base = per * rbf  # ProductKernel

    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type!r}")

    return gpytorch.kernels.ScaleKernel(base)


# ------------------------------------------------------------------
# SVGP model
# ------------------------------------------------------------------

class SVGPModel(ApproximateGP):
    """Sparse Variational GP with Cholesky variational distribution.

    Parameters
    ----------
    inducing_points : (M, 1+F) tensor — initial inducing locations in scaled space
    kernel_type     : "rbf" | "periodic" | "locally_periodic"
    n_sensor_feats  : F — dimensionality of the sensor sub-space
    period_scaled   : 24-h period in scaled time units
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        kernel_type: KernelType = "locally_periodic",
        n_sensor_feats: int = 12,
        period_scaled: float = 24.0,
    ) -> None:
        M = inducing_points.shape[0]
        variational_distribution = CholeskyVariationalDistribution(M)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = build_kernel(kernel_type, n_sensor_feats, period_scaled)
        self.kernel_type = kernel_type

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ------------------------------------------------------------------
# Inducing-point initialisation
# ------------------------------------------------------------------

def init_inducing_points(X: torch.Tensor, M: int) -> torch.Tensor:
    """K-Means initialisation of M inducing points in the full input space."""
    if X.dim() == 1:
        X = X.unsqueeze(-1)
    X_np = X.detach().cpu().numpy()
    M_eff = min(M, len(X_np))
    kmeans = KMeans(n_clusters=M_eff, random_state=42, n_init="auto")
    kmeans.fit(X_np)
    return torch.tensor(kmeans.cluster_centers_, dtype=X.dtype)


def _init_inducing_for_kernel(
    X_train: torch.Tensor,
    M: int,
    kernel_type: KernelType,
    period_scaled: float = 24.0,
) -> torch.Tensor:
    """Kernel-aware inducing point factory.

    For the pure *periodic* kernel the time column is the only dimension that
    matters.  K-Means over the full time range places many points at the same
    phase of the 24-h cycle (period_scaled << time range), collapsing K(Z,Z)
    to a near-identity matrix and causing Cholesky failures.  We instead space
    M_eff ≤ 24 points uniformly within a single period [0, period_scaled) so
    each has a distinct phase.  Feature columns are fixed at training medians
    and refined by gradient descent during ELBO optimisation.
    """
    if kernel_type == "periodic":
        M_eff = min(M, 24)
        # Uniformly cover one period: indices 0, 1/M_eff, 2/M_eff, ... of [0, p)
        t_vals = torch.linspace(0.0, period_scaled, M_eff + 1)[:-1]   # (M_eff,)
        feat_medians = X_train[:, 1:].median(dim=0).values             # (F,)
        feat_part = feat_medians.unsqueeze(0).expand(M_eff, -1)        # (M_eff, F)
        Z = torch.cat([t_vals.unsqueeze(-1), feat_part], dim=1)
        return Z.to(X_train.dtype)

    # RBF and locally_periodic: standard K-Means over full scaled space
    return init_inducing_points(X_train, M)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_svgp(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    M: int = 200,
    n_epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 256,
    kernel_type: KernelType = "locally_periodic",
    n_sensor_feats: int = 12,
    period_scaled: float = 24.0,
    verbose: bool = True,
) -> tuple[SVGPModel, gpytorch.likelihoods.GaussianLikelihood]:
    """Train an SVGP via mini-batch ELBO optimisation.

    Parameters
    ----------
    X_train       : (N, 1+F) scaled input [time_scaled | features_scaled]
    y_train       : (N,) standardised CO(GT) target
    M             : number of inducing points
    n_epochs      : training epochs
    lr            : Adam learning rate
    batch_size    : mini-batch size
    kernel_type   : kernel variant
    n_sensor_feats: F (number of sensor feature columns)
    period_scaled : 24-h period in scaled time units (= 24 / time_scaler_std)

    Returns
    -------
    (model, likelihood) both in eval mode
    """
    inducing_points = _init_inducing_for_kernel(
        X_train, M, kernel_type, period_scaled
    )
    model = SVGPModel(
        inducing_points,
        kernel_type=kernel_type,
        n_sensor_feats=n_sensor_feats,
        period_scaled=period_scaled,
    )
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr
    )
    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model, num_data=len(y_train)
    )

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Both context managers are needed: gpytorch.settings.cholesky_jitter
    # controls GPyTorch-level jitter, while li_settings.cholesky_jitter
    # controls the jitter inside linear_operator's psd_safe_cholesky (which
    # is what actually fails for the periodic kernel's Gram matrix in float32).
    with gpytorch.settings.cholesky_jitter(1e-2), li_settings.cholesky_jitter(float_value=1e-2, double_value=1e-2):
        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if verbose and epoch % 10 == 0:
                n_batches = len(loader)
                print(
                    f"  [{kernel_type}] Epoch {epoch:4d}/{n_epochs}"
                    f"  avg-ELBO={-epoch_loss / n_batches:.4f}"
                )

    model.eval()
    likelihood.eval()
    return model, likelihood


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

@torch.no_grad()
def predict_svgp(
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (predictive_mean, predictive_std) on X_test.

    Std is passed through the likelihood so it includes observation noise.
    """
    model.eval()
    likelihood.eval()
    with (
        gpytorch.settings.fast_pred_var(),
        gpytorch.settings.cholesky_jitter(1e-2),
        li_settings.cholesky_jitter(float_value=1e-2, double_value=1e-2),
    ):
        pred = likelihood(model(X_test))
    return pred.mean, pred.variance.sqrt()


@torch.no_grad()
def predict_svgp_decomposed(
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose predictive uncertainty into epistemic and aleatoric parts.

    Epistemic = GP posterior variance  (reduces with more data)
    Aleatoric = learned observation noise  (irreducible)
    Total     = Epistemic + Aleatoric

    Returns
    -------
    mean          : (N,) predictive mean
    total_std     : (N,) sqrt(epistemic_var + aleatoric_var)
    epistemic_std : (N,) sqrt of GP posterior variance
    aleatoric_std : (N,) sqrt of observation noise variance
    """
    model.eval()
    likelihood.eval()
    with (
        gpytorch.settings.fast_pred_var(),
        gpytorch.settings.cholesky_jitter(1e-2),
        li_settings.cholesky_jitter(float_value=1e-2, double_value=1e-2),
    ):
        f_pred = model(X_test)           # function posterior (no noise)
        y_pred = likelihood(f_pred)      # observation posterior (with noise)

    epistemic_var = f_pred.variance.clamp(min=0)
    total_var = y_pred.variance.clamp(min=0)
    aleatoric_var = (total_var - epistemic_var).clamp(min=0)

    return (
        y_pred.mean,
        total_var.sqrt(),
        epistemic_var.sqrt(),
        aleatoric_var.sqrt(),
    )
