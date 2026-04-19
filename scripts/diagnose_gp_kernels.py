"""Diagnostics for understanding how each GP kernel affects predictions.

This script complements ``scripts/train_gp.py`` without changing the training
pipeline. It trains the same three SVGP variants, then reports:

1. Learned kernel hyperparameters.
2. Pairwise prediction differences across kernels.
3. Counterfactual sensitivity to time with features fixed.
4. Counterfactual sensitivity to features with time fixed.

Outputs are written to ``data/processed``:

- ``gp_kernel_diagnostics.txt``
- ``gp_prediction_differences.csv``
- ``gp_counterfactual_sensitivity.csv``

Usage
-----
    python -m scripts.diagnose_gp_kernels
"""

from __future__ import annotations

import torch
import numpy as np
import pandas as pd

from scripts.train_gp import BATCH_SIZE, LR, M_INDUCING, N_EPOCHS, prepare_data
from src.config import PROCESSED_DATA_DIR
from src.models.sparse_gp import KernelType, predict_svgp, train_svgp

KERNEL_TYPES: list[KernelType] = ["rbf", "periodic", "locally_periodic"]


def _denormalize(mean: torch.Tensor, std: torch.Tensor, y_mean: float, y_std: float) -> tuple[np.ndarray, np.ndarray]:
    mean_orig = mean.detach().cpu().numpy() * y_std + y_mean
    std_orig = std.detach().cpu().numpy() * y_std
    return mean_orig, std_orig


def _extract_kernel_params(model, likelihood) -> dict[str, float]:
    params: dict[str, float] = {
        "outputscale": float(model.covar_module.outputscale.detach().cpu().item()),
        "noise": float(likelihood.noise.detach().cpu().item()),
        "mean_constant": float(model.mean_module.constant.detach().cpu().item()),
    }

    base = model.covar_module.base_kernel

    if model.kernel_type == "rbf":
        lengthscale = base.lengthscale.detach().cpu().view(-1).numpy()
        params["rbf_lengthscale_mean"] = float(lengthscale.mean())
        params["rbf_lengthscale_min"] = float(lengthscale.min())
        params["rbf_lengthscale_max"] = float(lengthscale.max())
    elif model.kernel_type == "periodic":
        params["period_length"] = float(base.period_length.detach().cpu().item())
        params["periodic_lengthscale"] = float(base.lengthscale.detach().cpu().item())
    elif model.kernel_type == "locally_periodic":
        per_kernel = base.kernels[0]
        rbf_kernel = base.kernels[1]
        rbf_lengthscale = rbf_kernel.lengthscale.detach().cpu().view(-1).numpy()
        params["period_length"] = float(per_kernel.period_length.detach().cpu().item())
        params["periodic_lengthscale"] = float(per_kernel.lengthscale.detach().cpu().item())
        params["rbf_lengthscale_mean"] = float(rbf_lengthscale.mean())
        params["rbf_lengthscale_min"] = float(rbf_lengthscale.min())
        params["rbf_lengthscale_max"] = float(rbf_lengthscale.max())

    return params


def _counterfactual_time_grid(X_train: torch.Tensor, X_test: torch.Tensor, n_hours: int = 48) -> torch.Tensor:
    """Vary time while keeping sensor features fixed to the median test feature vector."""
    feat_anchor = X_test[:, 1:].median(dim=0).values
    start_time = float(X_train[-1, 0].item())
    t_vals = torch.arange(start_time, start_time + n_hours, dtype=X_train.dtype)
    feat_block = feat_anchor.unsqueeze(0).expand(len(t_vals), -1)
    return torch.cat([t_vals.unsqueeze(-1), feat_block], dim=1)


def _counterfactual_feature_grid(X_train: torch.Tensor, n_points: int = 256) -> torch.Tensor:
    """Vary features by sampling training rows while keeping time fixed."""
    n_points = min(n_points, len(X_train))
    idx = torch.linspace(0, len(X_train) - 1, n_points).round().long()
    feat_samples = X_train[idx, 1:]
    time_anchor = X_train[:, 0].median()
    time_block = time_anchor.expand(n_points, 1)
    return torch.cat([time_block, feat_samples], dim=1)


def _pairwise_prediction_differences(predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    kernel_names = list(predictions.keys())
    for i, left in enumerate(kernel_names):
        for right in kernel_names[i + 1 :]:
            diff = predictions[left] - predictions[right]
            rows.append(
                {
                    "kernel_left": left,
                    "kernel_right": right,
                    "mean_abs_diff": float(np.mean(np.abs(diff))),
                    "max_abs_diff": float(np.max(np.abs(diff))),
                    "corr": float(np.corrcoef(predictions[left], predictions[right])[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_mean,
        y_std,
        raw_co,
        period_scaled,
        n_sensor_feats,
        split,
    ) = prepare_data()

    test_predictions: dict[str, np.ndarray] = {}
    hyper_rows: list[dict[str, float | str]] = []
    sensitivity_rows: list[dict[str, float | str]] = []

    X_cf_time = _counterfactual_time_grid(X_train, X_test)
    X_cf_feat = _counterfactual_feature_grid(X_train)

    for kernel_type in KERNEL_TYPES:
        print(f"Training diagnostic model: {kernel_type}")
        model, likelihood = train_svgp(
            X_train,
            y_train,
            M=M_INDUCING,
            n_epochs=N_EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            kernel_type=kernel_type,
            n_sensor_feats=n_sensor_feats,
            period_scaled=period_scaled,
            verbose=False,
        )

        mean_test, std_test = predict_svgp(model, likelihood, X_test)
        mean_test_orig, _ = _denormalize(mean_test, std_test, y_mean, y_std)
        test_predictions[kernel_type] = mean_test_orig

        mean_time, std_time = predict_svgp(model, likelihood, X_cf_time)
        mean_time_orig, _ = _denormalize(mean_time, std_time, y_mean, y_std)

        mean_feat, std_feat = predict_svgp(model, likelihood, X_cf_feat)
        mean_feat_orig, _ = _denormalize(mean_feat, std_feat, y_mean, y_std)

        hyper_row = {"kernel": kernel_type}
        hyper_row.update(_extract_kernel_params(model, likelihood))
        hyper_rows.append(hyper_row)

        sensitivity_rows.append(
            {
                "kernel": kernel_type,
                "time_cf_std": float(np.std(mean_time_orig)),
                "time_cf_range": float(np.max(mean_time_orig) - np.min(mean_time_orig)),
                "feature_cf_std": float(np.std(mean_feat_orig)),
                "feature_cf_range": float(np.max(mean_feat_orig) - np.min(mean_feat_orig)),
                "time_to_feature_std_ratio": float(np.std(mean_time_orig) / (np.std(mean_feat_orig) + 1e-12)),
                "time_to_feature_range_ratio": float(
                    (np.max(mean_time_orig) - np.min(mean_time_orig))
                    / ((np.max(mean_feat_orig) - np.min(mean_feat_orig)) + 1e-12)
                ),
            }
        )

    df_hyper = pd.DataFrame(hyper_rows).set_index("kernel")
    df_pairwise = _pairwise_prediction_differences(test_predictions)
    df_sensitivity = pd.DataFrame(sensitivity_rows).set_index("kernel")

    hyper_path = PROCESSED_DATA_DIR / "gp_kernel_diagnostics.txt"
    pairwise_path = PROCESSED_DATA_DIR / "gp_prediction_differences.csv"
    sensitivity_path = PROCESSED_DATA_DIR / "gp_counterfactual_sensitivity.csv"

    with hyper_path.open("w", encoding="utf-8") as fh:
        fh.write("Learned GP hyperparameters\n")
        fh.write("=" * 80 + "\n")
        fh.write(df_hyper.to_string(float_format=lambda x: f"{x:.6f}"))
        fh.write("\n\nCounterfactual sensitivity summary\n")
        fh.write("=" * 80 + "\n")
        fh.write(df_sensitivity.to_string(float_format=lambda x: f"{x:.6f}"))
        fh.write("\n")

    df_pairwise.to_csv(pairwise_path, index=False)
    df_sensitivity.to_csv(sensitivity_path)

    print(f"Wrote diagnostics -> {hyper_path}")
    print(f"Wrote pairwise differences -> {pairwise_path}")
    print(f"Wrote sensitivity summary -> {sensitivity_path}")


if __name__ == "__main__":
    main()
