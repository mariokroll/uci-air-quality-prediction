"""Evaluation metrics for probabilistic regression.

All functions accept plain numpy arrays and return scalar floats unless
noted.  They are model-agnostic — work for both SVGP and BNN outputs.
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def gaussian_nll(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    """Mean Gaussian negative log-likelihood.

    NLL = 0.5 * mean(log(2π σ²) + (y - μ)² / σ²)
    """
    var = std ** 2 + 1e-9
    return float(np.mean(0.5 * (np.log(2 * np.pi * var) + (y_true - mean) ** 2 / var)))


def empirical_coverage(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    k: float = 1.0,
) -> float:
    """Fraction of true values falling within the ±k·σ predictive interval.

    A well-calibrated model should achieve ≈68 % at k=1 and ≈95 % at k=2.

    Parameters
    ----------
    y_true : (N,) observed values
    mean   : (N,) predictive means
    std    : (N,) predictive standard deviations (>= 0)
    k      : half-width multiplier

    Returns
    -------
    coverage in [0, 1]
    """
    in_band = np.abs(y_true - mean) <= k * std
    return float(in_band.mean())


def compute_all_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float]:
    """Convenience wrapper returning RMSE, NLL, and coverage at 1σ / 2σ."""
    return {
        "RMSE": rmse(y_true, mean),
        "NLL": gaussian_nll(y_true, mean, std),
        "Coverage@1σ": empirical_coverage(y_true, mean, std, k=1.0),
        "Coverage@2σ": empirical_coverage(y_true, mean, std, k=2.0),
    }
