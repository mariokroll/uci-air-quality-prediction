"""Uncertainty-based decision policy for flagging unreliable predictions.

A prediction is flagged for human review when the predictive std exceeds a
threshold.  The threshold can be:

* Fixed (absolute, in the target's original units).
* Percentile-based: set to the p-th percentile of the training-set std
  distribution so that at most (1-p/100) fraction of test points are flagged.

Usage
-----
    policy = UncertaintyPolicy(threshold=0.5)          # fixed
    policy = UncertaintyPolicy.from_percentile(        # percentile-based
        train_std, percentile=90
    )
    flags = policy.flag(test_std)                      # (N,) bool array
    report = policy.summary(y_true, mean, test_std)    # dict with metrics
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_all_metrics


class UncertaintyPolicy:
    """Flag predictions whose predictive std exceeds a threshold.

    Parameters
    ----------
    threshold : std threshold in the target's original units.
                Predictions with std > threshold are flagged.
    """

    def __init__(self, threshold: float) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self.threshold = float(threshold)

    @classmethod
    def from_percentile(
        cls,
        std_values: np.ndarray,
        percentile: float = 90.0,
    ) -> "UncertaintyPolicy":
        """Set threshold to the p-th percentile of `std_values`.

        Parameters
        ----------
        std_values  : array of predictive stds (e.g., from the training set
                      or a held-out calibration set)
        percentile  : value in (0, 100); higher means fewer flags
        """
        if not (0 < percentile < 100):
            raise ValueError(f"percentile must be in (0, 100), got {percentile}")
        threshold = float(np.percentile(std_values, percentile))
        return cls(threshold)

    def flag(self, std: np.ndarray) -> np.ndarray:
        """Return a boolean mask — True where std > threshold."""
        return std > self.threshold

    def flagged_fraction(self, std: np.ndarray) -> float:
        """Fraction of predictions flagged."""
        return float(self.flag(std).mean())

    def summary(
        self,
        y_true: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> dict[str, float]:
        """Full evaluation report split into flagged and accepted subsets.

        Returns
        -------
        dict with keys:
            threshold          — the policy threshold
            flagged_fraction   — fraction of points flagged
            accepted_RMSE      — RMSE on accepted (low-uncertainty) points
            accepted_NLL       — NLL on accepted points
            accepted_Coverage@1σ / @2σ
            flagged_RMSE       — RMSE on flagged points (if any)
        """
        flags = self.flag(std)
        accepted = ~flags

        report: dict[str, float] = {
            "threshold": self.threshold,
            "flagged_fraction": float(flags.mean()),
        }

        if accepted.any():
            acc_metrics = compute_all_metrics(
                y_true[accepted], mean[accepted], std[accepted]
            )
            for k, v in acc_metrics.items():
                report[f"accepted_{k}"] = v

        if flags.any():
            report["flagged_RMSE"] = float(
                np.sqrt(np.mean((y_true[flags] - mean[flags]) ** 2))
            )

        return report
