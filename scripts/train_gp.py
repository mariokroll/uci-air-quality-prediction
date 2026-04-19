"""Train and compare SVGP models with three kernel variants.

Input: X = [scaled_time | scaled_PF_imputed_sensor_features] — (N, 1+F)
Target: standardised CO(GT) from Phase-2 PF output.

Usage
-----
    python -m scripts.train_gp

Prerequisites
-------------
    data/processed/pf_imputed.csv  (run scripts/run_imputation.py first)

Outputs
-------
    data/processed/gp_predictions.png        — per-kernel prediction plot
    data/processed/gp_metrics.csv            — RMSE / NLL comparison table
    data/processed/svgp_<kernel>.pt          — saved model + likelihood state dicts
    data/processed/svgp_meta.pt              — y_mean, y_std, n_sensor_feats, period_scaled
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DATA_DIR, TARGET_COL
from src.data.loader import load_raw
from src.models.sparse_gp import KernelType, predict_svgp, train_svgp

warnings.filterwarnings("ignore", category=UserWarning)

# ── Hyper-parameters ────────────────────────────────────────────────────────
M_INDUCING = 200
N_EPOCHS = 100
LR = 0.01
BATCH_SIZE = 256
TRAIN_FRAC = 0.80
KERNEL_TYPES: list[KernelType] = ["rbf", "periodic", "locally_periodic"]


# ── Metrics ─────────────────────────────────────────────────────────────────


def gaussian_nll(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    var = std**2 + 1e-9
    return float(np.mean(0.5 * (np.log(2 * np.pi * var) + (y_true - mean) ** 2 / var)))


# ── Data preparation ─────────────────────────────────────────────────────────


def prepare_data(train_frac: float = TRAIN_FRAC):
    """Build the multi-dimensional input matrix and standardise it.

    X layout
    --------
    Col 0   : time index in hours (0, 1, …, T-1), then StandardScaled
    Cols 1-F: PF-imputed sensor features (all columns except CO(GT)), scaled

    Returns
    -------
    X_train, X_test, y_train, y_test : tensors in scaled / normalised space
    y_mean, y_std                    : floats for denormalisation
    raw_co                           : original CO(GT) series (for eval)
    period_scaled                    : 24 h in scaled time units
    n_sensor_feats                   : F
    split                            : integer split index
    """
    pf_path = PROCESSED_DATA_DIR / "pf_imputed.csv"
    if not pf_path.exists():
        sys.exit(
            f"[train_gp] PF output not found at {pf_path}.\n"
            "Run `python -m scripts.run_imputation` first."
        )

    df_pf = pd.read_csv(pf_path, index_col=0, parse_dates=True)
    df_raw = load_raw()
    T = len(df_pf)

    # Sensor feature columns: everything except CO(GT)
    feat_cols = [c for c in df_pf.columns if c != TARGET_COL]
    n_sensor_feats = len(feat_cols)

    # Raw arrays
    time_raw = np.arange(T, dtype=np.float32).reshape(-1, 1)
    feats_raw = df_pf[feat_cols].values.astype(np.float32)
    y_raw = df_pf[TARGET_COL].values.astype(np.float32)

    split = int(T * train_frac)

    # Fit StandardScaler on training data only → prevents test-set leakage
    X_scaler = StandardScaler()
    X_scaler.fit(feats_raw[:split])
    X_scaled = X_scaler.transform(feats_raw).astype(np.float32)

    # Standardise y using training statistics
    y_mean = float(y_raw[:split].mean())
    y_std = float(y_raw[:split].std()) + 1e-9
    y_norm = ((y_raw - y_mean) / y_std).astype(np.float32)

    # Period in scaled time units: 24 h / (std of raw time column)

    X_scaled = np.hstack([time_raw, X_scaled])
    period_scaled = 24.0

    X = torch.tensor(X_scaled)
    y = torch.tensor(y_norm)

    return (
        X[:split],
        X[split:],
        y[:split],
        y[split:],
        y_mean,
        y_std,
        df_raw[TARGET_COL],
        period_scaled,
        n_sensor_feats,
        split,
        X_scaler,
        feat_cols,
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing data ...")
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
        X_scaler,
        feat_cols,
    ) = prepare_data()

    # Observed CO in test window (for ground-truth RMSE / NLL)
    raw_co_test = raw_co.iloc[split:]
    eval_mask = raw_co_test.notna().values
    y_true_eval = raw_co_test[eval_mask].values

    print(
        f"  Train: {len(X_train)}  |  Test: {len(X_test)}"
        f"  ({eval_mask.sum()} observed CO points)\n"
        f"  Input dims: {X_train.shape[1]}  "
        f"  Sensor features: {n_sensor_feats}  "
        f"  period_scaled = {period_scaled:.5f}\n"
    )

    results = []
    all_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for k_type in KERNEL_TYPES:
        print(f"Training kernel: {k_type}")
        model, likelihood = train_svgp(
            X_train,
            y_train,
            M=M_INDUCING,
            n_epochs=N_EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            kernel_type=k_type,
            n_sensor_feats=n_sensor_feats,
            period_scaled=period_scaled,
            verbose=True,
        )

        mean_norm, std_norm = predict_svgp(model, likelihood, X_test)

        # Denormalise predictions
        mean_orig = mean_norm.numpy() * y_std + y_mean
        std_orig = std_norm.numpy() * y_std

        all_predictions[k_type] = (mean_orig, std_orig)

        # Save model checkpoint
        ckpt_path = PROCESSED_DATA_DIR / f"svgp_{k_type}.pt"
        torch.save(
            {"model": model.state_dict(), "likelihood": likelihood.state_dict()},
            ckpt_path,
        )
        print(f"  Saved model -> {ckpt_path}")

        # Calibration threshold for the frontend (locally_periodic only)
        if k_type == "locally_periodic":
            print("  Computing calibration threshold on training set ...")
            _, std_tr_n = predict_svgp(model, likelihood, X_train)
            std_tr_orig = std_tr_n.numpy() * y_std
            lp_calib_threshold = float(np.percentile(std_tr_orig, 90.0))
            print(f"  Calibration threshold (90th pct): {lp_calib_threshold:.4f} mg/m3")

        # Evaluate on originally-observed CO values only
        mean_eval = mean_orig[eval_mask]
        std_eval = std_orig[eval_mask]
        rmse = float(np.sqrt(np.mean((y_true_eval - mean_eval) ** 2)))
        nll = gaussian_nll(y_true_eval, mean_eval, std_eval)

        results.append({"kernel": k_type, "RMSE": rmse, "NLL": nll})
        print(f"  -> RMSE={rmse:.4f}  NLL={nll:.4f}\n")

    # Summary table
    df_metrics = pd.DataFrame(results).set_index("kernel")
    print("=" * 50)
    print(df_metrics.to_string(float_format="%.4f"))
    print("=" * 50)
    df_metrics.to_csv(PROCESSED_DATA_DIR / "gp_metrics.csv")

    # Save preprocessing metadata for inference-time denormalisation
    meta_path = PROCESSED_DATA_DIR / "svgp_meta.pt"
    torch.save(
        {
            "y_mean": y_mean,
            "y_std": y_std,
            "n_sensor_feats": n_sensor_feats,
            "period_scaled": period_scaled,
            "feat_cols": feat_cols,
            "x_scaler_mean": X_scaler.mean_,    # shape (F,) — features only
            "x_scaler_scale": X_scaler.scale_,  # shape (F,)
            "calib_threshold": lp_calib_threshold,
        },
        meta_path,
    )
    print(f"Saved preprocessing meta -> {meta_path}")

    _plot_predictions(X_test, all_predictions, raw_co, split)


def _plot_predictions(
    X_test: torch.Tensor,
    all_predictions: dict,
    raw_co: pd.Series,
    split: int,
) -> None:
    """Three-panel figure: one row per kernel, shared time axis."""
    colors = {
        "rbf": "steelblue",
        "periodic": "firebrick",
        "locally_periodic": "seagreen",
    }
    # Time axis uses the raw hour index stored in the original (unscaled) row
    t_test = np.arange(split, split + len(X_test))

    # Restrict plot to first 2 weeks (336 hours) of the test window
    PLOT_HOURS = 336
    plot_n = min(PLOT_HOURS, len(t_test))
    t_plot = t_test[:plot_n]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, k_type in zip(axes, KERNEL_TYPES):
        mean, std = all_predictions[k_type]
        col = colors[k_type]

        ax.fill_between(
            t_plot,
            mean[:plot_n] - std[:plot_n],
            mean[:plot_n] + std[:plot_n],
            alpha=0.25,
            color=col,
            label="$\\pm 1\\sigma$",
        )
        ax.plot(t_plot, mean[:plot_n], color=col, lw=1.2, label=k_type)

        obs_mask = raw_co.iloc[split : split + plot_n].notna()
        t_obs = t_plot[obs_mask.values]
        y_obs = raw_co.iloc[split : split + plot_n][obs_mask].values
        ax.scatter(
            t_obs,
            y_obs,
            s=4,
            color="black",
            alpha=0.6,
            label="Observed CO(GT)",
            zorder=5,
        )

        ax.set_ylabel("CO(GT) [mg/m3]", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Kernel: {k_type}", fontsize=10)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Hour index", fontsize=9)
    fig.suptitle("SVGP kernel comparison — first 2 weeks of test set", fontsize=12)
    plt.tight_layout()

    out = PROCESSED_DATA_DIR / "gp_predictions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {out}")


if __name__ == "__main__":
    main()
