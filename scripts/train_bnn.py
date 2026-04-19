"""Train a Bayesian Neural Network for CO(GT) prediction.

Input: X = [scaled_time | scaled_PF_imputed_sensor_features] — (N, 1+F)
Target: standardised CO(GT) from Phase-2 PF output.

Usage
-----
    python -m scripts.train_bnn

Prerequisites
-------------
    data/processed/pf_imputed.csv  (run scripts/run_imputation.py first)

Outputs
-------
    data/processed/bnn_predictions.png  — prediction plot (first 2 weeks)
    data/processed/bnn_metrics.csv      — RMSE / NLL
    data/processed/bnn_model.pt         — model state dict + preprocessing meta
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
from src.models.bnn_vi import predict_bnn, train_bnn

warnings.filterwarnings("ignore", category=UserWarning)

# ── Hyper-parameters ────────────────────────────────────────────────────────
HIDDEN_SIZES = [64, 64]
PRIOR_STD = 1.0
N_EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 256
TRAIN_FRAC = 0.80
N_PRED_SAMPLES = 200
PLOT_HOURS = 336  # first 2 weeks of the test window


# ── Metrics ─────────────────────────────────────────────────────────────────


def gaussian_nll(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    var = std**2 + 1e-9
    return float(np.mean(0.5 * (np.log(2 * np.pi * var) + (y_true - mean) ** 2 / var)))


# ── Data preparation (mirrors train_gp.py) ──────────────────────────────────


def prepare_data(train_frac: float = TRAIN_FRAC):
    pf_path = PROCESSED_DATA_DIR / "pf_imputed.csv"
    if not pf_path.exists():
        sys.exit(
            f"[train_bnn] PF output not found at {pf_path}.\n"
            "Run `python -m scripts.run_imputation` first."
        )

    df_pf = pd.read_csv(pf_path, index_col=0, parse_dates=True)
    df_raw = load_raw()
    T = len(df_pf)

    feat_cols = [c for c in df_pf.columns if c != TARGET_COL]
    n_sensor_feats = len(feat_cols)

    time_raw = np.arange(T, dtype=np.float32).reshape(-1, 1)
    feats_raw = df_pf[feat_cols].values.astype(np.float32)
    X_raw = np.hstack([time_raw, feats_raw])
    y_raw = df_pf[TARGET_COL].values.astype(np.float32)

    split = int(T * train_frac)

    X_scaler = StandardScaler()
    X_scaler.fit(X_raw[:split])
    X_scaled = X_scaler.transform(X_raw).astype(np.float32)

    y_mean = float(y_raw[:split].mean())
    y_std = float(y_raw[:split].std()) + 1e-9
    y_norm = ((y_raw - y_mean) / y_std).astype(np.float32)

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
        n_sensor_feats,
        split,
        X_scaler,
        feat_cols,
    ) = prepare_data()

    raw_co_test = raw_co.iloc[split:]
    eval_mask = raw_co_test.notna().values
    y_true_eval = raw_co_test[eval_mask].values

    print(
        f"  Train: {len(X_train)}  |  Test: {len(X_test)}"
        f"  ({eval_mask.sum()} observed CO points)\n"
        f"  Input dims: {X_train.shape[1]}  Sensor features: {n_sensor_feats}\n"
    )

    print("Training BNN ...")
    model = train_bnn(
        X_train,
        y_train,
        hidden_sizes=HIDDEN_SIZES,
        prior_std=PRIOR_STD,
        n_epochs=N_EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        verbose=True,
    )

    print(f"\nGenerating predictions ({N_PRED_SAMPLES} MC samples) ...")
    mean_norm, std_norm = predict_bnn(model, X_test, n_samples=N_PRED_SAMPLES)

    mean_orig = mean_norm.numpy() * y_std + y_mean
    std_orig = std_norm.numpy() * y_std

    mean_eval = mean_orig[eval_mask]
    std_eval = std_orig[eval_mask]
    rmse = float(np.sqrt(np.mean((y_true_eval - mean_eval) ** 2)))
    nll = gaussian_nll(y_true_eval, mean_eval, std_eval)

    df_metrics = pd.DataFrame([{"model": "BNN", "RMSE": rmse, "NLL": nll}]).set_index(
        "model"
    )
    print("=" * 40)
    print(df_metrics.to_string(float_format="%.4f"))
    print("=" * 40)
    df_metrics.to_csv(PROCESSED_DATA_DIR / "bnn_metrics.csv")

    # Calibration threshold: 90th percentile of training-set predictive std
    print("Computing calibration threshold on training set ...")
    mean_tr_n, std_tr_n = predict_bnn(model, X_train, n_samples=100)
    std_tr_orig = std_tr_n.numpy() * y_std
    calib_threshold = float(np.percentile(std_tr_orig, 90.0))
    print(f"  Calibration threshold (90th pct of train std): {calib_threshold:.4f} mg/m3")

    # Save model checkpoint with preprocessing metadata and scaler
    ckpt_path = PROCESSED_DATA_DIR / "bnn_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden_sizes": HIDDEN_SIZES,
            "in_features": X_train.shape[1],
            "y_mean": y_mean,
            "y_std": y_std,
            "n_sensor_feats": n_sensor_feats,
            "feat_cols": feat_cols,
            "x_scaler_mean": X_scaler.mean_,
            "x_scaler_scale": X_scaler.scale_,
            "calib_threshold": calib_threshold,
        },
        ckpt_path,
    )
    print(f"Model saved -> {ckpt_path}")

    _plot_predictions(mean_orig, std_orig, raw_co, split)


def _plot_predictions(
    mean: np.ndarray,
    std: np.ndarray,
    raw_co: pd.Series,
    split: int,
) -> None:
    t_test = np.arange(split, split + len(mean))
    plot_n = min(PLOT_HOURS, len(t_test))
    t_plot = t_test[:plot_n]

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.fill_between(
        t_plot,
        mean[:plot_n] - std[:plot_n],
        mean[:plot_n] + std[:plot_n],
        alpha=0.25,
        color="darkorange",
        label="$\\pm 1\\sigma$",
    )
    ax.plot(t_plot, mean[:plot_n], color="darkorange", lw=1.2, label="BNN mean")

    obs_mask = raw_co.iloc[split : split + plot_n].notna()
    t_obs = t_plot[obs_mask.values]
    y_obs = raw_co.iloc[split : split + plot_n][obs_mask].values
    ax.scatter(
        t_obs, y_obs, s=4, color="black", alpha=0.6, label="Observed CO(GT)", zorder=5
    )

    ax.set_xlabel("Hour index", fontsize=9)
    ax.set_ylabel("CO(GT) [mg/m3]", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("BNN prediction — first 2 weeks of test set", fontsize=10)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = PROCESSED_DATA_DIR / "bnn_predictions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {out}")


if __name__ == "__main__":
    main()
