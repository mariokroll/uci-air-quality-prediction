"""Phase 5 — Unified evaluation of SVGP and BNN models.

Trains all four models (SVGP ×3 kernels + BNN) on the same train/test split,
computes RMSE / NLL / Coverage@1σ / Coverage@2σ, applies an uncertainty-based
decision policy, and writes comparison plots and a metrics CSV.

Usage
-----
    python -m scripts.evaluate

Prerequisites
-------------
    data/processed/pf_imputed.csv  (run scripts/run_imputation.py first)

Outputs
-------
    data/processed/eval_metrics.csv         — full comparison table
    data/processed/eval_predictions.png     — 4-panel prediction plot (2 weeks)
    data/processed/eval_decision_policy.png — uncertainty + flagged points
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
from src.evaluation.decision_policy import UncertaintyPolicy
from src.evaluation.metrics import compute_all_metrics
from src.models.bnn_vi import predict_bnn, train_bnn
from src.models.sparse_gp import KernelType, predict_svgp, train_svgp

warnings.filterwarnings("ignore", category=UserWarning)

# ── Hyper-parameters ────────────────────────────────────────────────────────
TRAIN_FRAC = 0.80
PLOT_HOURS = 336  # first 2 weeks of test window

# SVGP
M_INDUCING = 200
GP_EPOCHS = 100
GP_LR = 0.01
GP_BATCH = 256
KERNEL_TYPES: list[KernelType] = ["rbf", "periodic", "locally_periodic"]

# BNN
BNN_HIDDEN = [64, 64]
BNN_EPOCHS = 100
BNN_LR = 1e-3
BNN_BATCH = 256
BNN_SAMPLES = 200

# Decision policy: flag top 10% most uncertain predictions
POLICY_PERCENTILE = 90.0


# ── Data preparation ─────────────────────────────────────────────────────────


def prepare_data(train_frac: float = TRAIN_FRAC):
    """Shared data prep for all models.

    X layout: col 0 = raw time index (hours); cols 1..F = StandardScaled
    sensor features.  Time is left unscaled so period_scaled = 24.0.
    """
    pf_path = PROCESSED_DATA_DIR / "pf_imputed.csv"
    if not pf_path.exists():
        sys.exit(
            f"[evaluate] PF output not found at {pf_path}.\n"
            "Run `python -m scripts.run_imputation` first."
        )

    df_pf = pd.read_csv(pf_path, index_col=0, parse_dates=True)
    df_raw = load_raw()
    T = len(df_pf)

    feat_cols = [c for c in df_pf.columns if c != TARGET_COL]
    n_sensor_feats = len(feat_cols)

    time_raw = np.arange(T, dtype=np.float32).reshape(-1, 1)
    feats_raw = df_pf[feat_cols].values.astype(np.float32)
    y_raw = df_pf[TARGET_COL].values.astype(np.float32)

    split = int(T * train_frac)

    feat_scaler = StandardScaler()
    feat_scaler.fit(feats_raw[:split])
    X_scaled = np.hstack(
        [time_raw, feat_scaler.transform(feats_raw).astype(np.float32)]
    )

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
    )


def prepare_data_bnn(train_frac: float = TRAIN_FRAC):
    """
    Same as prepare_data() but with time column included in scaling (mirrors train_bnn.py)."""
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
    ) = prepare_data()

    raw_co_test = raw_co.iloc[split:]
    eval_mask = raw_co_test.notna().values
    y_true_eval = raw_co_test[eval_mask].values

    print(
        f"  Train: {len(X_train)}  |  Test: {len(X_test)}"
        f"  ({eval_mask.sum()} observed CO points for evaluation)\n"
    )

    all_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ── SVGP ────────────────────────────────────────────────────────────────
    for k_type in KERNEL_TYPES:
        print(f"Training SVGP [{k_type}] ...")
        model, likelihood = train_svgp(
            X_train,
            y_train,
            M=M_INDUCING,
            n_epochs=GP_EPOCHS,
            lr=GP_LR,
            batch_size=GP_BATCH,
            kernel_type=k_type,
            n_sensor_feats=n_sensor_feats,
            period_scaled=24.0,
            verbose=True,
        )
        mean_n, std_n = predict_svgp(model, likelihood, X_test)
        mean_o = mean_n.numpy() * y_std + y_mean
        std_o = std_n.numpy() * y_std
        all_preds[f"svgp_{k_type}"] = (mean_o, std_o)
        m = compute_all_metrics(y_true_eval, mean_o[eval_mask], std_o[eval_mask])
        print(
            f"  -> RMSE={m['RMSE']:.4f}  NLL={m['NLL']:.4f}"
            f"  Cov@1σ={m['Coverage@1σ']:.3f}  Cov@2σ={m['Coverage@2σ']:.3f}\n"
        )

    # ── BNN ─────────────────────────────────────────────────────────────────
    print("Training BNN ...")
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
    ) = prepare_data_bnn()  # time column included in scaling for BNN
    bnn_model = train_bnn(
        X_train,
        y_train,
        hidden_sizes=BNN_HIDDEN,
        n_epochs=BNN_EPOCHS,
        lr=BNN_LR,
        batch_size=BNN_BATCH,
        verbose=True,
    )
    mean_n, std_n = predict_bnn(bnn_model, X_test, n_samples=BNN_SAMPLES)
    mean_o = mean_n.numpy() * y_std + y_mean
    std_o = std_n.numpy() * y_std
    all_preds["bnn"] = (mean_o, std_o)
    m = compute_all_metrics(y_true_eval, mean_o[eval_mask], std_o[eval_mask])
    print(
        f"  -> RMSE={m['RMSE']:.4f}  NLL={m['NLL']:.4f}"
        f"  Cov@1σ={m['Coverage@1σ']:.3f}  Cov@2σ={m['Coverage@2σ']:.3f}\n"
    )

    # ── Metrics table ────────────────────────────────────────────────────────
    rows = []
    for name, (mean_o, std_o) in all_preds.items():
        m = compute_all_metrics(y_true_eval, mean_o[eval_mask], std_o[eval_mask])
        rows.append({"model": name, **m})

    df_metrics = pd.DataFrame(rows).set_index("model")
    print("=" * 65)
    print(df_metrics.to_string(float_format="%.4f"))
    print("=" * 65)
    df_metrics.to_csv(PROCESSED_DATA_DIR / "eval_metrics.csv")

    # ── Decision policy (applied to best model by RMSE) ─────────────────────
    best_name = df_metrics["RMSE"].idxmin()
    best_mean, best_std = all_preds[best_name]
    print(
        f"\nDecision policy on best model: {best_name}"
        f"  (threshold = {POLICY_PERCENTILE}th percentile of test std)"
    )

    policy = UncertaintyPolicy.from_percentile(
        best_std[eval_mask], percentile=POLICY_PERCENTILE
    )
    report = policy.summary(y_true_eval, best_mean[eval_mask], best_std[eval_mask])
    print(
        f"  threshold={report['threshold']:.4f}"
        f"  flagged={report['flagged_fraction']:.1%}"
        f"  accepted_RMSE={report.get('accepted_RMSE', float('nan')):.4f}"
    )

    # ── Plots ────────────────────────────────────────────────────────────────
    _plot_predictions(all_preds, raw_co, split)
    _plot_decision_policy(best_mean, best_std, raw_co, split, policy, best_name)


# ── Plotting ─────────────────────────────────────────────────────────────────

_COLORS = {
    "svgp_rbf": "steelblue",
    "svgp_periodic": "firebrick",
    "svgp_locally_periodic": "seagreen",
    "bnn": "darkorange",
}


def _plot_predictions(
    all_preds: dict[str, tuple[np.ndarray, np.ndarray]],
    raw_co: pd.Series,
    split: int,
) -> None:
    n_models = len(all_preds)
    t_test = np.arange(split, split + len(next(iter(all_preds.values()))[0]))
    plot_n = min(PLOT_HOURS, len(t_test))
    t_plot = t_test[:plot_n]

    fig, axes = plt.subplots(n_models, 1, figsize=(14, 3.5 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    obs_mask = raw_co.iloc[split : split + plot_n].notna()
    t_obs = t_plot[obs_mask.values]
    y_obs = raw_co.iloc[split : split + plot_n][obs_mask].values

    for ax, (name, (mean, std)) in zip(axes, all_preds.items()):
        col = _COLORS.get(name, "purple")
        ax.fill_between(
            t_plot,
            mean[:plot_n] - std[:plot_n],
            mean[:plot_n] + std[:plot_n],
            alpha=0.25,
            color=col,
            label="$\\pm 1\\sigma$",
        )
        ax.plot(t_plot, mean[:plot_n], color=col, lw=1.2, label=name)
        ax.scatter(
            t_obs, y_obs, s=4, color="black", alpha=0.6, label="Observed", zorder=5
        )
        ax.set_ylabel("CO(GT) [mg/m3]", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Hour index", fontsize=9)
    fig.suptitle("Model comparison — first 2 weeks of test set", fontsize=11)
    plt.tight_layout()
    out = PROCESSED_DATA_DIR / "eval_predictions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction plot saved -> {out}")


def _plot_decision_policy(
    mean: np.ndarray,
    std: np.ndarray,
    raw_co: pd.Series,
    split: int,
    policy: UncertaintyPolicy,
    model_name: str,
) -> None:
    t_test = np.arange(split, split + len(mean))
    plot_n = min(PLOT_HOURS, len(t_test))
    t_plot = t_test[:plot_n]
    m_plot = mean[:plot_n]
    s_plot = std[:plot_n]

    flags = policy.flag(s_plot)

    obs_mask = raw_co.iloc[split : split + plot_n].notna()
    t_obs = t_plot[obs_mask.values]
    y_obs = raw_co.iloc[split : split + plot_n][obs_mask].values

    fig, (ax_pred, ax_std) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Top panel: predictions with flagged regions shaded
    col = _COLORS.get(model_name, "purple")
    ax_pred.fill_between(
        t_plot,
        m_plot - s_plot,
        m_plot + s_plot,
        alpha=0.2,
        color=col,
        label="$\\pm 1\\sigma$",
    )
    ax_pred.plot(t_plot, m_plot, color=col, lw=1.2, label=model_name)
    ax_pred.scatter(
        t_obs, y_obs, s=4, color="black", alpha=0.7, label="Observed", zorder=5
    )
    # Shade flagged time steps
    for i, flagged in enumerate(flags):
        if flagged:
            ax_pred.axvspan(
                t_plot[i] - 0.5, t_plot[i] + 0.5, alpha=0.15, color="red", linewidth=0
            )
    ax_pred.set_ylabel("CO(GT) [mg/m3]", fontsize=9)
    ax_pred.set_title(
        f"Decision policy ({model_name}) — flagged={flags.mean():.1%}"
        f"  threshold={policy.threshold:.3f} mg/m3",
        fontsize=9,
    )
    ax_pred.legend(loc="upper right", fontsize=8)

    # Bottom panel: predictive std with threshold line
    ax_std.plot(t_plot, s_plot, color=col, lw=0.8, label="Predictive std")
    ax_std.axhline(
        policy.threshold,
        color="red",
        lw=1.2,
        ls="--",
        label=f"Threshold ({policy.threshold:.3f})",
    )
    ax_std.scatter(
        t_plot[flags], s_plot[flags], s=12, color="red", zorder=5, label="Flagged"
    )
    ax_std.set_ylabel("Predictive std [mg/m3]", fontsize=9)
    ax_std.set_xlabel("Hour index", fontsize=9)
    ax_std.legend(loc="upper right", fontsize=8)
    ax_std.tick_params(labelsize=8)

    plt.tight_layout()
    out = PROCESSED_DATA_DIR / "eval_decision_policy.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Decision policy plot saved -> {out}")


if __name__ == "__main__":
    main()
