"""Compare BLR vs Particle Filter imputation with visualisation.

Usage:
    python -m scripts.run_imputation

Outputs:
    data/processed/imputation_comparison_blr.png  — BLR per-feature plot
    data/processed/imputation_comparison_pf.png   — PF per-feature plot
    data/processed/pf_imputed.csv                 — PF mean imputation
    stdout                                        — RMSE comparison table
"""

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import NMHC_COL, PROCESSED_DATA_DIR, TARGET_COL
from src.data.loader import load_raw
from src.data.preprocessor import missing_summary
from src.imputation.bayesian_linear import BayesianLinearImputer
from src.imputation.particle_filter import MultivariateParticleFilter

# ── Holdout seed for reproducible RMSE evaluation ──────────────────────────
HOLDOUT_SEED = 0
HOLDOUT_FRAC = 0.10  # mask 10 % of observed non-target, non-NMHC values

EVAL_COLS = [
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]


def make_holdout_mask(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Boolean mask: True = held-out cell (was observed, now masked)."""
    rng = np.random.default_rng(seed)
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in EVAL_COLS:
        observed_idx = df.index[df[col].notna()]
        n_hold = max(1, int(len(observed_idx) * frac))
        chosen = rng.choice(len(observed_idx), size=n_hold, replace=False)
        mask.loc[observed_idx[chosen], col] = True
    return mask


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw data ────────────────────────────────────────────────────
    print("Loading data …")
    df = load_raw()
    print(f"  Shape: {df.shape}\n")

    # ── 2. Create holdout mask for quantitative comparison ──────────────────
    holdout = make_holdout_mask(df, HOLDOUT_FRAC, HOLDOUT_SEED)
    df_masked = df.copy()
    df_masked[holdout] = np.nan  # artificially hide holdout cells

    # ── 3. BLR imputation on masked data ────────────────────────────────────
    print("Running Bayesian Linear Regression imputation …")
    blr = BayesianLinearImputer()
    df_blr = blr.fit_transform(df_masked)

    # ── 4. Fit PF on bayesian output, run on masked data ──────────
    print("Fitting Particle Filter parameters …")
    pf = MultivariateParticleFilter(n_particles=1000, random_state=42)
    pf.fit(df_blr)

    print("Running Particle Filter (this takes ~30 s for N=1000, T=9357) …")
    df_pf_mean, df_pf_std = pf.run(df_blr, hide_target=True)

    # ── 5. RMSE comparison ──────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"{'Column':<22}  {'BLR RMSE':>10}  {'PF RMSE':>10}  {'Winner':>7}")
    print("-" * 62)
    blr_wins = pf_wins = 0
    for col in EVAL_COLS:
        mask = holdout[col].values
        gt = df.loc[holdout[col], col].values
        blr_pred = df_blr.loc[holdout[col], col].values
        pf_pred = df_pf_mean.loc[holdout[col], col].values
        r_blr = rmse(gt, blr_pred)
        r_pf = rmse(gt, pf_pred)
        winner = "BLR" if r_blr <= r_pf else "PF "
        blr_wins += winner.strip() == "BLR"
        pf_wins += winner.strip() == "PF"
        print(f"{col:<22}  {r_blr:>10.4f}  {r_pf:>10.4f}  {winner:>7}")
    print("=" * 62)
    print(f"BLR wins: {blr_wins}   PF wins: {pf_wins}\n")

    # ── 6. Missing-value summary after imputation ───────────────────────────
    print("Missing values after PF imputation:")
    print(missing_summary(df_pf_mean).to_string())
    print()

    # ── 7. Visualisation ────────────────────────────────────────────────────
    _plot_blr(df, df_blr)
    _plot_pf(df, df_pf_mean, df_pf_std)

    # ── 8. Save PF output ───────────────────────────────────────────────────
    out_path = PROCESSED_DATA_DIR / "pf_imputed.csv"
    df_pf_mean.to_csv(out_path)
    print(f"PF imputed data saved -> {out_path}")


def _make_grid(n_cols_data: int) -> tuple:
    """Return (fig, axes_flat, x, idx_slice) for the standard 2-week grid."""
    ncols = 3
    nrows = (n_cols_data + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.2), sharex=False)
    axes_flat = axes.flatten()
    t_end = 24 * 14
    x = np.arange(t_end)
    return fig, axes_flat, x, slice(0, t_end)


def _plot_blr(df_raw: pd.DataFrame, df_blr: pd.DataFrame) -> None:
    """One figure showing the BLR imputation for every feature."""
    cols = [c for c in df_raw.columns if c != NMHC_COL]
    fig, axes_flat, x, idx_slice = _make_grid(len(cols))

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        raw_vals = df_raw[col].iloc[idx_slice].values
        blr_vals = df_blr[col].iloc[idx_slice].values

        ax.plot(x, blr_vals, color="firebrick", lw=1.2, label="BLR")
        obs_mask = ~np.isnan(raw_vals)
        ax.scatter(
            x[obs_mask],
            raw_vals[obs_mask],
            s=4,
            color="black",
            alpha=0.5,
            label="Observed",
            zorder=5,
        )
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Hour offset", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(len(cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9, ncol=2)
    fig.suptitle(
        "Bayesian Linear Regression Imputation — first 2 weeks", fontsize=13, y=1.01
    )
    plt.tight_layout()

    out_path = PROCESSED_DATA_DIR / "imputation_comparison_blr.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"BLR plot saved -> {out_path}")


def _plot_pf(
    df_raw: pd.DataFrame,
    df_pf_mean: pd.DataFrame,
    df_pf_std: pd.DataFrame,
) -> None:
    """One figure showing the Particle Filter imputation for every feature."""
    cols = [c for c in df_raw.columns if c != NMHC_COL]
    fig, axes_flat, x, idx_slice = _make_grid(len(cols))

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        raw_vals = df_raw[col].iloc[idx_slice].values
        pf_mean = df_pf_mean[col].iloc[idx_slice].values
        pf_std = df_pf_std[col].iloc[idx_slice].values

        ax.fill_between(
            x,
            pf_mean - pf_std,
            pf_mean + pf_std,
            alpha=0.25,
            color="steelblue",
            label="PF ±1σ",
        )
        ax.plot(x, pf_mean, color="steelblue", lw=1.2, label="PF mean")
        obs_mask = ~np.isnan(raw_vals)
        ax.scatter(
            x[obs_mask],
            raw_vals[obs_mask],
            s=4,
            color="black",
            alpha=0.5,
            label="Observed",
            zorder=5,
        )
        title = col if col != TARGET_COL else f"{col}  ★ target (hidden during PF)"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Hour offset", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(len(cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9, ncol=3)
    fig.suptitle("Particle Filter Imputation — first 2 weeks", fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = PROCESSED_DATA_DIR / "imputation_comparison_pf.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"PF plot saved -> {out_path}")


if __name__ == "__main__":
    main()
