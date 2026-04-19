"""Phase 6 — Scalability profiling for BNN and SVGP models.

Loads trained checkpoints from data/processed/, reconstructs each model,
and measures:
  1. Estimated training time (s) — timed on PROBE_EPOCHS, scaled to 100.
  2. Inference latency (ms / sample) — mean over N_REPS runs.
  3. Peak RAM during inference (MB) — via tracemalloc.

Usage
-----
    python -m scripts.profile_scalability

Prerequisites
-------------
    data/processed/bnn_model.pt
    data/processed/svgp_<kernel>.pt   (for each kernel)
    data/processed/svgp_meta.pt
    (run scripts/train_bnn and scripts/train_gp first)

Outputs
-------
    data/processed/scalability_metrics.csv
"""

from __future__ import annotations

import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

import gpytorch

from src.config import PROCESSED_DATA_DIR
from src.models.bnn_vi import BNNRegressor, predict_bnn, train_bnn
from src.models.sparse_gp import KernelType, SVGPModel, predict_svgp, train_svgp

# ── Profiling hyper-parameters ───────────────────────────────────────────────
PROBE_EPOCHS = 5       # epochs used to estimate per-epoch training cost
FULL_EPOCHS = 100      # target epoch count for extrapolation
N_WARMUP = 3           # inference warm-up runs (not timed)
N_REPS = 20            # timed inference repetitions
N_SAMPLES_BNN = 200    # MC samples for BNN inference
KERNEL_TYPES: list[KernelType] = ["rbf", "periodic", "locally_periodic"]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_bnn_checkpoint(ckpt_path: Path) -> tuple[BNNRegressor, dict]:
    """Reconstruct BNNRegressor from a saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BNNRegressor(
        in_features=ckpt["in_features"],
        hidden_sizes=ckpt["hidden_sizes"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_svgp_checkpoint(
    ckpt_path: Path,
    k_type: KernelType,
    n_sensor_feats: int,
    period_scaled: float,
) -> tuple[SVGPModel, gpytorch.likelihoods.GaussianLikelihood]:
    """Reconstruct SVGPModel + likelihood from a saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model"]
    # Inducing point tensor gives us M and D without storing them explicitly
    inducing_pts = model_state["variational_strategy.inducing_points"]
    model = SVGPModel(
        inducing_pts,
        kernel_type=k_type,
        n_sensor_feats=n_sensor_feats,
        period_scaled=period_scaled,
    )
    model.load_state_dict(model_state)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.load_state_dict(ckpt["likelihood"])
    model.eval()
    likelihood.eval()
    return model, likelihood


# ── Measurement helpers ───────────────────────────────────────────────────────

def measure_inference_latency(fn, *args) -> float:
    """Mean inference wall-time in **ms per run** over N_REPS repetitions."""
    for _ in range(N_WARMUP):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        fn(*args)
    elapsed = time.perf_counter() - t0
    return elapsed / N_REPS * 1_000  # → ms


def measure_peak_ram_mb(fn, *args) -> float:
    """Peak RAM in MB allocated during a single call to fn(*args)."""
    tracemalloc.start()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


def estimate_training_time(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    model_type: str,
    **kwargs,
) -> float:
    """Probe PROBE_EPOCHS of training and scale to FULL_EPOCHS. Returns seconds."""
    if model_type == "bnn":
        t0 = time.perf_counter()
        train_bnn(X_train, y_train, n_epochs=PROBE_EPOCHS, verbose=False, **kwargs)
    elif model_type == "svgp":
        t0 = time.perf_counter()
        train_svgp(X_train, y_train, n_epochs=PROBE_EPOCHS, verbose=False, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    return (time.perf_counter() - t0) / PROBE_EPOCHS * FULL_EPOCHS


# ── Data loading (minimal — only need tensors for profiling) ──────────────────

def _load_tensors():
    """Return (X_train_bnn, y_train_bnn, X_test_bnn, X_test_gp) as tensors."""
    from sklearn.preprocessing import StandardScaler
    from src.data.loader import load_raw
    from src.config import TARGET_COL

    pf_path = PROCESSED_DATA_DIR / "pf_imputed.csv"
    if not pf_path.exists():
        sys.exit(f"[profile] {pf_path} not found. Run run_imputation first.")

    import pandas as pd

    df_pf = pd.read_csv(pf_path, index_col=0, parse_dates=True)
    T = len(df_pf)
    feat_cols = [c for c in df_pf.columns if c != TARGET_COL]

    time_raw = np.arange(T, dtype=np.float32).reshape(-1, 1)
    feats_raw = df_pf[feat_cols].values.astype(np.float32)
    y_raw = df_pf[TARGET_COL].values.astype(np.float32)

    split = int(T * 0.80)

    # BNN: scale everything including time
    X_raw_bnn = np.hstack([time_raw, feats_raw])
    scaler_bnn = StandardScaler().fit(X_raw_bnn[:split])
    X_bnn = torch.tensor(scaler_bnn.transform(X_raw_bnn).astype(np.float32))

    # GP: leave time unscaled, scale features only
    scaler_gp = StandardScaler().fit(feats_raw[:split])
    X_gp = torch.tensor(
        np.hstack([time_raw, scaler_gp.transform(feats_raw).astype(np.float32)])
    )

    y_mean = float(y_raw[:split].mean())
    y_std = float(y_raw[:split].std()) + 1e-9
    y_norm = torch.tensor(((y_raw - y_mean) / y_std).astype(np.float32))

    return (
        X_bnn[:split], y_norm[:split], X_bnn[split:],   # BNN train/test
        X_gp[:split],  y_norm[:split], X_gp[split:],    # GP  train/test
        len(feat_cols),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    bnn_ckpt = PROCESSED_DATA_DIR / "bnn_model.pt"
    meta_ckpt = PROCESSED_DATA_DIR / "svgp_meta.pt"

    missing = [p for p in [bnn_ckpt, meta_ckpt] if not p.exists()]
    for k in KERNEL_TYPES:
        p = PROCESSED_DATA_DIR / f"svgp_{k}.pt"
        if not p.exists():
            missing.append(p)
    if missing:
        sys.exit(
            "[profile] Missing checkpoints:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nRun train_bnn and train_gp first."
        )

    print("Loading data ...")
    (
        X_train_bnn, y_train_bnn, X_test_bnn,
        X_train_gp,  y_train_gp,  X_test_gp,
        n_sensor_feats,
    ) = _load_tensors()

    meta = torch.load(meta_ckpt, map_location="cpu", weights_only=False)
    n_sensor_feats = meta["n_sensor_feats"]
    period_scaled = meta["period_scaled"]

    print(f"  Train size: {len(X_train_bnn)}  Test size: {len(X_test_bnn)}\n")

    rows = []

    # ── BNN ──────────────────────────────────────────────────────────────────
    print("Profiling BNN ...")
    bnn_model, bnn_ckpt_data = load_bnn_checkpoint(bnn_ckpt)

    latency_bnn = measure_inference_latency(
        predict_bnn, bnn_model, X_test_bnn, N_SAMPLES_BNN
    )
    ram_bnn = measure_peak_ram_mb(
        predict_bnn, bnn_model, X_test_bnn, N_SAMPLES_BNN
    )
    train_time_bnn = estimate_training_time(
        X_train_bnn, y_train_bnn, "bnn",
        hidden_sizes=bnn_ckpt_data["hidden_sizes"],
    )
    ms_per_sample_bnn = latency_bnn / len(X_test_bnn)

    rows.append({
        "model": "bnn",
        "train_time_s": round(train_time_bnn, 2),
        "inference_ms_total": round(latency_bnn, 3),
        "inference_us_per_sample": round(ms_per_sample_bnn * 1000, 4),
        "peak_ram_mb": round(ram_bnn, 3),
    })
    print(
        f"  train_time={train_time_bnn:.1f}s  "
        f"latency={latency_bnn:.1f}ms  "
        f"peak_ram={ram_bnn:.1f}MB\n"
    )

    # ── SVGP ─────────────────────────────────────────────────────────────────
    for k_type in KERNEL_TYPES:
        print(f"Profiling SVGP [{k_type}] ...")
        gp_ckpt = PROCESSED_DATA_DIR / f"svgp_{k_type}.pt"
        model, likelihood = load_svgp_checkpoint(
            gp_ckpt, k_type, n_sensor_feats, period_scaled
        )

        latency_gp = measure_inference_latency(
            predict_svgp, model, likelihood, X_test_gp
        )
        ram_gp = measure_peak_ram_mb(
            predict_svgp, model, likelihood, X_test_gp
        )
        M = model.variational_strategy.inducing_points.shape[0]
        train_time_gp = estimate_training_time(
            X_train_gp, y_train_gp, "svgp",
            M=M, kernel_type=k_type,
            n_sensor_feats=n_sensor_feats, period_scaled=period_scaled,
        )
        ms_per_sample_gp = latency_gp / len(X_test_gp)

        rows.append({
            "model": f"svgp_{k_type}",
            "train_time_s": round(train_time_gp, 2),
            "inference_ms_total": round(latency_gp, 3),
            "inference_us_per_sample": round(ms_per_sample_gp * 1000, 4),
            "peak_ram_mb": round(ram_gp, 3),
        })
        print(
            f"  train_time={train_time_gp:.1f}s  "
            f"latency={latency_gp:.1f}ms  "
            f"peak_ram={ram_gp:.1f}MB\n"
        )

    df = pd.DataFrame(rows).set_index("model")
    print("=" * 75)
    print(df.to_string())
    print("=" * 75)

    out = PROCESSED_DATA_DIR / "scalability_metrics.csv"
    df.to_csv(out)
    print(f"\nScalability metrics saved -> {out}")


if __name__ == "__main__":
    main()
