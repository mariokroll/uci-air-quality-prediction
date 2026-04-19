"""Phase 7 — Interactive Streamlit frontend for probabilistic CO(GT) prediction.

Supports two models selectable via toggle buttons at the top of the page:
  • BNN  — Bayesian Neural Network (Bayes by Backprop)
  • GP   — Sparse Variational GP with Locally-Periodic kernel

Both models display:
  - Predicted CO(GT) with ±1σ confidence interval
  - Epistemic vs. Aleatoric uncertainty bar chart
  - Decision-policy warning when total std > calibration threshold

Usage
-----
    streamlit run src/frontend/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import streamlit as st

# ── Project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import PROCESSED_DATA_DIR
from src.models.bnn_vi import BNNRegressor, predict_bnn_decomposed
from src.models.sparse_gp import SVGPModel, predict_svgp_decomposed
from src.evaluation.decision_policy import UncertaintyPolicy
import gpytorch

# ── Paths ─────────────────────────────────────────────────────────────────────
BNN_CKPT = PROCESSED_DATA_DIR / "bnn_model.pt"
GP_CKPT  = PROCESSED_DATA_DIR / "svgp_locally_periodic.pt"
GP_META  = PROCESSED_DATA_DIR / "svgp_meta.pt"

# ── Feature metadata (UCI Air Quality dataset typical ranges) ─────────────────
_FEATURE_RANGES: dict[str, tuple[float, float, float]] = {
    "PT08.S1(CO)":   (500.0,  2800.0, 1350.0),
    "NMHC(GT)":      (0.0,    1500.0,  200.0),
    "C6H6(GT)":      (0.0,    63.0,    7.5),
    "PT08.S2(NMHC)": (300.0,  2400.0,  950.0),
    "NOx(GT)":       (0.0,    1600.0,  250.0),
    "PT08.S3(NOx)":  (300.0,  2700.0,  800.0),
    "NO2(GT)":       (0.0,    500.0,   110.0),
    "PT08.S4(NO2)":  (500.0,  2900.0,  1450.0),
    "PT08.S5(O3)":   (200.0,  2800.0,  950.0),
    "T":             (-5.0,   45.0,    18.0),
    "RH":            (5.0,    100.0,   55.0),
    "AH":            (0.2,    2.5,     1.0),
}

_UNITS: dict[str, str] = {
    "PT08.S1(CO)":   "sensor units",
    "NMHC(GT)":      "mg/m³",
    "C6H6(GT)":      "µg/m³",
    "PT08.S2(NMHC)": "sensor units",
    "NOx(GT)":       "ppb",
    "PT08.S3(NOx)":  "sensor units",
    "NO2(GT)":       "µg/m³",
    "PT08.S4(NO2)":  "sensor units",
    "PT08.S5(O3)":   "sensor units",
    "T":             "°C",
    "RH":            "%",
    "AH":            "g/m³",
}

N_MC_SAMPLES = 200  # BNN Monte-Carlo forward passes


# ── Checkpoint loaders (cached across reruns) ─────────────────────────────────

@st.cache_resource(show_spinner="Loading BNN …")
def load_bnn():
    if not BNN_CKPT.exists():
        return None, None
    ckpt = torch.load(BNN_CKPT, map_location="cpu", weights_only=False)
    model = BNNRegressor(
        in_features=ckpt["in_features"],
        hidden_sizes=ckpt["hidden_sizes"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@st.cache_resource(show_spinner="Loading GP …")
def load_gp():
    if not GP_CKPT.exists() or not GP_META.exists():
        return None, None, None
    meta = torch.load(GP_META, map_location="cpu", weights_only=False)
    ckpt = torch.load(GP_CKPT, map_location="cpu", weights_only=False)
    model_state = ckpt["model"]
    inducing_pts = model_state["variational_strategy.inducing_points"]
    model = SVGPModel(
        inducing_pts,
        kernel_type="locally_periodic",
        n_sensor_feats=meta["n_sensor_feats"],
        period_scaled=meta["period_scaled"],
    )
    model.load_state_dict(model_state)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.load_state_dict(ckpt["likelihood"])
    model.eval()
    likelihood.eval()
    return model, likelihood, meta


# ── Input preprocessing ───────────────────────────────────────────────────────

def _scale_bnn(hour_index: float, feat_values: list[float], ckpt: dict) -> torch.Tensor:
    """BNN: scaler was fit on [time | features] together."""
    raw = np.array([hour_index] + feat_values, dtype=np.float32)
    scaled = (raw - ckpt["x_scaler_mean"]) / ckpt["x_scaler_scale"]
    return torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, 1+F)


def _scale_gp(hour_index: float, feat_values: list[float], meta: dict) -> torch.Tensor:
    """GP: scaler was fit on features only; time is prepended as raw hours."""
    feats = np.array(feat_values, dtype=np.float32)
    feats_scaled = (feats - meta["x_scaler_mean"]) / meta["x_scaler_scale"]
    raw = np.concatenate([[hour_index], feats_scaled]).astype(np.float32)
    return torch.tensor(raw, dtype=torch.float32).unsqueeze(0)  # (1, 1+F)


# ── Shared results renderer ───────────────────────────────────────────────────

def _render_results(
    pred_mean: float,
    pred_total_std: float,
    pred_epi_std: float,
    pred_ale_std: float,
    calib_threshold: float,
) -> None:
    """Render the prediction metric, table, bar chart, and policy banner."""
    import matplotlib.pyplot as plt

    flagged = pred_total_std > calib_threshold

    res_col, chart_col = st.columns([1, 1])

    with res_col:
        st.metric(
            label="Predicted CO(GT)",
            value=f"{pred_mean:.3f} mg/m³",
            delta=f"±{pred_total_std:.3f} mg/m³  (1σ)",
            delta_color="off",
        )
        st.markdown(
            f"""
| Component | Value |
|-----------|-------|
| **Epistemic std** | {pred_epi_std:.4f} mg/m³ |
| **Aleatoric std** | {pred_ale_std:.4f} mg/m³ |
| **Total std** | {pred_total_std:.4f} mg/m³ |
| **Threshold** | {calib_threshold:.4f} mg/m³ |
"""
        )
        if flagged:
            st.warning(
                "⚠️ **Prediction Flagged for Human Review due to High Uncertainty.**\n\n"
                f"Total std ({pred_total_std:.4f}) exceeds the calibration threshold "
                f"({calib_threshold:.4f} mg/m³). Treat this estimate with caution.",
                icon="⚠️",
            )
        else:
            st.success(
                f"✅ Prediction accepted "
                f"(std {pred_total_std:.4f} ≤ threshold {calib_threshold:.4f}).",
                icon="✅",
            )

    with chart_col:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        components = ["Epistemic", "Aleatoric", "Total"]
        values     = [pred_epi_std, pred_ale_std, pred_total_std]
        colors     = ["#4e8df5", "#f5814e", "#888888"]
        bars = ax.bar(components, values, color=colors, width=0.5, edgecolor="white")
        ax.axhline(
            calib_threshold, color="red", lw=1.5, ls="--",
            label=f"Threshold ({calib_threshold:.3f})",
        )
        ax.set_ylabel("Standard deviation [mg/m³]", fontsize=9)
        ax.set_title("Uncertainty decomposition", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8,
            )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        f"95% confidence interval: "
        f"[{pred_mean - 2*pred_total_std:.3f}, {pred_mean + 2*pred_total_std:.3f}] mg/m³"
    )


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌫️",
    layout="wide",
)

st.title("🌫️ Air Quality CO(GT) Predictor")
st.markdown(
    "Enter current sensor readings, choose a model, and click **Predict** to obtain "
    "a probabilistic CO concentration estimate with full uncertainty breakdown."
)

# ── Model toggle ──────────────────────────────────────────────────────────────
st.subheader("Select model")
btn_col1, btn_col2 = st.columns(2)

if "model_mode" not in st.session_state:
    st.session_state.model_mode = "bnn"

if btn_col1.button(
    "🧠  BNN — Bayesian Neural Network",
    use_container_width=True,
    type="primary" if st.session_state.model_mode == "bnn" else "secondary",
):
    st.session_state.model_mode = "bnn"
    st.rerun()

if btn_col2.button(
    "📈  GP — Locally Periodic SVGP",
    use_container_width=True,
    type="primary" if st.session_state.model_mode == "gp" else "secondary",
):
    st.session_state.model_mode = "gp"
    st.rerun()

mode = st.session_state.model_mode

# ── Load selected model ───────────────────────────────────────────────────────
if mode == "bnn":
    bnn_model, bnn_ckpt = load_bnn()
    if bnn_model is None:
        st.error(
            f"**BNN checkpoint not found** at `{BNN_CKPT}`.\n\n"
            "Run `python -m scripts.train_bnn` first, then refresh."
        )
        st.stop()
    feat_cols      = bnn_ckpt["feat_cols"]
    calib_threshold = bnn_ckpt["calib_threshold"]
    arch_info      = f"Hidden layers: {bnn_ckpt['hidden_sizes']}  ·  MC samples: {N_MC_SAMPLES}"
else:
    gp_model, gp_likelihood, gp_meta = load_gp()
    if gp_model is None:
        st.error(
            f"**GP checkpoint not found** at `{GP_CKPT}` or `{GP_META}`.\n\n"
            "Run `python -m scripts.train_gp` first, then refresh."
        )
        st.stop()
    feat_cols      = gp_meta["feat_cols"]
    calib_threshold = gp_meta["calib_threshold"]
    M = gp_model.variational_strategy.inducing_points.shape[0]
    arch_info      = f"Kernel: Locally Periodic  ·  Inducing points: {M}"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model info")
    st.markdown(f"**Active model:** {'BNN' if mode == 'bnn' else 'GP (Locally Periodic)'}")
    st.markdown(f"**{arch_info}**")
    st.markdown(
        f"**Input features:** 1 time index + {len(feat_cols)} sensors"
    )
    st.markdown(
        f"**Decision threshold:** {calib_threshold:.4f} mg/m³  "
        f"(90th pct of training std)"
    )
    st.divider()
    st.caption("Trained on UCI Air Quality dataset · Phase 7")

# ── Feature inputs ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Sensor inputs")

hour_index = st.number_input(
    "Hour Index (position in dataset, 0 = first measurement)",
    min_value=0,
    max_value=10_000,
    value=7_500,
    step=1,
    help="Raw hourly index used as the time feature by both models.",
)

col_left, col_right = st.columns(2)
feature_values: dict[str, float] = {}

for i, col_name in enumerate(feat_cols):
    lo, hi, default = _FEATURE_RANGES.get(col_name, (0.0, 1000.0, 100.0))
    unit = _UNITS.get(col_name, "")
    label = f"{col_name}  [{unit}]" if unit else col_name
    container = col_left if i % 2 == 0 else col_right
    feature_values[col_name] = container.slider(
        label,
        min_value=float(lo),
        max_value=float(hi),
        value=float(default),
        step=float((hi - lo) / 200),
        format="%.2f",
    )

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔍  Predict", use_container_width=True, type="primary")

if predict_btn:
    feat_list = [feature_values[c] for c in feat_cols]

    if mode == "bnn":
        X_input = _scale_bnn(float(hour_index), feat_list, bnn_ckpt)
        y_mean_ckpt = bnn_ckpt["y_mean"]
        y_std_ckpt  = bnn_ckpt["y_std"]
        with st.spinner("Running Monte Carlo BNN inference …"):
            m_n, t_n, e_n, a_n = predict_bnn_decomposed(
                bnn_model, X_input, n_samples=N_MC_SAMPLES
            )
    else:
        X_input = _scale_gp(float(hour_index), feat_list, gp_meta)
        y_mean_ckpt = gp_meta["y_mean"]
        y_std_ckpt  = gp_meta["y_std"]
        with st.spinner("Running GP inference …"):
            m_n, t_n, e_n, a_n = predict_svgp_decomposed(
                gp_model, gp_likelihood, X_input
            )

    pred_mean      = float(m_n[0].item()) * y_std_ckpt + y_mean_ckpt
    pred_total_std = float(t_n[0].item()) * y_std_ckpt
    pred_epi_std   = float(e_n[0].item()) * y_std_ckpt
    pred_ale_std   = float(a_n[0].item()) * y_std_ckpt

    st.subheader("Prediction result")
    _render_results(pred_mean, pred_total_std, pred_epi_std, pred_ale_std, calib_threshold)
