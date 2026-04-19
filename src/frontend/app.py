"""Phase 7 — Interactive Streamlit frontend for BNN CO(GT) prediction.

Loads the trained BNN checkpoint and lets the user supply sensor readings
via sliders, then displays:
  - Predicted CO(GT) with ±1σ confidence interval
  - Epistemic vs. Aleatoric uncertainty bar chart
  - Decision-policy warning when total std > calibration threshold

Usage
-----
    streamlit run src/frontend/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Streamlit import — must come before any st.* calls
import streamlit as st

# ── Project imports ───────────────────────────────────────────────────────────
# Resolve project root so the script works regardless of cwd
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import PROCESSED_DATA_DIR
from src.models.bnn_vi import BNNRegressor, predict_bnn_decomposed
from src.evaluation.decision_policy import UncertaintyPolicy

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = PROCESSED_DATA_DIR / "bnn_model.pt"
N_MC_SAMPLES = 200

# Realistic min / max / default values derived from the UCI Air Quality dataset
# (used only when the checkpoint has not been loaded yet for the defaults)
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


# ── Model loading (cached across reruns) ──────────────────────────────────────

@st.cache_resource(show_spinner="Loading BNN model …")
def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return None
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = BNNRegressor(
        in_features=ckpt["in_features"],
        hidden_sizes=ckpt["hidden_sizes"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def _scale_input(raw_vec: np.ndarray, ckpt: dict) -> torch.Tensor:
    """Apply the saved StandardScaler to a single (1+F,) input vector."""
    scaled = (raw_vec - ckpt["x_scaler_mean"]) / ckpt["x_scaler_scale"]
    return torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, 1+F)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌫️",
    layout="wide",
)

st.title("🌫️ Air Quality CO(GT) Predictor")
st.markdown(
    "Enter current sensor readings and click **Predict** to obtain a "
    "probabilistic CO concentration estimate with uncertainty breakdown."
)

# ── Load model ────────────────────────────────────────────────────────────────

result = load_checkpoint()
if result is None:
    st.error(
        f"**Model checkpoint not found** at `{CHECKPOINT_PATH}`.\n\n"
        "Run `python -m scripts.train_bnn` first, then refresh this page."
    )
    st.stop()

model, ckpt = result
feat_cols: list[str] = ckpt["feat_cols"]
y_mean: float = ckpt["y_mean"]
y_std: float = ckpt["y_std"]
calib_threshold: float = ckpt["calib_threshold"]
policy = UncertaintyPolicy(threshold=calib_threshold)

# ── Sidebar — model info ──────────────────────────────────────────────────────

with st.sidebar:
    st.header("Model info")
    st.markdown(f"**Architecture:** BNN {ckpt['hidden_sizes']}")
    st.markdown(f"**Input features:** {ckpt['in_features']} (time + {len(feat_cols)} sensors)")
    st.markdown(f"**MC samples:** {N_MC_SAMPLES}")
    st.markdown(f"**Decision threshold:** {calib_threshold:.4f} mg/m³ (90th pct of training std)")
    st.divider()
    st.caption("Trained on UCI Air Quality dataset · Phase 7")

# ── Feature inputs ────────────────────────────────────────────────────────────

st.subheader("Sensor inputs")

# Hour index: raw position in the dataset (0 … ~9 400)
# We default to 7 500 which sits in the test portion of the data.
hour_index = st.number_input(
    "Hour Index (position in dataset, 0 = first measurement)",
    min_value=0,
    max_value=10_000,
    value=7_500,
    step=1,
    help="Raw hourly index used by the model as the time feature.",
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

# ── Prediction ────────────────────────────────────────────────────────────────

st.divider()
predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

if predict_btn:
    # Build raw input vector: [hour_index, feat_1, ..., feat_F]
    raw_vec = np.array(
        [float(hour_index)] + [feature_values[c] for c in feat_cols],
        dtype=np.float32,
    )
    X_input = _scale_input(raw_vec, ckpt)  # (1, 1+F)

    with st.spinner("Running Monte Carlo inference …"):
        mean_n, total_std_n, epi_std_n, ale_std_n = predict_bnn_decomposed(
            model, X_input, n_samples=N_MC_SAMPLES
        )

    # Denormalise to original CO(GT) units (mg/m³)
    pred_mean = float(mean_n[0].item()) * y_std + y_mean
    pred_total_std = float(total_std_n[0].item()) * y_std
    pred_epi_std = float(epi_std_n[0].item()) * y_std
    pred_ale_std = float(ale_std_n[0].item()) * y_std

    # ── Decision policy ───────────────────────────────────────────────────────
    flagged = policy.flag(np.array([pred_total_std]))[0]

    # ── Result cards ─────────────────────────────────────────────────────────
    st.subheader("Prediction result")

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
                f"✅ Prediction accepted (std {pred_total_std:.4f} ≤ threshold {calib_threshold:.4f}).",
                icon="✅",
            )

    with chart_col:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3.5))

        components = ["Epistemic", "Aleatoric", "Total"]
        values = [pred_epi_std, pred_ale_std, pred_total_std]
        colors = ["#4e8df5", "#f5814e", "#888888"]

        bars = ax.bar(components, values, color=colors, width=0.5, edgecolor="white")
        ax.axhline(calib_threshold, color="red", lw=1.5, ls="--", label=f"Threshold ({calib_threshold:.3f})")
        ax.set_ylabel("Standard deviation [mg/m³]", fontsize=9)
        ax.set_title("Uncertainty decomposition", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Confidence interval ───────────────────────────────────────────────────
    st.caption(
        f"95% confidence interval: "
        f"[{pred_mean - 2*pred_total_std:.3f}, {pred_mean + 2*pred_total_std:.3f}] mg/m³"
    )
