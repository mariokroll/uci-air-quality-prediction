# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Install uv from the official distroless image (single binary, no extra deps)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
# Copy manifests first so this expensive layer is cached unless deps change.
COPY pyproject.toml uv.lock ./

# Sync production dependencies exactly as locked.
# uv respects the [[tool.uv.index]] pytorch entry in pyproject.toml.
# To build a CPU-only image (smaller), change the pytorch index URL in
# pyproject.toml to https://download.pytorch.org/whl/cpu before building.
RUN uv sync --no-dev --frozen

# ── Application source ────────────────────────────────────────────────────────
COPY src/ ./src/

# ── Model checkpoint ──────────────────────────────────────────────────────────
# Run `python -m scripts.train_bnn` before building so this file exists.
COPY data/processed/bnn_model.pt ./data/processed/bnn_model.pt

COPY data/processed/svgp_locally_periodic.pt ./data/processed/svgp_locally_periodic.pt

# ── Runtime configuration ─────────────────────────────────────────────────────
EXPOSE 8501

# Disable telemetry and suppress the "open browser" prompt inside containers.
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

CMD ["uv", "run", "streamlit", "run", "src/frontend/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
