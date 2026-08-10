# =============================================================================
# CNC Tool Wear Prediction & Scrap Risk Signal System
# Production container image for the Streamlit dashboard.
#
# Build:  docker build -t cnc-tool-wear-system .
# Run:    docker run --rm -p 8501:8501 --env-file .env cnc-tool-wear-system
# =============================================================================
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System dependencies: build tools for xgboost/lightgbm/shap native extensions,
# curl for the container health check.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer -> only rebuilds on
# requirements.txt changes, not on every source-code edit).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application source.
COPY . .

# Run as a non-root user.
RUN useradd --create-home --uid 1000 cncapp \
    && chown -R cncapp:cncapp /app
USER cncapp

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
CMD ["--server.port=8501", "--server.address=0.0.0.0"]
