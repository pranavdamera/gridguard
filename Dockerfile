FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies first (cached unless pyproject.toml changes).
# Separating this from the package install keeps the layer cache stable across
# source-only changes.
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "pandas>=2.1" "numpy>=1.26" "scikit-learn>=1.4" "xgboost>=2.0" "lightgbm>=4.0" \
    "shap>=0.44" "fastapi>=0.111" "uvicorn[standard]>=0.29" "streamlit>=1.35" \
    "plotly>=5.20" "httpx>=0.27" "pydantic>=2.7" "pydantic-settings>=2.3" \
    "python-dotenv>=1.0" "requests>=2.31" "joblib>=1.4" "pyarrow>=16"

# Copy source and register the package (--no-deps avoids reinstalling above)
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/
COPY config/ ./config/
RUN pip install --no-cache-dir --no-deps -e .

# Artifacts and data are mounted at runtime via docker-compose volumes

EXPOSE 8000

CMD ["uvicorn", "gridguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
