FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying code (layer cache efficiency)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "."

# Copy source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/

# Artifacts and data are mounted at runtime via docker-compose volumes

EXPOSE 8000

CMD ["uvicorn", "gridguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
