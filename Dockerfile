# GridGuard backend image.
#
# The dependency list is derived from pyproject.toml rather than restated here.
# The previous version of this file kept a hand-maintained copy of the list, and
# it had drifted: it omitted pvlib and scipy while installing lightgbm (imported
# nowhere) and streamlit. Since `gridguard.api.main` imports `data.synthetic`,
# which imports pvlib at module level, the resulting image could not start the
# API at all — it died on import. Reading the real dependency set removes the
# class of bug rather than the instance.

# ---------------------------------------------------------------------------
# base — runtime dependencies and the installed package
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer, cached until pyproject.toml itself changes. tomllib is in
# the 3.11 standard library, so this needs nothing installed to run.
COPY pyproject.toml ./
RUN python -c "import tomllib,pathlib;p=pathlib.Path('pyproject.toml');d=tomllib.loads(p.read_text())['project']['dependencies'];pathlib.Path('/tmp/requirements.txt').write_text(chr(10).join(d))" \
 && pip install -r /tmp/requirements.txt

# Source, entry points, the site registry, and the curated measured telemetry.
# Shipping data/curated means the image runs the real-data pipeline with no
# download and no credentials, exactly as a fresh clone does.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY data/curated/ ./data/curated/

# --no-deps: everything is already installed above; this only registers the
# package. It is also why the drift above was fatal rather than self-healing.
RUN pip install --no-deps -e .

# ---------------------------------------------------------------------------
# api — the deployed backend. Never trains; loads prebuilt artifacts.
# ---------------------------------------------------------------------------
FROM base AS api

EXPOSE 8000

# Probed with Python rather than curl: the slim image has no curl, so the
# previous HEALTHCHECK could only ever report unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request;sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=5).status==200 else 1)"

CMD ["uvicorn", "gridguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
