.PHONY: install install-dashboard lint format test \
        build-artifacts build-real build-synthetic data-real \
        api web dashboard demo \
        docker-build docker-up docker-down docker-research clean

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:
	pip install -e ".[dev]"
	cd web && npm ci

install-dashboard:
	pip install -e ".[dashboard]"

# ---------------------------------------------------------------------------
# Quality gates — the same checks CI runs
# ---------------------------------------------------------------------------

lint:
	ruff check .
	black --check .
	cd web && npm run lint && npm run typecheck

format:
	ruff check --fix .
	black .

test:
	pytest --cov=gridguard --cov-report=term-missing

# ---------------------------------------------------------------------------
# Artifacts
#
# build-artifacts is the one canonical command: it verifies/downloads data,
# preprocesses, trains, calibrates uncertainty, evaluates, and writes a
# manifest recording exactly what was built. The deployed backend never trains
# — it loads what this produced.
# ---------------------------------------------------------------------------

build-artifacts:
	python scripts/build_artifacts.py

build-real:
	python scripts/build_artifacts.py --mode real

build-synthetic:
	python scripts/build_artifacts.py --mode synthetic

# Refresh the curated measured datasets from the OEDI data lake. Only needed to
# change the window or add a site — the curated parquet files are committed, so
# a fresh clone runs on real data with no download.
data-real:
	python scripts/download_data.py --refresh-curated

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

api:
	uvicorn gridguard.api.main:app --reload --host 0.0.0.0 --port 8000

web:
	cd web && npm run dev

# Streamlit is the internal research/diagnostics surface, not the product.
dashboard:
	streamlit run dashboard/app.py --server.port 8501

demo:
	@echo "GridGuard — run these in separate terminals:"
	@echo ""
	@echo "  make build-artifacts   (once: builds models + calibration + manifest)"
	@echo "  make api               -> http://localhost:8000/docs"
	@echo "  make web               -> http://localhost:3000"
	@echo ""
	@echo "  Optional research dashboard:"
	@echo "  make dashboard         -> http://localhost:8501"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

# `docker compose up` is the whole demo: it builds artifacts once, then starts
# the API and the web app in dependency order.

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down -v

# Adds the Streamlit research surface on :8501. Not part of the default path.
docker-research:
	docker compose --profile research up -d

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage coverage.xml htmlcov .ruff_cache artifacts/
	@echo "Removed derived artifacts. data/curated/ is version-controlled and kept."
