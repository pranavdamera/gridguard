.PHONY: install lint test data train data-gmu train-gmu api dashboard web demo-reset clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/ scripts/ dashboard/
	black --check src/ tests/ scripts/ dashboard/

format:
	ruff check --fix src/ tests/ scripts/ dashboard/
	black src/ tests/ scripts/ dashboard/

test:
	pytest tests/ -v --cov=gridguard --cov-report=term-missing

# Generate synthetic data (no API key needed)
data-synthetic:
	python scripts/download_data.py --source synthetic

# Download real NREL PVDAQ data (requires NREL_API_KEY in .env)
data-nrel:
	python scripts/download_data.py --source nrel

# Generate site-specific synthetic data for GMU Fairfax (250 kW, 38.83°N)
data-gmu:
	python scripts/download_data.py --source synthetic --site-id gmu_fairfax

train:
	python scripts/run_pipeline.py

# Train using GMU Fairfax synthetic data
train-gmu:
	python scripts/run_pipeline.py --site-id gmu_fairfax

api:
	uvicorn gridguard.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py --server.port 8501

web:
	cd web && npm run dev

# Regenerate deterministic demo bundle (data + models + artifacts) from scratch
demo-reset:
	python scripts/reset_demo.py

# Run API + frontend (requires separate terminals or tmux)
demo:
	@echo "Run in separate terminals:"
	@echo "  make demo-reset   (once, to generate artifacts)"
	@echo "  make api          (FastAPI backend)"
	@echo "  make web          (Next.js frontend)"
	@echo "  make dashboard    (Streamlit research dashboard)"
	@echo ""
	@echo "  API docs:         http://localhost:8000/docs"
	@echo "  Public frontend:  http://localhost:3000"
	@echo "  Demo endpoint:    http://localhost:8000/demo/scenario"
	@echo "  Dashboard:        http://localhost:8501"

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov artifacts/
