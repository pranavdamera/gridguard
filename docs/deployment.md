# GridGuard Deployment Guide

GridGuard has two deployable pieces:
- **FastAPI backend** (Python) — deployable on Render, Railway, Fly.io, or Cloud Run
- **Next.js frontend** (Node.js) — deployable on Vercel

---

## Local development

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git

### One-time setup

```bash
git clone https://github.com/pranav-damera/gridguard
cd gridguard

# Backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Frontend
cd web && npm install && cd ..
```

### Run the full demo locally

```bash
# 1. Generate artifacts (one time, then again if you want to reset)
make demo-reset

# 2. Start backend (in terminal 1)
make api

# 3. Start frontend (in terminal 2)
make web

# Open:
#   http://localhost:3000       — public frontend
#   http://localhost:8000/docs  — API docs
#   http://localhost:8501       — Streamlit research dashboard (optional)
```

---

## Environment variables

### Backend (`.env`)

```env
# Data source
DATA_SOURCE=synthetic           # "synthetic" | "nrel"
NREL_API_KEY=DEMO_KEY           # Your NREL key if using real data

# CORS — comma-separated origins for the frontend
ALLOWED_ORIGINS=http://localhost:3000,https://your-vercel-app.vercel.app

# Anomaly detection
ANOMALY_THRESHOLD_SIGMA=2.0     # Alert threshold (z-score)

# Paths (relative to repo root)
MODEL_DIR=artifacts/models
DATA_PROCESSED_DIR=data/processed
```

Copy `.env.example` to `.env` and fill in your values.

### Frontend (`web/.env.local`)

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

For production, set this to your deployed backend URL.

---

## Deploying the backend

### Render (recommended for ease)

1. Create a new **Web Service** on Render
2. Connect your GitHub repo
3. Set:
   - Build command: `pip install -e .`
   - Start command: `uvicorn gridguard.api.main:app --host 0.0.0.0 --port $PORT`
   - Environment: `ALLOWED_ORIGINS=https://your-vercel-app.vercel.app`
4. Add a Disk mount at `/opt/render/project/src/artifacts` (for model artifacts)

> **Note**: You must run `make demo-reset` locally and commit/upload the generated artifacts, or run the reset script as part of the build command.

### Railway

```bash
railway login
railway init
railway up
```

Set `ALLOWED_ORIGINS` in the Railway dashboard.

### Fly.io

```bash
fly launch
fly deploy
fly secrets set ALLOWED_ORIGINS="https://your-vercel-app.vercel.app"
```

### Docker

A `Dockerfile` and `docker-compose.yml` are included:

```bash
docker compose up -d
```

The API will be available at `http://localhost:8000`.

To generate artifacts inside Docker:
```bash
docker compose exec api python scripts/reset_demo.py
```

---

## Deploying the frontend (Vercel)

### Automatic (recommended)

1. Push the repo to GitHub
2. Import the repo on [vercel.com](https://vercel.com)
3. Set the **Root Directory** to `web`
4. Set environment variable: `NEXT_PUBLIC_API_BASE_URL=https://your-backend.onrender.com`
5. Deploy

### Manual CLI

```bash
npm install -g vercel
cd web
vercel --prod
```

Set the env var in Vercel dashboard or via CLI:
```bash
vercel env add NEXT_PUBLIC_API_BASE_URL
```

---

## Backend start command

For any PaaS that needs an explicit start command:

```bash
uvicorn gridguard.api.main:app --host 0.0.0.0 --port $PORT
```

Or with gunicorn for production:

```bash
gunicorn gridguard.api.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

---

## Checklist before deploying

- [ ] `make demo-reset` generates artifacts successfully
- [ ] `pytest` passes (92 tests)
- [ ] `cd web && npm run build` passes
- [ ] `ALLOWED_ORIGINS` includes your frontend URL
- [ ] `NEXT_PUBLIC_API_BASE_URL` points to your deployed backend
- [ ] Model artifacts are committed or copied to the deployment environment
