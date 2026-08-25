# Deployment

GridGuard deploys as two independent services:

```
GitHub
  ├── Vercel  ──  Next.js frontend  (static + server components)
  └── Render  ──  FastAPI backend   (loads prebuilt artifacts)
```

They are deployed and scaled separately, and the only coupling between them is
one environment variable on each side.

---

## The artifact model

**The backend never trains.** It loads artifacts produced by a single canonical
command and serves them:

```bash
make build-artifacts      # python scripts/build_artifacts.py
```

That command verifies/loads data → preprocesses → trains → calibrates
uncertainty → evaluates → writes `artifacts/models/manifest.json` recording the
git commit, dataset window, hyperparameters, calibration configuration and every
evaluation score behind the build.

Artifacts are **not committed** — they total roughly 150 MB. The *curated
measured datasets* (~3 MB) **are** committed, which is what makes this work: the
build needs no network access and no credentials, because the real data is
already in the repository and the simulated fleet is regenerated
deterministically from a fixed seed.

So the deployment story is simply: **run the build during the build step.**

| What | Where | Size | Committed? |
|---|---|---|---|
| Curated measured telemetry | `data/curated/` | ~3 MB | Yes |
| Model artifacts + calibration | `artifacts/models/` | ~150 MB | No — built at deploy |
| Derived frames (detected, events) | `data/processed/` | ~25 MB | No — built at deploy |

If your host's build step is too constrained for a full build, restrict the
fleet:

```bash
python scripts/build_artifacts.py --mode real       # 3 sites instead of 10
python scripts/build_artifacts.py --no-physics      # skip the pvlib models
```

---

## Local

```bash
git clone https://github.com/pranavdamera/gridguard
cd gridguard

python -m venv .venv && source .venv/bin/activate
make install                 # backend + frontend dependencies

make build-artifacts         # once, ~4 minutes; no network needed

make api                     # terminal 1 -> http://localhost:8000/docs
make web                     # terminal 2 -> http://localhost:3000
```

Offline research diagnostics (`pip install -e ".[experiments]"` first):

```bash
make diagnostics             # -> artifacts/reports/diagnostics/
```

---

## Backend — Render

A [`render.yaml`](../render.yaml) blueprint is included. Point Render's
**New Blueprint Instance** at the repository and it will pick it up.

Or configure a Web Service manually:

| Setting | Value |
|---|---|
| Runtime | Python 3.12 |
| Build command | `pip install -e . && python scripts/build_artifacts.py` |
| Start command | `uvicorn gridguard.api.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |

The start command **must** bind `$PORT` — Render assigns it, and a service that
binds a fixed port will fail its health check.

### Backend environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `CORS_ALLOWED_ORIGINS` | **Yes** | `http://localhost:3000` | Comma-separated frontend origins. Set to your Vercel production URL. |
| `CORS_ALLOW_VERCEL_PREVIEWS` | No | `false` | Allows any `*.vercel.app` origin. Convenient for preview deploys; widens the origin set. |
| `DATA_MODE` | No | `synthetic` | Default fleet view. Both modes always available per-site. |
| `ANOMALY_METHOD` | No | `conformal` | `conformal` or `sigma`. |
| `CONFORMAL_ALPHA` | No | `0.05` | Miscoverage level. |

There are no secrets in the default configuration. The real-data path reads a
public dataset over anonymous HTTPS.

### Other hosts

Any platform that can run a Python build step and bind `$PORT` works. Fly.io:

```bash
fly launch
fly secrets set CORS_ALLOWED_ORIGINS="https://your-app.vercel.app"
fly deploy
```

Railway: same build and start commands; set the variables in the dashboard.

For production traffic, run under gunicorn with uvicorn workers:

```bash
gunicorn gridguard.api.main:app -k uvicorn.workers.UvicornWorker \
  -w 2 --bind 0.0.0.0:$PORT
```

Two workers each hold the artifacts in memory (a few hundred MB total), so size
the instance accordingly rather than raising the worker count freely.

---

## Frontend — Vercel

1. Import the repository at [vercel.com/new](https://vercel.com/new).
2. Set **Root Directory** to `web`. Vercel then auto-detects Next.js.
3. Add the environment variable:

   | Variable | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | `https://your-service.onrender.com` |

4. Deploy.

`NEXT_PUBLIC_*` variables are **inlined at build time**, so changing this value
requires a redeploy, not just a restart. Nothing secret may go in it — it ships
in the client bundle.

Or from the CLI:

```bash
npm install -g vercel
cd web
vercel env add NEXT_PUBLIC_API_URL
vercel --prod
```

---

## CORS

The API refuses to guess. `CORS_ALLOWED_ORIGINS` is an explicit list:

```env
CORS_ALLOWED_ORIGINS=https://gridguard.vercel.app,https://gridguard.example.com
```

`*` is accepted but logs a warning on startup and should not be deployed —
with credentials disabled it is not a critical vulnerability, but it does let
any site read your API from a user's browser.

Preview deployments get a fresh subdomain per branch, so they can only be
matched by pattern. Set `CORS_ALLOW_VERCEL_PREVIEWS=true` to allow
`https://<anything>.vercel.app` if you want previews to work against production
data, understanding that it opens the API to every Vercel-hosted site.

---

## Verifying a deployment

```bash
curl https://your-service.onrender.com/health
```

```json
{
  "status": "ok",
  "sites_loaded": 10,
  "sites_with_models": 10,
  "sites_with_calibration": 10,
  "data_rows": 270921,
  "manifest_present": true,
  "artifact_built_at": "2026-08-18T18:42:55+00:00",
  "artifact_commit": "98defe3...",
  "detection_method": "conformal",
  "warnings": []
}
```

Check that:

- `status` is `ok`, not `degraded` (degraded means no artifacts loaded).
- `manifest_present` is `true`.
- `artifact_commit` matches the commit you deployed.
- `warnings` is empty.

The response deliberately contains no filesystem paths, hostnames or
configuration values, so it is safe to expose publicly.

Then load the frontend and open **/data** — if provenance renders, the frontend
is reaching the API and the artifacts carry their provenance records.

---

## Docker

```bash
docker compose up
```

That is the whole thing. Three services start in order:

| Service | Role |
| --- | --- |
| `artifacts` | One-shot. Trains, calibrates, evaluates, writes the manifest, exits. |
| `api` | Waits for `artifacts` to exit successfully, then serves them on :8000. |
| `web` | Waits for `api` to report healthy, then serves the frontend on :3000. |

### How dependencies are installed

`Dockerfile` extracts its dependency list *from* `pyproject.toml` at build time
rather than restating it. This is deliberate. The previous version kept a
hand-maintained copy, and it had drifted: it omitted `pvlib` and `scipy` while
installing `lightgbm`, which nothing imports. Because `gridguard.api.main`
imports `pvlib` transitively at module level, the image could not start the API
at all. `tests/test_packaging.py` now asserts both halves of the invariant — the
API's import graph is fully declared, and the Dockerfile does not pin
requirements inline.

Dependencies still install in their own layer, so source-only changes do not
reinstall the scientific stack.

---

## Refreshing the curated datasets

Only needed to change the data window or add a measured site:

```bash
make data-real        # re-downloads from the OEDI data lake, rewrites data/curated/
git diff --stat data/curated/
```

Review the diff before committing — these files are the repository's only
committed data, and they are what makes a fresh clone runnable offline.
