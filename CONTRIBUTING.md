# Contributing to GridGuard

Thanks for your interest. This is primarily a portfolio and learning project,
but contributions that improve correctness, add real dataset adapters, or
extend the anomaly detection methodology are welcome.

## Ground rules

- Open an issue before starting large changes so we can align.
- Keep PRs focused on one thing.
- Do not overclaim accuracy. If you add a new model, include honest benchmark
  numbers against the persistence baseline.
- Detection metrics come from *injected* faults. Never describe them as
  field-validated.
- Never commit API keys, credentials, or PII. Measured telemetry belongs in
  `data/curated/` only if it is small, public, and carries a provenance record.

## Local setup

```bash
git clone https://github.com/YOUR_USERNAME/gridguard.git
cd gridguard
python -m venv .venv && source .venv/bin/activate
make install            # backend + frontend dependencies
make build-artifacts   # ~4 min; needs no network and no credentials
make test              # all tests should pass before opening a PR
```

## Code standards

- Python 3.11+
- `ruff` for linting, `black` for formatting — run `make format` before committing
- Every new function needs at least one test in `tests/`
- Temporal train/test splits are mandatory — never use random splits on time series
- The anomaly detector must never see lagged power. A degraded system produces low
  output, so its lagged power is low, so a lag-aware model calls the degradation
  normal. Use `mode="weather_only"` for anything feeding detection.
- Anything that displays generation numbers must also carry its `data_mode`.
  Measured and simulated data are never presented interchangeably.

## Pull request checklist

- [ ] `make lint` passes (ruff, black, eslint, tsc)
- [ ] `make test` passes
- [ ] No new dependencies unless clearly justified
- [ ] Docstring on new public functions
- [ ] `CHANGELOG.md` entry (if a notable change)

## What's in scope

- New model baselines (LightGBM, LSTM, Prophet)
- Real dataset adapters (Open Power System Data, Ausgrid, PVOutput)
- Better anomaly detection (conformal prediction intervals, CUSUM)
- Multi-site fleet-level analysis
- Dashboard improvements
- Documentation fixes

## What's out of scope

- Production deployment tooling (Kubernetes, cloud infra)
- Proprietary data integrations
- Anything that assumes access to real-time sensor streams
