# Contributing to GridGuard

Thanks for your interest. This is primarily a portfolio and learning project,
but contributions that improve correctness, add real dataset adapters, or
extend the anomaly detection methodology are welcome.

## Ground rules

- Open an issue before starting large changes so we can align.
- Keep PRs focused on one thing.
- Do not overclaim accuracy. If you add a new model, include honest benchmark
  numbers against the persistence baseline.
- Never commit real sensor data, API keys, or PII.

## Local setup

```bash
git clone https://github.com/YOUR_USERNAME/gridguard.git
cd gridguard
python -m venv .venv && source .venv/bin/activate
make install     # installs with dev extras
cp .env.example .env
make data-synthetic
make train
make test        # all tests should pass before opening a PR
```

## Code standards

- Python 3.11+
- `ruff` for linting, `black` for formatting — run `make format` before committing
- Every new function needs at least one test in `tests/`
- Temporal train/test splits are mandatory — never use random splits on time-series

## Pull request checklist

- [ ] `make lint` passes
- [ ] `make test` passes with coverage ≥ 80% on new code
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
