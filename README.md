# Football Predictor ⚽

Match outcome predictor using a bivariate Poisson model (Dixon & Coles 1997), XGBoost ensemble, and Monte Carlo simulation — built with Clean Architecture.

Predicts 1X2 probabilities, expected goals, exact score distributions, and betting market indicators (over/under, both teams to score, clean sheets) for 10 leagues including the Premier League, La Liga, Champions League, and World Cup.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Agustin091218/football-predictor.git && cd football-predictor

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env
# Set FOOTBALL_DATA_TOKEN (free tier: https://www.football-data.org/client/register)

# 4. Fetch data
football fetch --league PL --season 2024

# 5. Compute stats
football compute-stats --league PL --season 2024

# 6. Predict
football predict <match_id>
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `football fetch` | Download matches from football-data.org |
| `football compute-stats` | Aggregate team statistics from finished matches |
| `football predict <id>` | Generate prediction for a match |
| `football list-upcoming` | Show upcoming fixtures with IDs |
| `football evaluate` | Compare past predictions against real results |
| `football backtest` | Historical evaluation without data leakage |
| `football set-weights` | Adjust Poisson/XGBoost ensemble weights |
| `football scheduler-start` | Start automated fetch/stats/predict/retrain jobs |
| `football wc-calibrate` | World Cup 2026 calibration report |
| `football wc-add-result` | Register a World Cup result |
| `football wc-standings` | World Cup group standings |

## API Endpoints

Start the server:
```bash
uvicorn football_predictor.interface.api:app --reload
# Swagger UI → http://localhost:8000/docs
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health & config |
| GET | `/leagues` | List supported leagues |
| GET | `/leagues/{id}/upcoming` | Upcoming matches |
| GET | `/leagues/{id}/matches` | Finished matches |
| POST | `/predictions/{match_id}` | Generate prediction |
| GET | `/predictions/{match_id}` | Get stored prediction |
| GET | `/predictions/history` | Recent predictions |
| GET | `/leagues/{id}/accuracy` | Accuracy stats per league |
| POST | `/leagues/{id}/fetch` | Trigger data fetch |
| POST | `/leagues/{id}/train` | Train XGBoost model |
| POST | `/leagues/{id}/backtest` | Run backtest |
| GET | `/world-cup/standings` | World Cup group standings |
| POST | `/world-cup/results` | Add World Cup result |
| POST | `/predict/custom` | Predict any two teams |

## Architecture

```
interface/        ← CLI (Typer) + REST API (FastAPI)
    ↓
application/      ← Use cases: predict, train, backtest, fetch
    ↓
domain/           ← Pure business logic: Poisson, ELO, Monte Carlo
    ↓
infrastructure/   ← SQLite repos, XGBoost, feature engineering, API clients
```

- **Domain layer** has zero framework dependencies — pure math. Implements Dixon & Coles bivariate Poisson with low-scoring correction, ELO-based strength estimation, and deterministic Monte Carlo simulation (no numpy).
- **Application layer** orchestrates use cases through abstract repository interfaces.
- **Infrastructure layer** provides SQLite persistence (SQLAlchemy), XGBoost training/prediction, feature engineering with signal nodes, football-data.org API client, and optional Gemini LLM analysis.
- **Dependency injection** via `Container` singleton — API and CLI share the same instances.

## How Predictions Work

1. **Signal Nodes** produce independent assessments:
   - Form (recent results, trends)
   - ELO ratings (strength differential)
   - Head-to-head history
   - Poisson model (Dixon & Coles)
   - Context (league position, match importance)

2. **Ensemble** combines Poisson + XGBoost predictions with configurable weights.

3. **Monte Carlo** simulates 10,000 matches from Poisson distributions to produce:
   - 1X2 probabilities with 95% confidence intervals
   - Top 10 most likely scorelines
   - Goal distributions and percentiles
   - Market probabilities: over 2.5, BTTS, clean sheets

4. **LLM** (optional Gemini) provides natural-language match analysis enriched with web search for recent form.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FOOTBALL_DATA_TOKEN` | — | **Required**. API key from football-data.org |
| `GEMINI_API_KEY` | — | Optional. Enables LLM match analysis |
| `FOOTBALL_DB_PATH` | `football.db` | SQLite database path |
| `MODELS_DIR` | `models` | Trained model storage |
| `ELO_RATINGS_PATH` | `archive/eloratings.csv` | ELO ratings CSV |
| `POISSON_WEIGHT` | `0.5` | Poisson weight in ensemble |
| `XGBOOST_WEIGHT` | `0.5` | XGBoost weight in ensemble |
| `N_SIMULATIONS` | `10000` | Monte Carlo iterations |
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_KEY` | — | Optional. Protects API endpoints |
| `ENABLE_SCHEDULER` | `true` | Auto fetch/stats/predict/retrain |

## Docker

```bash
docker compose up -d
# API → http://localhost:8000
# Health → http://localhost:8000/health
```

## Development

```bash
pip install -e ".[dev]"   # Install with pytest, ruff
ruff check .              # Lint
ruff format .             # Format
pytest                    # Run 39 tests
```

## Supported Leagues

Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Primeira Liga, Eredivisie, Brasileirão, Championship, Champions League.
