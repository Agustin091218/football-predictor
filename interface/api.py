"""REST API with FastAPI — prediction, leagues, ML training endpoints.

Uses the Container singleton for all dependencies.
Auto-generated Swagger UI at /docs.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from football_predictor.application.backtesting import BacktestingUseCase
from football_predictor.application.predict_match import MatchNotFoundError
from football_predictor.domain.entities import Match, Prediction
from football_predictor.infrastructure.config import SUPPORTED_LEAGUES
from football_predictor.infrastructure.container import container
from football_predictor.infrastructure.ml.calibration import ProbabilityCalibrator
from football_predictor.infrastructure.scheduler import PredictionScheduler
from football_predictor.interface.world_cup_api import router as wc_router

logger = logging.getLogger("api.requests")

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class TeamResponse(BaseModel):
    id: str
    name: str
    short_name: str
    tla: str
    country: str


class LeagueResponse(BaseModel):
    id: str
    name: str
    country: str
    season: str | None = None


class ScoreResponse(BaseModel):
    home_goals: int | None = None
    away_goals: int | None = None
    result: str | None = None


class MatchResponse(BaseModel):
    id: str
    home_team: TeamResponse
    away_team: TeamResponse
    league: LeagueResponse
    match_date: str
    status: str
    score: ScoreResponse
    matchday: int | None = None


class MonteCarloResponse(BaseModel):
    n_simulations: int
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    ci_home_win: list[float]
    most_likely_score: str
    top_scores: list[dict]
    prob_over_2_5: float
    prob_btts: float
    prob_clean_sheet_home: float | None = None
    prob_clean_sheet_away: float | None = None
    goals_p50_home: float
    goals_p50_away: float


class PredictionResponse(BaseModel):
    match_id: str
    match: MatchResponse
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    predicted_result: str
    expected_goals_home: float
    expected_goals_away: float
    confidence: float
    model_version: str
    predicted_at: str
    simulation: MonteCarloResponse | None = None
    llm_explanation: str | None = None
    llm_actions: list[str] | None = None


class AccuracyByResult(BaseModel):
    predicted: int
    correct: int
    accuracy: float


class AccuracyResponse(BaseModel):
    league_id: str
    total_evaluated: int
    correct: int
    accuracy: float
    by_result: dict[str, AccuracyByResult]


class HealthResponse(BaseModel):
    status: str
    db_path: str
    xgboost_trained: bool
    llm_available: bool
    supported_leagues: int
    scheduler_running: bool = False
    scheduler_jobs: int = 0


# ---------------------------------------------------------------------------
# Mappers
# ---------------------------------------------------------------------------


def _match_to_response(match: Match) -> MatchResponse:
    result_str = None
    if match.result is not None:
        result_str = match.result.value

    return MatchResponse(
        id=str(match.id),
        home_team=TeamResponse(
            id=str(match.home_team.id),
            name=match.home_team.name,
            short_name=match.home_team.short_name,
            tla=match.home_team.tla,
            country=match.home_team.country,
        ),
        away_team=TeamResponse(
            id=str(match.away_team.id),
            name=match.away_team.name,
            short_name=match.away_team.short_name,
            tla=match.away_team.tla,
            country=match.away_team.country,
        ),
        league=LeagueResponse(
            id=str(match.league.id),
            name=match.league.name or str(match.league.id),
            country=match.league.country,
            season=match.league.season,
        ),
        match_date=match.match_date.isoformat(),
        status=match.status.value,
        score=ScoreResponse(
            home_goals=match.score.home_goals if match.score else None,
            away_goals=match.score.away_goals if match.score else None,
            result=result_str,
        ),
        matchday=match.matchday,
    )


def _prediction_to_response(pred: Prediction) -> PredictionResponse:
    sim = None
    if pred.simulation:
        sim_dict = pred.simulation.as_dict()
        sim = MonteCarloResponse(**sim_dict)

    result_map = {"home_win": "1", "draw": "X", "away_win": "2"}
    predicted_str = result_map.get(pred.predicted_result.value, pred.predicted_result.value)

    return PredictionResponse(
        match_id=str(pred.match.id),
        match=_match_to_response(pred.match),
        prob_home_win=pred.prob_home_win,
        prob_draw=pred.prob_draw,
        prob_away_win=pred.prob_away_win,
        predicted_result=predicted_str,
        expected_goals_home=pred.expected_goals_home,
        expected_goals_away=pred.expected_goals_away,
        confidence=pred.confidence,
        model_version=pred.model_version,
        predicted_at=pred.predicted_at.isoformat(),
        simulation=sim,
        llm_explanation=pred.llm_explanation,
        llm_actions=pred.llm_actions,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Football Predictor API",
    description="Predicción de partidos con Poisson + XGBoost + Monte Carlo",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(
        "%s %s → %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


app.include_router(wc_router)


_API_KEY = os.getenv("API_KEY", "")


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if not _API_KEY:
        return await call_next(request)
    if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
        return await call_next(request)
    if request.url.path.startswith("/assets/") or request.url.path.startswith("/vite.svg"):
        return await call_next(request)
    if request.headers.get("X-API-Key") != _API_KEY:
        return JSONResponse(status_code=401, content={"detail": "X-API-Key header required"})
    return await call_next(request)


scheduler = PredictionScheduler(container)
calibrator = ProbabilityCalibrator()
backtest_uc = BacktestingUseCase(
    match_repo=container.match_repo,
    stats_repo=container.stats_repo,
    poisson_service=container.poisson_service,
    feature_engineer=container.feature_engineer,
    xgboost_predictor=container.xgboost_predictor,
    monte_carlo=container.mc_simulator,
    calibrator=calibrator,
)


@app.on_event("startup")
async def startup_event():
    if os.getenv("ENABLE_SCHEDULER", "true").lower() == "true":
        scheduler.start()
        logging.getLogger(__name__).info("API iniciada con scheduler activo")


@app.on_event("shutdown")
async def shutdown_event():
    scheduler.stop()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        db_path=container.db_path,
        xgboost_trained=container.xgboost_predictor.is_trained,
        llm_available=container.llm_orchestrator is not None,
        supported_leagues=len(SUPPORTED_LEAGUES),
        scheduler_running=scheduler._scheduler.running,
        scheduler_jobs=len(scheduler.get_jobs_status()),
    )


@app.get("/leagues", response_model=list[LeagueResponse])
async def list_leagues():
    return [
        LeagueResponse(
            id=cfg["id"],
            name=cfg["name"],
            country=cfg["country"],
        )
        for cfg in SUPPORTED_LEAGUES.values()
    ]


@app.get("/leagues/{league_id}/upcoming", response_model=list[MatchResponse])
async def upcoming_matches(
    league_id: str,
    days: int = Query(default=7, ge=1, le=30),
):
    if league_id not in SUPPORTED_LEAGUES:
        raise HTTPException(404, f"Liga {league_id} no soportada")
    matches = container.match_repo.get_upcoming_matches(league_id, days_ahead=days)
    return [_match_to_response(m) for m in matches]


@app.get("/leagues/{league_id}/matches", response_model=list[MatchResponse])
async def finished_matches(
    league_id: str,
    season: str = Query(default="2024"),
    limit: int = Query(default=20, ge=1, le=100),
):
    if league_id not in SUPPORTED_LEAGUES:
        raise HTTPException(404, f"Liga {league_id} no soportada")
    matches = container.match_repo.get_finished_matches(league_id, season, limit=limit)
    return [_match_to_response(m) for m in matches]


@app.get("/predictions/history")
async def predictions_history(limit: int = Query(default=20, ge=1, le=100)):
    preds = container.prediction_repo.get_recent(limit=limit)
    return [_prediction_to_response(p) for p in preds]


@app.post("/predictions/{match_id}", response_model=PredictionResponse)
async def create_prediction(match_id: str):
    try:
        prediction = container.predict_uc.execute(match_id)
        return _prediction_to_response(prediction)
    except MatchNotFoundError:
        raise HTTPException(404, f"Partido {match_id} no encontrado") from None
    except Exception:
        logger.exception("Error generating prediction for %s", match_id)
        raise HTTPException(500, "Error interno generando predicción") from None


@app.get("/predictions/{match_id}", response_model=PredictionResponse)
async def get_prediction(match_id: str):
    prediction = container.prediction_repo.get_by_match_id(match_id)
    if prediction is None:
        raise HTTPException(404, f"No hay predicción para {match_id}")
    return _prediction_to_response(prediction)


@app.get("/leagues/{league_id}/accuracy", response_model=AccuracyResponse)
async def league_accuracy(
    league_id: str,
    season: str | None = None,
):
    stats = container.prediction_repo.get_accuracy_stats(league_id=league_id)
    if stats["total_evaluated"] == 0:
        raise HTTPException(404, f"No hay predicciones evaluadas para {league_id}")

    by_result = {
        key: AccuracyByResult(
            predicted=val["predicted"],
            correct=val["correct"],
            accuracy=val["accuracy"],
        )
        for key, val in stats["by_result"].items()
    }

    return AccuracyResponse(
        league_id=league_id,
        total_evaluated=stats["total_evaluated"],
        correct=stats["correct"],
        accuracy=stats["accuracy"],
        by_result=by_result,
    )


@app.post("/leagues/{league_id}/fetch", response_model=dict)
async def fetch_league(league_id: str, season: str = Query(default="2024")):
    if not container.football_data_token:
        raise HTTPException(400, "FOOTBALL_DATA_TOKEN no configurado")
    result = container.fetch_and_store_uc.execute(league_id, season)
    return result


@app.post("/leagues/{league_id}/compute-stats", response_model=dict)
async def compute_stats(league_id: str, season: str = Query(default="2024")):
    result = container.compute_stats_uc.execute(league_id, season)
    return result


@app.post("/leagues/{league_id}/train", response_model=dict)
async def train_model(league_id: str, season: str = Query(default="2024")):
    result = container.train_uc.execute(league_id, season)
    return result


@app.get("/scheduler/status", response_model=list[dict])
async def scheduler_status():
    return scheduler.get_jobs_status()


@app.post("/leagues/{league_id}/backtest", response_model=dict)
async def backtest_league(
    league_id: str,
    season: str = Query(default="2024"),
    train_until_matchday: int = Query(default=20, ge=5, le=35),
):
    try:
        result = backtest_uc.execute(league_id, season, train_until_matchday)
        return result.as_dict()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logging.getLogger(__name__).error("Backtest error: %s", exc)
        raise HTTPException(500, "Error ejecutando backtest") from exc


@app.post("/predict/custom")
async def predict_custom(home: str, away: str):
    import csv
    from datetime import datetime

    from football_predictor.domain.entities import CompetitionType, League, Match, MatchStatus, Team

    elo = {}
    elo_path = os.getenv("ELO_RATINGS_PATH", "archive/eloratings.csv")
    try:
        with open(elo_path) as f:
            for row in csv.DictReader(f):
                try:
                    r = float(row.get("rating") or 0)
                    if r > 0:
                        elo[row["team"]] = r
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        pass

    mid = int(datetime.now().timestamp() * 1000) % 10_000_000
    league = League(
        id="CUSTOM",
        name="Custom Match",
        country="International",
        competition_type=CompetitionType.INTERNATIONAL,
        season=str(datetime.now().year),
    )
    hteam = Team(id=mid, name=home, short_name=home[:3].upper(), tla=home[:3].upper(), country="?")
    ateam = Team(
        id=mid + 1, name=away, short_name=away[:3].upper(), tla=away[:3].upper(), country="?"
    )
    match = Match(
        id=mid,
        home_team=hteam,
        away_team=ateam,
        league=league,
        match_date=datetime.now(),
        status=MatchStatus.SCHEDULED,
    )

    container.match_repo.save(match)
    prediction = container.predict_uc.execute(str(mid))
    return _prediction_to_response(prediction)


_frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(_frontend_dist):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(_frontend_dist, "assets")),
        name="react_assets",
    )

    @app.get("/test-predict")
    async def serve_test_predict():
        test_path = os.path.join(_frontend_dist, "test-predict.html")
        if os.path.exists(test_path):
            with open(test_path) as f:
                return HTMLResponse(f.read())
        raise HTTPException(404)

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        for prefix in ("predict", "leagues", "world-cup", "health", "docs", "openapi.json"):
            if full_path.startswith(prefix):
                raise HTTPException(404)
        index_path = os.path.join(_frontend_dist, "index.html")
        if os.path.exists(index_path):
            with open(index_path) as f:
                return HTMLResponse(f.read())
        raise HTTPException(404)

    @app.get("/")
    async def serve_react_root():
        index_path = os.path.join(_frontend_dist, "index.html")
        if os.path.exists(index_path):
            with open(index_path) as f:
                return HTMLResponse(f.read())
        raise HTTPException(404)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
