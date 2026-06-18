"""World Cup 2026 API router — calibration, standings, upsets, predictions.

All state is persisted in SQLite via container.wc_results_repo.
Results survive server restarts.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from football_predictor.infrastructure.container import container

router = APIRouter(prefix="/world-cup", tags=["World Cup 2026"])
logger = logging.getLogger(__name__)

_FIXTURES_PATH = os.getenv(
    "WORLD_CUP_FIXTURES_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "world_cup_fixtures.json"),
)


def _load_fixtures() -> list[dict[str, str]]:
    try:
        with open(_FIXTURES_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("World Cup fixtures not found at %s", _FIXTURES_PATH)
        return []


UPCOMING_WORLD_CUP = _load_fixtures()


class ResultBody(BaseModel):
    match_id: str | None = None
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    group: str
    match_date: str
    entered_by: str = "user"


@router.get("/calibration")
async def world_cup_calibration():
    result = container.wc_calibration_uc.execute()
    return result


@router.post("/results")
async def add_result(body: ResultBody):
    result = container.wc_tracker.add_result(
        home_team=body.home_team,
        away_team=body.away_team,
        home_goals=body.home_goals,
        away_goals=body.away_goals,
        group=body.group,
        match_date=body.match_date,
        match_id=body.match_id,
        entered_by=body.entered_by,
    )
    return result


@router.get("/results")
async def list_results():
    results = container.wc_results_repo.get_all_results()
    stats = container.wc_results_repo.get_stats()
    return {"results": results, "stats": stats, "persisted_in": container.db_path}


@router.get("/results/{match_id}")
async def get_result(match_id: str):
    result = container.wc_results_repo.get_result(match_id)
    if not result:
        raise HTTPException(404, f"Sin resultado para {match_id}")
    return result


@router.delete("/results/{match_id}")
async def delete_result(match_id: str):
    result = container.wc_tracker.delete_result(match_id)
    if not result["deleted"]:
        raise HTTPException(404, f"No existe resultado para {match_id}")
    return result


@router.get("/standings")
async def world_cup_standings():
    return container.wc_tracker.get_standings()


@router.get("/upsets")
async def world_cup_upsets(min_upset_score: float = Query(default=0.4, ge=0.0, le=1.0)):
    result = container.wc_calibration_uc.execute()
    upsets = [u for u in result.get("upsets", []) if u.get("upset_score", 0) >= min_upset_score]
    return {"upsets": upsets, "total": len(upsets)}


@router.get("/team/{team_name}/predictions")
async def team_predictions(team_name: str):
    recent = container.prediction_repo.get_recent(limit=100)
    matches = []
    tn = team_name.lower()
    for p in recent:
        hn = p.match.home_team.name.lower()
        an = p.match.away_team.name.lower()
        if tn in hn or tn in an:
            matches.append(
                {
                    "match_id": str(p.match.id),
                    "home": p.match.home_team.name,
                    "away": p.match.away_team.name,
                    "predicted": p.predicted_result.value,
                    "prob_home": round(p.prob_home_win, 4),
                    "prob_draw": round(p.prob_draw, 4),
                    "prob_away": round(p.prob_away_win, 4),
                    "confidence": p.confidence,
                    "predicted_at": p.predicted_at.isoformat(),
                }
            )
    return {"team": team_name, "predictions": len(matches), "matches": matches}


@router.post("/predict-next")
async def predict_next_matches():
    from football_predictor.domain.entities import CompetitionType, League, Match, MatchStatus, Team

    generated: list[dict[str, Any]] = []
    for i, m in enumerate(UPCOMING_WORLD_CUP):
        mid = 80000 + i
        try:
            hteam = Team(
                id=mid,
                name=m["home"],
                short_name=m["home"][:3].upper(),
                tla=m["home"][:3].upper(),
                country="?",
            )
            ateam = Team(
                id=mid + 1000,
                name=m["away"],
                short_name=m["away"][:3].upper(),
                tla=m["away"][:3].upper(),
                country="?",
            )
            league = League(
                id="WC2026",
                name="World Cup 2026",
                country="International",
                competition_type=CompetitionType.INTERNATIONAL,
                season="2026",
            )
            dt = datetime.strptime(m["date"], "%Y-%m-%d")
            match = Match(
                id=mid,
                home_team=hteam,
                away_team=ateam,
                league=league,
                match_date=dt,
                status=MatchStatus.SCHEDULED,
                matchday=1,
            )
            container.match_repo.save(match)
            pred = container.predict_uc.execute(str(mid))
            generated.append(
                {
                    "match": f"{m['home']} vs {m['away']}",
                    "group": m["group"],
                    "date": m["date"],
                    "prob_home": round(pred.prob_home_win, 4),
                    "prob_draw": round(pred.prob_draw, 4),
                    "prob_away": round(pred.prob_away_win, 4),
                    "predicted": pred.predicted_result.value,
                    "confidence": pred.confidence,
                }
            )
        except Exception as exc:
            generated.append(
                {"match": f"{m['home']} vs {m['away']}", "status": "error", "error": str(exc)}
            )
    return {"generated": len(generated), "predictions": generated}
