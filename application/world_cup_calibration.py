"""World Cup 2026 calibration: compare predictions against real results.

Hardcodes known match results and evaluates the prediction system
against ground truth. Recalibrates probability estimates when enough
data is available.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from football_predictor.domain.repositories import (
    MatchRepository,
    PredictionRepository,
    WorldCupResultsRepository,
)
from football_predictor.infrastructure.ml.calibration import ProbabilityCalibrator

logger = logging.getLogger(__name__)

WORLD_CUP_2026_RESULTS: list[dict[str, Any]] = [
    {
        "match_id": "wc2026_A1",
        "home_team": "Mexico",
        "away_team": "South Africa",
        "group": "A",
        "match_date": "2026-06-11",
        "home_goals": 2,
        "away_goals": 0,
        "actual_result": "1",
    },
    {
        "match_id": "wc2026_A2",
        "home_team": "South Korea",
        "away_team": "Czech Republic",
        "group": "A",
        "match_date": "2026-06-11",
        "home_goals": 2,
        "away_goals": 1,
        "actual_result": "1",
    },
    {
        "match_id": "wc2026_B1",
        "home_team": "Canada",
        "away_team": "Bosnia Herzegovina",
        "group": "B",
        "match_date": "2026-06-12",
        "home_goals": 1,
        "away_goals": 1,
        "actual_result": "X",
    },
    {
        "match_id": "wc2026_B2",
        "home_team": "Qatar",
        "away_team": "Switzerland",
        "group": "B",
        "match_date": "2026-06-12",
        "home_goals": 1,
        "away_goals": 1,
        "actual_result": "X",
    },
    {
        "match_id": "wc2026_D1",
        "home_team": "United States",
        "away_team": "Paraguay",
        "group": "D",
        "match_date": "2026-06-12",
        "home_goals": 4,
        "away_goals": 1,
        "actual_result": "1",
    },
    {
        "match_id": "wc2026_C1",
        "home_team": "Brazil",
        "away_team": "Morocco",
        "group": "C",
        "match_date": "2026-06-13",
        "home_goals": 1,
        "away_goals": 1,
        "actual_result": "X",
    },
    {
        "match_id": "wc2026_E1",
        "home_team": "Australia",
        "away_team": "Turkey",
        "group": "E",
        "match_date": "2026-06-13",
        "home_goals": 2,
        "away_goals": 0,
        "actual_result": "1",
    },
    {
        "match_id": "wc2026_F1",
        "home_team": "Germany",
        "away_team": "Curaçao",
        "group": "F",
        "match_date": "2026-06-14",
        "home_goals": None,
        "away_goals": None,
        "actual_result": None,
    },
    {
        "match_id": "wc2026_F2",
        "home_team": "Netherlands",
        "away_team": "Japan",
        "group": "F",
        "match_date": "2026-06-14",
        "home_goals": None,
        "away_goals": None,
        "actual_result": None,
    },
]

_RESULT_IDX = {"1": 0, "X": 1, "2": 2}
_ENUM_TO_DISPLAY = {"home_win": "1", "draw": "X", "away_win": "2"}


@dataclass
class WorldCupMatchResult:
    match_id: str
    home_team: str
    away_team: str
    group: str
    match_date: str
    home_goals: int | None
    away_goals: int | None
    actual_result: str | None
    predicted_result: str | None = None
    prob_home_win: float | None = None
    prob_draw: float | None = None
    prob_away_win: float | None = None
    confidence: float | None = None
    was_correct: bool | None = None
    log_loss_contribution: float | None = None
    brier_contribution: float | None = None
    upset_score: float | None = None


class WorldCupCalibrationUseCase:
    def __init__(
        self,
        prediction_repo: PredictionRepository,
        match_repo: MatchRepository,
        calibrator: ProbabilityCalibrator,
    ) -> None:
        self._prediction_repo = prediction_repo
        self._match_repo = match_repo
        self._calibrator = calibrator

    def execute(self, ground_truth: list[dict] | None = None) -> dict[str, Any]:
        if ground_truth is None:
            ground_truth = list(WORLD_CUP_2026_RESULTS)

        results: list[WorldCupMatchResult] = []
        recent = self._prediction_repo.get_recent(limit=200)

        for gt in ground_truth:
            r = WorldCupMatchResult(
                match_id=gt["match_id"],
                home_team=gt["home_team"],
                away_team=gt["away_team"],
                group=gt["group"],
                match_date=gt["match_date"],
                home_goals=gt.get("home_goals"),
                away_goals=gt.get("away_goals"),
                actual_result=gt.get("actual_result"),
            )
            if r.actual_result is None:
                results.append(r)
                continue

            pred = self._find_prediction(recent, gt)
            if pred is None:
                results.append(r)
                continue

            r.predicted_result = _ENUM_TO_DISPLAY.get(
                pred.predicted_result.value, pred.predicted_result.value
            )
            r.prob_home_win = pred.prob_home_win
            r.prob_draw = pred.prob_draw
            r.prob_away_win = pred.prob_away_win
            r.confidence = pred.confidence

            idx = _RESULT_IDX[r.actual_result]
            probs = [r.prob_home_win, r.prob_draw, r.prob_away_win]
            r.was_correct = r.predicted_result == r.actual_result
            r.log_loss_contribution = -math.log(max(probs[idx], 1e-10))
            actual_vec = [0.0, 0.0, 0.0]
            actual_vec[idx] = 1.0
            r.brier_contribution = sum((probs[i] - actual_vec[i]) ** 2 for i in range(3))
            r.upset_score = 1.0 - probs[idx]
            results.append(r)

        evaluated = [r for r in results if r.was_correct is not None]
        n = len(evaluated)
        metrics: dict[str, Any] = {
            "accuracy": 0.0,
            "log_loss": 0.0,
            "brier_score": 0.0,
            "n_correctos": 0,
            "n_evaluados": n,
        }
        upsets: list[WorldCupMatchResult] = []
        by_group: dict[str, dict] = {}

        if n > 0:
            corrects = sum(1 for r in evaluated if r.was_correct)
            metrics["accuracy"] = corrects / n
            metrics["log_loss"] = sum(r.log_loss_contribution for r in evaluated) / n  # type: ignore[arg-type]
            metrics["brier_score"] = sum(r.brier_contribution for r in evaluated) / n  # type: ignore[arg-type]
            metrics["n_correctos"] = corrects
            upsets = [r for r in evaluated if r.upset_score and r.upset_score > 0.5]

            for r in evaluated:
                g = r.group
                if g not in by_group:
                    by_group[g] = {"correct": 0, "total": 0}
                by_group[g]["total"] += 1
                if r.was_correct:
                    by_group[g]["correct"] += 1
            for g in by_group:
                by_group[g]["accuracy"] = by_group[g]["correct"] / by_group[g]["total"]

        cal_fitted = False
        cal_note = None
        if n >= 5:
            y_true = [_RESULT_IDX[r.actual_result] for r in evaluated]  # type: ignore[index]
            y_prob = [[r.prob_home_win, r.prob_draw, r.prob_away_win] for r in evaluated]
            self._calibrator.fit(y_true, y_prob)
            cal_fitted = True
            logger.info("Calibrator fitted on %d World Cup matches", n)
        else:
            cal_note = f"Solo {n} partidos. Mínimo 5 para calibrar."

        return {
            "status": "ok",
            "mundial_2026": {
                "partidos_jugados": len([r for r in ground_truth if r.get("actual_result")]),
                "partidos_con_prediccion": n,
                "cobertura": n / len([r for r in ground_truth if r.get("actual_result")])
                if n > 0
                else 0.0,
            },
            "metricas": metrics,
            "por_grupo": by_group,
            "upsets": [
                {
                    "partido": f"{r.home_team} vs {r.away_team}",
                    "resultado": f"{r.home_goals}-{r.away_goals}",
                    "prediccion": r.predicted_result,
                    "prob_resultado_real": round(1.0 - (r.upset_score or 0), 4),
                    "upset_score": round(r.upset_score or 0, 4),
                }
                for r in sorted(upsets, key=lambda x: x.upset_score or 0, reverse=True)
            ],
            "calibration": {
                "fitted": cal_fitted,
                "n_samples": n,
                "note": cal_note,
            },
        }

    def _find_prediction(self, recent: list, gt: dict) -> Any:
        for p in recent:
            if hasattr(p, "match"):
                hn = p.match.home_team.name.lower()
                an = p.match.away_team.name.lower()
                if hn == gt["home_team"].lower() and an == gt["away_team"].lower():
                    return p
                if hn and gt["home_team"].lower() in hn and an and gt["away_team"].lower() in an:
                    return p
        return None


class WorldCupTracker:
    def __init__(
        self,
        calibration_uc: WorldCupCalibrationUseCase,
        prediction_repo: PredictionRepository,
        wc_results_repo: WorldCupResultsRepository,
    ) -> None:
        self.calibration_uc = calibration_uc
        self._prediction_repo = prediction_repo
        self._wc_repo = wc_results_repo

    def add_result(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        group: str,
        match_date: str,
        match_id: str = None,
        entered_by: str = "user",
    ) -> dict[str, Any]:
        if not match_id:
            match_id = f"wc2026_{group}_{home_team[:3].upper()}"

        if home_goals > away_goals:
            actual = "1"
        elif home_goals == away_goals:
            actual = "X"
        else:
            actual = "2"

        self._wc_repo.save_result(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            group_letter=group,
            match_date=match_date,
            home_goals=home_goals,
            away_goals=away_goals,
            entered_by=entered_by,
        )

        all_results = self._merge_results()
        recal = self.calibration_uc.execute(all_results)
        return {
            "result_added": f"{home_team} {home_goals}-{away_goals} {away_team}",
            "match_id": match_id,
            "actual_result": actual,
            "total_results": len(self._wc_repo.get_all_results()),
            "recalibration": recal["calibration"],
            "current_accuracy": recal["metricas"].get("accuracy"),
            "persisted": True,
        }

    def _merge_results(self) -> list[dict[str, Any]]:
        base: dict[str, dict] = {r["match_id"]: dict(r) for r in WORLD_CUP_2026_RESULTS}
        for db_result in self._wc_repo.get_all_results():
            mid = db_result["match_id"]
            if mid in base:
                base[mid]["home_goals"] = db_result["home_goals"]
                base[mid]["away_goals"] = db_result["away_goals"]
                base[mid]["actual_result"] = db_result["actual_result"]
            else:
                base[mid] = {
                    "match_id": mid,
                    "home_team": db_result["home_team"],
                    "away_team": db_result["away_team"],
                    "group": db_result["group_letter"],
                    "match_date": db_result["match_date"],
                    "home_goals": db_result["home_goals"],
                    "away_goals": db_result["away_goals"],
                    "actual_result": db_result["actual_result"],
                }
        return list(base.values())

    def get_standings(self) -> dict[str, list[dict]]:
        return self._calc_standings(self._merge_results())

    @staticmethod
    def _calc_standings(results: list[dict]) -> dict[str, list[dict]]:
        groups: dict[str, dict[str, dict]] = {}
        for r in results:
            if r.get("actual_result") is None:
                continue
            g = r["group"]
            if g not in groups:
                groups[g] = {}
            for team, gf, ga, res in [
                (r["home_team"], r["home_goals"], r["away_goals"], r["actual_result"]),
                (
                    r["away_team"],
                    r["away_goals"],
                    r["home_goals"],
                    "2" if r["actual_result"] == "1" else "1" if r["actual_result"] == "2" else "X",
                ),
            ]:
                if team not in groups[g]:
                    groups[g][team] = {
                        "team": team,
                        "pj": 0,
                        "pg": 0,
                        "pe": 0,
                        "pp": 0,
                        "gf": 0,
                        "gc": 0,
                        "pts": 0,
                    }
                s = groups[g][team]
                s["pj"] += 1
                s["gf"] += gf
                s["gc"] += ga
                if res == "1":
                    s["pg"] += 1
                    s["pts"] += 3
                elif res == "X":
                    s["pe"] += 1
                    s["pts"] += 1
                else:
                    s["pp"] += 1
        result: dict[str, list[dict]] = {}
        for g in sorted(groups):
            result[g] = sorted(
                groups[g].values(),
                key=lambda x: (x["pts"], x["gf"] - x["gc"], x["gf"]),
                reverse=True,
            )
        return result

    def get_result(self, match_id: str) -> dict | None:
        return self._wc_repo.get_result(match_id)

    def delete_result(self, match_id: str) -> dict:
        deleted = self._wc_repo.delete_result(match_id)
        return {
            "deleted": deleted,
            "match_id": match_id,
            "message": "Resultado eliminado" if deleted else "No encontrado",
        }
