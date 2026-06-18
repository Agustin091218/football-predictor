"""Backtesting: evaluate model performance retroactively without data leakage.

Simulates how the model would have performed historically by training
on early-season data and testing on later matchdays.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from football_predictor.domain.repositories import MatchRepository, TeamStatsRepository
from football_predictor.domain.services import (
    MonteCarloSimulator,
    PoissonService,
    TeamStrengths,
)
from football_predictor.infrastructure.ml.calibration import ProbabilityCalibrator
from football_predictor.infrastructure.ml.feature_engineering import FeatureEngineer
from football_predictor.infrastructure.ml.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)

_RESULT_IDX = {"1": 0, "X": 1, "2": 2}
_IDX_RESULT = {0: "1", 1: "X", 2: "2"}


@dataclass
class BacktestResult:
    league_id: str
    season: str
    train_until_matchday: int
    n_matches_train: int
    n_matches_test: int
    accuracy: float
    log_loss: float
    brier_score: float
    breakdown: dict[str, dict[str, Any]]
    calibration_data: list[dict[str, Any]]
    best_threshold: float
    best_threshold_accuracy: float
    best_threshold_n: int
    roi_flat_betting: float
    model_used: str
    evaluated_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "league_id": self.league_id,
            "season": self.season,
            "train_until_matchday": self.train_until_matchday,
            "n_matches_train": self.n_matches_train,
            "n_matches_test": self.n_matches_test,
            "accuracy": round(self.accuracy, 4),
            "log_loss": round(self.log_loss, 4),
            "brier_score": round(self.brier_score, 4),
            "breakdown": self.breakdown,
            "calibration_data": self.calibration_data,
            "best_threshold": round(self.best_threshold, 2),
            "best_threshold_accuracy": round(self.best_threshold_accuracy, 4),
            "best_threshold_n": self.best_threshold_n,
            "roi_flat_betting": round(self.roi_flat_betting, 2),
            "model_used": self.model_used,
            "evaluated_at": self.evaluated_at,
        }


class BacktestingUseCase:
    def __init__(
        self,
        match_repo: MatchRepository,
        stats_repo: TeamStatsRepository,
        poisson_service: PoissonService,
        feature_engineer: FeatureEngineer,
        xgboost_predictor: XGBoostPredictor,
        monte_carlo: MonteCarloSimulator,
        calibrator: ProbabilityCalibrator | None = None,
        n_simulations_backtest: int = 1_000,
    ) -> None:
        self._match_repo = match_repo
        self._stats_repo = stats_repo
        self._poisson_svc = poisson_service
        self._feature_engineer = feature_engineer
        self._xgb_predictor = xgboost_predictor
        self._mc = monte_carlo
        self._calibrator = calibrator
        self._n_sim_backtest = n_simulations_backtest

    def execute(
        self,
        league_id: str,
        season: str,
        train_until_matchday: int = 20,
    ) -> BacktestResult:
        all_matches = self._match_repo.get_finished_matches(league_id, season)  # type: ignore[attr-defined]
        all_matches.sort(key=lambda m: m.match_date)

        if len(all_matches) < 30:
            raise ValueError(f"Se necesitan al menos 30 partidos. Hay {len(all_matches)}.")

        train_matches = [m for m in all_matches if (m.matchday or 0) <= train_until_matchday]
        test_matches = [m for m in all_matches if (m.matchday or 0) > train_until_matchday]

        if len(train_matches) < 15 or len(test_matches) < 10:
            raise ValueError(
                f"División train/test insuficiente: "
                f"train={len(train_matches)}, test={len(test_matches)}. "
                f"Ajustar train_until_matchday."
            )

        logger.info(
            "Backtest %s/%s: train=%d test=%d",
            league_id,
            season,
            len(train_matches),
            len(test_matches),
        )

        xgb_local = XGBoostPredictor()
        model_used = "poisson_only_backtest"
        try:
            X_train, y_train = self._feature_engineer.build_dataset(league_id, season)
            n_train = min(len(X_train), len(train_matches))
            X_train = X_train.iloc[:n_train]
            y_train = y_train.iloc[:n_train]
            if len(X_train) >= 15:
                xgb_local.train(X_train, y_train)
                model_used = "ensemble_backtest"
        except Exception as exc:
            logger.warning("XGBoost backtest failed: %s. Using Poisson only.", exc)

        predictions_data: list[dict[str, Any]] = []
        for match in test_matches:
            if match.score is None or match.score.result is None:
                continue
            try:
                past = [m for m in all_matches if m.match_date < match.match_date]
                home_stats = self._stats_repo.get_stats(  # type: ignore[attr-defined]
                    match.home_team.id, league_id, season
                )
                away_stats = self._stats_repo.get_stats(  # type: ignore[attr-defined]
                    match.away_team.id, league_id, season
                )

                hg_list = [m.score.home_goals for m in past if m.score and m.score.is_complete]  # type: ignore[union-attr]
                ag_list = [m.score.away_goals for m in past if m.score and m.score.is_complete]  # type: ignore[union-attr]
                params = self._poisson_svc.calculate_league_params_from_matches(
                    hg_list,
                    ag_list,
                    league_id,
                    season,  # type: ignore[arg-type]
                )

                if home_stats:
                    hs = self._poisson_svc.calculate_team_strengths_from_stats(home_stats, params)
                else:
                    hs = TeamStrengths(team_id=match.home_team.id)
                if away_stats:
                    astr = self._poisson_svc.calculate_team_strengths_from_stats(away_stats, params)
                else:
                    astr = TeamStrengths(team_id=match.away_team.id)

                poisson_r = self._poisson_svc.predict(hs, astr, params)
                ph = poisson_r["prob_home_win"]
                pd_ = poisson_r["prob_draw"]
                pa = poisson_r["prob_away_win"]

                if xgb_local.is_trained:
                    try:
                        feats = self._feature_engineer.build_features_for_match(
                            match, past, home_stats, away_stats
                        )
                        xgb_out = xgb_local.predict_single(feats)
                        ph = 0.5 * ph + 0.5 * xgb_out["prob_home_win"]
                        pd_ = 0.5 * pd_ + 0.5 * xgb_out["prob_draw"]
                        pa = 0.5 * pa + 0.5 * xgb_out["prob_away_win"]
                        total = ph + pd_ + pa
                        ph, pd_, pa = ph / total, pd_ / total, pa / total
                    except Exception:
                        pass

                if self._calibrator and self._calibrator.is_fitted:
                    ph, pd_, pa = self._calibrator.transform_single(ph, pd_, pa)

                entropy = -sum(p * math.log(p) for p in (ph, pd_, pa) if p > 0)
                confidence = round(1.0 - entropy / math.log(3), 4)

                probs = {"1": ph, "X": pd_, "2": pa}
                predicted = max(probs, key=probs.get)
                actual = _RESULT_MAP[match.score.result.value]

                predictions_data.append(
                    {
                        "predicted_result": predicted,
                        "actual_result": actual,
                        "prob_home": ph,
                        "prob_draw": pd_,
                        "prob_away": pa,
                        "confidence": confidence,
                    }
                )
            except Exception as exc:
                logger.warning("Error predicting %d in backtest: %s", match.id, exc)

        n = len(predictions_data)
        if n == 0:
            raise ValueError("No se pudieron generar predicciones en backtest")

        correct = sum(1 for p in predictions_data if p["predicted_result"] == p["actual_result"])
        accuracy = correct / n
        ll = self._calculate_log_loss(predictions_data)
        brier = self._calculate_brier_score(predictions_data)
        breakdown = self._calculate_breakdown(predictions_data)
        calib = self._calculate_calibration(predictions_data)
        best_t, best_a, best_n = self._find_best_threshold(predictions_data)
        roi = self._calculate_roi(predictions_data)

        return BacktestResult(
            league_id=league_id,
            season=season,
            train_until_matchday=train_until_matchday,
            n_matches_train=len(train_matches),
            n_matches_test=len(test_matches),
            accuracy=accuracy,
            log_loss=ll,
            brier_score=brier,
            breakdown=breakdown,
            calibration_data=calib,
            best_threshold=best_t,
            best_threshold_accuracy=best_a,
            best_threshold_n=best_n,
            roi_flat_betting=roi,
            model_used=model_used,
            evaluated_at=datetime.now().isoformat(),
        )

    @staticmethod
    def _calculate_log_loss(predictions_data: list[dict]) -> float:
        total_loss = 0.0
        for p in predictions_data:
            idx = _RESULT_IDX.get(p["actual_result"], 1)
            probs = [p["prob_home"], p["prob_draw"], p["prob_away"]]
            total_loss += -math.log(max(probs[idx], 1e-10))
        return total_loss / len(predictions_data)

    @staticmethod
    def _calculate_brier_score(predictions_data: list[dict]) -> float:
        total_bs = 0.0
        for p in predictions_data:
            idx = _RESULT_IDX.get(p["actual_result"], 1)
            actual_vec = [0.0, 0.0, 0.0]
            actual_vec[idx] = 1.0
            pred_vec = [p["prob_home"], p["prob_draw"], p["prob_away"]]
            total_bs += sum((pred_vec[i] - actual_vec[i]) ** 2 for i in range(3))
        return total_bs / len(predictions_data)

    @staticmethod
    def _calculate_breakdown(predictions_data: list[dict]) -> dict:
        breakdown: dict = {}
        for key in ("1", "X", "2"):
            predicted = sum(1 for p in predictions_data if p["predicted_result"] == key)
            correct = sum(
                1
                for p in predictions_data
                if p["predicted_result"] == key and p["actual_result"] == key
            )
            breakdown[key] = {
                "predicted": predicted,
                "correct": correct,
                "accuracy": round(correct / predicted, 4) if predicted > 0 else 0.0,
            }
        return breakdown

    @staticmethod
    def _calculate_calibration(predictions_data: list[dict]) -> list[dict]:
        bins: list[dict] = []
        for lo in range(0, 100, 10):
            hi = lo + 10
            subset = [p for p in predictions_data if lo / 100 <= p["confidence"] < hi / 100]
            if not subset:
                continue
            correct = sum(1 for p in subset if p["predicted_result"] == p["actual_result"])
            bins.append(
                {
                    "bin": f"{lo}-{hi}%",
                    "predicted": len(subset),
                    "correct": correct,
                    "real_accuracy": round(correct / len(subset), 4),
                    "confidence_mid": round((lo + hi) / 200, 2),
                }
            )
        return bins

    @staticmethod
    def _find_best_threshold(
        predictions_data: list[dict],
        thresholds: list[float] | None = None,
    ) -> tuple[float, float, int]:
        if thresholds is None:
            thresholds = [i / 10 for i in range(10)]
        best = (0.0, 0.0, 0)
        for t in thresholds:
            subset = [p for p in predictions_data if p["confidence"] >= t]
            if not subset:
                continue
            acc = sum(1 for p in subset if p["predicted_result"] == p["actual_result"]) / len(
                subset
            )
            if acc > best[1] or (acc == best[1] and len(subset) > best[2]):
                best = (t, acc, len(subset))
        return best

    @staticmethod
    def _calculate_roi(predictions_data: list[dict]) -> float:
        total_gain = 0.0
        for p in predictions_data:
            pred_idx = _RESULT_IDX.get(p["predicted_result"], 1)
            probs = [p["prob_home"], p["prob_draw"], p["prob_away"]]
            prob_pred = max(probs[pred_idx], 0.01)
            fair_odds = 1.0 / prob_pred
            if p["predicted_result"] == p["actual_result"]:
                total_gain += fair_odds - 1.0
            else:
                total_gain -= 1.0
        return (total_gain / len(predictions_data)) * 100


_RESULT_MAP = {"home_win": "1", "draw": "X", "away_win": "2"}
