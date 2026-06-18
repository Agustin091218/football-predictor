"""Ensemble predictor combining Poisson + XGBoost with configurable weights.

Produces a unified EnsembleResult with blended probabilities, Monte Carlo
simulation, and per-model breakdown for debugging.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from football_predictor.domain.entities import Match, MatchResult, TeamStats
from football_predictor.domain.services import MonteCarloResult, MonteCarloSimulator, PoissonService
from football_predictor.infrastructure.ml.feature_engineering import FeatureEngineer
from football_predictor.infrastructure.ml.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    lambda_home: float
    lambda_away: float
    monte_carlo: MonteCarloResult
    poisson_probs: dict[str, float]
    xgb_probs: dict[str, float] | None
    weights_used: dict[str, float]
    model_used: str
    confidence: float

    @property
    def predicted_result(self) -> MatchResult:
        probs = {
            MatchResult.HOME_WIN: self.prob_home_win,
            MatchResult.DRAW: self.prob_draw,
            MatchResult.AWAY_WIN: self.prob_away_win,
        }
        return max(probs, key=probs.get)

    @property
    def probabilities_are_valid(self) -> bool:
        total = self.prob_home_win + self.prob_draw + self.prob_away_win
        return abs(total - 1.0) < 0.01

    def as_dict(self) -> dict[str, Any]:
        return {
            "prob_home_win": round(self.prob_home_win, 4),
            "prob_draw": round(self.prob_draw, 4),
            "prob_away_win": round(self.prob_away_win, 4),
            "predicted_result": self.predicted_result.value,
            "lambda_home": round(self.lambda_home, 3),
            "lambda_away": round(self.lambda_away, 3),
            "model_used": self.model_used,
            "weights_used": self.weights_used,
            "confidence": self.confidence,
            "monte_carlo": self.monte_carlo.as_dict(),
            "poisson_probs": self.poisson_probs,
            "xgb_probs": self.xgb_probs,
        }


class EnsemblePredictor:
    """Blends Poisson and XGBoost predictions with configurable weights.

    Works even when XGBoost is not trained — falls back to Poisson-only.
    LLM-adjusted weights from the orchestrator can dynamically shift the
    balance between models on a per-match basis.
    """

    def __init__(
        self,
        poisson_service: PoissonService,
        xgboost_predictor: XGBoostPredictor,
        feature_engineer: FeatureEngineer,
        monte_carlo: MonteCarloSimulator,
        poisson_weight: float = 0.5,
        xgboost_weight: float = 0.5,
    ) -> None:
        total = poisson_weight + xgboost_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Los pesos deben sumar 1.0, suman {total:.3f}")
        self._poisson = poisson_service
        self._xgb = xgboost_predictor
        self._engineer = feature_engineer
        self._mc = monte_carlo
        self._pw = poisson_weight
        self._xw = xgboost_weight

    def predict(
        self,
        match: Match,
        signal_outputs: dict[str, Any],
        home_stats: TeamStats | None,
        away_stats: TeamStats | None,
        finished_matches: list[Match],
        llm_adjusted_weights: dict[str, float] | None = None,
    ) -> EnsembleResult:
        poisson_signal = signal_outputs.get("poisson", {})
        poisson_value = poisson_signal.get("value", {})

        poisson_probs = {
            "prob_home_win": float(poisson_value.get("prob_home_win", 0.0)),
            "prob_draw": float(poisson_value.get("prob_draw", 0.0)),
            "prob_away_win": float(poisson_value.get("prob_away_win", 0.0)),
        }
        lambda_home = float(poisson_value.get("lambda_home", 1.5))
        lambda_away = float(poisson_value.get("lambda_away", 1.1))

        xgb_probs: dict[str, float] | None = None
        xgb_available = False

        if self._xgb.is_trained:
            try:
                features = self._engineer.build_features_for_match(
                    match, finished_matches, home_stats, away_stats
                )
                xgb_result = self._xgb.predict_single(features)
                xgb_probs = {
                    "prob_home_win": float(xgb_result["prob_home_win"]),
                    "prob_draw": float(xgb_result["prob_draw"]),
                    "prob_away_win": float(xgb_result["prob_away_win"]),
                }
                xgb_available = True
            except Exception as exc:
                logger.warning("XGBoost predict failed: %s. Using Poisson only.", exc)

        if xgb_available and xgb_probs:
            pw = self._pw
            xw = self._xw

            if llm_adjusted_weights:
                poisson_node_weight = float(llm_adjusted_weights.get("poisson", 1.0))
                scale = max(0.0, min(poisson_node_weight, 3.0))
                pw_adj = pw * scale
                xw_adj = xw * (2.0 - scale)
                total_adj = pw_adj + xw_adj
                if total_adj > 0:
                    pw = pw_adj / total_adj
                    xw = xw_adj / total_adj
                pw = max(0.0, min(1.0, pw))
                xw = max(0.0, min(1.0, xw))
                w_total = pw + xw
                if w_total > 0:
                    pw /= w_total
                    xw /= w_total

            prob_home = pw * poisson_probs["prob_home_win"] + xw * xgb_probs["prob_home_win"]
            prob_draw = pw * poisson_probs["prob_draw"] + xw * xgb_probs["prob_draw"]
            prob_away = pw * poisson_probs["prob_away_win"] + xw * xgb_probs["prob_away_win"]

            weights_used = {"poisson": pw, "xgboost": xw}
            model_used = "ensemble"
        else:
            prob_home = poisson_probs["prob_home_win"]
            prob_draw = poisson_probs["prob_draw"]
            prob_away = poisson_probs["prob_away_win"]
            weights_used = {"poisson": 1.0, "xgboost": 0.0}
            model_used = "poisson_only"

        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total

        mc_result = self._mc.simulate(lambda_home, lambda_away)

        entropy = -sum(p * math.log(p) for p in (prob_home, prob_draw, prob_away) if p > 0)
        max_entropy = math.log(3)
        base_confidence = round(1.0 - entropy / max_entropy, 4)

        if xgb_available and xgb_probs:
            poisson_result = max(poisson_probs, key=poisson_probs.get)
            xgb_result_key = max(xgb_probs, key=xgb_probs.get)
            agreement_bonus = 0.05 if poisson_result == xgb_result_key else -0.05
            final_confidence = min(1.0, max(0.0, base_confidence + agreement_bonus))
        else:
            final_confidence = base_confidence

        return EnsembleResult(
            prob_home_win=prob_home,
            prob_draw=prob_draw,
            prob_away_win=prob_away,
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            monte_carlo=mc_result,
            poisson_probs=poisson_probs,
            xgb_probs=xgb_probs,
            weights_used=weights_used,
            model_used=model_used,
            confidence=final_confidence,
        )

    def set_weights(self, poisson_weight: float, xgboost_weight: float) -> None:
        total = poisson_weight + xgboost_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Los pesos deben sumar 1.0, suman {total:.3f}")
        self._pw = poisson_weight
        self._xw = xgboost_weight
        logger.info(
            "Pesos actualizados: Poisson=%.2f XGBoost=%.2f",
            self._pw,
            self._xw,
        )
