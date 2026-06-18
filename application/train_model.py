"""Use case: train the XGBoost model from historical match data."""

from __future__ import annotations

import logging
from typing import Any

from football_predictor.domain.repositories import MatchRepository, TeamStatsRepository
from football_predictor.infrastructure.ml.feature_engineering import FeatureEngineer
from football_predictor.infrastructure.ml.model_store_impl import LocalModelStore
from football_predictor.infrastructure.ml.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)

_MIN_MATCHES = 20


class TrainModelUseCase:
    def __init__(
        self,
        match_repo: MatchRepository,
        stats_repo: TeamStatsRepository,
        feature_engineer: FeatureEngineer,
        xgboost_predictor: XGBoostPredictor,
        models_dir: str = "models",
    ) -> None:
        self._match_repo = match_repo
        self._stats_repo = stats_repo
        self._engineer = feature_engineer
        self._xgb = xgboost_predictor
        self._model_store = LocalModelStore(models_dir=models_dir)

    def execute(self, league_id: str, season: str) -> dict[str, Any]:
        try:
            X, y = self._engineer.build_dataset(league_id, season)
        except ValueError as exc:
            return {"status": "insufficient_data", "error": str(exc)}

        if len(X) < _MIN_MATCHES:
            return {
                "status": "insufficient_data",
                "error": f"Need {_MIN_MATCHES} matches, got {len(X)}",
            }

        metrics = self._xgb.train(X, y)

        version = f"{league_id}_{season}"
        self._model_store.save_model(self._xgb, version=version)

        logger.info(
            "Model trained for %s/%s: acc=%.4f (%d samples)",
            league_id,
            season,
            metrics["cv_accuracy_mean"],
            len(X),
        )

        return {
            "status": "ok",
            "metrics": metrics,
            "n_samples": len(X),
            "n_features": len(X.columns),
            "model_version": version,
        }
