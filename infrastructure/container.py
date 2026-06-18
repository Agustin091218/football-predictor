"""Dependency injection container — singleton shared by API and CLI.

Centralises all wiring so FastAPI and Typer use the same instances.
Avoids duplicate connections and inconsistent configuration.
"""

from __future__ import annotations

import logging
import os

from football_predictor.infrastructure.logging import setup_logging

setup_logging()

from football_predictor.infrastructure.config import validate_settings

validate_settings()

from football_predictor.application.compute_stats import ComputeStatsUseCase
from football_predictor.application.fetch_and_store import FetchAndStoreUseCase
from football_predictor.application.predict_match import PredictMatchUseCase
from football_predictor.application.train_model import TrainModelUseCase
from football_predictor.domain.services import MonteCarloSimulator, PoissonService
from football_predictor.infrastructure.api_clients.football_data_client import (
    FootballDataOrgClient,
)
from football_predictor.infrastructure.config import DEFAULT_DB_PATH, DEFAULT_MODELS_DIR
from football_predictor.infrastructure.ml.ensemble import EnsemblePredictor
from football_predictor.infrastructure.ml.feature_engineering import FeatureEngineer
from football_predictor.infrastructure.ml.llm_orchestrator import LLMOrchestrator
from football_predictor.infrastructure.ml.model_store_impl import LocalModelStore
from football_predictor.infrastructure.ml.signal_nodes import (
    ContextNode,
    EloNode,
    FormNode,
    HeadToHeadNode,
    PoissonSignalNode,
)
from football_predictor.infrastructure.ml.xgboost_model import XGBoostPredictor
from football_predictor.infrastructure.repositories.sqlite_match_repo import (
    SqliteMatchRepository,
)
from football_predictor.infrastructure.repositories.sqlite_prediction_repo import (
    SqlitePredictionRepository,
)
from football_predictor.infrastructure.repositories.sqlite_stats_repo import (
    SqliteStatsRepository,
)

logger = logging.getLogger(__name__)


class Container:
    """Centralised dependency container.

    Reads configuration from environment variables, wires all
    infrastructure, domain services, ML components, and use cases.
    Intended as a module-level singleton.
    """

    def __init__(self) -> None:
        self.db_path = os.getenv("FOOTBALL_DB_PATH", DEFAULT_DB_PATH)
        self.models_dir = os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR)
        self.football_data_token = os.getenv("FOOTBALL_DATA_TOKEN", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.poisson_weight = float(os.getenv("POISSON_WEIGHT", "0.5"))
        self.xgboost_weight = float(os.getenv("XGBOOST_WEIGHT", "0.5"))
        self.n_simulations = int(os.getenv("N_SIMULATIONS", "10000"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        logging.basicConfig(level=getattr(logging, self.log_level, logging.INFO))
        self._build()

    def _build(self) -> None:
        # Repositories
        self.match_repo = SqliteMatchRepository(self.db_path)
        self.stats_repo = SqliteStatsRepository(self.db_path)
        self.prediction_repo = SqlitePredictionRepository(self.db_path)
        self.model_store = LocalModelStore(self.models_dir)

        # Domain services
        self.poisson_service = PoissonService(max_goals=8)
        self.mc_simulator = MonteCarloSimulator(n_simulations=self.n_simulations, seed=None)

        # Signal nodes
        self.signal_nodes = [
            FormNode(),
            EloNode(),
            HeadToHeadNode(),
            ContextNode(),
            PoissonSignalNode(self.poisson_service),
        ]

        # Feature engineering & ML
        self.feature_engineer = FeatureEngineer(
            match_repo=self.match_repo,
            stats_repo=self.stats_repo,
            poisson_service=self.poisson_service,
            signal_nodes=self.signal_nodes,
        )
        self.xgboost_predictor = XGBoostPredictor()
        self._load_trained_model()

        # LLM (optional)
        self.llm_orchestrator = None
        if self.gemini_api_key:
            try:
                self.llm_orchestrator = LLMOrchestrator(api_key=self.gemini_api_key)
            except Exception as exc:
                logger.warning("Gemini not available: %s", exc)

        # Ensemble
        self.ensemble = EnsemblePredictor(
            poisson_service=self.poisson_service,
            xgboost_predictor=self.xgboost_predictor,
            feature_engineer=self.feature_engineer,
            monte_carlo=self.mc_simulator,
            poisson_weight=self.poisson_weight,
            xgboost_weight=self.xgboost_weight,
        )

        # API client (optional)
        data_source = None
        if self.football_data_token:
            data_source = FootballDataOrgClient(token=self.football_data_token)

        # Use cases
        self.fetch_and_store_uc = FetchAndStoreUseCase(
            data_source=data_source, match_repo=self.match_repo
        )
        self.compute_stats_uc = ComputeStatsUseCase(
            match_repo=self.match_repo, stats_repo=self.stats_repo
        )
        self.predict_uc = PredictMatchUseCase(
            match_repo=self.match_repo,
            stats_repo=self.stats_repo,
            poisson_service=self.poisson_service,
            prediction_repo=self.prediction_repo,
            signal_nodes=self.signal_nodes,
            monte_carlo=self.mc_simulator,
            llm_orchestrator=self.llm_orchestrator,
            ensemble=self.ensemble,
        )
        self.train_uc = TrainModelUseCase(
            match_repo=self.match_repo,
            stats_repo=self.stats_repo,
            feature_engineer=self.feature_engineer,
            xgboost_predictor=self.xgboost_predictor,
            models_dir=self.models_dir,
        )

        from football_predictor.application.world_cup_calibration import (
            WorldCupCalibrationUseCase,
            WorldCupTracker,
        )
        from football_predictor.infrastructure.ml.calibration import ProbabilityCalibrator
        from football_predictor.infrastructure.repositories.sqlite_wc_results_repo import (
            SqliteWorldCupResultsRepository,
        )

        self.wc_results_repo = SqliteWorldCupResultsRepository(self.db_path)
        self.calibrator = ProbabilityCalibrator()
        self.wc_calibration_uc = WorldCupCalibrationUseCase(
            prediction_repo=self.prediction_repo,
            match_repo=self.match_repo,
            calibrator=self.calibrator,
        )
        self.wc_tracker = WorldCupTracker(
            calibration_uc=self.wc_calibration_uc,
            prediction_repo=self.prediction_repo,
            wc_results_repo=self.wc_results_repo,
        )

    def _load_trained_model(self) -> None:
        import os

        pattern = os.path.join(self.models_dir, "xgboost_intl.json")
        if os.path.exists(pattern):
            try:
                self.xgboost_predictor.load_model(pattern)
                logger.info("Loaded XGBoost model: %s", pattern)
            except Exception as exc:
                logger.warning("Could not load XGBoost model: %s", exc)


container = Container()
