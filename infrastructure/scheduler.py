"""Automated job scheduler for periodic model updates.

Uses APScheduler BackgroundScheduler — runs in a separate thread,
does not block the FastAPI/uvicorn event loop.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from football_predictor.infrastructure.config import get_all_league_ids
from football_predictor.infrastructure.container import Container

logger = logging.getLogger(__name__)


class PredictionScheduler:
    """Periodic background jobs: fetch, stats, predict, retrain.

    Each job calls use cases from the shared Container.
    Jobs are thread-safe because use cases open their own SQLite connections.
    """

    def __init__(self, container: Container) -> None:
        self._container = container
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._setup_jobs()

    def _setup_jobs(self) -> None:
        self._scheduler.add_job(
            func=self._fetch_all_leagues,
            trigger=CronTrigger(hour=6, minute=0),
            id="fetch_daily",
            name="Fetch partidos diario",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        self._scheduler.add_job(
            func=self._compute_all_stats,
            trigger=CronTrigger(day_of_week="mon", hour=7, minute=0),
            id="compute_stats_weekly",
            name="Calcular stats semanal",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        self._scheduler.add_job(
            func=self._predict_upcoming,
            trigger=CronTrigger(hour=8, minute=0),
            id="predict_upcoming_daily",
            name="Predecir próximos partidos",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        self._scheduler.add_job(
            func=self._retrain_all,
            trigger=CronTrigger(day_of_week="mon", hour=9, minute=0),
            id="retrain_weekly",
            name="Reentrenar modelos semanal",
            replace_existing=True,
            misfire_grace_time=7200,
        )

    def _fetch_all_leagues(self) -> None:
        if not self._container.football_data_token:
            logger.warning("Scheduler: FOOTBALL_DATA_TOKEN no configurado")
            return
        season = str(datetime.now().year)
        for league_id in get_all_league_ids():
            try:
                result = self._container.fetch_and_store_uc.execute(league_id, season)
                logger.info(
                    "Scheduler fetch %s: %s partidos",
                    league_id,
                    result.get("matches_fetched", 0),
                )
            except Exception as exc:
                logger.error("Scheduler fetch %s falló: %s", league_id, exc, exc_info=True)

    def _compute_all_stats(self) -> None:
        season = str(datetime.now().year)
        for league_id in get_all_league_ids():
            try:
                result = self._container.compute_stats_uc.execute(league_id, season)
                logger.info(
                    "Scheduler stats %s: %s equipos",
                    league_id,
                    result.get("teams_computed", 0),
                )
            except Exception as exc:
                logger.error("Scheduler stats %s falló: %s", league_id, exc, exc_info=True)

    def _predict_upcoming(self) -> None:
        for league_id in get_all_league_ids():
            try:
                upcoming = self._container.match_repo.get_upcoming_matches(league_id, days_ahead=3)
                predicted = 0
                for match in upcoming:
                    try:
                        self._container.predict_uc.execute(str(match.id))
                        predicted += 1
                    except Exception as exc:
                        logger.warning("Scheduler: no se pudo predecir %s: %s", match.id, exc)
                if predicted > 0:
                    logger.info("Scheduler predict %s: %d predicciones", league_id, predicted)
            except Exception as exc:
                logger.error("Scheduler predict %s falló: %s", league_id, exc, exc_info=True)

    def _retrain_all(self) -> None:
        season = str(datetime.now().year)
        for league_id in get_all_league_ids():
            try:
                finished = self._container.match_repo.get_finished_matches(league_id, season)
                if len(finished) < 50:
                    logger.info(
                        "Scheduler: %s tiene solo %d partidos — skip retrain",
                        league_id,
                        len(finished),
                    )
                    continue
                result = self._container.train_uc.execute(league_id, season)
                metrics = result.get("metrics", {})
                logger.info(
                    "Scheduler retrain %s: acc=%.4f log_loss=%.4f",
                    league_id,
                    metrics.get("cv_accuracy_mean", 0.0),
                    metrics.get("cv_log_loss_mean", 0.0),
                )
            except Exception as exc:
                logger.error("Scheduler retrain %s falló: %s", league_id, exc, exc_info=True)

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("Scheduler iniciado. Jobs activos:")
            for job in self._scheduler.get_jobs():
                next_run = getattr(job, "next_run_time", None)
                logger.info("  - %s | próxima: %s", job.name, next_run)

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler detenido.")

    def get_jobs_status(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for job in self._scheduler.get_jobs():
            next_run = getattr(job, "next_run_time", None)
            result.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": next_run.isoformat() if next_run else None,
                    "running": self._scheduler.running,
                }
            )
        return result
