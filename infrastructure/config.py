"""Centralized configuration for supported leagues and system constants.

All league metadata, season formatting, and validation logic in one place.
"""

from __future__ import annotations

import math
import os

DEFAULT_MODELS_DIR = "models"
DEFAULT_DB_PATH = "football.db"
DEFAULT_N_SIMULATIONS = 10_000
DEFAULT_POISSON_WEIGHT = 0.5
DEFAULT_XGBOOST_WEIGHT = 0.5
MIN_MATCHES_TO_TRAIN = 20
MIN_MATCHES_FOR_STATS = 5


class SettingValidationError(Exception):
    """Raised when environment configuration is invalid at startup."""


def validate_settings() -> None:
    pw = float(os.getenv("POISSON_WEIGHT", str(DEFAULT_POISSON_WEIGHT)))
    xw = float(os.getenv("XGBOOST_WEIGHT", str(DEFAULT_XGBOOST_WEIGHT)))
    if not math.isclose(pw + xw, 1.0, abs_tol=0.01):
        raise SettingValidationError(
            f"POISSON_WEIGHT ({pw}) + XGBOOST_WEIGHT ({xw}) = {pw + xw:.3f}. "
            f"Deben sumar 1.0 ± 0.01. "
            f"Ejemplo válido: POISSON_WEIGHT=0.4 XGBOOST_WEIGHT=0.6"
        )

    n_sim = int(os.getenv("N_SIMULATIONS", str(DEFAULT_N_SIMULATIONS)))
    if n_sim < 100:
        raise SettingValidationError(f"N_SIMULATIONS={n_sim} es demasiado bajo. Mínimo 100.")


SUPPORTED_LEAGUES: dict[str, dict] = {
    "PL": {
        "id": "PL",
        "name": "Premier League",
        "country": "England",
        "avg_matches_per_season": 380,
        "season_format": "calendar_year",
        "notes": "La liga más seguida del mundo. Alta paridad.",
    },
    "PD": {
        "id": "PD",
        "name": "La Liga",
        "country": "Spain",
        "avg_matches_per_season": 380,
        "season_format": "calendar_year",
        "notes": "Dominada históricamente por Real Madrid y Barcelona.",
    },
    "SA": {
        "id": "SA",
        "name": "Serie A",
        "country": "Italy",
        "avg_matches_per_season": 380,
        "season_format": "calendar_year",
        "notes": "Conocida por solidez defensiva. Menor promedio de goles.",
    },
    "BL1": {
        "id": "BL1",
        "name": "Bundesliga",
        "country": "Germany",
        "avg_matches_per_season": 306,
        "season_format": "calendar_year",
        "notes": "18 equipos. Mayor promedio de goles de las 5 grandes.",
    },
    "FL1": {
        "id": "FL1",
        "name": "Ligue 1",
        "country": "France",
        "avg_matches_per_season": 380,
        "season_format": "calendar_year",
        "notes": "Dominada por PSG en la última década.",
    },
    "PPL": {
        "id": "PPL",
        "name": "Primeira Liga",
        "country": "Portugal",
        "avg_matches_per_season": 306,
        "season_format": "calendar_year",
        "notes": "Porto, Benfica y Sporting dominan históricamente.",
    },
    "DED": {
        "id": "DED",
        "name": "Eredivisie",
        "country": "Netherlands",
        "avg_matches_per_season": 306,
        "season_format": "calendar_year",
        "notes": "Conocida por juego ofensivo y desarrollo de talentos.",
    },
    "BSA": {
        "id": "BSA",
        "name": "Brasileirao",
        "country": "Brazil",
        "avg_matches_per_season": 380,
        "season_format": "single_year",
        "notes": "Mayor liga de Sudamérica. 20 equipos.",
    },
    "CL": {
        "id": "CL",
        "name": "UEFA Champions League",
        "country": "Europe",
        "avg_matches_per_season": 125,
        "season_format": "calendar_year",
        "notes": "Fase de grupos + eliminatorias. Menos partidos por equipo.",
    },
    "ELC": {
        "id": "ELC",
        "name": "Championship",
        "country": "England",
        "avg_matches_per_season": 552,
        "season_format": "calendar_year",
        "notes": "Segunda división inglesa. 24 equipos, 46 jornadas.",
    },
}

_MATCHDAY_TOTALS: dict[str, int] = {
    "PL": 38,
    "PD": 38,
    "SA": 38,
    "FL1": 38,
    "BSA": 38,
    "BL1": 34,
    "PPL": 34,
    "DED": 34,
    "ELC": 46,
    "CL": 8,
}


def get_league_config(league_id: str) -> dict:
    if league_id not in SUPPORTED_LEAGUES:
        supported = ", ".join(sorted(SUPPORTED_LEAGUES))
        raise ValueError(f"Liga '{league_id}' no soportada. Ligas disponibles: {supported}")
    return SUPPORTED_LEAGUES[league_id]


def get_all_league_ids() -> list[str]:
    return sorted(SUPPORTED_LEAGUES)


def is_supported_league(league_id: str) -> bool:
    return league_id in SUPPORTED_LEAGUES


def get_season_display(league_id: str, season: str) -> str:
    config = get_league_config(league_id)
    fmt = config["season_format"]
    if fmt == "calendar_year":
        return f"{season}/{str(int(season) + 1)[-2:]}"
    return season


def get_matchday_total(league_id: str) -> int:
    return _MATCHDAY_TOTALS.get(league_id, 38)


def validate_season_format(season: str) -> bool:
    try:
        year = int(season)
    except (ValueError, TypeError):
        return False
    return 2000 <= year <= 2030
