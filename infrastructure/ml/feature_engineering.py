"""Feature engineering for XGBoost from SignalNode outputs.

Builds a flat tabular feature vector per match by running all signal
nodes and extracting/augmenting their scalar outputs.

Critical constraint: strictly temporal — a match at index i only sees
matches [0..i-1] as context to prevent data leakage.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from football_predictor.domain.entities import (
    Match,
    MatchResult,
    TeamStats,
)
from football_predictor.domain.repositories import MatchRepository, TeamStatsRepository
from football_predictor.domain.services import PoissonService
from football_predictor.infrastructure.ml.signal_nodes import SignalNode

logger = logging.getLogger(__name__)

_TARGET_MAP = {
    MatchResult.HOME_WIN.value: 0,
    MatchResult.DRAW.value: 1,
    MatchResult.AWAY_WIN.value: 2,
}

_MIN_MATCHES = 10


class FeatureEngineer:
    """Builds flat feature vectors from signal node outputs.

    Runs all signal nodes, extracts and flattens their scalar values,
    then augments with derived interaction features.
    """

    def __init__(
        self,
        match_repo: MatchRepository,
        stats_repo: TeamStatsRepository,
        poisson_service: PoissonService,
        signal_nodes: list[SignalNode],
    ) -> None:
        self._match_repo = match_repo
        self._stats_repo = stats_repo
        self._poisson = poisson_service
        self._signal_nodes = signal_nodes

    # ------------------------------------------------------------------
    # Per-match feature vector
    # ------------------------------------------------------------------

    def build_features_for_match(
        self,
        match: Match,
        finished_matches: list[Match],
        home_stats: TeamStats | None,
        away_stats: TeamStats | None,
    ) -> dict[str, float]:
        """Build a flat 40-feature dict for a single match.

        Runs every signal node, extracts + flattens its value dict,
        computes derived features, and fills missing values with 0.0.

        Args:
            match: The match to build features for.
            finished_matches: Historical matches (must be BEFORE this match).
            home_stats: TeamStats for the home team (can be None).
            away_stats: TeamStats for the away team (can be None).

        Returns:
            Flat dict with ~40 consistent keys, all floats.
        """
        context: dict[str, Any] = {
            "finished_matches": finished_matches,
            "home_stats": home_stats,
            "away_stats": away_stats,
            "elo_ratings": None,
        }

        features: dict[str, float] = {}

        for node in self._signal_nodes:
            try:
                output = node.compute(match, context)
            except Exception as exc:
                logger.warning("Signal node %s failed: %s", node.name, exc)
                features[f"{node.name}_node_confidence"] = 0.0
                continue

            value = output.get("value", {})
            confidence = output.get("confidence", 0.0)
            features[f"{node.name}_node_confidence"] = float(confidence)

            self._flatten_node(node.name, value, features)

        self._add_derived_features(features)
        self._fill_defaults(features)

        return features

    # ------------------------------------------------------------------
    # Flatten individual node outputs
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_node(node_name: str, value: dict[str, Any], features: dict[str, float]) -> None:
        """Extract scalar values from a node's value dict into features.

        Handles one-hot encoding for categorical fields (trend, advantage,
        last_result) and passes through numeric scalars directly.
        """
        prefix = f"{node_name}_"

        if node_name == "form":
            features[f"{prefix}home_form_5"] = float(value.get("home_form_5", 0))
            features[f"{prefix}home_form_10"] = float(value.get("home_form_10", 0))
            features[f"{prefix}away_form_5"] = float(value.get("away_form_5", 0))
            features[f"{prefix}away_form_10"] = float(value.get("away_form_10", 0))
            features[f"{prefix}home_goals_scored_last5"] = float(
                value.get("home_goals_scored_last5", 0)
            )
            features[f"{prefix}away_goals_scored_last5"] = float(
                value.get("away_goals_scored_last5", 0)
            )
            features[f"{prefix}home_goals_conceded_last5"] = float(
                value.get("home_goals_conceded_last5", 0)
            )
            features[f"{prefix}away_goals_conceded_last5"] = float(
                value.get("away_goals_conceded_last5", 0)
            )
            features[f"{prefix}home_trend_improving"] = (
                1.0 if value.get("home_trend") == "improving" else 0.0
            )
            features[f"{prefix}home_trend_declining"] = (
                1.0 if value.get("home_trend") == "declining" else 0.0
            )
            features[f"{prefix}away_trend_improving"] = (
                1.0 if value.get("away_trend") == "improving" else 0.0
            )
            features[f"{prefix}away_trend_declining"] = (
                1.0 if value.get("away_trend") == "declining" else 0.0
            )

        elif node_name == "elo":
            features[f"{prefix}home"] = float(value.get("home_elo", 1500))
            features[f"{prefix}away"] = float(value.get("away_elo", 1500))
            features[f"{prefix}diff"] = float(value.get("elo_diff", 0))
            features[f"{prefix}home_win_prob"] = float(value.get("home_win_prob_elo", 0.5))
            features[f"{prefix}advantage_home"] = (
                1.0 if value.get("elo_advantage") == "home" else 0.0
            )
            features[f"{prefix}advantage_away"] = (
                1.0 if value.get("elo_advantage") == "away" else 0.0
            )

        elif node_name == "h2h":
            features[f"{prefix}matches"] = float(value.get("h2h_matches", 0))
            features[f"{prefix}home_win_rate"] = float(value.get("home_win_rate", 0))
            features[f"{prefix}avg_total_goals"] = float(value.get("avg_total_goals", 0))
            features[f"{prefix}avg_home_goals"] = float(value.get("avg_home_goals", 0))
            features[f"{prefix}avg_away_goals"] = float(value.get("avg_away_goals", 0))
            last = value.get("last_result", "no_data")
            features[f"{prefix}last_was_home_win"] = 1.0 if last == "home_win" else 0.0
            features[f"{prefix}last_was_draw"] = 1.0 if last == "draw" else 0.0
            features[f"{prefix}last_was_away_win"] = 1.0 if last == "away_win" else 0.0

        elif node_name == "context":
            features[f"{prefix}home_days_rest"] = float(value.get("home_days_rest", 7))
            features[f"{prefix}away_days_rest"] = float(value.get("away_days_rest", 7))
            features[f"{prefix}rest_advantage_home"] = (
                1.0 if value.get("rest_advantage") == "home" else 0.0
            )
            features[f"{prefix}rest_advantage_away"] = (
                1.0 if value.get("rest_advantage") == "away" else 0.0
            )
            features[f"{prefix}matchday_normalized"] = float(value.get("matchday_normalized", 0))
            features[f"{prefix}is_late_season"] = 1.0 if value.get("is_late_season") else 0.0

        elif node_name == "poisson":
            features[f"{prefix}lambda_home"] = float(value.get("lambda_home", 1.5))
            features[f"{prefix}lambda_away"] = float(value.get("lambda_away", 1.1))
            features[f"{prefix}prob_home_win"] = float(value.get("prob_home_win", 1 / 3))
            features[f"{prefix}prob_draw"] = float(value.get("prob_draw", 1 / 3))
            features[f"{prefix}prob_away_win"] = float(value.get("prob_away_win", 1 / 3))

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------

    @staticmethod
    def _add_derived_features(features: dict[str, float]) -> None:
        """Compute interaction and composite features."""
        gf_home = features.get("form_home_goals_scored_last5", 0.0)
        gf_away = features.get("form_away_goals_scored_last5", 0.0)
        gc_home = features.get("form_home_goals_conceded_last5", 0.0)
        gc_away = features.get("form_away_goals_conceded_last5", 0.0)

        features["derived_goals_diff_form"] = gf_home - gf_away
        features["derived_defense_diff_form"] = gc_away - gc_home

        elo_diff = features.get("elo_diff", 0.0)
        form_home_5 = features.get("form_home_form_5", 0.0)
        features["derived_elo_x_form"] = elo_diff * form_home_5

        lam_h = features.get("poisson_lambda_home", 1.5)
        lam_a = features.get("poisson_lambda_away", 1.1)
        features["derived_total_expected_goals"] = lam_h + lam_a
        features["derived_goal_diff_expected"] = lam_h - lam_a

    # ------------------------------------------------------------------
    # Fill missing values with known defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_defaults(features: dict[str, float]) -> None:
        """Ensure all expected feature keys exist with sensible defaults."""
        defaults: dict[str, float] = {
            # Form
            "form_home_form_5": 0.0,
            "form_home_form_10": 0.0,
            "form_away_form_5": 0.0,
            "form_away_form_10": 0.0,
            "form_home_goals_scored_last5": 0.0,
            "form_away_goals_scored_last5": 0.0,
            "form_home_goals_conceded_last5": 0.0,
            "form_away_goals_conceded_last5": 0.0,
            "form_home_trend_improving": 0.0,
            "form_home_trend_declining": 0.0,
            "form_away_trend_improving": 0.0,
            "form_away_trend_declining": 0.0,
            "form_node_confidence": 0.0,
            # Elo
            "elo_home": 1500.0,
            "elo_away": 1500.0,
            "elo_diff": 0.0,
            "elo_home_win_prob": 0.5,
            "elo_advantage_home": 0.0,
            "elo_advantage_away": 0.0,
            "elo_node_confidence": 0.0,
            # H2H
            "h2h_matches": 0.0,
            "h2h_home_win_rate": 0.0,
            "h2h_avg_total_goals": 0.0,
            "h2h_avg_home_goals": 0.0,
            "h2h_avg_away_goals": 0.0,
            "h2h_last_was_home_win": 0.0,
            "h2h_last_was_draw": 0.0,
            "h2h_last_was_away_win": 0.0,
            "h2h_node_confidence": 0.0,
            # Context
            "ctx_home_days_rest": 7.0,
            "ctx_away_days_rest": 7.0,
            "ctx_rest_advantage_home": 0.0,
            "ctx_rest_advantage_away": 0.0,
            "ctx_matchday_normalized": 0.0,
            "ctx_is_late_season": 0.0,
            "ctx_node_confidence": 0.0,
            # Poisson
            "poisson_lambda_home": 1.5,
            "poisson_lambda_away": 1.1,
            "poisson_prob_home_win": 1 / 3,
            "poisson_prob_draw": 1 / 3,
            "poisson_prob_away_win": 1 / 3,
            "poisson_node_confidence": 0.0,
            # Derived
            "derived_goals_diff_form": 0.0,
            "derived_defense_diff_form": 0.0,
            "derived_elo_x_form": 0.0,
            "derived_total_expected_goals": 2.6,
            "derived_goal_diff_expected": 0.4,
        }
        for key, default in defaults.items():
            features.setdefault(key, default)

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        league_id: str,
        season: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build a full feature matrix and target vector for a league-season.

        Temporal constraint: match at index i only sees matches [0..i-1].
        This prevents data leakage in TimeSeriesSplit backtesting.

        Args:
            league_id: Competition code (e.g. "PL").
            season: Season identifier (e.g. "2024").

        Returns:
            Tuple of (X: DataFrame, y: Series) with aligned indices.

        Raises:
            ValueError: If fewer than MIN_MATCHES (10) matches are available.
        """
        matches = self._match_repo.get_finished_matches(league_id, season)  # type: ignore[attr-defined]

        if len(matches) < _MIN_MATCHES:
            raise ValueError(
                f"Need at least {_MIN_MATCHES} finished matches, "
                f"got {len(matches)} for {league_id}/{season}"
            )

        matches.sort(key=lambda m: m.match_date)

        feature_rows: list[dict[str, float]] = []
        targets: list[int] = []

        for i, match in enumerate(matches):
            if match.score is None or match.score.result is None:
                continue

            past_matches = matches[:i]

            home_stats = self._stats_repo.get_stats(  # type: ignore[attr-defined]
                match.home_team.id, league_id, season
            )
            away_stats = self._stats_repo.get_stats(  # type: ignore[attr-defined]
                match.away_team.id, league_id, season
            )

            features = self.build_features_for_match(match, past_matches, home_stats, away_stats)
            feature_rows.append(features)
            targets.append(_TARGET_MAP[match.score.result.value])

        X = pd.DataFrame(feature_rows)
        y = pd.Series(targets, name="target")

        # Clean NaNs
        X = X.fillna(0.0)
        y = y.dropna()
        X = X.loc[y.index]

        # Log distribution
        dist = y.value_counts().sort_index()
        logger.info(
            "Dataset built: %d matches, %d features. Targets: 1(W)=%d X(D)=%d 2(A)=%d",
            len(X),
            len(X.columns),
            dist.get(0, 0),
            dist.get(1, 0),
            dist.get(2, 0),
        )

        return X, y
