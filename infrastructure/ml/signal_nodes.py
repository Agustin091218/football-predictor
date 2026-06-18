"""Signal graph nodes for football match prediction.

Each node processes one dimension of the match (form, ELO, head-to-head,
context, Poisson model) and produces a structured signal dict.

Nodes are pure domain logic — no infrastructure imports.
The LLM orchestrator (phase 5B) adjusts weights dynamically.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from football_predictor.domain.entities import (
    GoalProbabilities,
    Match,
    MatchResult,
    TeamStats,
)
from football_predictor.domain.services import (
    PoissonModelParams,
    PoissonService,
)


def _poisson_probs(
    lambda_home: float, lambda_away: float, max_goals: int = 8
) -> tuple[float, float, float]:
    """Compute 1X2 probabilities from Poisson λ values using the joint distribution.

    For each possible score (i,j) with i,j ∈ [0, max_goals]:
      P(i,j) = Poisson(i|λ_home) × Poisson(j|λ_away)
    Then sum: i>j → home win, i==j → draw, i<j → away win.
    """
    home_pmf = [
        math.exp(-lambda_home) * (lambda_home**i) / math.factorial(i) for i in range(max_goals + 1)
    ]
    away_pmf = [
        math.exp(-lambda_away) * (lambda_away**j) / math.factorial(j) for j in range(max_goals + 1)
    ]

    prob_home = prob_draw = prob_away = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = home_pmf[i] * away_pmf[j]
            if i > j:
                prob_home += p
            elif i == j:
                prob_draw += p
            else:
                prob_away += p

    total = prob_home + prob_draw + prob_away
    if total > 0:
        return (prob_home / total, prob_draw / total, prob_away / total)
    return (0.4, 0.3, 0.3)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SignalNode(ABC):
    """Base class for all signal nodes in the learning graph.

    Attributes:
        name: Human-readable node identifier (set by subclass).
        weight: Current multiplier for this node's influence (0.0–3.0).
    """

    name: str = "base"

    def __init__(self) -> None:
        self.weight: float = 1.0

    @abstractmethod
    def compute(self, match: Match, context: dict[str, Any]) -> dict[str, Any]:
        """Process the signal and return a structured result dict.

        Args:
            match: The match being analysed.
            context: Dictionary with optional keys:
                - 'finished_matches': list[Match]
                - 'home_stats': Optional[TeamStats]
                - 'away_stats': Optional[TeamStats]
                - 'elo_ratings': Optional[dict[int, float]]

        Returns:
            Dict with keys: 'signal', 'weight', 'confidence', 'value', 'summary'.
        """
        ...

    def set_weight(self, weight: float) -> None:
        """Clamp and set the node weight in [0.0, 3.0]."""
        self.weight = max(0.0, min(weight, 3.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_team_matches(team_id: int, matches: list[Match]) -> list[Match]:
    """Return matches involving a team, sorted by date descending."""
    return sorted(
        [m for m in matches if m.home_team.id == team_id or m.away_team.id == team_id],
        key=lambda m: m.match_date,
        reverse=True,
    )


def _match_points_for_team(match: Match, team_id: int) -> float:
    """Return 3, 1, or 0 points for the team from this match."""
    if match.score is None or match.score.result is None:
        return 0.0
    is_home = match.home_team.id == team_id
    result = match.score.result
    if result == MatchResult.DRAW:
        return 1.0
    if (is_home and result == MatchResult.HOME_WIN) or (
        not is_home and result == MatchResult.AWAY_WIN
    ):
        return 3.0
    return 0.0


def _goals_for_team(match: Match, team_id: int) -> tuple[int, int]:
    """Return (scored, conceded) for the team from this match."""
    if match.score is None:
        return (0, 0)
    if match.home_team.id == team_id:
        return (match.score.home_goals or 0, match.score.away_goals or 0)
    return (match.score.away_goals or 0, match.score.home_goals or 0)


# ---------------------------------------------------------------------------
# FormNode
# ---------------------------------------------------------------------------


class FormNode(SignalNode):
    """Recent form analysis based on last 5 and 10 matches."""

    name = "form"

    def compute(self, match: Match, context: dict[str, Any]) -> dict[str, Any]:
        finished = context.get("finished_matches") or []
        if not isinstance(finished, list):
            finished = []

        home_matches = _get_team_matches(match.home_team.id, finished)
        away_matches = _get_team_matches(match.away_team.id, finished)
        has_data = bool(home_matches or away_matches)

        result: dict[str, Any] = {
            "home_form_5": 0.0,
            "home_form_10": 0.0,
            "away_form_5": 0.0,
            "away_form_10": 0.0,
            "home_goals_scored_last5": 0.0,
            "away_goals_scored_last5": 0.0,
            "home_goals_conceded_last5": 0.0,
            "away_goals_conceded_last5": 0.0,
            "home_trend": "no_data" if not home_matches else "stable",
            "away_trend": "no_data" if not away_matches else "stable",
        }

        if not has_data:
            return {
                "signal": self.name,
                "weight": self.weight,
                "confidence": 0.0,
                "value": result,
                "summary": f"Form: sin historial en DB para {match.home_team.name} ni {match.away_team.name}",
            }
            return self._build_return(match, result, 0, finished)

        # Home team form
        home_matches = _get_team_matches(match.home_team.id, finished)
        self._compute_form(home_matches, match.home_team.id, result, prefix="home")

        # Away team form
        away_matches = _get_team_matches(match.away_team.id, finished)
        self._compute_form(away_matches, match.away_team.id, result, prefix="away")

        matches_available = min(len(home_matches), len(away_matches))
        confidence = min(1.0, matches_available / 10)

        return self._build_return(match, result, confidence, finished)

    @staticmethod
    def _compute_form(team_matches: list[Match], team_id: int, result: dict, prefix: str) -> None:
        """Compute form metrics for a team and update result dict in-place.

        Args:
            team_matches: Matches involving the team, sorted by date DESC.
            team_id: The team being analysed.
            result: Output dict to update.
            prefix: Key prefix (e.g. "home" or "away").
        """
        if not team_matches:
            return

        # Points from last 5
        last_5 = team_matches[:5]
        points_5 = sum(_match_points_for_team(m, team_id) for m in last_5)
        result[f"{prefix}_form_5"] = points_5 / (3.0 * len(last_5))

        # Points from last 10
        last_10 = team_matches[:10]
        points_10 = sum(_match_points_for_team(m, team_id) for m in last_10)
        result[f"{prefix}_form_10"] = points_10 / (3.0 * len(last_10))

        # Goals in last 5
        goals_last5 = [_goals_for_team(m, team_id) for m in last_5]
        if last_5:
            result[f"{prefix}_goals_scored_last5"] = sum(g[0] for g in goals_last5) / len(last_5)
            result[f"{prefix}_goals_conceded_last5"] = sum(g[1] for g in goals_last5) / len(last_5)

        # Trend
        f5 = result[f"{prefix}_form_5"]
        f10 = result[f"{prefix}_form_10"]
        if f5 > f10 * 1.1:
            result[f"{prefix}_trend"] = "improving"
        elif f5 < f10 * 0.9:
            result[f"{prefix}_trend"] = "declining"
        else:
            result[f"{prefix}_trend"] = "stable"

    def _build_return(
        self, match: Match, value: dict, confidence: float, finished: list
    ) -> dict[str, Any]:
        return {
            "signal": self.name,
            "weight": self.weight,
            "confidence": confidence,
            "value": value,
            "summary": (
                f"{match.home_team.name} forma: {value['home_form_5']:.0%} (5P) | "
                f"{match.away_team.name} forma: {value['away_form_5']:.0%} (5P)"
            ),
        }


# ---------------------------------------------------------------------------
# EloNode
# ---------------------------------------------------------------------------


class EloNode(SignalNode):
    """ELO rating estimation from historical match results."""

    name = "elo"

    K_FACTOR: float = 32.0
    INITIAL_ELO: float = 1500.0

    def compute(self, match: Match, context: dict[str, Any]) -> dict[str, Any]:
        finished = context.get("finished_matches") or []
        if not isinstance(finished, list):
            finished = []
        elo_ratings_raw = context.get("elo_ratings")
        elo_ratings: dict[int, float] | None = (
            elo_ratings_raw if isinstance(elo_ratings_raw, dict) else None
        )

        if not elo_ratings:
            elo_ratings = self._compute_elo(finished)

        home_elo = elo_ratings.get(match.home_team.id, self.INITIAL_ELO)
        away_elo = elo_ratings.get(match.away_team.id, self.INITIAL_ELO)
        elo_diff = home_elo - away_elo
        home_win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        from football_predictor.domain.services import EloStrengthCalculator

        elo_calc = EloStrengthCalculator(home_advantage=1.1, elo_scale=400.0)
        avg_goals_home = float(context.get("avg_goals_home", 1.5))
        avg_goals_away = float(context.get("avg_goals_away", 1.1))
        lambda_home, lambda_away = elo_calc.calculate_lambdas(
            elo_home=home_elo,
            elo_away=away_elo,
            avg_goals_home=avg_goals_home,
            avg_goals_away=avg_goals_away,
        )
        home_share, _ = elo_calc.expected_goal_share(home_elo, away_elo)
        elo_ratio = round(math.exp(elo_diff / 400.0), 3)

        if elo_diff > 50:
            advantage = "home"
        elif elo_diff < -50:
            advantage = "away"
        else:
            advantage = "neutral"

        matches_available = len(finished)
        confidence = min(1.0, matches_available / 20)

        return {
            "signal": self.name,
            "weight": self.weight,
            "confidence": confidence,
            "value": {
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": elo_diff,
                "home_win_prob_elo": home_win_prob,
                "elo_advantage": advantage,
                "lambda_home_elo": lambda_home,
                "lambda_away_elo": lambda_away,
                "elo_ratio": elo_ratio,
                "home_goal_share": round(home_share, 3),
            },
            "summary": (
                f"ELO: {match.home_team.name} {home_elo:.0f} vs "
                f"{match.away_team.name} {away_elo:.0f} "
                f"(ratio: {elo_ratio:.2f}) "
                f"→ λ {lambda_home:.2f}-{lambda_away:.2f}"
            ),
        }

    def _compute_elo(self, matches: list[Match]) -> dict[int, float]:
        """Compute ELO ratings from scratch using chronological match order."""
        if not matches:
            return {}

        elo: dict[int, float] = {}
        # Sort chronologically
        sorted_matches = sorted(matches, key=lambda m: m.match_date)

        for m in sorted_matches:
            if m.score is None or m.score.result is None:
                continue
            hid = m.home_team.id
            aid = m.away_team.id

            home_elo = elo.setdefault(hid, self.INITIAL_ELO)
            away_elo = elo.setdefault(aid, self.INITIAL_ELO)

            expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))
            expected_away = 1.0 - expected_home

            result = m.score.result
            if result == MatchResult.HOME_WIN:
                score_home, score_away = 1.0, 0.0
            elif result == MatchResult.DRAW:
                score_home, score_away = 0.5, 0.5
            else:
                score_home, score_away = 0.0, 1.0

            elo[hid] = home_elo + self.K_FACTOR * (score_home - expected_home)
            elo[aid] = away_elo + self.K_FACTOR * (score_away - expected_away)

        return elo


# ---------------------------------------------------------------------------
# HeadToHeadNode
# ---------------------------------------------------------------------------


class HeadToHeadNode(SignalNode):
    """Head-to-head historical analysis between the two teams."""

    name = "h2h"

    def compute(self, match: Match, context: dict[str, Any]) -> dict[str, Any]:
        finished = context.get("finished_matches") or []
        if not isinstance(finished, list):
            finished = []

        hid = match.home_team.id
        aid = match.away_team.id

        # Filter H2H matches (either direction)
        h2h = [
            m
            for m in finished
            if (m.home_team.id == hid and m.away_team.id == aid)
            or (m.home_team.id == aid and m.away_team.id == hid)
        ]
        h2h.sort(key=lambda m: m.match_date, reverse=True)
        h2h = h2h[:5]

        if not h2h:
            return {
                "signal": self.name,
                "weight": self.weight,
                "confidence": 0.0,
                "value": {
                    "h2h_matches": 0,
                    "home_wins": 0,
                    "draws": 0,
                    "away_wins": 0,
                    "home_win_rate": 0.0,
                    "avg_total_goals": 0.0,
                    "avg_home_goals": 0.0,
                    "avg_away_goals": 0.0,
                    "last_result": "no_data",
                },
                "summary": "H2H: sin datos",
            }

        home_wins = 0
        draws = 0
        away_wins = 0
        home_goals_total = 0
        away_goals_total = 0

        for m in h2h:
            if m.score is None:
                continue
            # Determine result from the perspective of the CURRENT home team (hid)
            if m.home_team.id == hid:
                hg = m.score.home_goals or 0
                ag = m.score.away_goals or 0
                if m.score.result == MatchResult.HOME_WIN:
                    home_wins += 1
                elif m.score.result == MatchResult.DRAW:
                    draws += 1
                else:
                    away_wins += 1
            else:
                # Current home team (hid) played away in this historical match
                hg = m.score.away_goals or 0
                ag = m.score.home_goals or 0
                if m.score.result == MatchResult.AWAY_WIN:
                    home_wins += 1
                elif m.score.result == MatchResult.DRAW:
                    draws += 1
                else:
                    away_wins += 1

            home_goals_total += hg
            away_goals_total += ag

        n = len(h2h)

        # Last result
        last_m = h2h[0]
        if last_m.score and last_m.score.result:
            if last_m.home_team.id == hid:
                last_result = last_m.score.result.value
            else:
                # Invert perspective
                if last_m.score.result == MatchResult.HOME_WIN:
                    last_result = MatchResult.AWAY_WIN.value
                elif last_m.score.result == MatchResult.AWAY_WIN:
                    last_result = MatchResult.HOME_WIN.value
                else:
                    last_result = MatchResult.DRAW.value
        else:
            last_result = "no_data"

        value = {
            "h2h_matches": n,
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins,
            "home_win_rate": home_wins / n if n else 0.0,
            "avg_total_goals": (home_goals_total + away_goals_total) / n if n else 0.0,
            "avg_home_goals": home_goals_total / n if n else 0.0,
            "avg_away_goals": away_goals_total / n if n else 0.0,
            "last_result": last_result,
        }

        confidence = min(1.0, n / 5)

        return {
            "signal": self.name,
            "weight": self.weight,
            "confidence": confidence,
            "value": value,
            "summary": (f"H2H últimos {n}P: {home_wins}V-{draws}E-{away_wins}D"),
        }


# ---------------------------------------------------------------------------
# ContextNode
# ---------------------------------------------------------------------------


class ContextNode(SignalNode):
    """Context signals: rest days, matchday, season phase."""

    name = "context"

    def compute(self, match: Match, context: dict[str, Any]) -> dict[str, Any]:
        finished = context.get("finished_matches") or []
        if not isinstance(finished, list):
            finished = []

        home_rest = self._days_since_last(match.home_team.id, match.match_date, finished)
        away_rest = self._days_since_last(match.away_team.id, match.match_date, finished)

        if home_rest > away_rest + 2:
            rest_adv = "home"
        elif away_rest > home_rest + 2:
            rest_adv = "away"
        else:
            rest_adv = "neutral"

        matchday = match.matchday or 0
        matchday_norm = min(1.0, matchday / 38.0)

        return {
            "signal": self.name,
            "weight": self.weight,
            "confidence": 0.7,
            "value": {
                "home_days_rest": home_rest,
                "away_days_rest": away_rest,
                "rest_advantage": rest_adv,
                "matchday": matchday,
                "matchday_normalized": matchday_norm,
                "is_late_season": matchday_norm > 0.75,
            },
            "summary": (
                f"Descanso: {match.home_team.name} {home_rest}d vs "
                f"{match.away_team.name} {away_rest}d | Jornada {matchday}"
            ),
        }

    @staticmethod
    def _days_since_last(team_id: int, ref_date, finished: list[Match]) -> int:
        """Days since the team's most recent match before ref_date. Capped at 14."""
        previous = [
            m
            for m in finished
            if (m.home_team.id == team_id or m.away_team.id == team_id) and m.match_date < ref_date
        ]
        if not previous:
            return 7
        last_date = max(m.match_date for m in previous)
        delta = (ref_date - last_date).days
        return min(max(delta, 0), 14)


# ---------------------------------------------------------------------------
# PoissonSignalNode
# ---------------------------------------------------------------------------


class PoissonSignalNode(SignalNode):
    """Poisson model prediction signal.

    Uses PoissonService from the domain layer to compute expected goals
    and outcome probabilities from team strengths and league parameters.
    """

    name = "poisson"

    def __init__(self, poisson_service: PoissonService) -> None:
        super().__init__()
        self._poisson = poisson_service
        self.weight = 1.2

    def compute(self, match: Match, context: dict[str, Any]) -> dict[str, Any]:
        finished = context.get("finished_matches") or []
        if not isinstance(finished, list):
            finished = []
        home_stats_raw = context.get("home_stats")
        away_stats_raw = context.get("away_stats")
        home_stats: TeamStats | None = (
            home_stats_raw if isinstance(home_stats_raw, TeamStats) else None
        )
        away_stats: TeamStats | None = (
            away_stats_raw if isinstance(away_stats_raw, TeamStats) else None
        )

        league_id = str(match.league.id)
        season = match.league.season or ""
        lambda_source = "defaults"

        if home_stats is not None and away_stats is not None:
            league_matches = [m for m in finished if m.score is not None and m.score.is_complete]
            if league_matches:
                home_goals = [m.score.home_goals for m in league_matches]  # type: ignore[union-attr]
                away_goals = [m.score.away_goals for m in league_matches]  # type: ignore[union-attr]
                params = self._poisson.calculate_league_params_from_matches(
                    home_goals,
                    away_goals,
                    league_id,
                    season,  # type: ignore[arg-type]
                )
            else:
                params = PoissonModelParams(league_id=league_id, season=season)  # type: ignore[arg-type]

            home_strength = self._poisson.calculate_team_strengths_from_stats(home_stats, params)
            away_strength = self._poisson.calculate_team_strengths_from_stats(away_stats, params)
            pred = self._poisson.predict(home_strength, away_strength, params)
            lambda_source = "stats"

        elif (
            context.get("signal_outputs_so_far", {})
            .get("elo", {})
            .get("value", {})
            .get("lambda_home_elo")
        ):
            elo_val = context["signal_outputs_so_far"]["elo"]["value"]
            lambda_h = float(elo_val.get("lambda_home_elo", 1.5))
            lambda_a = float(elo_val.get("lambda_away_elo", 1.1))
            ph, pd, pa = _poisson_probs(lambda_h, lambda_a)
            pred = {
                "prob_home_win": ph,
                "prob_draw": pd,
                "prob_away_win": pa,
                "lambda_home": lambda_h,
                "lambda_away": lambda_a,
                "goal_probabilities": GoalProbabilities(
                    prob_matrix=[[0.0, 0.0], [0.0, 0.0]], max_goals=1
                ),
            }
            lambda_source = "elo"

        else:
            lambda_h = float(context.get("avg_goals_home", 1.5))
            lambda_a = float(context.get("avg_goals_away", 1.1))
            ph, pd, pa = _poisson_probs(lambda_h, lambda_a)
            pred = {
                "prob_home_win": ph,
                "prob_draw": pd,
                "prob_away_win": pa,
                "lambda_home": lambda_h,
                "lambda_away": lambda_a,
                "goal_probabilities": GoalProbabilities(
                    prob_matrix=[[0.0, 0.0], [0.0, 0.0]], max_goals=1
                ),
            }

        gp: GoalProbabilities = pred["goal_probabilities"]
        best_prob = -1.0
        best_i, best_j = 0, 0
        size = gp.max_goals + 1
        for i in range(size):
            for j in range(size):
                p = gp.get_prob(i, j)
                if p > best_prob:
                    best_prob = p
                    best_i, best_j = i, j
        most_likely = f"{best_i}-{best_j}"

        if (
            pred["prob_home_win"] >= pred["prob_draw"]
            and pred["prob_home_win"] >= pred["prob_away_win"]
        ):
            predicted = "1"
        elif pred["prob_draw"] >= pred["prob_away_win"]:
            predicted = "X"
        else:
            predicted = "2"

        confidence = min(1.0, len(finished) / 15)

        return {
            "signal": self.name,
            "weight": self.weight,
            "confidence": confidence,
            "value": {
                "lambda_home": pred["lambda_home"],
                "lambda_away": pred["lambda_away"],
                "prob_home_win": pred["prob_home_win"],
                "prob_draw": pred["prob_draw"],
                "prob_away_win": pred["prob_away_win"],
                "predicted_result": predicted,
                "most_likely_score": most_likely,
                "lambda_source": lambda_source,
            },
            "summary": (
                f"Poisson [{lambda_source}]: {pred['prob_home_win']:.0%}/"
                f"{pred['prob_draw']:.0%}/{pred['prob_away_win']:.0%} "
                f"| λ {pred['lambda_home']:.2f}-{pred['lambda_away']:.2f}"
            ),
        }
