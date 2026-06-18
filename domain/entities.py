"""Domain entities for football match prediction.

Pure business objects with no framework dependencies.
Uses dataclasses with frozen immutability where appropriate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from football_predictor.domain.services import MonteCarloResult

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MatchResult(Enum):
    """Outcome of a football match."""

    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"


class MatchStatus(Enum):
    """Lifecycle status of a match."""

    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class CompetitionType(Enum):
    """Type of football competition."""

    LEAGUE = "league"
    CUP = "cup"
    INTERNATIONAL = "international"


# ---------------------------------------------------------------------------
# Value objects & entities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class League:
    """A football league or competition.

    Attributes:
        id: Unique league identifier.
        name: Full name (e.g. "Premier League").
        country: Country where the league is played.
        competition_type: LEAGUE, CUP, or INTERNATIONAL.
        season: Optional season string (e.g. "2025/2026").
    """

    id: int
    name: str
    country: str
    competition_type: CompetitionType
    season: str | None = None


@dataclass(frozen=True)
class Team:
    """A football team.

    Attributes:
        id: Unique team identifier.
        name: Full team name (e.g. "Manchester United FC").
        short_name: Short display name (e.g. "Man Utd").
        tla: Three-letter acronym (e.g. "MUN").
        country: Country the team belongs to.
    """

    id: int
    name: str
    short_name: str
    tla: str
    country: str


@dataclass
class MatchScore:
    """Score of a football match.

    Attributes:
        home_goals: Goals scored by the home team (None if unknown).
        away_goals: Goals scored by the away team (None if unknown).
    """

    home_goals: int | None = None
    away_goals: int | None = None

    @property
    def is_complete(self) -> bool:
        """Whether both goal counts are known."""
        return self.home_goals is not None and self.away_goals is not None

    @property
    def result(self) -> MatchResult | None:
        """Derived match result from the score, or None if incomplete."""
        if not self.is_complete:
            return None
        if self.home_goals > self.away_goals:  # type: ignore[operator]
            return MatchResult.HOME_WIN
        if self.home_goals == self.away_goals:  # type: ignore[operator]
            return MatchResult.DRAW
        return MatchResult.AWAY_WIN

    @property
    def total_goals(self) -> int | None:
        """Sum of home and away goals, or None if incomplete."""
        if not self.is_complete:
            return None
        return self.home_goals + self.away_goals  # type: ignore[operator]


@dataclass
class Match:
    """A football match fixture.

    Attributes:
        id: Unique match identifier.
        home_team: Home team entity.
        away_team: Away team entity.
        league: League the match belongs to.
        match_date: Scheduled kick-off datetime.
        status: Current match status.
        score: Match score (None before kick-off or if unavailable).
        matchday: Matchday number within the season.
        venue: Stadium or location name (optional).
    """

    id: int
    home_team: Team
    away_team: Team
    league: League
    match_date: datetime
    status: MatchStatus
    score: MatchScore | None = None
    matchday: int | None = None
    venue: str | None = None

    @property
    def is_finished(self) -> bool:
        """Whether the match has concluded."""
        return self.status == MatchStatus.FINISHED

    @property
    def result(self) -> MatchResult | None:
        """Actual match result, derived from score when finished."""
        if not self.is_finished or self.score is None:
            return None
        return self.score.result


@dataclass
class GoalProbabilities:
    """Probability matrix for exact score combinations.

    Attributes:
        prob_matrix: 2D list where prob_matrix[i][j] = P(home=i, away=j).
        max_goals: Maximum goals considered per team (default 8).
    """

    prob_matrix: list[list[float]]
    max_goals: int = 8

    def __post_init__(self) -> None:
        """Validate matrix dimensions."""
        if len(self.prob_matrix) != self.max_goals + 1:
            raise ValueError(
                f"prob_matrix must have {self.max_goals + 1} rows, got {len(self.prob_matrix)}"
            )
        for i, row in enumerate(self.prob_matrix):
            if len(row) != self.max_goals + 1:
                raise ValueError(f"Row {i} must have {self.max_goals + 1} columns, got {len(row)}")

    def get_prob(self, home_goals: int, away_goals: int) -> float:
        """Return P(home_goals, away_goals) from the matrix.

        Args:
            home_goals: Goals scored by home team (0 .. max_goals).
            away_goals: Goals scored by away team (0 .. max_goals).

        Returns:
            Probability value at the given coordinates.

        Raises:
            IndexError: If either goal count exceeds max_goals.
        """
        return self.prob_matrix[home_goals][away_goals]


@dataclass
class Prediction:
    """A match outcome prediction produced by a model.

    Attributes:
        match: The match being predicted.
        prob_home_win: Probability of home win (0-1).
        prob_draw: Probability of draw (0-1).
        prob_away_win: Probability of away win (0-1).
        expected_goals_home: Expected goals for home team.
        expected_goals_away: Expected goals for away team.
        goal_probabilities: Optional exact-score probability matrix.
        confidence: Model confidence score (0-1).
        model_version: Identifier for the model version used.
        predicted_at: Timestamp when the prediction was made.
        was_correct: Whether the prediction was correct (None if match unfinished).
    """

    match: Match
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    expected_goals_home: float
    expected_goals_away: float
    goal_probabilities: GoalProbabilities | None = None
    simulation: MonteCarloResult | None = None
    signal_outputs: dict | None = None
    llm_explanation: str | None = None
    llm_actions: list | None = None
    extended_predictions: dict | None = None
    confidence: float = 0.0
    model_version: str = ""
    predicted_at: datetime = field(default_factory=datetime.utcnow)
    was_correct: bool | None = None

    @property
    def predicted_result(self) -> MatchResult:
        """The result with the highest predicted probability."""
        if self.prob_home_win >= self.prob_draw and self.prob_home_win >= self.prob_away_win:
            return MatchResult.HOME_WIN
        if self.prob_draw >= self.prob_away_win:
            return MatchResult.DRAW
        return MatchResult.AWAY_WIN

    @property
    def probabilities_are_valid(self) -> bool:
        """Check that win/draw/win probabilities sum to ~1.0."""
        total = self.prob_home_win + self.prob_draw + self.prob_away_win
        return math.isclose(total, 1.0, rel_tol=1e-6)

    def as_dict(self) -> dict:
        """Serialize prediction to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "match_id": self.match.id,
            "prob_home_win": self.prob_home_win,
            "prob_draw": self.prob_draw,
            "prob_away_win": self.prob_away_win,
            "expected_goals_home": self.expected_goals_home,
            "expected_goals_away": self.expected_goals_away,
            "predicted_result": self.predicted_result.value,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "predicted_at": self.predicted_at.isoformat(),
            "was_correct": self.was_correct,
            "simulation": self.simulation.as_dict() if self.simulation else None,
            "llm_explanation": self.llm_explanation,
            "llm_actions": self.llm_actions,
        }


@dataclass
class TeamStats:
    """Aggregated statistics for a team in a specific league season.

    Attributes:
        team: The team these stats belong to.
        league: The league context.
        season: Season identifier (e.g. "2025/2026").
        matches_played: Total matches played.
        matches_home: Matches played at home.
        matches_away: Matches played away.
        wins: Total wins.
        wins_home: Wins at home.
        wins_away: Wins away.
        draws: Total draws.
        draws_home: Draws at home.
        draws_away: Draws away.
        losses: Total losses.
        losses_home: Losses at home.
        losses_away: Losses away.
        goals_scored: Total goals scored.
        goals_scored_home: Goals scored at home.
        goals_scored_away: Goals scored away.
        goals_conceded: Total goals conceded.
        goals_conceded_home: Goals conceded at home.
        goals_conceded_away: Goals conceded away.
    """

    team: Team
    league: League
    season: str
    matches_played: int = 0
    matches_home: int = 0
    matches_away: int = 0
    wins: int = 0
    wins_home: int = 0
    wins_away: int = 0
    draws: int = 0
    draws_home: int = 0
    draws_away: int = 0
    losses: int = 0
    losses_home: int = 0
    losses_away: int = 0
    goals_scored: int = 0
    goals_scored_home: int = 0
    goals_scored_away: int = 0
    goals_conceded: int = 0
    goals_conceded_home: int = 0
    goals_conceded_away: int = 0

    @property
    def goals_per_match(self) -> float:
        """Average goals scored per match."""
        if self.matches_played == 0:
            return 0.0
        return self.goals_scored / self.matches_played

    @property
    def goals_conceded_per_match(self) -> float:
        """Average goals conceded per match."""
        if self.matches_played == 0:
            return 0.0
        return self.goals_conceded / self.matches_played

    @property
    def win_rate(self) -> float:
        """Win rate as a fraction (0-1)."""
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played
