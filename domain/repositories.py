"""Abstract repository interfaces for the domain layer.

Defines contracts that infrastructure adapters must fulfill.
All methods are abstract — no implementation details belong here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from football_predictor.domain.entities import (
    League,
    Match,
    MatchStatus,
    Prediction,
    Team,
    TeamStats,
)

# ---------------------------------------------------------------------------
# Match Repository
# ---------------------------------------------------------------------------


class MatchRepository(ABC):
    """Persistence contract for Match entities."""

    @abstractmethod
    def get_by_id(self, match_id: int) -> Match | None:
        """Retrieve a single match by its unique identifier.

        Args:
            match_id: Match identifier.

        Returns:
            The Match if found, None otherwise.
        """
        ...

    @abstractmethod
    def find_by_league(
        self,
        league: League,
        season: str | None = None,
        matchday: int | None = None,
    ) -> list[Match]:
        """Find all matches for a given league, optionally filtered.

        Args:
            league: The league to query.
            season: Optional season filter (e.g. "2025/2026").
            matchday: Optional matchday filter.

        Returns:
            List of matching Match entities.
        """
        ...

    @abstractmethod
    def find_by_team(self, team: Team, limit: int = 50) -> list[Match]:
        """Find recent matches involving a specific team (home or away).

        Args:
            team: The team to search for.
            limit: Maximum number of matches to return.

        Returns:
            List of Match entities, ordered by match_date descending.
        """
        ...

    @abstractmethod
    def find_by_status(self, status: MatchStatus) -> list[Match]:
        """Find all matches with a given lifecycle status.

        Args:
            status: The MatchStatus to filter by.

        Returns:
            List of matching Match entities.
        """
        ...

    @abstractmethod
    def find_by_date_range(self, start: datetime, end: datetime) -> list[Match]:
        """Find matches scheduled or played within a date range.

        Args:
            start: Inclusive start datetime.
            end: Inclusive end datetime.

        Returns:
            List of Match entities ordered by match_date.
        """
        ...

    @abstractmethod
    def save(self, match: Match) -> Match:
        """Persist a new or updated match.

        Args:
            match: The Match entity to save.

        Returns:
            The persisted Match.
        """
        ...

    @abstractmethod
    def save_many(self, matches: list[Match]) -> list[Match]:
        """Persist multiple matches in a single operation.

        Args:
            matches: List of Match entities to save.

        Returns:
            The persisted Match entities.
        """
        ...


# ---------------------------------------------------------------------------
# Team Repository
# ---------------------------------------------------------------------------


class TeamRepository(ABC):
    """Persistence contract for Team entities."""

    @abstractmethod
    def get_by_id(self, team_id: int) -> Team | None:
        """Retrieve a single team by its unique identifier.

        Args:
            team_id: Team identifier.

        Returns:
            The Team if found, None otherwise.
        """
        ...

    @abstractmethod
    def find_by_name(self, name: str) -> list[Team]:
        """Search teams by (partial) name.

        Args:
            name: Full or partial team name.

        Returns:
            List of matching Team entities.
        """
        ...

    @abstractmethod
    def find_by_country(self, country: str) -> list[Team]:
        """Find all teams from a given country.

        Args:
            country: Country name.

        Returns:
            List of Team entities.
        """
        ...

    @abstractmethod
    def find_by_tla(self, tla: str) -> Team | None:
        """Look up a team by its three-letter acronym.

        Args:
            tla: Three-letter acronym (e.g. "MUN").

        Returns:
            The Team if found, None otherwise.
        """
        ...

    @abstractmethod
    def get_all(self) -> list[Team]:
        """Retrieve every known team.

        Returns:
            Complete list of Team entities.
        """
        ...

    @abstractmethod
    def save(self, team: Team) -> Team:
        """Persist a team.

        Args:
            team: The Team entity to save.

        Returns:
            The persisted Team.
        """
        ...


# ---------------------------------------------------------------------------
# Prediction Repository
# ---------------------------------------------------------------------------


class PredictionRepository(ABC):
    """Persistence contract for Prediction entities."""

    @abstractmethod
    def get_by_match(self, match_id: int) -> Prediction | None:
        """Retrieve the most recent prediction for a match.

        Args:
            match_id: Match identifier.

        Returns:
            The Prediction if one exists, None otherwise.
        """
        ...

    @abstractmethod
    def find_by_model_version(self, model_version: str) -> list[Prediction]:
        """Find all predictions generated by a specific model version.

        Args:
            model_version: Model version identifier.

        Returns:
            List of Prediction entities.
        """
        ...

    @abstractmethod
    def find_pending_evaluation(self) -> list[Prediction]:
        """Find predictions for finished matches whose correctness
        has not yet been evaluated (was_correct is None).

        Returns:
            List of Prediction entities awaiting evaluation.
        """
        ...

    @abstractmethod
    def save(self, prediction: Prediction) -> Prediction:
        """Persist a prediction.

        Args:
            prediction: The Prediction to save.

        Returns:
            The persisted Prediction.
        """
        ...

    @abstractmethod
    def save_many(self, predictions: list[Prediction]) -> list[Prediction]:
        """Persist multiple predictions in a single operation.

        Args:
            predictions: List of Prediction entities to save.

        Returns:
            The persisted Prediction entities.
        """
        ...


# ---------------------------------------------------------------------------
# TeamStats Repository
# ---------------------------------------------------------------------------


class TeamStatsRepository(ABC):
    """Persistence contract for TeamStats entities."""

    @abstractmethod
    def get_by_team_and_season(self, team: Team, league: League, season: str) -> TeamStats | None:
        """Retrieve team statistics for a specific season.

        Args:
            team: The team.
            league: The league context.
            season: Season identifier (e.g. "2025/2026").

        Returns:
            TeamStats if available, None otherwise.
        """
        ...

    @abstractmethod
    def find_by_league_season(self, league: League, season: str) -> list[TeamStats]:
        """Retrieve stats for all teams in a league during a given season.

        Args:
            league: The league.
            season: Season identifier.

        Returns:
            List of TeamStats entities.
        """
        ...

    @abstractmethod
    def save(self, stats: TeamStats) -> TeamStats:
        """Persist team statistics.

        Args:
            stats: The TeamStats to save.

        Returns:
            The persisted TeamStats.
        """
        ...


# ---------------------------------------------------------------------------
# Football Data Source (external data provider contract)
# ---------------------------------------------------------------------------


class FootballDataSource(ABC):
    """Contract for an external football data provider (e.g. football-data.org).

    This is a *domain service* interface — the concrete HTTP client,
    caching layer, and API-key management belong in infrastructure.
    """

    @abstractmethod
    def fetch_league_standings(self, league: League, season: str | None = None) -> list[TeamStats]:
        """Fetch current league standings from the external source.

        Args:
            league: The league to query.
            season: Optional season filter.

        Returns:
            List of TeamStats as provided by the external API.
        """
        ...

    @abstractmethod
    def fetch_matches(
        self,
        league: League,
        season: str | None = None,
        matchday: int | None = None,
    ) -> list[Match]:
        """Fetch matches from the external source.

        Args:
            league: The league to query.
            season: Optional season filter.
            matchday: Optional matchday filter.

        Returns:
            List of Match entities populated from the external API.
        """
        ...

    @abstractmethod
    def fetch_teams(self, league: League, season: str | None = None) -> list[Team]:
        """Fetch all teams participating in a league.

        Args:
            league: The league to query.
            season: Optional season filter.

        Returns:
            List of Team entities.
        """
        ...


# ---------------------------------------------------------------------------
# Model Store (prediction model persistence)
# ---------------------------------------------------------------------------


class ModelStore(ABC):
    """Contract for storing and retrieving trained prediction models."""

    @abstractmethod
    def save_model(self, model: object, version: str) -> str:
        """Persist a trained model artifact.

        Args:
            model: The trained model object (framework-agnostic).
            version: Semantic version identifier for the model.

        Returns:
            The path or URI where the model was stored.
        """
        ...

    @abstractmethod
    def load_model(self, version: str) -> object:
        """Load a previously saved model artifact.

        Args:
            version: Model version to load. Use "latest" for the most recent.

        Returns:
            The deserialized model object.

        Raises:
            FileNotFoundError: If the model version does not exist.
        """
        ...

    @abstractmethod
    def list_versions(self) -> list[str]:
        """List all available model versions.

        Returns:
            Sorted list of version identifiers.
        """
        ...

    @abstractmethod
    def delete_model(self, version: str) -> None:
        """Delete a model artifact.

        Args:
            version: Model version to delete.

        Raises:
            FileNotFoundError: If the model version does not exist.
        """
        ...


# ---------------------------------------------------------------------------
# World Cup Results Repository
# ---------------------------------------------------------------------------


class WorldCupResultsRepository(ABC):
    """Persistence contract for World Cup match results."""

    @abstractmethod
    def save_result(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        group_letter: str,
        match_date: str,
        home_goals: int,
        away_goals: int,
        entered_by: str = "user",
    ) -> None: ...

    @abstractmethod
    def get_result(self, match_id: str) -> dict | None: ...

    @abstractmethod
    def get_all_results(self) -> list[dict]: ...

    @abstractmethod
    def get_results_by_group(self, group_letter: str) -> list[dict]: ...

    @abstractmethod
    def delete_result(self, match_id: str) -> bool: ...

    @abstractmethod
    def get_stats(self) -> dict: ...
