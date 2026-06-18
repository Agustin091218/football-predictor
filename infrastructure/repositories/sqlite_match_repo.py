"""SQLite-backed MatchRepository using SQLAlchemy Core (no ORM).

Stores match data in a flat denormalised table for fast reads.
All IDs are stored as TEXT to match the football-data.org API format.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    select,
)

from football_predictor.domain.entities import (
    CompetitionType,
    League,
    Match,
    MatchScore,
    MatchStatus,
    Team,
)
from football_predictor.domain.repositories import MatchRepository

# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------

metadata = MetaData()

matches_table = Table(
    "matches",
    metadata,
    Column("id", Text, primary_key=True),
    # Home team (denormalised)
    Column("home_team_id", Text, nullable=False),
    Column("home_team_name", Text, nullable=False),
    Column("home_team_short_name", Text, nullable=False),
    Column("home_team_tla", Text, nullable=False),
    Column("home_team_country", Text, nullable=False),
    # Away team (denormalised)
    Column("away_team_id", Text, nullable=False),
    Column("away_team_name", Text, nullable=False),
    Column("away_team_short_name", Text, nullable=False),
    Column("away_team_tla", Text, nullable=False),
    Column("away_team_country", Text, nullable=False),
    # League (denormalised)
    Column("league_id", Text, nullable=False),
    Column("league_name", Text, nullable=False),
    Column("league_country", Text, nullable=False),
    Column("league_competition_type", Text, nullable=False),
    Column("league_season", Text, nullable=True),
    # Match data
    Column("match_date", Text, nullable=False),
    Column("status", Text, nullable=False),
    # Score
    Column("home_goals", Integer, nullable=True),
    Column("away_goals", Integer, nullable=True),
    Column("home_goals_ht", Integer, nullable=True),
    Column("away_goals_ht", Integer, nullable=True),
    # Metadata
    Column("matchday", Integer, nullable=True),
    Column("venue", Text, nullable=True),
    Column("referee", Text, nullable=True),
)


# ---------------------------------------------------------------------------
# Repository implementation
# ---------------------------------------------------------------------------


class SqliteMatchRepository(MatchRepository):
    """SQLite persistence adapter for Match entities.

    Uses SQLAlchemy Core with explicit column definitions.
    No ORM, no declarative_base, no mapped classes.
    """

    def __init__(self, db_path: str = "football.db") -> None:
        engine_url = f"sqlite:///{db_path}"
        self._engine = create_engine(engine_url, echo=False)
        metadata.create_all(self._engine)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def get_by_id(self, match_id: int) -> Match | None:
        """Retrieve a single match by its unique identifier."""
        with self._engine.connect() as conn:
            stmt = select(matches_table).where(matches_table.c.id == str(match_id))
            row = conn.execute(stmt).first()
            if row is None:
                return None
            return self._row_to_match(row)

    def find_by_league(
        self,
        league: League,
        season: str | None = None,
        matchday: int | None = None,
    ) -> list[Match]:
        """Find all matches for a given league, optionally filtered."""
        with self._engine.connect() as conn:
            stmt = select(matches_table).where(matches_table.c.league_id == str(league.id))
            if season is not None:
                stmt = stmt.where(matches_table.c.league_season == season)
            if matchday is not None:
                stmt = stmt.where(matches_table.c.matchday == matchday)
            stmt = stmt.order_by(matches_table.c.match_date)
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_match(r) for r in rows]

    def find_by_team(self, team: Team, limit: int = 50) -> list[Match]:
        """Find recent matches involving a specific team (home or away)."""
        team_id = str(team.id)
        with self._engine.connect() as conn:
            stmt = (
                select(matches_table)
                .where(
                    (matches_table.c.home_team_id == team_id)
                    | (matches_table.c.away_team_id == team_id)
                )
                .order_by(matches_table.c.match_date.desc())
                .limit(limit)
            )
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_match(r) for r in rows]

    def find_by_status(self, status: MatchStatus) -> list[Match]:
        """Find all matches with a given lifecycle status."""
        with self._engine.connect() as conn:
            stmt = (
                select(matches_table)
                .where(matches_table.c.status == status.value)
                .order_by(matches_table.c.match_date)
            )
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_match(r) for r in rows]

    def find_by_date_range(self, start: datetime, end: datetime) -> list[Match]:
        """Find matches scheduled or played within a date range."""
        with self._engine.connect() as conn:
            stmt = (
                select(matches_table)
                .where(matches_table.c.match_date >= start.isoformat())
                .where(matches_table.c.match_date <= end.isoformat())
                .order_by(matches_table.c.match_date)
            )
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_match(r) for r in rows]

    def save(self, match: Match) -> Match:
        """Persist a new or updated match (upsert via INSERT OR REPLACE)."""
        values = self._match_to_row_values(match)
        with self._engine.connect() as conn:
            stmt = matches_table.insert().prefix_with("OR REPLACE").values(**values)
            conn.execute(stmt)
            conn.commit()
        return match

    def save_many(self, matches: list[Match]) -> list[Match]:
        """Persist multiple matches in a single transaction."""
        with self._engine.begin() as conn:
            for match in matches:
                values = self._match_to_row_values(match)
                stmt = matches_table.insert().prefix_with("OR REPLACE").values(**values)
                conn.execute(stmt)
        return matches

    # ------------------------------------------------------------------
    # Convenience query methods (not part of the abstract interface)
    # ------------------------------------------------------------------

    def get_by_league_and_season(self, league_id: str, season: str) -> list[Match]:
        """Find all matches for a league-season, ordered by date ascending.

        Args:
            league_id: League identifier (string code, e.g. "PL").
            season: Season identifier (e.g. "2025").

        Returns:
            List of Match entities ordered by match_date ASC.
        """
        with self._engine.connect() as conn:
            stmt = (
                select(matches_table)
                .where(matches_table.c.league_id == league_id)
                .where(matches_table.c.league_season == season)
                .order_by(matches_table.c.match_date.asc())
            )
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_match(r) for r in rows]

    def get_finished_matches(
        self,
        league_id: str,
        season: str,
        limit: int | None = None,
    ) -> list[Match]:
        """Find finished matches for a league-season, most recent first.

        Args:
            league_id: League identifier.
            season: Season identifier.
            limit: Maximum number of matches to return.

        Returns:
            List of finished Match entities ordered by match_date DESC.
        """
        with self._engine.connect() as conn:
            stmt = (
                select(matches_table)
                .where(matches_table.c.league_id == league_id)
                .where(matches_table.c.league_season == season)
                .where(matches_table.c.status == MatchStatus.FINISHED.value)
                .order_by(matches_table.c.match_date.desc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_match(r) for r in rows]

    def get_upcoming_matches(
        self,
        league_id: str,
        from_date: datetime | None = None,
        days_ahead: int = 7,
    ) -> list[Match]:
        """Find scheduled matches within a future window.

        Args:
            league_id: League identifier.
            from_date: Start of the window (defaults to now).
            days_ahead: Number of days to look ahead.

        Returns:
            List of scheduled Match entities ordered by match_date ASC.
        """
        if from_date is None:
            from_date = datetime.now()

        date_from = from_date.isoformat()
        date_to = (from_date + timedelta(days=days_ahead)).isoformat()

        with self._engine.connect() as conn:
            stmt = (
                select(matches_table)
                .where(matches_table.c.league_id == league_id)
                .where(matches_table.c.status == MatchStatus.SCHEDULED.value)
                .where(matches_table.c.match_date >= date_from)
                .where(matches_table.c.match_date <= date_to)
                .order_by(matches_table.c.match_date.asc())
            )
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_match(r) for r in rows]

    def get_head_to_head(
        self,
        home_team_id: str,
        away_team_id: str,
        limit: int = 10,
    ) -> list[Match]:
        """Find historical matches between two specific teams.

        Looks for matches where (home=X AND away=Y) OR (home=Y AND away=X),
        ordered by most recent first.

        Args:
            home_team_id: First team identifier.
            away_team_id: Second team identifier.
            limit: Maximum number of matches to return.

        Returns:
            List of Match entities ordered by match_date DESC.
        """
        with self._engine.connect() as conn:
            stmt = (
                select(matches_table)
                .where(
                    (
                        (matches_table.c.home_team_id == home_team_id)
                        & (matches_table.c.away_team_id == away_team_id)
                    )
                    | (
                        (matches_table.c.home_team_id == away_team_id)
                        & (matches_table.c.away_team_id == home_team_id)
                    )
                )
                .order_by(matches_table.c.match_date.desc())
                .limit(limit)
            )
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_match(r) for r in rows]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _row_to_match(self, row) -> Match:
        """Reconstruct a Match domain entity from a database row.

        Args:
            row: SQLAlchemy Row object from the matches table.

        Returns:
            Fully constructed Match entity.
        """
        home_team = Team(
            id=int(row.home_team_id),
            name=row.home_team_name,
            short_name=row.home_team_short_name,
            tla=row.home_team_tla,
            country=row.home_team_country,
        )
        away_team = Team(
            id=int(row.away_team_id),
            name=row.away_team_name,
            short_name=row.away_team_short_name,
            tla=row.away_team_tla,
            country=row.away_team_country,
        )
        league = League(
            id=row.league_id,  # may be str (e.g. "PL") — kept as-is per TEXT schema
            name=row.league_name,
            country=row.league_country,
            competition_type=CompetitionType(row.league_competition_type),
            season=row.league_season,
        )

        # Build MatchScore only if at least one goal value is present
        score: MatchScore | None = None
        if row.home_goals is not None or row.away_goals is not None:
            score = MatchScore(
                home_goals=row.home_goals,
                away_goals=row.away_goals,
            )

        return Match(
            id=int(row.id),
            home_team=home_team,
            away_team=away_team,
            league=league,
            match_date=datetime.fromisoformat(row.match_date),
            status=MatchStatus(row.status),
            score=score,
            matchday=row.matchday,
            venue=row.venue,
        )

    def _match_to_row_values(self, match: Match) -> dict:
        """Serialize a Match domain entity to a dictionary of column values.

        Args:
            match: The Match entity to persist.

        Returns:
            Dictionary keyed by column name for INSERT/REPLACE.
        """
        values: dict = {
            "id": str(match.id),
            # Home team
            "home_team_id": str(match.home_team.id),
            "home_team_name": match.home_team.name,
            "home_team_short_name": match.home_team.short_name,
            "home_team_tla": match.home_team.tla,
            "home_team_country": match.home_team.country,
            # Away team
            "away_team_id": str(match.away_team.id),
            "away_team_name": match.away_team.name,
            "away_team_short_name": match.away_team.short_name,
            "away_team_tla": match.away_team.tla,
            "away_team_country": match.away_team.country,
            # League
            "league_id": str(match.league.id),
            "league_name": match.league.name,
            "league_country": match.league.country,
            "league_competition_type": match.league.competition_type.value,
            "league_season": match.league.season,
            # Match data
            "match_date": match.match_date.isoformat(),
            "status": match.status.value,
            # Score
            "home_goals": match.score.home_goals if match.score else None,
            "away_goals": match.score.away_goals if match.score else None,
            "home_goals_ht": None,
            "away_goals_ht": None,
            # Metadata
            "matchday": match.matchday,
            "venue": match.venue,
            "referee": None,
        }
        return values
