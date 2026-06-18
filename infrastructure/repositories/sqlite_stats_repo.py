"""SQLite-backed TeamStatsRepository using SQLAlchemy Core (no ORM).

Shares the same database file as SqliteMatchRepository via db_path.
Uses a composite primary key: (team_id, league_id, season).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, Integer, MetaData, Table, Text, create_engine, select

from football_predictor.domain.entities import (
    CompetitionType,
    League,
    Team,
    TeamStats,
)
from football_predictor.domain.repositories import TeamStatsRepository

# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------

_stats_metadata = MetaData()

team_stats_table = Table(
    "team_stats",
    _stats_metadata,
    Column("team_id", Text, primary_key=True),
    Column("league_id", Text, primary_key=True),
    Column("season", Text, primary_key=True),
    # Denormalised team info
    Column("team_name", Text, nullable=False),
    Column("team_short_name", Text, nullable=False),
    Column("team_tla", Text, nullable=False),
    Column("team_country", Text, nullable=False),
    # Core stats
    Column("matches_played", Integer, default=0),
    Column("matches_home", Integer, default=0),
    Column("matches_away", Integer, default=0),
    Column("goals_scored", Integer, default=0),
    Column("goals_conceded", Integer, default=0),
    Column("goals_scored_home", Integer, default=0),
    Column("goals_scored_away", Integer, default=0),
    Column("goals_conceded_home", Integer, default=0),
    Column("goals_conceded_away", Integer, default=0),
    Column("wins", Integer, default=0),
    Column("draws", Integer, default=0),
    Column("losses", Integer, default=0),
    # Metadata
    Column("updated_at", Text, nullable=False),
)


# ---------------------------------------------------------------------------
# Repository implementation
# ---------------------------------------------------------------------------


class SqliteStatsRepository(TeamStatsRepository):
    """SQLite persistence adapter for TeamStats entities.

    Uses the same db_path convention as SqliteMatchRepository so both
    repos share the same database file (different tables).
    """

    def __init__(self, db_path: str = "football.db") -> None:
        engine_url = f"sqlite:///{db_path}"
        self._engine = create_engine(engine_url, echo=False)
        _stats_metadata.create_all(self._engine)

    # ------------------------------------------------------------------
    # ABC methods
    # ------------------------------------------------------------------

    def get_by_team_and_season(self, team: Team, league: League, season: str) -> TeamStats | None:
        """Retrieve TeamStats for a specific team in a league-season."""
        with self._engine.connect() as conn:
            stmt = (
                select(team_stats_table)
                .where(team_stats_table.c.team_id == str(team.id))
                .where(team_stats_table.c.league_id == str(league.id))
                .where(team_stats_table.c.season == season)
            )
            row = conn.execute(stmt).first()
            if row is None:
                return None
            return self._row_to_stats(row)

    def find_by_league_season(self, league: League, season: str) -> list[TeamStats]:
        """Retrieve all TeamStats for a league-season."""
        with self._engine.connect() as conn:
            stmt = (
                select(team_stats_table)
                .where(team_stats_table.c.league_id == str(league.id))
                .where(team_stats_table.c.season == season)
            )
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_stats(r) for r in rows]

    def save(self, stats: TeamStats) -> TeamStats:
        """Persist team statistics (upsert)."""
        values = self._stats_to_row_values(stats)
        with self._engine.connect() as conn:
            stmt = team_stats_table.insert().prefix_with("OR REPLACE").values(**values)
            conn.execute(stmt)
            conn.commit()
        return stats

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_stats(self, team_id: int, league_id: str, season: str) -> TeamStats | None:
        """Lookup by raw IDs (convenience, used by use cases)."""
        with self._engine.connect() as conn:
            stmt = (
                select(team_stats_table)
                .where(team_stats_table.c.team_id == str(team_id))
                .where(team_stats_table.c.league_id == league_id)
                .where(team_stats_table.c.season == season)
            )
            row = conn.execute(stmt).first()
            if row is None:
                return None
            return self._row_to_stats(row)

    def save_stats(self, stats: TeamStats) -> None:
        """Alias for save (convenience, used by use cases)."""
        self.save(stats)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _row_to_stats(self, row) -> TeamStats:
        """Reconstruct a TeamStats domain entity from a database row."""
        team = Team(
            id=int(row.team_id),
            name=row.team_name,
            short_name=row.team_short_name,
            tla=row.team_tla,
            country=row.team_country,
        )
        league = League(
            id=row.league_id,
            name="",
            country="",
            competition_type=CompetitionType.LEAGUE,
            season=row.season,
        )
        return TeamStats(
            team=team,
            league=league,
            season=row.season,
            matches_played=row.matches_played,
            matches_home=row.matches_home,
            matches_away=row.matches_away,
            goals_scored=row.goals_scored,
            goals_conceded=row.goals_conceded,
            goals_scored_home=row.goals_scored_home,
            goals_scored_away=row.goals_scored_away,
            goals_conceded_home=row.goals_conceded_home,
            goals_conceded_away=row.goals_conceded_away,
            wins=row.wins,
            draws=row.draws,
            losses=row.losses,
        )

    @staticmethod
    def _stats_to_row_values(stats: TeamStats) -> dict:
        """Serialize TeamStats to column values for INSERT/REPLACE."""
        return {
            "team_id": str(stats.team.id),
            "league_id": str(stats.league.id),
            "season": stats.season,
            "team_name": stats.team.name,
            "team_short_name": stats.team.short_name,
            "team_tla": stats.team.tla,
            "team_country": stats.team.country,
            "matches_played": stats.matches_played,
            "matches_home": stats.matches_home,
            "matches_away": stats.matches_away,
            "goals_scored": stats.goals_scored,
            "goals_conceded": stats.goals_conceded,
            "goals_scored_home": stats.goals_scored_home,
            "goals_scored_away": stats.goals_scored_away,
            "goals_conceded_home": stats.goals_conceded_home,
            "goals_conceded_away": stats.goals_conceded_away,
            "wins": stats.wins,
            "draws": stats.draws,
            "losses": stats.losses,
            "updated_at": datetime.now().isoformat(),
        }
