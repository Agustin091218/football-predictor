"""SQLite-backed World Cup results repository.

Persists actual match results (entered by user) in the same
database as matches, team_stats, and predictions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Column, Integer, MetaData, Table, Text, create_engine, select

_wc_metadata = MetaData()

wc_results_table = Table(
    "wc_results",
    _wc_metadata,
    Column("match_id", Text, primary_key=True),
    Column("home_team", Text, nullable=False),
    Column("away_team", Text, nullable=False),
    Column("group_letter", Text, nullable=False),
    Column("match_date", Text, nullable=False),
    Column("home_goals", Integer, nullable=False),
    Column("away_goals", Integer, nullable=False),
    Column("actual_result", Text, nullable=False),
    Column("entered_at", Text, nullable=False),
    Column("entered_by", Text, default="user"),
)


class SqliteWorldCupResultsRepository:
    def __init__(self, db_path: str = "football.db") -> None:
        engine_url = f"sqlite:///{db_path}"
        self._engine = create_engine(engine_url, echo=False)
        _wc_metadata.create_all(self._engine)

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
    ) -> None:
        if home_goals > away_goals:
            actual = "1"
        elif home_goals == away_goals:
            actual = "X"
        else:
            actual = "2"

        values = {
            "match_id": match_id,
            "home_team": home_team,
            "away_team": away_team,
            "group_letter": group_letter,
            "match_date": match_date,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "actual_result": actual,
            "entered_at": datetime.now().isoformat(),
            "entered_by": entered_by,
        }

        with self._engine.connect() as conn:
            stmt = wc_results_table.insert().prefix_with("OR REPLACE").values(**values)
            conn.execute(stmt)
            conn.commit()

    def get_result(self, match_id: str) -> dict[str, Any] | None:
        with self._engine.connect() as conn:
            stmt = select(wc_results_table).where(wc_results_table.c.match_id == match_id)
            row = conn.execute(stmt).first()
            if row is None:
                return None
            return dict(row._mapping)

    def get_all_results(self) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            stmt = select(wc_results_table).order_by(wc_results_table.c.match_date.asc())
            rows = conn.execute(stmt).fetchall()
            return [dict(r._mapping) for r in rows]

    def get_results_by_group(self, group_letter: str) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            stmt = (
                select(wc_results_table)
                .where(wc_results_table.c.group_letter == group_letter)
                .order_by(wc_results_table.c.match_date.asc())
            )
            rows = conn.execute(stmt).fetchall()
            return [dict(r._mapping) for r in rows]

    def delete_result(self, match_id: str) -> bool:
        with self._engine.connect() as conn:
            from sqlalchemy import delete

            stmt = delete(wc_results_table).where(wc_results_table.c.match_id == match_id)
            result = conn.execute(stmt)
            conn.commit()
            return result.rowcount > 0

    def get_stats(self) -> dict[str, Any]:
        with self._engine.connect() as conn:
            stmt = select(wc_results_table)
            rows = conn.execute(stmt).fetchall()

        if not rows:
            return {
                "total_results": 0,
                "by_group": {},
                "home_wins": 0,
                "draws": 0,
                "away_wins": 0,
                "last_entered_at": None,
            }

        by_group: dict[str, int] = {}
        home_wins = draws = away_wins = 0
        last_at = None

        for r in rows:
            g = r.group_letter
            by_group[g] = by_group.get(g, 0) + 1
            ar = r.actual_result
            if ar == "1":
                home_wins += 1
            elif ar == "X":
                draws += 1
            else:
                away_wins += 1
            if last_at is None or r.entered_at > last_at:
                last_at = r.entered_at

        return {
            "total_results": len(rows),
            "by_group": by_group,
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins,
            "last_entered_at": last_at,
        }
