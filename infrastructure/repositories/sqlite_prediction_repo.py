"""SQLite-backed PredictionRepository using SQLAlchemy Core (no ORM).

Persists predictions with minimal match metadata for display purposes.
Goal probability matrices are NOT persisted (too large for a flat table).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    Table,
    Text,
    and_,
    create_engine,
    select,
)

from football_predictor.domain.entities import (
    CompetitionType,
    GoalProbabilities,
    League,
    Match,
    MatchResult,
    MatchStatus,
    Prediction,
    Team,
)
from football_predictor.domain.repositories import PredictionRepository

# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------

_pred_metadata = MetaData()

predictions_table = Table(
    "predictions",
    _pred_metadata,
    Column("match_id", Text, primary_key=True),
    # Match metadata (minimal, for display)
    Column("home_team_name", Text, nullable=False),
    Column("away_team_name", Text, nullable=False),
    Column("league_id", Text, nullable=False),
    Column("league_name", Text, nullable=False),
    Column("match_date", Text, nullable=False),
    # Prediction data
    Column("prob_home_win", Float, nullable=False),
    Column("prob_draw", Float, nullable=False),
    Column("prob_away_win", Float, nullable=False),
    Column("expected_goals_home", Float, nullable=False),
    Column("expected_goals_away", Float, nullable=False),
    Column("predicted_result", Text, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("model_version", Text, nullable=False),
    Column("predicted_at", Text, nullable=False),
    # Evaluation
    Column("was_correct", Integer, nullable=True),
    Column("actual_result", Text, nullable=True),
    # LLM + signal graph fields (phase 5B)
    Column("signal_outputs", Text, nullable=True),
    Column("simulation_json", Text, nullable=True),
    Column("llm_explanation", Text, nullable=True),
    Column("llm_actions", Text, nullable=True),
    Column("extended_predictions", Text, nullable=True),
)

# Mapping for by_result display keys
_RESULT_KEYS = {
    MatchResult.HOME_WIN.value: "1",
    MatchResult.DRAW.value: "X",
    MatchResult.AWAY_WIN.value: "2",
}


# ---------------------------------------------------------------------------
# Repository implementation
# ---------------------------------------------------------------------------


class SqlitePredictionRepository(PredictionRepository):
    """SQLite persistence adapter for Prediction entities.

    Shares the same db_path convention so it lives alongside
    SqliteMatchRepository and SqliteStatsRepository in the same database.
    """

    def __init__(self, db_path: str = "football.db") -> None:
        engine_url = f"sqlite:///{db_path}"
        self._engine = create_engine(engine_url, echo=False)
        _pred_metadata.create_all(self._engine)

    # ------------------------------------------------------------------
    # ABC methods
    # ------------------------------------------------------------------

    def get_by_match(self, match_id: int) -> Prediction | None:
        """Retrieve the prediction for a match by its integer ID."""
        with self._engine.connect() as conn:
            stmt = select(predictions_table).where(predictions_table.c.match_id == str(match_id))
            row = conn.execute(stmt).first()
            if row is None:
                return None
            return self._row_to_prediction(row)

    def find_by_model_version(self, model_version: str) -> list[Prediction]:
        """Find all predictions from a specific model version."""
        with self._engine.connect() as conn:
            stmt = select(predictions_table).where(
                predictions_table.c.model_version == model_version
            )
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_prediction(r) for r in rows]

    def find_pending_evaluation(self) -> list[Prediction]:
        """Find predictions awaiting correctness evaluation."""
        with self._engine.connect() as conn:
            stmt = select(predictions_table).where(predictions_table.c.was_correct.is_(None))
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_prediction(r) for r in rows]

    def save(self, prediction: Prediction) -> Prediction:
        """Persist a prediction (upsert via INSERT OR REPLACE)."""
        values = self._prediction_to_row_values(prediction)
        with self._engine.connect() as conn:
            stmt = predictions_table.insert().prefix_with("OR REPLACE").values(**values)
            conn.execute(stmt)
            conn.commit()
        return prediction

    def save_many(self, predictions: list[Prediction]) -> list[Prediction]:
        """Persist multiple predictions in a single transaction."""
        with self._engine.begin() as conn:
            for pred in predictions:
                values = self._prediction_to_row_values(pred)
                stmt = predictions_table.insert().prefix_with("OR REPLACE").values(**values)
                conn.execute(stmt)
        return predictions

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_by_match_id(self, match_id: str) -> Prediction | None:
        """Retrieve prediction by match ID string (convenience)."""
        return self.get_by_match(int(match_id))

    def get_recent(self, limit: int = 50) -> list[Prediction]:
        """Return most recent predictions ordered by predicted_at DESC."""
        with self._engine.connect() as conn:
            stmt = (
                select(predictions_table)
                .order_by(predictions_table.c.predicted_at.desc())
                .limit(limit)
            )
            rows = conn.execute(stmt).fetchall()
            return [self._row_to_prediction(r) for r in rows]

    def get_accuracy_stats(
        self,
        league_id: str | None = None,
        from_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Compute aggregate accuracy statistics.

        Only considers predictions that have been evaluated (was_correct IS NOT NULL).

        Args:
            league_id: Optional league filter.
            from_date: Optional minimum prediction date.

        Returns:
            Dictionary with total_evaluated, correct, accuracy, and by_result breakdown.
        """
        with self._engine.connect() as conn:
            conditions = [predictions_table.c.was_correct.isnot(None)]
            if league_id is not None:
                conditions.append(predictions_table.c.league_id == league_id)
            if from_date is not None:
                conditions.append(predictions_table.c.predicted_at >= from_date.isoformat())

            stmt = select(predictions_table).where(and_(*conditions))
            rows = conn.execute(stmt).fetchall()

        total = len(rows)
        if total == 0:
            return {
                "total_evaluated": 0,
                "correct": 0,
                "accuracy": 0.0,
                "by_result": {
                    "1": {"predicted": 0, "correct": 0, "accuracy": 0.0},
                    "X": {"predicted": 0, "correct": 0, "accuracy": 0.0},
                    "2": {"predicted": 0, "correct": 0, "accuracy": 0.0},
                },
            }

        correct_count = sum(1 for r in rows if r.was_correct == 1)

        by_result: dict[str, dict[str, Any]] = {}
        for key in ("1", "X", "2"):
            by_result[key] = {"predicted": 0, "correct": 0, "accuracy": 0.0}

        for row in rows:
            display_key = _RESULT_KEYS.get(row.predicted_result, row.predicted_result)
            if display_key in by_result:
                by_result[display_key]["predicted"] += 1
                if row.was_correct == 1:
                    by_result[display_key]["correct"] += 1

        for key in ("1", "X", "2"):
            n = by_result[key]["predicted"]
            if n > 0:
                by_result[key]["accuracy"] = round(by_result[key]["correct"] / n, 4)

        return {
            "total_evaluated": total,
            "correct": correct_count,
            "accuracy": round(correct_count / total, 4),
            "by_result": by_result,
        }

    def update_result(self, match_id: str, actual_result: MatchResult) -> None:
        """Evaluate a prediction against the actual match result.

        Sets actual_result and was_correct on the prediction row.
        was_correct = 1 if predicted_result matches actual_result.value,
        otherwise 0.

        Args:
            match_id: Match identifier string.
            actual_result: The actual MatchResult from the finished match.
        """
        with self._engine.connect() as conn:
            # Read current predicted_result
            stmt = select(predictions_table.c.predicted_result).where(
                predictions_table.c.match_id == match_id
            )
            row = conn.execute(stmt).first()
            if row is None:
                return

            was_correct = 1 if row.predicted_result == actual_result.value else 0

            from sqlalchemy import update

            upd = (
                update(predictions_table)
                .where(predictions_table.c.match_id == match_id)
                .values(
                    actual_result=actual_result.value,
                    was_correct=was_correct,
                )
            )
            conn.execute(upd)
            conn.commit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _row_to_prediction(self, row) -> Prediction:
        """Reconstruct a Prediction from a database row.

        Match metadata is minimal (only what was stored).
        GoalProbabilities are NOT persisted — returned as an empty stub.
        """
        home_team = Team(
            id=0,
            name=row.home_team_name,
            short_name="",
            tla="",
            country="",
        )
        away_team = Team(
            id=0,
            name=row.away_team_name,
            short_name="",
            tla="",
            country="",
        )
        league = League(
            id=row.league_id,
            name=row.league_name,
            country="",
            competition_type=CompetitionType.LEAGUE,
        )
        match = Match(
            id=int(row.match_id),
            home_team=home_team,
            away_team=away_team,
            league=league,
            match_date=datetime.fromisoformat(row.match_date),
            status=MatchStatus.SCHEDULED,
        )

        was_correct = None
        if row.was_correct is not None:
            was_correct = bool(row.was_correct)

        return Prediction(
            match=match,
            prob_home_win=row.prob_home_win,
            prob_draw=row.prob_draw,
            prob_away_win=row.prob_away_win,
            expected_goals_home=row.expected_goals_home,
            expected_goals_away=row.expected_goals_away,
            goal_probabilities=GoalProbabilities(prob_matrix=[[0.0, 0.0], [0.0, 0.0]], max_goals=1),
            confidence=row.confidence,
            model_version=row.model_version,
            predicted_at=datetime.fromisoformat(row.predicted_at),
            was_correct=was_correct,
            signal_outputs=_safe_json_loads(row.signal_outputs),
            llm_explanation=row.llm_explanation,
            llm_actions=_safe_json_loads(row.llm_actions),
            extended_predictions=_safe_json_loads(row.extended_predictions),
        )

    @staticmethod
    def _prediction_to_row_values(prediction: Prediction) -> dict:
        """Serialize Prediction to column values for INSERT/REPLACE."""
        return {
            "match_id": str(prediction.match.id),
            "home_team_name": prediction.match.home_team.name,
            "away_team_name": prediction.match.away_team.name,
            "league_id": str(prediction.match.league.id),
            "league_name": prediction.match.league.name or "",
            "match_date": prediction.match.match_date.isoformat(),
            "prob_home_win": prediction.prob_home_win,
            "prob_draw": prediction.prob_draw,
            "prob_away_win": prediction.prob_away_win,
            "expected_goals_home": prediction.expected_goals_home,
            "expected_goals_away": prediction.expected_goals_away,
            "predicted_result": prediction.predicted_result.value,
            "confidence": prediction.confidence,
            "model_version": prediction.model_version,
            "predicted_at": prediction.predicted_at.isoformat(),
            "was_correct": prediction.was_correct,
            "actual_result": None,
            "signal_outputs": json.dumps(prediction.signal_outputs)
            if prediction.signal_outputs
            else None,
            "simulation_json": json.dumps(prediction.simulation.as_dict())
            if prediction.simulation
            else None,
            "llm_explanation": prediction.llm_explanation,
            "llm_actions": json.dumps(prediction.llm_actions) if prediction.llm_actions else None,
            "extended_predictions": json.dumps(prediction.extended_predictions)
            if prediction.extended_predictions
            else None,
        }


def _safe_json_loads(value: str | None) -> Any:
    """Parse a JSON string, returning None on failure or empty input."""
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
