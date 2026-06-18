"""Use case: compute team statistics from finished matches."""

from __future__ import annotations

import logging
from typing import Any

from football_predictor.domain.entities import (
    League,
    MatchResult,
    TeamStats,
)
from football_predictor.domain.repositories import MatchRepository, TeamStatsRepository

logger = logging.getLogger(__name__)

# Minimum matches required before computing statistics
MIN_MATCHES = 5


class ComputeStatsUseCase:
    """Aggregates finished match data into per-team statistics.

    Iterates over finished matches, accumulates goals, results, and
    home/away splits for each team, then persists TeamStats entities.
    """

    def __init__(
        self,
        match_repo: MatchRepository,
        stats_repo: TeamStatsRepository,
    ) -> None:
        self._match_repo = match_repo
        self._stats_repo = stats_repo

    def execute(self, league_id: str, season: str) -> dict[str, Any]:
        """Compute and persist TeamStats for all teams in a league-season.

        Args:
            league_id: Competition code (e.g. "PL").
            season: Season identifier (e.g. "2024").

        Returns:
            Dictionary with status, teams_computed, and matches_used.
        """
        matches = self._match_repo.get_finished_matches(league_id, season)  # type: ignore[attr-defined]

        if len(matches) < MIN_MATCHES:
            logger.warning(
                "Insufficient data for %s/%s: %d matches (need %d)",
                league_id,
                season,
                len(matches),
                MIN_MATCHES,
            )
            return {"status": "insufficient_data", "matches_found": len(matches)}

        team_data: dict[int, dict[str, Any]] = {}

        for match in matches:
            if match.score is None or match.score.result is None:
                continue

            home_id = match.home_team.id
            away_id = match.away_team.id
            result = match.score.result

            # Ensure both teams are registered
            for team, is_home in [(match.home_team, True), (match.away_team, False)]:
                if team.id not in team_data:
                    team_data[team.id] = {
                        "team": team,
                        "matches_home": 0,
                        "matches_away": 0,
                        "goals_scored_home": 0,
                        "goals_scored_away": 0,
                        "goals_conceded_home": 0,
                        "goals_conceded_away": 0,
                        "wins": 0,
                        "draws": 0,
                        "losses": 0,
                        "wins_home": 0,
                        "wins_away": 0,
                        "draws_home": 0,
                        "draws_away": 0,
                        "losses_home": 0,
                        "losses_away": 0,
                    }

            # Home team accumulation
            td_home = team_data[home_id]
            td_home["matches_home"] += 1
            td_home["goals_scored_home"] += match.score.home_goals or 0
            td_home["goals_conceded_home"] += match.score.away_goals or 0

            # Away team accumulation
            td_away = team_data[away_id]
            td_away["matches_away"] += 1
            td_away["goals_scored_away"] += match.score.away_goals or 0
            td_away["goals_conceded_away"] += match.score.home_goals or 0

            # Win / draw / loss
            if result == MatchResult.HOME_WIN:
                td_home["wins"] += 1
                td_home["wins_home"] += 1
                td_away["losses"] += 1
                td_away["losses_away"] += 1
            elif result == MatchResult.DRAW:
                td_home["draws"] += 1
                td_home["draws_home"] += 1
                td_away["draws"] += 1
                td_away["draws_away"] += 1
            else:  # AWAY_WIN
                td_away["wins"] += 1
                td_away["wins_away"] += 1
                td_home["losses"] += 1
                td_home["losses_home"] += 1

        # Use the league from the first match as context
        reference_league = matches[0].league

        # Build and persist TeamStats for each team
        for team_id, td in team_data.items():
            stats = TeamStats(
                team=td["team"],
                league=League(
                    id=reference_league.id,
                    name=reference_league.name,
                    country=reference_league.country,
                    competition_type=reference_league.competition_type,
                    season=season,
                ),
                season=season,
                matches_played=td["matches_home"] + td["matches_away"],
                matches_home=td["matches_home"],
                matches_away=td["matches_away"],
                wins=td["wins"],
                wins_home=td["wins_home"],
                wins_away=td["wins_away"],
                draws=td["draws"],
                draws_home=td["draws_home"],
                draws_away=td["draws_away"],
                losses=td["losses"],
                losses_home=td["losses_home"],
                losses_away=td["losses_away"],
                goals_scored=td["goals_scored_home"] + td["goals_scored_away"],
                goals_scored_home=td["goals_scored_home"],
                goals_scored_away=td["goals_scored_away"],
                goals_conceded=td["goals_conceded_home"] + td["goals_conceded_away"],
                goals_conceded_home=td["goals_conceded_home"],
                goals_conceded_away=td["goals_conceded_away"],
            )
            self._stats_repo.save(stats)

        logger.info(
            "Computed stats for %d teams using %d matches (%s/%s)",
            len(team_data),
            len(matches),
            league_id,
            season,
        )
        return {
            "status": "ok",
            "teams_computed": len(team_data),
            "matches_used": len(matches),
        }
