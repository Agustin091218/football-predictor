"""Use case: fetch matches from external API and persist them locally."""

from __future__ import annotations

import logging
from typing import Any

from football_predictor.domain.entities import CompetitionType, League
from football_predictor.domain.repositories import FootballDataSource, MatchRepository

logger = logging.getLogger(__name__)


class FetchAndStoreUseCase:
    """Downloads matches from a football data provider and stores them.

    Orchestrates the data flow: external API → domain entities → persistence.
    """

    def __init__(
        self,
        data_source: FootballDataSource,
        match_repo: MatchRepository,
    ) -> None:
        self._data_source = data_source
        self._match_repo = match_repo

    def execute(self, league_id: str, season: str) -> dict[str, Any]:
        """Fetch matches for a league-season and save them to the repository.

        Args:
            league_id: Competition code (e.g. "PL", "PD").
            season: Season start year as string (e.g. "2024").

        Returns:
            Dictionary with league_id, season, matches_fetched, and status.
        """
        try:
            league = League(
                id=league_id,
                name="",
                country="",
                competition_type=CompetitionType.LEAGUE,
                season=season,
            )
            matches = self._data_source.fetch_matches(league, season=season)
            self._match_repo.save_many(matches)

            logger.info(
                "Fetched and stored %d matches for %s/%s",
                len(matches),
                league_id,
                season,
            )
            return {
                "league_id": league_id,
                "season": season,
                "matches_fetched": len(matches),
                "status": "ok",
            }
        except Exception as exc:
            logger.error(
                "Failed to fetch/store matches for %s/%s: %s",
                league_id,
                season,
                exc,
            )
            return {"status": "error", "error": str(exc)}
