"""HTTP client for the football-data.org v4 REST API.

Implements the FootballDataSource abstract interface using httpx.
Handles rate-limiting, error recovery, and JSON→domain mapping.

Token gratuito: https://www.football-data.org/client/register
Límite: 10 requests/minuto en plan gratuito
Ligas disponibles gratis: PL, PD, SA, BL1, FL1, CL, DED, PPL, BSA
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from football_predictor.domain.entities import (
    CompetitionType,
    League,
    Match,
    MatchScore,
    MatchStatus,
    Team,
    TeamStats,
)
from football_predictor.domain.repositories import FootballDataSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.football-data.org/v4"
DATE_FORMAT = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class FootballDataOrgClient(FootballDataSource):
    """Concrete adapter for the football-data.org REST API.

    Wraps httpx.Client with auth headers and timeout.
    Maps JSON responses to domain entities.
    """

    def __init__(self, token: str, timeout: int = 30) -> None:
        self._token = token
        self._timeout = timeout
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"X-Auth-Token": token},
            timeout=timeout,
        )
        logger.info("FootballDataOrgClient initialized (timeout=%ds)", timeout)

    # ------------------------------------------------------------------
    # FootballDataSource abstract methods
    # ------------------------------------------------------------------

    def fetch_league_standings(self, league: League, season: str | None = None) -> list[TeamStats]:
        """Fetch league standings and map to TeamStats entities.

        GET /competitions/{id}/standings?season={season}

        Extracts the TOTAL standings table and maps each row to a
        minimal TeamStats object with win/draw/loss/goal data.
        """
        params = {}
        if season is not None:
            params["season"] = season

        data = self._get(f"/competitions/{league.id}/standings", params=params)

        # Build a lightweight League for the TeamStats context
        league_ref = League(
            id=league.id,
            name=league.name,
            country=league.country,
            competition_type=CompetitionType.LEAGUE,
            season=season,
        )

        results: list[TeamStats] = []
        for standing_group in data.get("standings", []):
            if standing_group.get("type") != "TOTAL":
                continue
            for row in standing_group.get("table", []):
                team_data = row["team"]
                team = Team(
                    id=team_data["id"],
                    name=team_data.get("name", "Unknown"),
                    short_name=team_data.get("shortName", team_data.get("name", "")),
                    tla=team_data.get("tla", "???"),
                    country=team_data.get("area", {}).get("name", "Unknown"),
                )
                stats = TeamStats(
                    team=team,
                    league=league_ref,
                    season=season or "",
                    matches_played=row.get("playedGames", 0),
                    wins=row.get("won", 0),
                    draws=row.get("draw", 0),
                    losses=row.get("lost", 0),
                    goals_scored=row.get("goalsFor", 0),
                    goals_conceded=row.get("goalsAgainst", 0),
                )
                results.append(stats)
        return results

    def fetch_matches(
        self,
        league: League,
        season: str | None = None,
        matchday: int | None = None,
    ) -> list[Match]:
        """Fetch matches for a league-season from the API.

        GET /competitions/{id}/matches?season={season}

        Args:
            league: The league to query (uses league.id as the competition code).
            season: Optional season filter (e.g. "2025").
            matchday: Optional matchday filter.

        Returns:
            List of Match entities.
        """
        params: dict = {}
        if season is not None:
            params["season"] = season
        if matchday is not None:
            params["matchday"] = matchday

        data = self._get(f"/competitions/{league.id}/matches", params=params)
        league_id_str = str(league.id)
        return [
            self._parse_match(item, league_id_str, season or "") for item in data.get("matches", [])
        ]

    def fetch_teams(self, league: League, season: str | None = None) -> list[Team]:
        """Fetch all teams participating in a league.

        GET /competitions/{id}/teams?season={season}
        """
        params = {}
        if season is not None:
            params["season"] = season

        data = self._get(f"/competitions/{league.id}/teams", params=params)
        return [self._parse_team(item) for item in data.get("teams", [])]

    # ------------------------------------------------------------------
    # Additional convenience methods
    # ------------------------------------------------------------------

    def fetch_leagues(self) -> list[League]:
        """Fetch all available competitions from the API.

        GET /competitions

        Maps each item to a League domain entity.
        Only includes competitions with a non-null ``code`` field.

        Returns:
            List of League entities.
        """
        data = self._get("/competitions")
        leagues: list[League] = []
        for item in data.get("competitions", []):
            code = item.get("code")
            if code is None:
                continue
            comp_type = item.get("type", "").upper()
            if comp_type == "LEAGUE":
                competition_type = CompetitionType.LEAGUE
            elif comp_type == "CUP":
                competition_type = CompetitionType.CUP
            else:
                competition_type = CompetitionType.INTERNATIONAL

            league = League(
                id=code,
                name=item.get("name", "Unknown"),
                country=item.get("area", {}).get("name", "Unknown"),
                competition_type=competition_type,
            )
            leagues.append(league)
        return leagues

    def fetch_upcoming_matches(self, league_id: str, days_ahead: int = 7) -> list[Match]:
        """Fetch scheduled matches within a future window.

        GET /competitions/{id}/matches?status=SCHEDULED&dateFrom=X&dateTo=Y

        Args:
            league_id: Competition code (e.g. "PL").
            days_ahead: Number of days to look ahead from today.

        Returns:
            List of Match entities.
        """
        today = datetime.now(timezone.utc)
        date_from = today.strftime(DATE_FORMAT)
        date_to = (today + timedelta(days=days_ahead)).strftime(DATE_FORMAT)

        params = {
            "status": "SCHEDULED",
            "dateFrom": date_from,
            "dateTo": date_to,
        }
        data = self._get(f"/competitions/{league_id}/matches", params=params)
        return [self._parse_match(item, league_id, "") for item in data.get("matches", [])]

    # ------------------------------------------------------------------
    # Private: HTTP + error handling
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Perform a GET request with error handling.

        Args:
            path: API path (e.g. "/competitions/PL/matches").
            params: Optional query parameters.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            httpx.HTTPStatusError: On non-404, non-429 errors.
            RuntimeError: On rate limit (429).
        """
        try:
            response = self._client.get(path, params=params)
        except httpx.RequestError as exc:
            logger.error("Request failed for %s: %s", path, exc)
            raise

        if response.status_code == 429:
            logger.warning("Rate limit alcanzado, esperando... (%s)", path)
            raise RuntimeError(
                f"Rate limit exceeded on {path}. Free tier allows 10 requests/minute."
            )

        if response.status_code == 404:
            logger.warning("Recurso no encontrado: %s (status 404)", path)
            return {}

        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Private: JSON → domain mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_team(item: dict) -> Team:
        """Map a team JSON object to a Team domain entity.

        Args:
            item: Team JSON from the API response.

        Returns:
            Team entity.
        """
        return Team(
            id=item["id"],
            name=item.get("name", "Unknown"),
            short_name=item.get("shortName", item.get("name", "Unknown")),
            tla=item.get("tla", item.get("name", "???")[:3].upper()),
            country=item.get("area", {}).get("name", "Unknown"),
        )

    @staticmethod
    def _parse_match(item: dict, league_id: str, season: str) -> Match:
        """Map a match JSON object to a Match domain entity.

        Args:
            item: Match JSON from the API response.
            league_id: Competition code (e.g. "PL").
            season: Season string provided in the request.

        Returns:
            Fully constructed Match entity.
        """
        home_data = item["homeTeam"]
        away_data = item["awayTeam"]

        home_team = Team(
            id=home_data["id"],
            name=home_data.get("name", "Unknown"),
            short_name=home_data.get("shortName", home_data.get("name", "Unknown")),
            tla=home_data.get("tla", "???"),
            country="Unknown",
        )
        away_team = Team(
            id=away_data["id"],
            name=away_data.get("name", "Unknown"),
            short_name=away_data.get("shortName", away_data.get("name", "Unknown")),
            tla=away_data.get("tla", "???"),
            country="Unknown",
        )

        league = League(
            id=league_id,
            name="",
            country="",
            competition_type=CompetitionType.LEAGUE,
            season=season,
        )

        # Parse UTC date (handles both "Z" and "+00:00" suffixes)
        utc_str = item["utcDate"].replace("Z", "+00:00")
        match_date = datetime.fromisoformat(utc_str)

        status = FootballDataOrgClient._parse_status(item["status"])

        # Build score from fullTime data
        score: MatchScore | None = None
        full_time = item.get("score", {}).get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")
        if home_goals is not None or away_goals is not None:
            score = MatchScore(home_goals=home_goals, away_goals=away_goals)

        return Match(
            id=item["id"],
            home_team=home_team,
            away_team=away_team,
            league=league,
            match_date=match_date,
            status=status,
            score=score,
            matchday=item.get("matchday"),
            venue=item.get("venue"),
        )

    @staticmethod
    def _parse_status(status_str: str) -> MatchStatus:
        """Map API status string to MatchStatus enum.

        Mapping:
            SCHEDULED, TIMED → SCHEDULED
            IN_PLAY, PAUSED  → LIVE
            FINISHED         → FINISHED
            POSTPONED        → POSTPONED
            Any other        → SCHEDULED (safe default)

        Args:
            status_str: Status string from the API.

        Returns:
            Corresponding MatchStatus enum value.
        """
        upper = status_str.upper()
        if upper in ("SCHEDULED", "TIMED"):
            return MatchStatus.SCHEDULED
        if upper in ("IN_PLAY", "PAUSED"):
            return MatchStatus.LIVE
        if upper == "FINISHED":
            return MatchStatus.FINISHED
        if upper == "POSTPONED":
            return MatchStatus.POSTPONED
        if upper == "CANCELLED":
            return MatchStatus.CANCELLED
        return MatchStatus.SCHEDULED
