"""Use case: generate a Poisson-based match outcome prediction."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from typing import Any

from football_predictor.domain.entities import (
    Match,
    Prediction,
)
from football_predictor.domain.repositories import (
    MatchRepository,
    PredictionRepository,
    TeamStatsRepository,
)
from football_predictor.domain.services import (
    MonteCarloSimulator,
    PoissonModelParams,
    PoissonService,
    TeamStrengths,
)
from football_predictor.infrastructure.ml.ensemble import EnsemblePredictor
from football_predictor.infrastructure.ml.llm_orchestrator import LLMOrchestrator
from football_predictor.infrastructure.ml.signal_nodes import SignalNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class MatchNotFoundError(Exception):
    """Raised when a match ID cannot be found in the repository."""


class InsufficientDataError(Exception):
    """Raised when there is not enough data to generate a prediction."""


# ---------------------------------------------------------------------------
# Use case
# ---------------------------------------------------------------------------


class PredictMatchUseCase:
    """Generates a match outcome prediction using the bivariate Poisson model.

    Orchestrates:
      1. Lookup match and team statistics.
      2. Calibrate Poisson parameters from league history.
      3. Compute team strengths relative to league average.
      4. Run the Poisson model and build a Prediction entity.
    """

    def __init__(
        self,
        match_repo: MatchRepository,
        stats_repo: TeamStatsRepository,
        poisson_service: PoissonService,
        prediction_repo: PredictionRepository,
        signal_nodes: list[SignalNode] | None = None,
        monte_carlo: MonteCarloSimulator | None = None,
        llm_orchestrator: LLMOrchestrator | None = None,
        ensemble: EnsemblePredictor | None = None,
        n_simulations: int = 10_000,
    ) -> None:
        self._match_repo = match_repo
        self._stats_repo = stats_repo
        self._poisson = poisson_service
        self._prediction_repo = prediction_repo
        self._signal_nodes = signal_nodes
        self._mc = monte_carlo or MonteCarloSimulator(n_simulations=n_simulations)
        self._llm = llm_orchestrator
        self._ensemble = ensemble

    def execute(self, match_id: str) -> Prediction:
        """Generate and persist a prediction for the given match.

        Args:
            match_id: Match identifier (string, converted to int internally).

        Returns:
            The generated Prediction entity.

        Raises:
            MatchNotFoundError: If the match does not exist.
            InsufficientDataError: If league parameters cannot be calibrated.
        """
        # 1. Look up the match
        mid = int(match_id)
        match: Match | None = self._match_repo.get_by_id(mid)
        if match is None:
            raise MatchNotFoundError(f"Partido {match_id} no encontrado")

        league_id = str(match.league.id)
        season = match.league.season or ""

        # 2. Load team statistics
        home_stats = self._stats_repo.get_stats(  # type: ignore[attr-defined]
            match.home_team.id, league_id, season
        )
        away_stats = self._stats_repo.get_stats(  # type: ignore[attr-defined]
            match.away_team.id, league_id, season
        )

        # 3. Handle missing stats with ELO fallback or defaults
        home_default = away_default = False
        if home_stats is None:
            logger.warning(
                "No stats for home team %s (%s/%s), checking ELO",
                match.home_team.name,
                league_id,
                season,
            )
            home_default = True
        if away_stats is None:
            logger.warning(
                "No stats for away team %s (%s/%s), checking ELO",
                match.away_team.name,
                league_id,
                season,
            )
            away_default = True

        elo_ratings = self._load_elo_ratings()
        home_elo = elo_ratings.get(match.home_team.name, 1500.0)
        away_elo = elo_ratings.get(match.away_team.name, 1500.0)
        if home_elo != 1500.0 or away_elo != 1500.0:
            logger.info(
                "ELO ratings: %s=%.0f, %s=%.0f",
                match.home_team.name,
                home_elo,
                match.away_team.name,
                away_elo,
            )

        # 4. Calibrate PoissonModelParams from league history
        if not home_default and not away_default:
            league_matches = self._match_repo.get_finished_matches(  # type: ignore[attr-defined]
                league_id, season
            )
            home_goals_list: list[int] = []
            away_goals_list: list[int] = []
            for m in league_matches:
                if m.score is not None and m.score.is_complete:
                    home_goals_list.append(m.score.home_goals)  # type: ignore[arg-type]
                    away_goals_list.append(m.score.away_goals)  # type: ignore[arg-type]

            params = self._poisson.calculate_league_params_from_matches(
                home_goals_list, away_goals_list, league_id, season
            )
        else:
            params = PoissonModelParams(league_id=league_id, season=season)

        # 4b. If no stats, query Gemini for real-time match data
        _gemini_matches = None
        if home_default and away_default and self._llm is not None:
            gemini_form = self._query_gemini_for_form(match)
            if gemini_form:
                params.avg_goals_home = gemini_form.get("avg_goals_home", params.avg_goals_home)
                params.avg_goals_away = gemini_form.get("avg_goals_away", params.avg_goals_away)
                if gemini_form.get("home_matches") or gemini_form.get("away_matches"):
                    _gemini_matches = {
                        "home": gemini_form.get("home_matches", []),
                        "away": gemini_form.get("away_matches", []),
                    }
            logger.info(
                "Gemini form data: home_goals=%.2f away_goals=%.2f",
                params.avg_goals_home,
                params.avg_goals_away,
            )

        # 5. Compute team strengths from stats (or ELO, or defaults)
        if home_default:
            home_strength = self._strength_from_elo(home_elo, is_home=True)
        else:
            home_strength = self._poisson.calculate_team_strengths_from_stats(
                home_stats,
                params,  # type: ignore[arg-type]
            )

        if away_default:
            away_strength = self._strength_from_elo(away_elo, is_home=False)
        else:
            away_strength = self._poisson.calculate_team_strengths_from_stats(
                away_stats,
                params,  # type: ignore[arg-type]
            )

        # 6. Run the Poisson model
        result = self._poisson.predict(home_strength, away_strength, params)

        if home_default and away_default and (home_elo != 1500.0 or away_elo != 1500.0):
            # Both stats missing but ELO available → compute lambdas directly from ELO
            from football_predictor.domain.services import EloStrengthCalculator

            elo_calc = EloStrengthCalculator(home_advantage=1.1, elo_scale=400.0)
            avg_h = 1.25 if home_elo > 1800 else 1.5
            avg_a = 0.7 if home_elo > 1800 else 1.1
            lambda_h, lambda_a = elo_calc.calculate_lambdas(home_elo, away_elo, avg_h, avg_a)
            prob_h, prob_d, prob_a = self._poisson.probabilities_from_lambdas(lambda_h, lambda_a)
            result["prob_home_win"] = prob_h
            result["prob_draw"] = prob_d
            result["prob_away_win"] = prob_a
            result["lambda_home"] = lambda_h
            result["lambda_away"] = lambda_a
            self._ensemble = None  # force legacy path for this prediction

        # 7. Run signal nodes
        signal_context: dict[str, Any] = {
            "finished_matches": league_matches if not home_default and not away_default else [],
            "home_stats": home_stats,
            "away_stats": away_stats,
            "elo_ratings": {
                match.home_team.id: home_elo,
                match.away_team.id: away_elo,
            }
            if home_elo != 1500.0 or away_elo != 1500.0
            else None,
        }

        if self._signal_nodes:
            signal_outputs: dict[str, Any] = {}
            for node in self._signal_nodes:
                signal_context["signal_outputs_so_far"] = signal_outputs.copy()
                try:
                    output = node.compute(match, signal_context)
                    signal_outputs[node.name] = output
                    logger.info(
                        "Node %s: confidence=%.2f, summary=%s",
                        node.name,
                        output.get("confidence", 0),
                        output.get("summary", "?"),
                    )
                except Exception as exc:
                    logger.warning("Node %s failed: %s", node.name, exc, exc_info=True)
        else:
            from football_predictor.infrastructure.ml.signal_nodes import (
                ContextNode,
                EloNode,
                FormNode,
                HeadToHeadNode,
                PoissonSignalNode,
            )

            default_nodes = {
                "form": FormNode(),
                "elo": EloNode(),
                "h2h": HeadToHeadNode(),
                "context": ContextNode(),
                "poisson": PoissonSignalNode(self._poisson),
            }
            signal_outputs = {}
            for name, node in default_nodes.items():
                signal_context["signal_outputs_so_far"] = signal_outputs.copy()
                try:
                    output = node.compute(match, signal_context)
                    signal_outputs[name] = output
                    logger.info(
                        "Node %s: confidence=%.2f, summary=%s",
                        name,
                        output.get("confidence", 0),
                        output.get("summary", "?"),
                    )
                except Exception as exc:
                    logger.warning("Node %s failed: %s", name, exc, exc_info=True)

        # 8. LLM orchestration (if available)
        llm_output: dict[str, Any] | None = None
        if self._llm:
            try:
                lambda_h = (
                    signal_outputs.get("poisson", {}).get("value", {}).get("lambda_home", 1.5)
                )
                lambda_a = (
                    signal_outputs.get("poisson", {}).get("value", {}).get("lambda_away", 1.1)
                )
                mc_preview = self._mc.simulate(lambda_h, lambda_a)
                llm_output = self._llm.orchestrate(match, signal_outputs, mc_preview)
            except Exception as exc:
                logger.warning("LLM orchestration skipped: %s", exc)

        # 9. Ensemble or legacy prediction path
        if self._ensemble is not None:
            finished_ctx = signal_context["finished_matches"]
            ensemble_result = self._ensemble.predict(
                match=match,
                signal_outputs=signal_outputs,
                home_stats=home_stats,
                away_stats=away_stats,
                finished_matches=finished_ctx,
                llm_adjusted_weights=llm_output.get("adjusted_weights") if llm_output else None,
            )

            confidence = ensemble_result.confidence
            if llm_output:
                assessment = llm_output.get("confidence_assessment", "medium")
                if assessment == "high":
                    confidence = min(1.0, confidence * 1.1)
                elif assessment == "low":
                    confidence = confidence * 0.85
                signal_outputs["_llm_analysis"] = llm_output

            if _gemini_matches:
                signal_outputs["_gemini_matches"] = _gemini_matches

            model_version = f"{ensemble_result.model_used}-v1"

            prediction = Prediction(
                match=match,
                prob_home_win=ensemble_result.prob_home_win,
                prob_draw=ensemble_result.prob_draw,
                prob_away_win=ensemble_result.prob_away_win,
                expected_goals_home=ensemble_result.lambda_home,
                expected_goals_away=ensemble_result.lambda_away,
                simulation=ensemble_result.monte_carlo,
                signal_outputs=signal_outputs,
                llm_explanation=llm_output.get("explanation") if llm_output else None,
                llm_actions=llm_output.get("actions") if llm_output else None,
                extended_predictions=llm_output.get("extended_predictions") if llm_output else None,
                confidence=confidence,
                model_version=model_version,
            )
        else:
            monte_carlo = self._mc.simulate(result["lambda_home"], result["lambda_away"])
            if llm_output:
                signal_outputs["_llm_analysis"] = llm_output
            if _gemini_matches:
                signal_outputs["_gemini_matches"] = _gemini_matches
            prediction = Prediction(
                match=match,
                prob_home_win=result["prob_home_win"],
                prob_draw=result["prob_draw"],
                prob_away_win=result["prob_away_win"],
                expected_goals_home=result["lambda_home"],
                expected_goals_away=result["lambda_away"],
                goal_probabilities=result["goal_probabilities"],
                simulation=monte_carlo,
                signal_outputs=signal_outputs,
                llm_explanation=llm_output.get("explanation") if llm_output else None,
                llm_actions=llm_output.get("actions") if llm_output else None,
                extended_predictions=llm_output.get("extended_predictions") if llm_output else None,
                confidence=self._calculate_confidence(
                    result["prob_home_win"],
                    result["prob_draw"],
                    result["prob_away_win"],
                ),
                model_version="poisson-v1",
            )

        # 10. Persist
        self._prediction_repo.save(prediction)

        return prediction

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(ph: float, pd: float, pa: float) -> float:
        """Normalized Shannon entropy as a confidence metric.

        High confidence → probabilities concentrated on one outcome.
        Low confidence  → probabilities spread evenly (toss-up match).

        Args:
            ph: Probability of home win.
            pd: Probability of draw.
            pa: Probability of away win.

        Returns:
            Confidence score in [0, 1], rounded to 4 decimal places.
        """
        entropy = -sum(p * math.log(p) for p in (ph, pd, pa) if p > 0)
        max_entropy = math.log(3)
        return round(1.0 - (entropy / max_entropy), 4)

    @staticmethod
    def _load_elo_ratings() -> dict[str, float]:
        elo_path = os.getenv("ELO_RATINGS_PATH", "archive/eloratings.csv")
        try:
            with open(elo_path) as f:
                reader = csv.DictReader(f)
                ratings: dict[str, float] = {}
                for row in reader:
                    try:
                        r = float(row.get("rating") or 0)
                        if r > 0:
                            ratings[row["team"]] = r
                    except (ValueError, KeyError):
                        continue
                return ratings
        except FileNotFoundError:
            return {}

    @staticmethod
    def _strength_from_elo(elo: float, is_home: bool) -> TeamStrengths:
        factor = 1.0 + (elo - 1500.0) / 400.0
        home_bonus = 0.3 if is_home else 0.0
        return TeamStrengths(
            team_id=0,
            attack_home=min(3.0, max(0.3, factor + home_bonus)),
            attack_away=min(3.0, max(0.3, factor * 0.75)),
            defense_home=min(2.0, max(0.3, 1.0 / factor)),
            defense_away=min(2.0, max(0.3, 1.0 / factor)),
        )

    def _query_gemini_for_form(self, match: Match) -> dict[str, float] | None:
        try:
            from google import genai as google_genai
            from google.genai import types

            client = google_genai.Client(
                api_key=self._llm.gemini_api_key
                if hasattr(self._llm, "gemini_api_key")
                else os.getenv("GEMINI_API_KEY", "")
            )
            if not hasattr(client, "_api_key") and not os.getenv("GEMINI_API_KEY"):
                return None

            prompt = f"""Search the internet for the most recent 10 matches of {match.home_team.name} and {match.away_team.name} national football teams.
For each team, find: opponent, score, date, competition.
Then compute:
- Average goals scored per match by {match.home_team.name} in last 10 matches
- Average goals conceded per match by {match.home_team.name} in last 10 matches
- Average goals scored per match by {match.away_team.name} in last 10 matches
- Average goals conceded per match by {match.away_team.name} in last 10 matches

Also include the full list of matches found.

Return ONLY this JSON (no markdown):
{{"home_avg_scored": 2.0, "home_avg_conceded": 0.6, "away_avg_scored": 1.8, "away_avg_conceded": 1.2, "home_matches": [{{"opponent": "Italy", "score": "2-0", "date": "2025-11-15", "competition": "Nations League"}}], "away_matches": [{{"opponent": "Serbia", "score": "1-1", "date": "2025-11-15", "competition": "Nations League"}}]}}"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                ),
            )
            text = response.text.strip()
            if "```" in text:
                text = re.sub(r"```\w*\n?", "", text).replace("```", "").strip()

            data = json.loads(text)
            home_scored = float(data.get("home_avg_scored", 0))
            away_scored = float(data.get("away_avg_scored", 0))
            home_matches = data.get("home_matches", [])
            away_matches = data.get("away_matches", [])

            result: dict = {}
            if home_scored > 0 and away_scored > 0:
                result["avg_goals_home"] = max(1.5, home_scored)
                result["avg_goals_away"] = max(0.6, away_scored * 0.7)
            if home_matches or away_matches:
                result["home_matches"] = home_matches[:10]
                result["away_matches"] = away_matches[:10]
            return result if result else None
        except Exception as exc:
            logger.warning("Gemini form query failed: %s", exc)
            return None
