"""LLM orchestrator for the football predictor signal graph.

Uses Google Gemini API to analyse match signals and produce:
- Explanations of model predictions
- Adjusted signal weights for continuous learning
- Action recommendations for model improvement
"""

from __future__ import annotations

import json
import logging
from typing import Any

from football_predictor.domain.entities import Match
from football_predictor.domain.services import MonteCarloResult

logger = logging.getLogger(__name__)

# Default signal weights (used when LLM fails or returns neutral)
_DEFAULT_WEIGHTS = {
    "form": 1.0,
    "elo": 0.8,
    "h2h": 0.6,
    "context": 0.5,
    "poisson": 1.2,
}

_MAX_FEEDBACK_ITEMS = 50


class LLMOrchestrator:
    """Orchestrates the signal graph via Gemini LLM analysis.

    Receives outputs from all SignalNodes plus Monte Carlo simulation
    results, sends them to Gemini for analysis, and parses the JSON
    response into structured actions and adjusted weights.

    Maintains a feedback history so the LLM can learn from past mistakes
    and adjust its reasoning over time.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._logger = logging.getLogger(__name__)
        self._feedback_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def orchestrate(
        self,
        match: Match,
        signal_outputs: dict[str, Any],
        monte_carlo: MonteCarloResult,
        model_version: str = "ensemble-v1",
    ) -> dict[str, Any]:
        """Call Gemini with all match data and return structured analysis.

        Uses Google Search grounding for real-time data when available."""
        prompt = self._build_prompt(match, signal_outputs, monte_carlo)

        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                ),
            )
            parsed = self._parse_response(response.text)
            self._logger.info("LLM orchestration OK for match %s", match.id)
            return parsed
        except Exception as exc:
            self._logger.error("LLM orchestration failed for match %s: %s", match.id, exc)
            return self._neutral_response()

    # ------------------------------------------------------------------
    # Learning from results
    # ------------------------------------------------------------------

    def learn_from_result(
        self,
        match_id: str,
        signal_outputs: dict[str, Any],
        predicted_result: str,
        actual_result: str,
    ) -> None:
        """Record the actual match result for future LLM context.

        Identifies which signal was most contradicted by the actual outcome
        and stores it for the next orchestration cycle.

        Args:
            match_id: Match identifier.
            signal_outputs: Original signal outputs from the prediction.
            predicted_result: The result the model predicted (e.g. "1", "X", "2").
            actual_result: What actually happened ("1", "X", "2").
        """
        error_signal = self._find_most_wrong_signal(signal_outputs, actual_result)

        self._feedback_history.append(
            {
                "match_id": match_id,
                "predicted": predicted_result,
                "actual": actual_result,
                "error_signal": error_signal,
                "signal_summaries": {k: v.get("summary", "") for k, v in signal_outputs.items()},
            }
        )

        if len(self._feedback_history) > _MAX_FEEDBACK_ITEMS:
            self._feedback_history = self._feedback_history[-_MAX_FEEDBACK_ITEMS:]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        match: Match,
        signal_outputs: dict[str, Any],
        monte_carlo: MonteCarloResult,
    ) -> str:
        """Build the complete Gemini prompt with match context and signals."""
        top_3 = monte_carlo.top_scores[:3]
        top_scores_str = ", ".join(f"{s['score']} ({s['probability']:.1%})" for s in top_3)

        return f"""Sos un analista experto en fútbol y estadística deportiva.
Hacé un análisis detallado de este partido usando los datos del modelo.

PARTIDO: {match.home_team.name} vs {match.away_team.name}
Liga: {match.league.name or match.league.id} | Fecha: {match.match_date.strftime("%d/%m/%Y %H:%M")}

SIMULACIÓN MONTE CARLO — ESTOS SON LOS DATOS OFICIALES DEL MODELO ({monte_carlo.n_simulations:,} simulaciones):
⚠️ NO CONTRADIGAS ESTOS NÚMEROS. Son la fuente de verdad. Tu análisis debe explicarlos, no refutarlos.
- Victoria local: {monte_carlo.prob_home_win:.1%}
- Empate: {monte_carlo.prob_draw:.1%}
- Victoria visitante: {monte_carlo.prob_away_win:.1%}
- IC 95% victoria local: [{monte_carlo.ci_home_win_low:.1%}, {monte_carlo.ci_home_win_high:.1%}]
- Marcador más probable: {monte_carlo.most_likely_score}
- Top 3 marcadores: {top_scores_str}
- Over 2.5: {monte_carlo.prob_over_2_5:.1%}
- Ambos marcan: {monte_carlo.prob_btts:.1%}
- Mediana goles: {monte_carlo.home_goals_p50:.1f} - {monte_carlo.away_goals_p50:.1f}
- Clean sheet local: {monte_carlo.prob_clean_sheet_home:.1%}
- Clean sheet visitante: {monte_carlo.prob_clean_sheet_away:.1%}

⚠️ REGLA OBLIGATORIA: Si el partido está parejo (diferencia <15% entre local y visitante), tu explanation DEBE decir "partido parejo" o "ligera ventaja". NUNCA digas "clara victoria" ni inventes porcentajes distintos a los de arriba. Si la confianza es baja (<25%), mencioná "alta incertidumbre" y no fuerces un favorito.

SEÑALES DEL GRAFO (datos complementarios — si están SIN DATOS, ignorá y basate 100% en Monte Carlo):
{self._format_signals(signal_outputs)}

HISTORIAL DE APRENDIZAJE RECIENTE:
{self._format_feedback_history()}

INSTRUCCIÓN IMPORTANTE:
Las señales con confianza 0% o "SIN DATOS" no tienen información real. Si la mayoría de las señales están sin datos, tu ÚNICA fuente es Monte Carlo. No uses tu "conocimiento general" de qué equipo es mejor — solo los números.
Nunca digas "ELO idéntico" o "forma 0%" si son SIN DATOS — simplemente no las menciones.

⚠️ USÁ LOS NÚMEROS EXACTOS del modelo, NO los inventes:
- Las probabilidades 1X2 y xG vienen de la SIMULACIÓN MONTE CARLO de arriba. Usá esos porcentajes exactos.
- El marcador más probable y el clean sheet % también vienen de Monte Carlo. No los cambies.
- Si el partido está parejo (diferencia <15% entre 1 y 2), decí claramente "partido parejo" o "ligera ventaja", NO "clara victoria".
- Si la confianza es baja (<25%), mencioná que "el modelo muestra alta incertidumbre".
- NUNCA digas "clara ventaja para X" si los números de Monte Carlo muestran un partido parejo. ESO ES ALUCINACIÓN.
- NUNCA digas porcentajes distintos a los que aparecen en SIMULACIÓN MONTE CARLO.

Respondé ÚNICAMENTE en este JSON (sin markdown):
{{
  "explanation": "Análisis detallado de 4-6 oraciones explicando POR QUÉ el modelo predice este resultado. Mencioná el diferencial de ELO, los goles esperados, la ventaja de localía, y qué factores inclinan la balanza. Si el modelo predice victoria clara, explicá por qué. Si está parejo, explicá la incertidumbre.",
  "alternative_scenarios": ["Escenario 1: si el equipo visitante marca primero, el partido podría...", "Escenario 2: si hay expulsión temprana..."],
  "goal_timing": "Análisis de cuándo se esperan los goles: ¿primer o segundo tiempo? ¿gol tempranero o partido cerrado? Basado en los percentiles de goles y la distribución.",
  "match_flow": "Cómo se espera que se desarrolle el partido: ¿dominio claro de un equipo? ¿partido trabado? ¿ida y vuelta? Basado en xG, posesión implícita por ELO, y clean sheet probability.",
  "key_matchups": "Qué duelos individuales o sectoriales pueden definir el partido según las fortalezas/debilidades relativas de cada equipo.",
  "discipline": "Probabilidad de tarjetas basada en el contexto: ¿partido tenso con mucho en juego? ¿diferencia grande que lleva a frustración? ¿rivalidad histórica?",
  "key_factors": ["factor 1", "factor 2", "factor 3", "factor 4"],
  "confidence_assessment": "high|medium|low",
  "adjusted_weights": {{
    "form": 1.0, "elo": 0.8, "h2h": 0.6, "context": 0.5, "poisson": 1.2
  }},
  "actions": ["acción_1"],
  "risk_flags": ["flag si hay algo inusual, o lista vacía"]
}}

Los adjusted_weights entre 0.0 y 3.0.

PREDICCIONES ADICIONALES REQUERIDAS:
Incluí también predicciones numéricas para estas dimensiones:

  "extended_predictions": {{
    "total_goals": {{
      "expected": 3.2,
      "over_2_5": true,
      "most_likely_range": "2-3"
    }},
    "cards": {{
      "total_cards_expected": 3.5,
      "red_card_prob": 0.12,
      "high_intensity": false
    }},
    "game_situations": {{
      "early_goal_prob": 0.35,
      "comeback_possible": false,
      "penalty_prob": 0.08,
      "dominance": "home",
      "key_factor": "superioridad técnica"
    }},
    "confidence_by_dimension": {{
      "result": 0.85,
      "goals": 0.70,
      "cards": 0.40,
      "situations": 0.55
    }}
  }}

Basate en ELO (diferencia grande → dominancia), xG calculados (mejor estimación de goles), H2H (partidos físicos → más tarjetas), fase del torneo (eliminatorias → más tensión). Si no tenés datos específicos, usá promedios FIFA: Mundial ~2.6 goles/partido, ~4.2 tarjetas/partido."""

    @staticmethod
    def _format_signals(signal_outputs: dict[str, Any]) -> str:
        """Format signal outputs as human-readable text for the LLM."""
        if not signal_outputs:
            return "No hay señales disponibles"

        lines: list[str] = []
        for name, signal in signal_outputs.items():
            weight = signal.get("weight", 1.0)
            confidence = signal.get("confidence", 0.0)
            summary = signal.get("summary", "sin resumen")
            if confidence < 0.01:
                lines.append(
                    f"{name.upper()}: SIN DATOS (confianza 0%) — ignorar esta señal. "
                    f"Usar valores por defecto o descartar."
                )
            else:
                lines.append(
                    f"{name.upper()}: {summary} (peso: {weight:.1f}, confianza: {confidence:.0%})"
                )
        return "\n".join(lines)

    def _format_feedback_history(self) -> str:
        """Format the last 5 feedback entries for LLM context."""
        recent = self._feedback_history[-5:]
        if not recent:
            return "Sin historial previo"

        lines: list[str] = []
        for entry in recent:
            lines.append(
                f"Partido {entry['match_id']}: predije {entry['predicted']}, "
                f"resultó {entry['actual']}. "
                f"Señal que más erró: {entry['error_signal']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        """Parse the LLM JSON response with validation and fallbacks.

        Strips markdown code fences, validates required keys,
        and clamps adjusted weights to [0.0, 3.0].
        """
        cleaned = text.strip()

        # Remove possible ```json ... ``` wrappers
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM response as JSON: %s", exc)
            return LLMOrchestrator._neutral_response()

        if not isinstance(data, dict):
            return LLMOrchestrator._neutral_response()

        # Ensure required keys exist
        data.setdefault("explanation", "Análisis no disponible.")
        data.setdefault("alternative_scenarios", [])
        data.setdefault("goal_timing", "")
        data.setdefault("match_flow", "")
        data.setdefault("key_matchups", "")
        data.setdefault("discipline", "")
        data.setdefault("key_factors", [])
        data.setdefault("confidence_assessment", "medium")
        data.setdefault("actions", [])
        data.setdefault("risk_flags", [])
        data.setdefault("extended_predictions", None)

        # Validate and clamp adjusted_weights
        weights = data.get("adjusted_weights", {})
        if not isinstance(weights, dict):
            weights = {}
        clamped: dict[str, float] = {}
        for name in _DEFAULT_WEIGHTS:
            w = weights.get(name, _DEFAULT_WEIGHTS[name])
            if isinstance(w, (int, float)):
                clamped[name] = max(0.0, min(float(w), 3.0))
            else:
                clamped[name] = _DEFAULT_WEIGHTS[name]
        data["adjusted_weights"] = clamped

        return data

    @staticmethod
    def _neutral_response() -> dict[str, Any]:
        """Safe fallback when the LLM fails or returns unparseable output."""
        return {
            "explanation": "Análisis basado en modelo estadístico.",
            "alternative_scenarios": [],
            "goal_timing": "",
            "match_flow": "",
            "key_matchups": "",
            "discipline": "",
            "key_factors": [],
            "confidence_assessment": "medium",
            "adjusted_weights": dict(_DEFAULT_WEIGHTS),
            "actions": [],
            "risk_flags": [],
        }

    # ------------------------------------------------------------------
    # Signal analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _find_most_wrong_signal(
        signal_outputs: dict[str, Any],
        actual_result: str,
    ) -> str:
        """Identify which signal most contradicted the actual result.

        Looks for the signal whose predicted_result (if available) differs
        most from the actual outcome. Falls back to checking Poisson first.

        Args:
            signal_outputs: Dict of node_name → node_output.
            actual_result: "1", "X", or "2".

        Returns:
            Name of the signal that was most wrong, or "unknown".
        """
        most_wrong = "unknown"
        worst_confidence = -1.0

        for name, signal in signal_outputs.items():
            value = signal.get("value", {})
            predicted = value.get("predicted_result")
            if predicted is None:
                continue

            if predicted == actual_result:
                continue

            confidence = signal.get("confidence", 0.0)
            if confidence > worst_confidence:
                worst_confidence = confidence
                most_wrong = name

        return most_wrong
