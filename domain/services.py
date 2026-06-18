"""Domain services — pure business logic for Poisson-based match prediction.

Implements the Dixon & Coles (1997) bivariate Poisson model with
low-scoring correction for football match outcome prediction.

No external imports beyond math and domain entities.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from football_predictor.domain.entities import GoalProbabilities, MatchResult

# ---------------------------------------------------------------------------
# Model parameter & strength value objects
# ---------------------------------------------------------------------------


@dataclass
class PoissonModelParams:
    """Parameters for a Poisson goal model calibrated to a league-season.

    Attributes:
        league_id: Identifier of the league.
        season: Season string (e.g. "2025/2026").
        avg_goals_home: League-average goals scored by the home team per match.
        avg_goals_away: League-average goals scored by the away team per match.
        rho: Dixon-Coles dependence parameter (0.05–0.15). Positive values
            reduce the probability of 0-0 and 1-1 scorelines while increasing
            1-0 and 0-1, matching observed football data.
        n_matches_used: Number of matches used to calibrate these parameters.
    """

    league_id: int
    season: str
    avg_goals_home: float = 1.5
    avg_goals_away: float = 1.1
    rho: float = 0.1
    n_matches_used: int = 0


@dataclass
class TeamStrengths:
    """Attack and defense strength factors for a team.

    All values are multiplicative factors relative to league average (1.0).
    attack > 1.0 → team scores more than average.
    defense < 1.0 → team concedes less than average.

    Attributes:
        team_id: Identifier of the team.
        attack_home: Home attack strength factor.
        attack_away: Away attack strength factor.
        defense_home: Home defense strength factor.
        defense_away: Away defense strength factor.
    """

    team_id: int
    attack_home: float = 1.0
    attack_away: float = 1.0
    defense_home: float = 1.0
    defense_away: float = 1.0


# ---------------------------------------------------------------------------
# Pure mathematical helpers
# ---------------------------------------------------------------------------


def poisson_pmf(k: int, lam: float) -> float:
    """Probability mass function of the Poisson distribution.

    Uses math.lgamma to avoid numerical overflow for moderate k and lam.
    Formula: exp(k * log(lam) - lam - lgamma(k + 1))

    Args:
        k: Number of events (non-negative integer).
        lam: Expected rate (lambda > 0).

    Returns:
        P(X = k) under Poisson(lam).

    Raises:
        ValueError: If lam <= 0 or k < 0.
    """
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def dixon_coles_correction(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    """Dixon & Coles (1997) low-scoring correction factor τ.

    Applied only to scorelines (0,0), (1,0), (0,1), (1,1).
    With positive rho:
      - 0-0 is REDUCED  (factor = 1 − λ_home·λ_away·ρ)
      - 1-0 is INCREASED (factor = 1 + λ_home·ρ)
      - 0-1 is INCREASED (factor = 1 + λ_away·ρ)
      - 1-1 is REDUCED  (factor = 1 − ρ)

    All other scorelines return 1.0 (no correction).

    Args:
        home_goals: Goals scored by the home team.
        away_goals: Goals scored by the away team.
        lambda_home: Expected home goals (Poisson rate).
        lambda_away: Expected away goals (Poisson rate).
        rho: Dependence parameter (0.0 = no correction).

    Returns:
        Multiplicative correction factor.
    """
    if rho == 0.0:
        return 1.0

    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_home * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho

    return 1.0


# ---------------------------------------------------------------------------
# Poisson Service
# ---------------------------------------------------------------------------


class PoissonService:
    """Bivariate Poisson model for football match prediction.

    Implements the Dixon & Coles approach:
    1. Estimate expected goals (lambda) from team strengths and league params.
    2. Build a joint probability matrix for exact scorelines.
    3. Apply the τ correction for low-scoring games.
    4. Derive 1X2 result probabilities and expected goals.

    Attributes:
        max_goals: Upper bound for goals per team in the probability matrix.
    """

    def __init__(self, max_goals: int = 8) -> None:
        if max_goals < 2:
            raise ValueError(f"max_goals must be >= 2, got {max_goals}")
        self.max_goals = max_goals

    def probabilities_from_lambdas(
        self, lambda_home: float, lambda_away: float
    ) -> tuple[float, float, float]:
        home_pmf = [poisson_pmf(i, lambda_home) for i in range(self.max_goals + 1)]
        away_pmf = [poisson_pmf(j, lambda_away) for j in range(self.max_goals + 1)]
        prob_home = prob_draw = prob_away = 0.0
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                p = home_pmf[i] * away_pmf[j]
                if i > j:
                    prob_home += p
                elif i == j:
                    prob_draw += p
                else:
                    prob_away += p
        total = prob_home + prob_draw + prob_away
        if total > 0:
            return (prob_home / total, prob_draw / total, prob_away / total)
        return (0.4, 0.3, 0.3)

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def calculate_lambdas(
        self,
        home_strength: TeamStrengths,
        away_strength: TeamStrengths,
        params: PoissonModelParams,
    ) -> tuple[float, float]:
        """Compute expected home and away goals (lambda values).

        λ_home = attack_home × defense_away × avg_goals_home
        λ_away = attack_away × defense_home × avg_goals_away

        Both values are clamped to [0.1, 10.0] to prevent degenerate cases.

        Args:
            home_strength: Home team attack/defense factors.
            away_strength: Away team attack/defense factors.
            params: League-season calibrated parameters.

        Returns:
            Tuple of (lambda_home, lambda_away).
        """
        lambda_home = home_strength.attack_home * away_strength.defense_away * params.avg_goals_home
        lambda_away = away_strength.attack_away * home_strength.defense_home * params.avg_goals_away
        return (
            max(0.1, min(10.0, lambda_home)),
            max(0.1, min(10.0, lambda_away)),
        )

    def build_score_matrix(
        self,
        lambda_home: float,
        lambda_away: float,
        rho: float = 0.0,
    ) -> GoalProbabilities:
        """Build the joint goal probability matrix P(home=i, away=j).

        For each cell (i, j):
          P(i, j) = Poisson(i|λ_home) × Poisson(j|λ_away) × τ(i, j)

        The matrix is then normalized so all cells sum to 1.0.

        Args:
            lambda_home: Expected home goals.
            lambda_away: Expected away goals.
            rho: Dixon-Coles dependence parameter (0 = independent Poisson).

        Returns:
            GoalProbabilities with normalized joint probabilities.
        """
        size = self.max_goals + 1
        matrix: list[list[float]] = []

        # Step 1: pre-compute Poisson PMFs for both teams
        home_pmf = [poisson_pmf(i, lambda_home) for i in range(size)]
        away_pmf = [poisson_pmf(j, lambda_away) for j in range(size)]

        # Step 2: joint probabilities with Dixon-Coles correction
        for i in range(size):
            row: list[float] = []
            for j in range(size):
                prob = home_pmf[i] * away_pmf[j]
                if rho != 0.0:
                    prob *= dixon_coles_correction(i, j, lambda_home, lambda_away, rho)
                row.append(max(0.0, prob))
            matrix.append(row)

        # Step 3: normalize to sum to 1.0
        total = sum(sum(row) for row in matrix)
        if total > 0:
            for i in range(size):
                for j in range(size):
                    matrix[i][j] /= total

        return GoalProbabilities(prob_matrix=matrix, max_goals=self.max_goals)

    def calculate_result_probabilities(
        self, goal_probs: GoalProbabilities
    ) -> tuple[float, float, float]:
        """Derive 1X2 probabilities from a goal probability matrix.

        Sums over i > j (home win), i == j (draw), i < j (away win),
        then normalizes so the three values sum to exactly 1.0.

        Args:
            goal_probs: Joint goal probability matrix.

        Returns:
            Tuple of (prob_home_win, prob_draw, prob_away_win).
        """
        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0

        size = goal_probs.max_goals + 1
        for i in range(size):
            for j in range(size):
                p = goal_probs.get_prob(i, j)
                if i > j:
                    prob_home += p
                elif i == j:
                    prob_draw += p
                else:
                    prob_away += p

        total = prob_home + prob_draw + prob_away
        if total > 0:
            prob_home /= total
            prob_draw /= total
            prob_away /= total

        return (prob_home, prob_draw, prob_away)

    # ------------------------------------------------------------------
    # Strength estimation from statistics
    # ------------------------------------------------------------------

    def calculate_team_strengths_from_stats(
        self,
        stats: TeamStats,  # type: ignore[name-defined]  # noqa: F821
        params: PoissonModelParams,
    ) -> TeamStrengths:
        """Estimate attack/defense strengths from TeamStats.

        Attack strength = team_goals_per_match / league_avg_goals.
        Defense strength = team_conceded_per_match / league_avg_goals (opponent).

        Returns strengths of 1.0 (league average) when no matches have been played.

        Args:
            stats: Aggregated team statistics for a league-season.
            params: League-season calibrated parameters.

        Returns:
            TeamStrengths with computed factors.
        """
        if stats.matches_home == 0 or stats.matches_away == 0:
            return TeamStrengths(team_id=stats.team.id)

        attack_home = (stats.goals_scored_home / stats.matches_home) / params.avg_goals_home
        attack_away = (stats.goals_scored_away / stats.matches_away) / params.avg_goals_away
        defense_home = (stats.goals_conceded_home / stats.matches_home) / params.avg_goals_away
        defense_away = (stats.goals_conceded_away / stats.matches_away) / params.avg_goals_home

        return TeamStrengths(
            team_id=stats.team.id,
            attack_home=attack_home,
            attack_away=attack_away,
            defense_home=defense_home,
            defense_away=defense_away,
        )

    # ------------------------------------------------------------------
    # League calibration
    # ------------------------------------------------------------------

    def calculate_league_params_from_matches(
        self,
        home_goals_list: list[int],
        away_goals_list: list[int],
        league_id: int,
        season: str,
    ) -> PoissonModelParams:
        """Calibrate PoissonModelParams from lists of observed goals.

        Computes the mean home and away goals across finished matches.
        Returns default parameters when the input lists are empty.

        Args:
            home_goals_list: Home goals from each finished match.
            away_goals_list: Away goals from each finished match (same length).
            league_id: League identifier.
            season: Season identifier.

        Returns:
            PoissonModelParams with computed averages.

        Raises:
            ValueError: If the two lists have different lengths.
        """
        if len(home_goals_list) != len(away_goals_list):
            raise ValueError(
                f"home_goals_list and away_goals_list must have the same length, "
                f"got {len(home_goals_list)} vs {len(away_goals_list)}"
            )

        n = len(home_goals_list)
        if n == 0:
            return PoissonModelParams(league_id=league_id, season=season)

        avg_home = sum(home_goals_list) / n
        avg_away = sum(away_goals_list) / n

        return PoissonModelParams(
            league_id=league_id,
            season=season,
            avg_goals_home=avg_home,
            avg_goals_away=avg_away,
            n_matches_used=n,
        )

    # ------------------------------------------------------------------
    # High-level prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        home_strength: TeamStrengths,
        away_strength: TeamStrengths,
        params: PoissonModelParams,
        apply_dixon_coles: bool = True,
    ) -> dict:
        """Generate a full match prediction from team strengths and league params.

        Orchestrates:
          1. calculate_lambdas  → expected goals (λ)
          2. build_score_matrix → joint goal probability matrix
          3. calculate_result_probabilities → 1X2 probabilities

        Args:
            home_strength: Home team attack/defense factors.
            away_strength: Away team attack/defense factors.
            params: League-season calibrated parameters.
            apply_dixon_coles: Whether to apply the τ correction.

        Returns:
            Dictionary with keys:
              - lambda_home: float
              - lambda_away: float
              - prob_home_win: float
              - prob_draw: float
              - prob_away_win: float
              - expected_goals_home: float
              - expected_goals_away: float
              - goal_probabilities: GoalProbabilities
        """
        lambda_home, lambda_away = self.calculate_lambdas(home_strength, away_strength, params)

        rho = params.rho if apply_dixon_coles else 0.0
        goal_probs = self.build_score_matrix(lambda_home, lambda_away, rho)

        prob_home, prob_draw, prob_away = self.calculate_result_probabilities(goal_probs)

        # Expected goals computed directly from the probability matrix
        size = self.max_goals + 1
        expected_home = 0.0
        expected_away = 0.0
        for i in range(size):
            for j in range(size):
                p = goal_probs.get_prob(i, j)
                expected_home += i * p
                expected_away += j * p

        return {
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "prob_home_win": prob_home,
            "prob_draw": prob_draw,
            "prob_away_win": prob_away,
            "expected_goals_home": expected_home,
            "expected_goals_away": expected_away,
            "goal_probabilities": goal_probs,
        }


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    """Aggregated results from a Monte Carlo match simulation.

    Attributes:
        n_simulations: Total number of simulated matches.
        home_wins / draws / away_wins: Raw counts of each outcome.
        prob_home_win / prob_draw / prob_away_win: Empirical probabilities.
        ci_home_win_low / ci_home_win_high: 95% confidence interval for home win prob.
        top_scores: Top 10 most frequent exact scorelines with counts and probs.
        goals_distribution: Map from total goals to probability.
        home_goals_p25 / p50 / p75: Home goals percentiles.
        away_goals_p25 / p50 / p75: Away goals percentiles.
        prob_over_2_5: Probability of >2.5 total goals.
        prob_btts: Probability both teams score.
        prob_clean_sheet_home / prob_clean_sheet_away: Clean sheet probabilities.
    """

    n_simulations: int
    home_wins: int
    draws: int
    away_wins: int
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    ci_home_win_low: float
    ci_home_win_high: float
    top_scores: list[dict[str, Any]]
    goals_distribution: dict[int, float]
    home_goals_p25: float
    home_goals_p50: float
    home_goals_p75: float
    away_goals_p25: float
    away_goals_p50: float
    away_goals_p75: float
    prob_over_2_5: float
    prob_btts: float
    prob_clean_sheet_home: float
    prob_clean_sheet_away: float

    @property
    def most_likely_score(self) -> str:
        """The scoreline with the highest frequency."""
        if not self.top_scores:
            return "0-0"
        return self.top_scores[0]["score"]

    @property
    def most_likely_result(self) -> MatchResult:
        """The outcome with the highest empirical probability."""
        probs = {
            MatchResult.HOME_WIN: self.prob_home_win,
            MatchResult.DRAW: self.prob_draw,
            MatchResult.AWAY_WIN: self.prob_away_win,
        }
        return max(probs, key=probs.get)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a compact dictionary for API/JSON output."""
        return {
            "n_simulations": self.n_simulations,
            "prob_home_win": round(self.prob_home_win, 4),
            "prob_draw": round(self.prob_draw, 4),
            "prob_away_win": round(self.prob_away_win, 4),
            "ci_home_win": [
                round(self.ci_home_win_low, 4),
                round(self.ci_home_win_high, 4),
            ],
            "most_likely_score": self.most_likely_score,
            "top_scores": self.top_scores[:5],
            "prob_over_2_5": round(self.prob_over_2_5, 4),
            "prob_btts": round(self.prob_btts, 4),
            "prob_clean_sheet_home": round(self.prob_clean_sheet_home, 4),
            "prob_clean_sheet_away": round(self.prob_clean_sheet_away, 4),
            "goals_p50_home": self.home_goals_p50,
            "goals_p50_away": self.away_goals_p50,
        }


class MonteCarloSimulator:
    """Independent Poisson-based Monte Carlo match simulator.

    Runs N independent simulations, each drawing goals from
    Poisson(lambda_home) and Poisson(lambda_away).
    Aggregates results into a MonteCarloResult.

    Uses Knuth's algorithm for Poisson sampling — no numpy required.
    Deterministic when seeded.
    """

    def __init__(self, n_simulations: int = 10_000, seed: int = None) -> None:
        self._n = n_simulations
        self._rng = random.Random(seed)

    def simulate(
        self,
        lambda_home: float,
        lambda_away: float,
    ) -> MonteCarloResult:
        """Run the simulation and return aggregated results.

        Args:
            lambda_home: Expected home goals (Poisson rate).
            lambda_away: Expected away goals (Poisson rate).

        Returns:
            MonteCarloResult with empirical probabilities, top scores,
            goal distributions, percentiles, and market probabilities.
        """
        home_goals_list: list[int] = []
        away_goals_list: list[int] = []
        score_counts: dict[str, int] = {}
        home_wins = draws = away_wins = 0
        over_2_5 = btts = clean_home = clean_away = 0

        for _ in range(self._n):
            gh = self._poisson_sample(lambda_home)
            ga = self._poisson_sample(lambda_away)

            home_goals_list.append(gh)
            away_goals_list.append(ga)

            key = f"{gh}-{ga}"
            score_counts[key] = score_counts.get(key, 0) + 1

            if gh > ga:
                home_wins += 1
            elif gh == ga:
                draws += 1
            else:
                away_wins += 1

            if gh + ga > 2.5:
                over_2_5 += 1
            if gh > 0 and ga > 0:
                btts += 1
            if ga == 0:
                clean_home += 1
            if gh == 0:
                clean_away += 1

        n = self._n
        ph = home_wins / n
        pd = draws / n
        pa = away_wins / n

        # 95% Wald confidence interval for home win probability
        margin = 1.96 * math.sqrt(ph * (1.0 - ph) / n)

        # Top 10 scores sorted by count descending
        top_scores = sorted(
            [
                {
                    "score": k,
                    "count": v,
                    "probability": round(v / n, 4),
                }
                for k, v in score_counts.items()
            ],
            key=lambda x: x["count"],
            reverse=True,
        )[:10]

        # Goals distribution
        goals_dist: dict[int, int] = {}
        for gh, ga in zip(home_goals_list, away_goals_list):
            t = gh + ga
            goals_dist[t] = goals_dist.get(t, 0) + 1
        goals_distribution = {k: round(v / n, 4) for k, v in sorted(goals_dist.items())}

        return MonteCarloResult(
            n_simulations=n,
            home_wins=home_wins,
            draws=draws,
            away_wins=away_wins,
            prob_home_win=ph,
            prob_draw=pd,
            prob_away_win=pa,
            ci_home_win_low=max(0.0, ph - margin),
            ci_home_win_high=min(1.0, ph + margin),
            top_scores=top_scores,
            goals_distribution=goals_distribution,
            home_goals_p25=_percentile(home_goals_list, 0.25),
            home_goals_p50=_percentile(home_goals_list, 0.50),
            home_goals_p75=_percentile(home_goals_list, 0.75),
            away_goals_p25=_percentile(away_goals_list, 0.25),
            away_goals_p50=_percentile(away_goals_list, 0.50),
            away_goals_p75=_percentile(away_goals_list, 0.75),
            prob_over_2_5=over_2_5 / n,
            prob_btts=btts / n,
            prob_clean_sheet_home=clean_home / n,
            prob_clean_sheet_away=clean_away / n,
        )

    def _poisson_sample(self, lam: float) -> int:
        """Knuth's algorithm for sampling from Poisson(lam).

        Numerically stable for all positive lambda values.
        No external dependencies required.
        """
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self._rng.random()
        return k - 1


def _percentile(data: list[int], p: float) -> float:
    """Compute the p-th percentile without numpy.

    Args:
        data: List of integer values.
        p: Percentile as a fraction (e.g. 0.5 for median).

    Returns:
        The value at the given percentile.
    """
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(p * len(s)), len(s) - 1)
    return float(s[idx])


# ---------------------------------------------------------------------------
# ELO-based lambda calculator (Dixon & Robinson 1998 style)
# ---------------------------------------------------------------------------


class EloStrengthCalculator:
    """Calculates expected goals (lambdas) directly from ELO ratings.

    Uses an exponential ratio formula that stays naturally bounded,
    avoiding the clamping required by the multiplicative model.

    Dixon & Robinson (1998) style:
      ratio = exp((ELO_home - ELO_away) / elo_scale)
      lambdas = softmax over expected total goals × home advantage
    """

    def __init__(
        self,
        home_advantage: float = 1.1,
        elo_scale: float = 400.0,
    ) -> None:
        self._home_adv = home_advantage
        self._elo_scale = elo_scale

    def calculate_lambdas(
        self,
        elo_home: float,
        elo_away: float,
        avg_goals_home: float = 1.5,
        avg_goals_away: float = 1.1,
    ) -> tuple[float, float]:
        total_expected = avg_goals_home + avg_goals_away
        home_share, _ = self.expected_goal_share(elo_home, elo_away)

        lambda_home = total_expected * home_share * self._home_adv
        lambda_away = total_expected * (1.0 - home_share) / self._home_adv

        total_lambda = lambda_home + lambda_away
        if total_lambda > 0:
            scale_factor = total_expected / total_lambda
            lambda_home *= scale_factor
            lambda_away *= scale_factor

        lambda_home = max(0.1, min(lambda_home, 8.0))
        lambda_away = max(0.1, min(lambda_away, 8.0))
        return lambda_home, lambda_away

    def expected_goal_share(self, elo_home: float, elo_away: float) -> tuple[float, float]:
        ratio = math.exp((elo_home - elo_away) / self._elo_scale)
        home_share = ratio / (ratio + 1.0 / ratio)
        return home_share, 1.0 - home_share
