"""Integration tests — full pipeline with synthetic data.

Tests the complete stack: repositories → stats → signal nodes →
feature engineering → Monte Carlo → ensemble → config.

No external APIs, no Gemini. Reproducible via random.seed(42).
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import pytest

from football_predictor.application.compute_stats import ComputeStatsUseCase
from football_predictor.application.train_model import TrainModelUseCase
from football_predictor.domain.entities import (
    CompetitionType,
    League,
    Match,
    MatchScore,
    MatchStatus,
    Team,
)
from football_predictor.domain.services import MonteCarloSimulator, PoissonService
from football_predictor.infrastructure.config import (
    SUPPORTED_LEAGUES,
    get_league_config,
    get_matchday_total,
    get_season_display,
    is_supported_league,
    validate_season_format,
)
from football_predictor.infrastructure.ml.ensemble import EnsemblePredictor
from football_predictor.infrastructure.ml.feature_engineering import FeatureEngineer
from football_predictor.infrastructure.ml.signal_nodes import (
    ContextNode,
    EloNode,
    FormNode,
    HeadToHeadNode,
    PoissonSignalNode,
)
from football_predictor.infrastructure.ml.xgboost_model import XGBoostPredictor
from football_predictor.infrastructure.repositories.sqlite_match_repo import (
    SqliteMatchRepository,
)
from football_predictor.infrastructure.repositories.sqlite_prediction_repo import (
    SqlitePredictionRepository,
)
from football_predictor.infrastructure.repositories.sqlite_stats_repo import (
    SqliteStatsRepository,
)


def _poisson_sample(lam: float, rng: random.Random) -> int:
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_league() -> League:
    return League(
        id="TEST",
        name="Test League",
        country="Testland",
        competition_type=CompetitionType.LEAGUE,
        season="2024",
    )


@pytest.fixture
def fake_teams() -> list[Team]:
    return [
        Team(
            id=100 + i,
            name=f"Team {i}",
            short_name=f"T{i}",
            tla=f"T{i:02d}",
            country="Testland",
        )
        for i in range(10)
    ]


@pytest.fixture
def fake_matches(fake_teams, fake_league) -> list[Match]:
    rng = random.Random(42)
    matches: list[Match] = []
    match_id = 1000
    matchday = 1
    day_offset = 0

    for i, home in enumerate(fake_teams):
        for j, away in enumerate(fake_teams):
            if i == j:
                continue
            hg = _poisson_sample(1.5, rng)
            ag = _poisson_sample(1.1, rng)
            match = Match(
                id=match_id,
                home_team=home,
                away_team=away,
                league=fake_league,
                match_date=datetime(2024, 8, 1) + timedelta(days=day_offset),
                status=MatchStatus.FINISHED,
                score=MatchScore(home_goals=hg, away_goals=ag),
                matchday=matchday,
            )
            matches.append(match)
            match_id += 1
            day_offset += 1
            if day_offset % 10 == 0:
                matchday += 1

    matches.sort(key=lambda m: m.match_date)
    return matches


@pytest.fixture
def db_repos(tmp_path) -> dict:
    db_path = str(tmp_path / "test.db")
    return {
        "match_repo": SqliteMatchRepository(db_path=db_path),
        "stats_repo": SqliteStatsRepository(db_path=db_path),
        "prediction_repo": SqlitePredictionRepository(db_path=db_path),
    }


@pytest.fixture
def poisson_svc() -> PoissonService:
    return PoissonService(max_goals=8)


@pytest.fixture
def mc_simulator() -> MonteCarloSimulator:
    return MonteCarloSimulator(n_simulations=1_000, seed=42)


@pytest.fixture
def signal_nodes_list(poisson_svc) -> list:
    return [
        FormNode(),
        EloNode(),
        HeadToHeadNode(),
        ContextNode(),
        PoissonSignalNode(poisson_svc),
    ]


@pytest.fixture
def feature_eng(db_repos, poisson_svc, signal_nodes_list) -> FeatureEngineer:
    return FeatureEngineer(
        match_repo=db_repos["match_repo"],
        stats_repo=db_repos["stats_repo"],
        poisson_service=poisson_svc,
        signal_nodes=signal_nodes_list,
    )


@pytest.fixture
def full_setup(
    db_repos,
    fake_matches,
    fake_league,
    poisson_svc,
    mc_simulator,
    signal_nodes_list,
    feature_eng,
    tmp_path,
) -> dict:
    match_repo = db_repos["match_repo"]
    stats_repo = db_repos["stats_repo"]
    prediction_repo = db_repos["prediction_repo"]

    match_repo.save_many(fake_matches)

    compute_uc = ComputeStatsUseCase(match_repo, stats_repo)
    compute_uc.execute(fake_league.id, fake_league.season)

    return {
        "match_repo": match_repo,
        "stats_repo": stats_repo,
        "prediction_repo": prediction_repo,
        "matches": fake_matches,
        "league": fake_league,
        "poisson_svc": poisson_svc,
        "mc_simulator": mc_simulator,
        "signal_nodes": signal_nodes_list,
        "feature_eng": feature_eng,
        "models_dir": str(tmp_path / "models"),
    }


# ---------------------------------------------------------------------------
# TestSQLiteRepositories
# ---------------------------------------------------------------------------


class TestSQLiteRepositories:
    def test_save_and_retrieve_match(self, db_repos, fake_matches):
        match = fake_matches[0]
        db_repos["match_repo"].save(match)
        retrieved = db_repos["match_repo"].get_by_id(match.id)

        assert retrieved is not None
        assert retrieved.id == match.id
        assert retrieved.home_team.id == match.home_team.id
        assert retrieved.away_team.id == match.away_team.id
        assert retrieved.score.home_goals == match.score.home_goals
        assert retrieved.score.away_goals == match.score.away_goals
        assert retrieved.status == match.status

    def test_save_many_and_count(self, db_repos, fake_matches, fake_league):
        db_repos["match_repo"].save_many(fake_matches)
        finished = db_repos["match_repo"].get_finished_matches(fake_league.id, fake_league.season)
        assert len(finished) == len(fake_matches)

    def test_get_upcoming_empty_when_all_finished(self, db_repos, fake_matches, fake_league):
        db_repos["match_repo"].save_many(fake_matches)
        upcoming = db_repos["match_repo"].get_upcoming_matches(fake_league.id)
        assert upcoming == []

    def test_head_to_head(self, db_repos, fake_matches, fake_league):
        db_repos["match_repo"].save_many(fake_matches)
        h2h = db_repos["match_repo"].get_head_to_head(str(100), str(101), limit=10)
        assert len(h2h) == 2
        for m in h2h:
            ids = {m.home_team.id, m.away_team.id}
            assert ids == {100, 101}

    def test_stats_roundtrip(self, db_repos, fake_matches, fake_league, fake_teams):
        db_repos["match_repo"].save_many(fake_matches)
        compute = ComputeStatsUseCase(db_repos["match_repo"], db_repos["stats_repo"])
        result = compute.execute(fake_league.id, fake_league.season)
        assert result["status"] == "ok"
        assert result["teams_computed"] == 10

        stats = db_repos["stats_repo"].get_stats(100, fake_league.id, fake_league.season)
        assert stats is not None
        assert stats.matches_played > 0
        assert stats.goals_scored >= 0
        assert stats.matches_home + stats.matches_away == stats.matches_played
        assert stats.wins + stats.draws + stats.losses == stats.matches_played


# ---------------------------------------------------------------------------
# TestSignalNodes
# ---------------------------------------------------------------------------


class TestSignalNodes:
    def test_all_nodes_return_required_keys(self, full_setup, signal_nodes_list):
        match = full_setup["matches"][50]
        context = _build_context(full_setup, match, 50)

        required_keys = {"signal", "weight", "confidence", "value", "summary"}
        for node in signal_nodes_list:
            output = node.compute(match, context)
            missing = required_keys - output.keys()
            assert not missing, f"Nodo {node.name} falta keys: {missing}"
            assert 0.0 <= output["confidence"] <= 1.0
            assert isinstance(output["summary"], str)
            assert len(output["summary"]) > 0

    def test_nodes_survive_empty_context(self, full_setup, signal_nodes_list):
        match = full_setup["matches"][0]
        for node in signal_nodes_list:
            output = node.compute(match, {})
            assert "value" in output

    def test_form_node_values_in_range(self, full_setup):
        node = FormNode()
        match = full_setup["matches"][20]
        context = _build_context(full_setup, match, 20)
        output = node.compute(match, context)
        val = output["value"]

        assert 0.0 <= val["home_form_5"] <= 1.0
        assert 0.0 <= val["home_form_10"] <= 1.0
        assert 0.0 <= val["away_form_5"] <= 1.0
        assert val["home_trend"] in ("improving", "declining", "stable")

    def test_elo_node_calculates_from_history(self, full_setup):
        node = EloNode()
        match = full_setup["matches"][30]
        context = _build_context(full_setup, match, 30)
        output = node.compute(match, context)
        val = output["value"]

        assert val["elo_diff"] == pytest.approx(val["home_elo"] - val["away_elo"], abs=0.01)
        assert 0.0 <= val["home_win_prob_elo"] <= 1.0
        assert val["elo_advantage"] in ("home", "away", "neutral")

    def test_poisson_node_probs_sum_to_one(self, full_setup, poisson_svc):
        node = PoissonSignalNode(poisson_svc)
        match = full_setup["matches"][40]
        context = _build_context(full_setup, match, 40)
        output = node.compute(match, context)
        val = output["value"]

        total = val["prob_home_win"] + val["prob_draw"] + val["prob_away_win"]
        assert total == pytest.approx(1.0, abs=0.01)
        assert val["lambda_home"] > 0
        assert val["lambda_away"] > 0


# ---------------------------------------------------------------------------
# TestFeatureEngineering
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    def test_features_have_consistent_keys(self, full_setup, feature_eng):
        matches = full_setup["matches"]
        stats_repo = full_setup["stats_repo"]
        league = full_setup["league"]

        key_sets = []
        for i in [10, 30, 50, 70]:
            match = matches[i]
            home_stats = stats_repo.get_stats(match.home_team.id, league.id, league.season)
            away_stats = stats_repo.get_stats(match.away_team.id, league.id, league.season)
            features = feature_eng.build_features_for_match(
                match, matches[:i], home_stats, away_stats
            )
            key_sets.append(set(features.keys()))

        assert all(ks == key_sets[0] for ks in key_sets), (
            "Features keys are not consistent across matches"
        )

    def test_no_data_leakage_in_dataset(self, full_setup, feature_eng):
        league = full_setup["league"]
        call_counts: list[int] = []
        original = feature_eng.build_features_for_match

        def capturing_build(match, finished, home_stats, away_stats):
            call_counts.append(len(finished))
            return original(match, finished, home_stats, away_stats)

        feature_eng.build_features_for_match = capturing_build
        try:
            X, y = feature_eng.build_dataset(league.id, league.season)
        finally:
            feature_eng.build_features_for_match = original

        assert call_counts == sorted(call_counts), "Context matches not monotonically increasing"
        assert call_counts[0] == 0, "First match must receive 0 context matches"

    def test_dataset_shape(self, full_setup, feature_eng):
        league = full_setup["league"]
        X, y = feature_eng.build_dataset(league.id, league.season)

        assert len(X) == len(y)
        assert len(X) == 90
        assert len(X.columns) >= 40
        assert set(y.unique()).issubset({0, 1, 2})
        assert X.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# TestMonteCarloSimulator
# ---------------------------------------------------------------------------


class TestMonteCarloSimulator:
    def test_probabilities_sum_to_one(self, mc_simulator):
        result = mc_simulator.simulate(1.5, 1.1)
        total = result.prob_home_win + result.prob_draw + result.prob_away_win
        assert total == pytest.approx(1.0, abs=0.001)

    def test_n_simulations_correct(self, mc_simulator):
        result = mc_simulator.simulate(1.5, 1.1)
        assert result.n_simulations == 1_000
        assert result.home_wins + result.draws + result.away_wins == 1_000

    def test_home_advantage_reflected(self, mc_simulator):
        result = mc_simulator.simulate(2.0, 0.8)
        assert result.prob_home_win > result.prob_away_win

    def test_top_scores_ordered(self, mc_simulator):
        result = mc_simulator.simulate(1.5, 1.1)
        counts = [s["count"] for s in result.top_scores]
        assert counts == sorted(counts, reverse=True)

    def test_over_2_5_coherent(self, mc_simulator):
        result = mc_simulator.simulate(2.5, 2.0)
        assert result.prob_over_2_5 > 0.5

    def test_reproducible_with_seed(self):
        mc1 = MonteCarloSimulator(n_simulations=500, seed=123)
        mc2 = MonteCarloSimulator(n_simulations=500, seed=123)
        r1 = mc1.simulate(1.5, 1.1)
        r2 = mc2.simulate(1.5, 1.1)
        assert r1.prob_home_win == r2.prob_home_win
        assert r1.most_likely_score == r2.most_likely_score

    def test_ci_contains_probability(self, mc_simulator):
        result = mc_simulator.simulate(1.5, 1.1)
        assert result.ci_home_win_low <= result.prob_home_win
        assert result.prob_home_win <= result.ci_home_win_high


# ---------------------------------------------------------------------------
# TestEnsemblePredictor
# ---------------------------------------------------------------------------


class TestEnsemblePredictor:
    def test_ensemble_poisson_only_when_xgb_not_trained(
        self, full_setup, poisson_svc, mc_simulator, feature_eng
    ):
        xgb = XGBoostPredictor()
        ensemble = EnsemblePredictor(
            poisson_service=poisson_svc,
            xgboost_predictor=xgb,
            feature_engineer=feature_eng,
            monte_carlo=mc_simulator,
        )

        match = full_setup["matches"][50]
        context = _build_context(full_setup, match, 50)
        signal_outputs = _build_signal_outputs(full_setup, match, context)

        result = ensemble.predict(
            match=match,
            signal_outputs=signal_outputs,
            home_stats=context["home_stats"],
            away_stats=context["away_stats"],
            finished_matches=full_setup["matches"][:50],
        )

        assert result.model_used == "poisson_only"
        assert result.xgb_probs is None
        assert result.probabilities_are_valid

    def test_ensemble_with_trained_xgb(
        self, full_setup, poisson_svc, mc_simulator, feature_eng, tmp_path
    ):
        xgb = XGBoostPredictor()
        train_uc = TrainModelUseCase(
            match_repo=full_setup["match_repo"],
            stats_repo=full_setup["stats_repo"],
            feature_engineer=feature_eng,
            xgboost_predictor=xgb,
            models_dir=str(tmp_path / "models"),
        )
        result_train = train_uc.execute(full_setup["league"].id, full_setup["league"].season)
        assert result_train["status"] == "ok"

        ensemble = EnsemblePredictor(
            poisson_service=poisson_svc,
            xgboost_predictor=xgb,
            feature_engineer=feature_eng,
            monte_carlo=mc_simulator,
        )

        match = full_setup["matches"][60]
        context = _build_context(full_setup, match, 60)
        signal_outputs = _build_signal_outputs(full_setup, match, context)

        result = ensemble.predict(
            match=match,
            signal_outputs=signal_outputs,
            home_stats=context["home_stats"],
            away_stats=context["away_stats"],
            finished_matches=full_setup["matches"][:60],
        )

        assert result.model_used == "ensemble"
        assert result.xgb_probs is not None
        assert result.probabilities_are_valid
        assert 0.0 <= result.confidence <= 1.0

    def test_ensemble_probs_valid_with_extreme_llm_weights(
        self, full_setup, poisson_svc, mc_simulator, feature_eng
    ):
        xgb = XGBoostPredictor()
        ensemble = EnsemblePredictor(
            poisson_service=poisson_svc,
            xgboost_predictor=xgb,
            feature_engineer=feature_eng,
            monte_carlo=mc_simulator,
        )

        match = full_setup["matches"][50]
        context = _build_context(full_setup, match, 50)
        signal_outputs = _build_signal_outputs(full_setup, match, context)

        for extreme_weight in [0.0, 3.0, -1.0, 5.0, None]:
            llm_w = {"poisson": extreme_weight} if extreme_weight is not None else None
            result = ensemble.predict(
                match=match,
                signal_outputs=signal_outputs,
                home_stats=context["home_stats"],
                away_stats=context["away_stats"],
                finished_matches=full_setup["matches"][:50],
                llm_adjusted_weights=llm_w,
            )
            assert result.probabilities_are_valid, f"Invalid probs with weight={extreme_weight}"


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_all_leagues_have_required_keys(self):
        required = {"id", "name", "country", "season_format", "avg_matches_per_season"}
        for lid, config in SUPPORTED_LEAGUES.items():
            missing = required - config.keys()
            assert not missing, f"League {lid} missing keys: {missing}"

    def test_get_league_config_valid(self):
        config = get_league_config("PL")
        assert config["name"] == "Premier League"
        assert config["country"] == "England"

    def test_get_league_config_invalid(self):
        with pytest.raises(ValueError, match="no soportada"):
            get_league_config("FAKE_LEAGUE")

    def test_season_display_calendar_year(self):
        assert get_season_display("PL", "2024") == "2024/25"

    def test_season_display_single_year(self):
        assert get_season_display("BSA", "2024") == "2024"

    def test_validate_season_format(self):
        assert validate_season_format("2024") is True
        assert validate_season_format("1999") is False
        assert validate_season_format("2031") is False
        assert validate_season_format("abcd") is False
        assert validate_season_format("24") is False

    def test_is_supported_league(self):
        assert is_supported_league("PL") is True
        assert is_supported_league("FAKE") is False

    def test_matchday_total_known_leagues(self):
        assert get_matchday_total("PL") == 38
        assert get_matchday_total("BL1") == 34
        assert get_matchday_total("ELC") == 46


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_context(full_setup, match, num_past):
    stats = full_setup["stats_repo"]
    league = full_setup["league"]
    return {
        "finished_matches": full_setup["matches"][:num_past],
        "home_stats": stats.get_stats(match.home_team.id, league.id, league.season),
        "away_stats": stats.get_stats(match.away_team.id, league.id, league.season),
        "elo_ratings": None,
    }


def _build_signal_outputs(full_setup, match, context):
    return {node.name: node.compute(match, context) for node in full_setup["signal_nodes"]}
