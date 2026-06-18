import pytest

from football_predictor.domain.services import EloStrengthCalculator, MonteCarloSimulator


class TestEloStrengthCalculator:
    @pytest.fixture
    def calc(self):
        return EloStrengthCalculator(home_advantage=1.1, elo_scale=400.0)

    def test_equal_teams_split_goals_evenly(self, calc):
        lh, la = calc.calculate_lambdas(1500, 1500, 1.5, 1.1)
        assert abs(lh + la - (1.5 + 1.1)) < 0.01
        assert lh > la

    def test_stronger_home_team_gets_more_goals(self, calc):
        lh_strong, la_strong = calc.calculate_lambdas(1800, 1500, 1.5, 1.1)
        lh_equal, la_equal = calc.calculate_lambdas(1500, 1500, 1.5, 1.1)
        assert lh_strong > lh_equal
        assert la_strong < la_equal

    def test_no_clamping_needed_for_extreme_elo_diff(self, calc):
        lh, la = calc.calculate_lambdas(1910, 1467, 3.6, 0.84)
        assert lh < 8.0, f"lambda_home={lh:.2f} still hitting clamp"
        assert la > 0.1
        assert lh > la
        total = lh + la
        expected_total = 3.6 + 0.84
        assert abs(total - expected_total) < 0.5

    def test_symmetry(self, calc):
        lh_ab, la_ab = calc.calculate_lambdas(1800, 1500, 1.5, 1.1)
        lh_ba, la_ba = calc.calculate_lambdas(1500, 1800, 1.5, 1.1)
        assert lh_ab > la_ab
        assert lh_ba < la_ba

    def test_lambdas_always_positive(self, calc):
        casos = [
            (1000, 2000, 0.5, 0.3),
            (2000, 1000, 4.0, 3.0),
            (1500, 1500, 0.1, 0.1),
            (1500, 1500, 5.0, 5.0),
        ]
        for elo_h, elo_a, avg_h, avg_a in casos:
            lh, la = calc.calculate_lambdas(elo_h, elo_a, avg_h, avg_a)
            assert lh > 0
            assert la > 0

    def test_total_goals_conserved(self, calc):
        casos = [
            (1800, 1500, 1.5, 1.1),
            (1910, 1467, 3.6, 0.84),
            (1600, 1600, 2.0, 1.5),
        ]
        for elo_h, elo_a, avg_h, avg_a in casos:
            lh, la = calc.calculate_lambdas(elo_h, elo_a, avg_h, avg_a)
            expected_total = avg_h + avg_a
            actual_total = lh + la
            assert abs(actual_total - expected_total) < 0.1

    def test_home_goal_share_with_known_elo(self, calc):
        h_equal, a_equal = calc.expected_goal_share(1500, 1500)
        assert abs(h_equal - 0.5) < 0.001
        assert abs(a_equal - 0.5) < 0.001
        h_strong, a_strong = calc.expected_goal_share(1900, 1500)
        assert h_strong > 0.85
        assert a_strong < 0.15

    def test_germany_vs_curacao_realistic_output(self, calc):
        lh, la = calc.calculate_lambdas(
            elo_home=1910,
            elo_away=1467,
            avg_goals_home=3.6,
            avg_goals_away=0.84,
        )
        assert 2.0 <= lh <= 6.0
        assert 0.1 <= la <= 2.0
        mc = MonteCarloSimulator(n_simulations=5_000, seed=42)
        result = mc.simulate(lh, la)
        assert result.prob_home_win > 0.80
        assert result.prob_home_win < 0.99
