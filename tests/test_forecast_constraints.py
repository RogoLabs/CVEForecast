"""Tests for forecast_constraints.py."""
import pytest
from forecast_constraints import ForecastConstraints


@pytest.fixture
def constraints():
    config = {
        'min_annual_growth_rate': 0.05,
        'max_annual_growth_rate': 0.40,
        'historical_avg_growth': 0.18,
        'enable_growth_floor': True,
        'enable_trend_adjustment': True,
        'enable_ytd_floor': True,
        'trend_adjustment_confidence': 0.7,
        'trend_adjustment_threshold': 0.75,
        'ytd_minimum_factor': 0.85,
    }
    return ForecastConstraints(config)


class TestGrowthFloor:
    def test_below_minimum_raises_floor(self, constraints):
        # 40000 with 5% min = 42000 minimum
        result = constraints.apply_growth_floor(40000, 40000)
        assert result >= 42000

    def test_above_maximum_capped(self, constraints):
        # 40000 with 40% max = 56000 maximum
        result = constraints.apply_growth_floor(100000, 40000)
        assert result <= 56000

    def test_within_range_unchanged(self, constraints):
        # 12.5% growth - within 5-40% range
        result = constraints.apply_growth_floor(45000, 40000)
        assert result == 45000

    def test_disabled_returns_original(self, constraints):
        constraints.enable_floor = False
        result = constraints.apply_growth_floor(1, 40000)
        assert result == 1

    def test_exact_minimum(self, constraints):
        # Exactly 5% growth should pass through
        result = constraints.apply_growth_floor(42000, 40000)
        assert result == 42000


class TestApplyConstraints:
    def test_empty_input_returns_empty(self, constraints):
        result = constraints.apply_constraints({})
        assert result == {}

    def test_with_previous_year_actuals(self, constraints):
        yearly = {2026: {'ModelA': 30000}}
        actuals = {2025: 40000}
        result = constraints.apply_constraints(
            yearly, previous_year_actuals=actuals
        )
        # 30000 is below 5% growth from 40000 (42000), so should be raised
        assert result[2026]['ModelA'] >= 42000

    def test_no_baseline_passes_through(self, constraints):
        yearly = {2030: {'ModelA': 50000}}
        result = constraints.apply_constraints(yearly)
        assert result[2030]['ModelA'] == 50000

    def test_multiple_models(self, constraints):
        yearly = {2026: {'ModelA': 30000, 'ModelB': 50000}}
        actuals = {2025: 40000}
        result = constraints.apply_constraints(
            yearly, previous_year_actuals=actuals
        )
        assert result[2026]['ModelA'] >= 42000  # Constrained up
        assert result[2026]['ModelB'] == 50000  # Within range, unchanged
