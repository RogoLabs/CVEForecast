"""
Tests for the v0.12 forecasting core: covariates, transforms, backtest, intervals.

These pin the properties the pipeline depends on rather than specific accuracy
numbers, which move as data arrives.
"""

import numpy as np
import pandas as pd
import pytest
from core.covariates import (
    build_future_covariates,
    business_days_for_index,
    business_days_in_month,
    denormalise_by_business_days,
    normalise_by_business_days,
)
from core.intervals import apply_intervals, build_intervals, pooled_residuals, validate_coverage
from core.transforms import damp_forecast_path, from_log_space, to_log_space, trim_to_window
from darts import TimeSeries
from validation.rolling_origin import (
    BacktestResult,
    RollingOriginBacktest,
    mark_naive_baselines,
    rank_models,
    seasonal_naive_mae,
)


def make_series(n=60, start=1000.0, growth=1.02):
    values = [start * growth**i for i in range(n)]
    index = pd.date_range('2019-01-31', periods=n, freq='ME')
    return TimeSeries.from_series(pd.Series(values, index=index))


class TestCovariates:
    def test_business_days_in_expected_range(self):
        index = pd.date_range('2020-01-31', periods=48, freq='ME')
        bdays = business_days_for_index(index)
        assert bdays.min() >= 19
        assert bdays.max() <= 23

    def test_known_month(self):
        # February 2026: 1st is a Sunday, 28 days -> 20 business days
        assert business_days_in_month(pd.Timestamp('2026-02-28')) == 20

    def test_normalise_roundtrip(self):
        series = make_series(24)
        restored = denormalise_by_business_days(normalise_by_business_days(series))
        np.testing.assert_allclose(restored.values().flatten(), series.values().flatten(), rtol=1e-9)

    def test_future_covariates_shape(self):
        cov = build_future_covariates(pd.Timestamp('2024-01-31'), pd.Timestamp('2025-12-31'))
        assert len(cov) == 24
        assert cov.width == 13  # 12 month dummies + business days

    def test_future_covariates_without_business_days(self):
        cov = build_future_covariates(
            pd.Timestamp('2024-01-31'), pd.Timestamp('2024-12-31'), include_business_days=False
        )
        assert cov.width == 12

    def test_empty_range_rejected(self):
        with pytest.raises(ValueError):
            build_future_covariates(pd.Timestamp('2025-01-31'), pd.Timestamp('2024-01-31'))


class TestTransforms:
    def test_log_roundtrip(self):
        series = make_series(24)
        restored = from_log_space(to_log_space(series))
        np.testing.assert_allclose(restored.values().flatten(), series.values().flatten(), rtol=1e-9)

    def test_log_handles_zero(self):
        index = pd.date_range('2020-01-31', periods=3, freq='ME')
        series = TimeSeries.from_series(pd.Series([0.0, 5.0, 10.0], index=index))
        assert np.all(np.isfinite(to_log_space(series).values()))

    def test_negative_values_rejected(self):
        index = pd.date_range('2020-01-31', periods=2, freq='ME')
        with pytest.raises(ValueError):
            to_log_space(TimeSeries.from_series(pd.Series([-1.0, 5.0], index=index)))

    def test_smearing_correction_raises_the_estimate(self):
        series = to_log_space(make_series(12))
        assert from_log_space(series, sigma=0.5).values().sum() > from_log_space(series).values().sum()

    def test_damping_shrinks_an_extrapolated_trend(self):
        index = pd.date_range('2026-01-31', periods=6, freq='ME')
        forecast = TimeSeries.from_series(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=index))
        damped = damp_forecast_path(forecast, last_observed=0.0, phi=0.9)
        assert damped.values().flatten()[-1] < 6.0
        # the first step is barely touched, later steps increasingly so
        assert damped.values().flatten()[0] == pytest.approx(0.9, abs=1e-9)

    def test_phi_one_is_a_noop(self):
        index = pd.date_range('2026-01-31', periods=3, freq='ME')
        forecast = TimeSeries.from_series(pd.Series([1.0, 2.0, 3.0], index=index))
        np.testing.assert_allclose(damp_forecast_path(forecast, 0.0, phi=1.0).values(), forecast.values())

    def test_damping_in_levels_space_stays_positive(self):
        index = pd.date_range('2026-01-31', periods=4, freq='ME')
        forecast = TimeSeries.from_series(pd.Series([110.0, 130.0, 160.0, 200.0], index=index))
        damped = damp_forecast_path(forecast, last_observed=100.0, phi=0.8, in_log_space=False)
        values = damped.values().flatten()
        assert np.all(values > 0)
        assert values[-1] < 200.0

    def test_invalid_phi_rejected(self):
        index = pd.date_range('2026-01-31', periods=2, freq='ME')
        forecast = TimeSeries.from_series(pd.Series([1.0, 2.0], index=index))
        with pytest.raises(ValueError):
            damp_forecast_path(forecast, 0.0, phi=1.5)

    def test_trim_window(self):
        series = make_series(60)
        assert len(trim_to_window(series, 24)) == 24
        assert len(trim_to_window(series, None)) == 60
        assert len(trim_to_window(series, 999)) == 60


class TestRollingOrigin:
    def test_seasonal_naive_scale(self):
        values = np.arange(24, dtype=float)
        assert seasonal_naive_mae(values) == pytest.approx(12.0)

    def test_constant_series_does_not_divide_by_zero(self):
        assert seasonal_naive_mae(np.ones(24)) == 1.0

    def test_origins_respect_min_train(self):
        backtest = RollingOriginBacktest(horizon=6, min_train=24, max_origins=None)
        origins = backtest.origins_for(make_series(60))
        assert min(origins) >= 24
        assert max(origins) <= 59

    def test_max_origins_keeps_the_most_recent(self):
        backtest = RollingOriginBacktest(horizon=6, min_train=24, max_origins=5)
        origins = backtest.origins_for(make_series(60))
        assert len(origins) == 5
        assert max(origins) == 59

    def test_perfect_forecaster_scores_zero(self):
        series = make_series(60)
        values = series.values().flatten()

        def oracle(train, horizon):
            start = len(train)
            future = values[start : start + horizon]
            index = series.time_index[start : start + len(future)]
            return TimeSeries.from_times_and_values(index, future)

        result = RollingOriginBacktest(horizon=6, min_train=36, max_origins=6).evaluate(series, oracle, 'Oracle')
        assert result.is_valid
        assert result.mase == pytest.approx(0.0, abs=1e-9)

    def test_failed_forecasts_are_recorded_not_hidden(self):
        result = RollingOriginBacktest(horizon=6, min_train=36, max_origins=4).evaluate(
            make_series(60), lambda train, horizon: None, 'Broken'
        )
        assert not result.is_valid
        assert result.errors

    def test_beats_naive_is_relative_to_the_scored_baseline(self):
        """MASE > 1 does not mean 'lost to naive' - the baseline is what matters."""
        results = {
            'Good': BacktestResult('Good', n_origins=10, mase=2.1),
            'Bad': BacktestResult('Bad', n_origins=10, mase=3.6),
            'NaiveDrift': BacktestResult('NaiveDrift', n_origins=10, mase=2.7),
        }
        assert mark_naive_baselines(results) == pytest.approx(2.7)
        assert results['Good'].beats_naive is True
        assert results['Bad'].beats_naive is False
        assert results['NaiveDrift'].beats_naive is None

    def test_no_baseline_leaves_verdict_unknown(self):
        results = {'Only': BacktestResult('Only', n_origins=10, mase=2.0)}
        assert mark_naive_baselines(results) is None
        assert results['Only'].beats_naive is None

    def test_ranking_puts_failures_last(self):
        results = {
            'Broken': BacktestResult('Broken'),
            'Best': BacktestResult('Best', n_origins=10, mase=1.0),
            'Worse': BacktestResult('Worse', n_origins=10, mase=2.0),
        }
        assert [r.model_name for r in rank_models(results)] == ['Best', 'Worse', 'Broken']


class TestIntervals:
    @staticmethod
    def residuals(n=40, scale=0.2):
        rng = np.random.default_rng(0)
        return {h: list(rng.normal(0, scale * h**0.5, n)) for h in range(1, 13)}

    def test_bands_bracket_the_point_forecast(self):
        bands = build_intervals(self.residuals())
        low, high = bands.for_horizon(1)['80']
        assert low < 1.0 < high

    def test_bands_widen_with_horizon(self):
        bands = build_intervals(self.residuals())
        width = lambda h: bands.for_horizon(h)['80'][1] - bands.for_horizon(h)['80'][0]  # noqa: E731
        assert width(12) >= width(6) >= width(1)

    def test_95_is_wider_than_80(self):
        bands = build_intervals(self.residuals())
        f = bands.for_horizon(6)
        assert f['95'][0] <= f['80'][0] and f['95'][1] >= f['80'][1]

    def test_coverage_is_near_nominal(self):
        residuals = self.residuals(n=200)
        coverage = validate_coverage(residuals, build_intervals(residuals))
        assert coverage['80']['calibrated']
        assert coverage['95']['calibrated']

    def test_no_residuals_yields_no_bands(self):
        bands = build_intervals({})
        assert bands.factors == {}
        assert apply_intervals({'2026-01': 100.0}, bands) == {}

    def test_too_few_residuals_yields_no_bands(self):
        assert build_intervals({1: [0.1, 0.2]}).factors == {}

    def test_applied_intervals_scale_the_point(self):
        bands = build_intervals(self.residuals())
        applied = apply_intervals({'2026-01': 1000.0, '2026-02': 1000.0}, bands)
        assert applied['2026-01']['lower_80'] < 1000 < applied['2026-01']['upper_80']
        assert set(applied['2026-01']) == {'lower_80', 'upper_80', 'lower_95', 'upper_95'}

    def test_horizon_beyond_fitted_range_reuses_the_longest(self):
        bands = build_intervals(self.residuals())
        assert bands.for_horizon(99) == bands.for_horizon(12)

    def test_pooling_selects_named_models(self):
        per_model = {'a': {1: [0.1, 0.2]}, 'b': {1: [0.3]}}
        assert sorted(pooled_residuals(per_model, ['a'])[1]) == [0.1, 0.2]
        assert len(pooled_residuals(per_model)[1]) == 3


class TestCumulativeBand:
    """
    The shaded chart band is computed server-side so the month each cumulative
    step belongs to cannot drift. These pin that alignment.
    """

    @staticmethod
    def band(timeline, step_intervals):
        from adapters.cve_adapter import CVEForecaster

        # The method touches no instance state; call it unbound to avoid needing
        # a configured forecaster (and a cvelistV5 checkout) for a pure-maths test.
        return CVEForecaster._generate_cumulative_band(None, timeline, step_intervals)

    def test_band_brackets_the_line(self):
        timeline = [
            {'date': '2026-01-01T00:00:00Z', 'cumulative_total': 0},
            {'date': '2026-09-17T12:00:00Z', 'cumulative_total': 66401},
            {'date': '2026-10-01T00:00:00Z', 'cumulative_total': 70047},
            {'date': '2026-11-01T00:00:00Z', 'cumulative_total': 79445},
        ]
        steps = {
            '2026-09': {'lower_80': 3121, 'upper_80': 5179},
            '2026-10': {'lower_80': 8043, 'upper_80': 14350},
        }
        result = self.band(timeline, steps)
        lower = {e['date']: e['cumulative_total'] for e in result['lower']}
        upper = {e['date']: e['cumulative_total'] for e in result['upper']}

        for entry in timeline:
            assert lower[entry['date']] <= entry['cumulative_total'] <= upper[entry['date']]

    def test_anchor_point_carries_no_uncertainty(self):
        """The anchor is an observed count, not a forecast."""
        timeline = [
            {'date': '2026-01-01T00:00:00Z', 'cumulative_total': 0},
            {'date': '2026-09-17T12:00:00Z', 'cumulative_total': 66401},
            {'date': '2026-10-01T00:00:00Z', 'cumulative_total': 70047},
        ]
        result = self.band(timeline, {'2026-09': {'lower_80': 3121, 'upper_80': 5179}})
        assert result['lower'][1]['cumulative_total'] == 66401
        assert result['upper'][1]['cumulative_total'] == 66401

    def test_step_uses_the_month_it_came_from(self):
        """Sep 17 -> Oct 1 is September's remainder, not October's forecast."""
        timeline = [
            {'date': '2026-01-01T00:00:00Z', 'cumulative_total': 0},
            {'date': '2026-09-17T12:00:00Z', 'cumulative_total': 66401},
            {'date': '2026-10-01T00:00:00Z', 'cumulative_total': 70047},
        ]
        steps = {'2026-09': {'lower_80': 3000, 'upper_80': 5000}, '2026-10': {'lower_80': 0, 'upper_80': 99999}}
        result = self.band(timeline, steps)
        step = 70047 - 66401
        assert result['lower'][2]['cumulative_total'] == 70047 + (3000 - step)
        assert result['upper'][2]['cumulative_total'] == 70047 + (5000 - step)

    def test_year_boundary_resets_accumulated_uncertainty(self):
        timeline = [
            {'date': '2026-12-01T00:00:00Z', 'cumulative_total': 88450},
            {'date': '2026-12-31T23:59:59Z', 'cumulative_total': 97885},
            {'date': '2027-01-01T00:00:00Z', 'cumulative_total': 0},
            {'date': '2027-02-01T00:00:00Z', 'cumulative_total': 9000},
        ]
        steps = {
            '2026-12': {'lower_80': 8000, 'upper_80': 12000},
            '2027-01': {'lower_80': 7000, 'upper_80': 11000},
        }
        result = self.band(timeline, steps)
        lower = {e['date']: e['cumulative_total'] for e in result['lower']}
        assert lower['2027-01-01T00:00:00Z'] == 0
        assert lower['2027-02-01T00:00:00Z'] == 9000 + (7000 - 9000)

    def test_no_intervals_yields_no_band(self):
        assert self.band([{'date': '2026-10-01T00:00:00Z', 'cumulative_total': 1}], {}) == {}
