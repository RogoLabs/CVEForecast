"""
Tests for empirical prediction intervals.

These pin the properties the CNA bands rest on, and the two failures that
building them actually surfaced: a horizon lookup that assumed the fitted
horizons were contiguous, and a band silently reused far past the range it was
fitted over.
"""

import numpy as np
import pytest
from core.intervals import (
    MIN_RESIDUALS_PER_HORIZON,
    IntervalBands,
    apply_intervals,
    build_intervals,
    build_shared_shape,
    robust_scale,
    scale_bands,
    validate_coverage,
)


def residuals(n, spread=0.2, horizons=range(1, 7), seed=0):
    """Log residuals that widen with horizon, as real ones do."""
    rng = np.random.default_rng(seed)
    return {h: list(rng.normal(0, spread * (1 + 0.15 * h), n)) for h in horizons}


class TestBuildIntervals:
    def test_bands_widen_with_horizon(self):
        bands = build_intervals(residuals(60))
        widths = [bands.factors[h]['80'][1] / bands.factors[h]['80'][0] for h in sorted(bands.factors)]
        assert widths == sorted(widths)

    def test_95_contains_80(self):
        bands = build_intervals(residuals(60))
        for per_level in bands.factors.values():
            assert per_level['95'][0] <= per_level['80'][0]
            assert per_level['95'][1] >= per_level['80'][1]

    def test_too_few_residuals_yields_no_band(self):
        thin = {h: [0.1, -0.1] for h in range(1, 7)}
        assert build_intervals(thin).factors == {}

    def test_a_horizon_below_the_threshold_borrows_rather_than_fabricates(self):
        mixed = dict(residuals(MIN_RESIDUALS_PER_HORIZON * 4, horizons=[1, 2]))
        mixed[3] = [0.05, -0.05]
        bands = build_intervals(mixed)
        assert bands.factors[3]['80'] == bands.factors[2]['80']


class TestForHorizon:
    """
    A residual is only recorded where actual and forecast are both positive, so a
    series with sporadic zero months loses whole horizons. 127 of 140 CNAs have
    at least one zero month; the aggregate CVE series has none, which is why
    indexing at min(max(factors), horizon) survived this long.
    """

    @pytest.fixture
    def gapped(self):
        bands = IntervalBands()
        bands.factors = {
            1: {'80': (0.9, 1.1)},
            2: {'80': (0.8, 1.2)},
            5: {'80': (0.7, 1.3)},
            9: {'80': (0.5, 1.5)},
        }
        bands.max_horizon = 9
        return bands

    def test_a_missing_interior_horizon_does_not_raise(self, gapped):
        assert gapped.for_horizon(3)['80'] == (0.8, 1.2)
        assert gapped.for_horizon(7)['80'] == (0.7, 1.3)

    def test_falls_back_to_the_nearest_fitted_horizon_below(self, gapped):
        assert gapped.for_horizon(8)['80'] == (0.7, 1.3)

    def test_beyond_the_fitted_range_reuses_the_longest(self, gapped):
        assert gapped.for_horizon(30)['80'] == gapped.factors[9]['80']

    def test_before_the_first_fitted_horizon_uses_it(self):
        bands = IntervalBands()
        bands.factors = {4: {'80': (0.7, 1.3)}}
        assert bands.for_horizon(1)['80'] == (0.7, 1.3)

    def test_empty_bands_return_nothing(self):
        assert IntervalBands().for_horizon(1) == {}


class TestRoundTrip:
    """
    The run that builds a CNA's band is almost never the run that publishes it:
    scoring is capped at a dozen CNAs a run, so bands travel through the cache.
    """

    def test_survives_serialisation(self):
        bands = build_intervals(residuals(60))
        restored = IntervalBands.from_dict(bands.to_dict())
        assert restored.to_dict() == bands.to_dict()
        assert restored.levels == bands.levels
        assert restored.max_horizon == bands.max_horizon

    def test_empty_payloads_are_tolerated(self):
        assert IntervalBands.from_dict({}).factors == {}
        assert IntervalBands.from_dict(None).factors == {}


class TestSharedShape:
    """
    The band's shape is pooled across CNAs and only its width is per-CNA, because
    a rolling-origin backtest holds at most max_origins - h + 1 residuals at
    horizon h - so no single short series has anything to fit at long horizons.
    """

    def test_a_wider_member_gets_a_wider_band(self):
        groups = {
            'steady': residuals(40, spread=0.1, seed=1),
            'erratic': residuals(40, spread=0.8, seed=2),
        }
        shape, scales = build_shared_shape(groups)
        assert scales['erratic'] > scales['steady']

        steady = scale_bands(shape, scales['steady'])
        erratic = scale_bands(shape, scales['erratic'])
        width = lambda b, h: b.factors[h]['80'][1] / b.factors[h]['80'][0]  # noqa: E731
        assert width(erratic, 1) > width(steady, 1)

    def test_scaling_preserves_widening_with_horizon(self):
        shape, scales = build_shared_shape({'a': residuals(40, seed=3), 'b': residuals(40, seed=4)})
        band = scale_bands(shape, scales['a'])
        widths = [band.factors[h]['80'][1] / band.factors[h]['80'][0] for h in sorted(band.factors)]
        assert widths == sorted(widths)

    def test_the_shape_reaches_horizons_no_single_member_could_fit(self):
        # One residual each at h=12 is useless alone and ample pooled, which is
        # exactly the case the next-year column depends on.
        groups = {
            f'cna{i}': {1: list(np.random.default_rng(i).normal(0, 0.3, 20)), 12: [0.1 * (i - 10)]} for i in range(30)
        }
        shape, scales = build_shared_shape(groups)
        assert 12 in shape.factors

    def test_a_member_with_too_little_to_measure_is_left_out(self):
        groups = {'ok': residuals(40, seed=5), 'thin': {1: [0.1, -0.1]}}
        _shape, scales = build_shared_shape(groups)
        assert 'ok' in scales
        assert 'thin' not in scales

    def test_no_members_yields_nothing(self):
        shape, scales = build_shared_shape({})
        assert shape.factors == {} and scales == {}

    def test_an_unusable_scale_yields_no_band(self):
        shape, _ = build_shared_shape({'a': residuals(40, seed=6), 'b': residuals(40, seed=7)})
        assert scale_bands(shape, float('nan')).factors == {}
        assert scale_bands(shape, 0.0).factors == {}


class TestRobustScale:
    def test_is_not_moved_much_by_one_bad_month(self):
        clean = list(np.random.default_rng(0).normal(0, 0.3, 40))
        assert robust_scale(clean + [12.0]) == pytest.approx(robust_scale(clean), rel=0.15)

    def test_too_few_points_is_not_a_measurement(self):
        assert np.isnan(robust_scale([0.1, 0.2]))


class TestApplyIntervals:
    def test_bounds_bracket_their_own_point(self):
        bands = build_intervals(residuals(60))
        forecasts = {f'2027-{m:02d}': 100.0 for m in range(1, 7)}
        for month, band in apply_intervals(forecasts, bands).items():
            assert band['lower_80'] <= forecasts[month] <= band['upper_80']

    def test_no_bands_means_no_intervals(self):
        assert apply_intervals({'2027-01': 100.0}, IntervalBands()) == {}


class TestCoverage:
    def test_reports_close_to_nominal_on_the_residuals_that_built_it(self):
        data = residuals(200)
        summary = validate_coverage(data, build_intervals(data))
        assert summary['80']['empirical'] == pytest.approx(0.80, abs=0.06)
        assert summary['95']['empirical'] == pytest.approx(0.95, abs=0.04)

    def test_no_bands_means_no_coverage(self):
        assert validate_coverage(residuals(60), IntervalBands()) == {}
