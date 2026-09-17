"""
Tests for forecast sanity guards and year-total construction.

Rewritten for v0.12. The previous suite asserted the behaviour of the growth
floor - that a forecast below 5% growth was raised to it. That behaviour was the
defect: it compared a partial-year remainder against a full prior year and
collapsed every model onto the same number. These tests pin the replacement,
including a regression test for the specific 2026 failure.
"""

import pytest
from forecast_constraints import (
    ForecastConstraints,
    YearProjection,
    build_year_projections,
    combine_model_forecasts,
)


@pytest.fixture
def constraints():
    return ForecastConstraints({})


class TestYearProjections:
    def test_splits_actuals_from_forecast(self):
        actuals = {'2026-01': 4302, '2026-02': 4616}
        forecasts = {'2026-03': 5000, '2026-04': 5200}
        result = build_year_projections(actuals, forecasts)

        proj = result[2026]
        assert proj.actual_ytd == 8918
        assert proj.forecast_remainder == 10200
        assert proj.total == 19118
        assert proj.months_actual == 2
        assert proj.months_forecast == 2
        assert not proj.is_complete

    def test_published_month_is_never_overwritten_by_its_own_forecast(self):
        actuals = {'2026-01': 4302}
        forecasts = {'2026-01': 9999, '2026-02': 5000}
        result = build_year_projections(actuals, forecasts)

        assert result[2026].actual_ytd == 4302
        assert result[2026].forecast_remainder == 5000
        assert result[2026].months_forecast == 1

    def test_spans_multiple_years(self):
        result = build_year_projections({'2026-01': 100}, {'2026-02': 200, '2027-01': 300})

        assert result[2026].total == 300
        assert result[2027].total == 300
        assert result[2027].actual_ytd == 0
        assert result[2027].is_complete is False

    def test_year_band_sums_monthly_bands_onto_actuals(self):
        result = build_year_projections(
            {'2026-01': 1000},
            {'2026-02': 500, '2026-03': 600},
            intervals={
                '2026-02': {'lower_80': 400, 'upper_80': 700},
                '2026-03': {'lower_80': 480, 'upper_80': 840},
            },
        )
        proj = result[2026]
        assert proj.lower_80 == 1000 + 880
        assert proj.upper_80 == 1000 + 1540
        assert proj.lower_80 < proj.total < proj.upper_80

    def test_empty_inputs(self):
        assert build_year_projections({}, {}) == {}

    def test_partial_month_counts_actual_plus_remainder(self):
        """The in-progress month is a nowcast: published so far + days remaining."""
        result = build_year_projections(
            {'2026-01': 4302, '2026-02': 2000},  # Feb still running
            {'2026-02': 900, '2026-03': 5000},  # Feb entry is the remainder
            partial_month='2026-02',
        )
        proj = result[2026]
        assert proj.actual_ytd == 6302
        assert proj.forecast_remainder == 5900
        assert proj.total == 12202
        # Feb is counted once as an actual month, not twice
        assert proj.months_actual == 2
        assert proj.months_forecast == 1

    def test_partial_month_band_must_not_double_count_published_days(self):
        """
        The year band adds what each month still CONTRIBUTES. Feeding it the
        current month's published figure instead of its remainder pushes the lower
        bound above the total, because actual_ytd already holds those days.
        """
        result = build_year_projections(
            {'2026-01': 4000, '2026-02': 2000},
            {'2026-02': 900, '2026-03': 5000},
            intervals={
                # remainder bands, not full-month bands
                '2026-02': {'lower_80': 700, 'upper_80': 1200},
                '2026-03': {'lower_80': 4200, 'upper_80': 6100},
            },
            partial_month='2026-02',
        )
        proj = result[2026]
        assert proj.total == 11900
        assert proj.lower_80 == 6000 + 700 + 4200
        assert proj.upper_80 == 6000 + 1200 + 6100
        assert proj.lower_80 <= proj.total <= proj.upper_80

    def test_without_partial_month_the_forecast_is_still_discarded(self):
        result = build_year_projections(
            {'2026-01': 4302, '2026-02': 2000},
            {'2026-02': 900, '2026-03': 5000},
        )
        assert result[2026].forecast_remainder == 5000
        assert result[2026].total == 11302

    def test_a_measured_annual_band_is_preferred_over_summing_the_months(self):
        """
        Summing monthly bounds assumes the model errs in the same direction all
        year. It does not: measured on these series, monthly residual spread runs
        about 3.4x the spread of the same model's error on a 16-month total,
        where 4.0x would mean the months cancel completely. So a band measured on
        the total wins wherever one exists, and it is much the narrower.
        """
        from core.intervals import IntervalBands

        band = IntervalBands()
        band.factors = {1: {'80': (0.9, 1.15)}}
        band.max_horizon = 1

        summed = {
            '2026-02': {'lower_80': 300, 'upper_80': 1800},
            '2026-03': {'lower_80': 2000, 'upper_80': 9000},
        }
        result = build_year_projections(
            {'2026-01': 4000},
            {'2026-02': 1000, '2026-03': 5000},
            intervals=summed,
            annual_bands={'2026': band},
        )
        proj = result[2026]
        # 6000 of forecast, banded as a whole: 4000 published + [0.9x, 1.15x].
        assert proj.lower_80 == 4000 + 5400
        assert proj.upper_80 == 4000 + 6900
        assert proj.lower_80 <= proj.total <= proj.upper_80
        # Narrower than what summing the months would have produced.
        assert (proj.upper_80 - proj.lower_80) < (summed['2026-02']['upper_80'] - summed['2026-02']['lower_80']) + (
            summed['2026-03']['upper_80'] - summed['2026-03']['lower_80']
        )

    def test_summing_still_applies_where_no_annual_band_was_measured(self):
        result = build_year_projections(
            {'2026-01': 4000},
            {'2026-02': 1000, '2026-03': 5000},
            intervals={
                '2026-02': {'lower_80': 800, 'upper_80': 1300},
                '2026-03': {'lower_80': 4200, 'upper_80': 6100},
            },
            annual_bands={'2029': None},
        )
        proj = result[2026]
        assert proj.lower_80 == 4000 + 800 + 4200
        assert proj.upper_80 == 4000 + 1300 + 6100


class TestSanityGuards:
    def test_plausible_forecast_passes_silently(self, constraints):
        proj = YearProjection(year=2026, actual_ytd=57872, forecast_remainder=27000)
        assert constraints.check_annual(proj, previous_year_total=48153) == []

    def test_2026_actual_growth_must_not_trip_the_guard(self, constraints):
        """2026 ran ~+137% year over year. A guard that flags reality is useless."""
        proj = YearProjection(year=2026, actual_ytd=57872, forecast_remainder=56493)
        assert constraints.check_annual(proj, previous_year_total=48153) == []

    def test_divergence_is_flagged(self, constraints):
        proj = YearProjection(year=2026, actual_ytd=0, forecast_remainder=5_000_000)
        assert constraints.check_annual(proj, previous_year_total=48153)

    def test_collapse_is_flagged(self, constraints):
        proj = YearProjection(year=2026, actual_ytd=0, forecast_remainder=100)
        assert constraints.check_annual(proj, previous_year_total=48153)

    def test_no_baseline_means_no_opinion(self, constraints):
        proj = YearProjection(year=2026, actual_ytd=0, forecast_remainder=999_999)
        assert constraints.check_annual(proj, previous_year_total=None) == []

    def test_monthly_spike_flagged(self, constraints):
        assert constraints.check_monthly([90000], recent_history=[10000, 11000, 12000])

    def test_monthly_within_range_passes(self, constraints):
        assert constraints.check_monthly([15000, 16000], recent_history=[10000, 11000, 12000]) == []

    def test_guards_can_be_disabled(self):
        disabled = ForecastConstraints({'enable_sanity_guards': False})
        proj = YearProjection(year=2026, actual_ytd=0, forecast_remainder=5_000_000)
        assert disabled.check_annual(proj, previous_year_total=48153) == []


class TestConfigWiring:
    def test_reads_its_own_block(self):
        c = ForecastConstraints({'max_annual_growth': 2.0, 'min_annual_growth': 0.5})
        assert c.max_annual_growth == 2.0
        assert c.min_annual_growth == 0.5

    def test_full_config_document_is_unwrapped_not_ignored(self):
        """v0.11 passed the whole config here and silently got defaults."""
        c = ForecastConstraints({'models': {}, 'forecast_constraints': {'max_annual_growth': 7.5}})
        assert c.max_annual_growth == 7.5


class TestEnsemble:
    def test_trimmed_mean_drops_the_extremes(self):
        per_model = {
            'a': {'2026-01': 100},
            'b': {'2026-01': 110},
            'c': {'2026-01': 120},
            'diverged': {'2026-01': 100000},
        }
        combined = combine_model_forecasts(per_model, method='trimmed_mean')
        assert combined['2026-01'] == pytest.approx(115.0)

    def test_members_restrict_the_pool(self):
        per_model = {'good': {'2026-01': 100}, 'bad': {'2026-01': 900}}
        assert combine_model_forecasts(per_model, members=['good'])['2026-01'] == pytest.approx(100.0)

    def test_small_pools_fall_back_to_median(self):
        per_model = {'a': {'2026-01': 100}, 'b': {'2026-01': 200}}
        assert combine_model_forecasts(per_model, method='trimmed_mean')['2026-01'] == pytest.approx(150.0)

    def test_empty_pool(self):
        assert combine_model_forecasts({}) == {}
