"""
Tests for the CNA side of prediction intervals.

Covers the pieces that decide what actually reaches the page: which spans of the
forecast get measured as totals, when a forecast is rejected as a runaway, and
that the residuals a band is built from survive the cache.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from cna_model_cache import ModelSelectionCache
from core.base_forecaster import BaseForecaster
from darts import TimeSeries
from validation.rolling_origin import RollingOriginBacktest


class FakeHorizon(BaseForecaster):
    """
    A forecaster with only the horizon behaviour the span derivation needs.

    Subclassed rather than stubbed because cumulative_windows builds on
    publication_windows, so the two have to stay in step - which is the property
    several of these tests are about.
    """

    def __init__(self, start, end):
        self._start, self._end = start, end

    def get_forecast_horizon(self):
        return self._start, self._end

    # Unused here; declared because the base class requires them.
    def load_data(self):
        raise NotImplementedError

    def get_model_list(self):
        raise NotImplementedError

    def create_model(self, model_name, hyperparameters):
        raise NotImplementedError

    def apply_constraints(self, forecasts):
        raise NotImplementedError

    def save_results(self, forecasts):
        raise NotImplementedError


def _at(year, month):
    return FakeHorizon(
        datetime(year, month, 1, tzinfo=timezone.utc),
        datetime(year + 1, 12, 31, tzinfo=timezone.utc),
    )


def windows_for(year, month):
    """Publication windows as they would be derived in a given month."""
    return _at(year, month).publication_windows()


def cumulative_windows_for(year, month):
    """Cumulative spans as they would be derived in a given month."""
    return _at(year, month).cumulative_windows()


class TestPublicationWindows:
    """
    The forecast runs from the current month to the end of next year, so it is
    between 13 and 24 months long depending on when it is generated and the year
    boundary inside it moves with it. Hard-coding either would silently measure
    the wrong span for most of the year.
    """

    def test_september_splits_four_months_then_twelve(self):
        assert windows_for(2026, 9) == {'2026': (1, 4), '2027': (5, 16)}

    def test_january_is_a_full_year_then_a_full_year(self):
        assert windows_for(2026, 1) == {'2026': (1, 12), '2027': (13, 24)}

    def test_december_is_one_month_then_twelve(self):
        assert windows_for(2026, 12) == {'2026': (1, 1), '2027': (2, 13)}

    @pytest.mark.parametrize('month', range(1, 13))
    def test_windows_tile_the_forecast_without_gap_or_overlap(self, month):
        windows = windows_for(2026, month)
        spans = sorted(windows.values())
        assert spans[0][0] == 1
        for (_a, prev_end), (next_start, _b) in zip(spans, spans[1:]):
            assert next_start == prev_end + 1
        # The last window ends at the forecast's last month.
        assert spans[-1][1] == 25 - month

    @pytest.mark.parametrize('month', range(1, 13))
    def test_next_year_is_always_a_full_twelve_months(self, month):
        first, last = windows_for(2026, month)['2027']
        assert last - first + 1 == 12


class TestWindowResiduals:
    """A total is scored as a total, because the error on a sum is not the sum of
    the errors on its parts."""

    def test_a_window_is_scored_on_the_sum_not_the_months(self):
        # A series a drift model tracks in aggregate while missing month to
        # month: the monthly errors are large and cancel, the annual one is small.
        rng = np.random.default_rng(0)
        values = [100 + 30 * ((-1) ** i) + rng.normal(0, 5) for i in range(60)]
        idx = pd.date_range('2020-01-01', periods=60, freq='MS')
        series = TimeSeries.from_dataframe(pd.DataFrame({'value': values}, index=idx), freq='MS')

        bt = RollingOriginBacktest(horizon=12, min_train=24, step=1, max_origins=12)
        result = bt.evaluate(
            series,
            lambda train, h: TimeSeries.from_times_and_values(
                pd.date_range(train.end_time() + pd.DateOffset(months=1), periods=h, freq='MS'),
                np.full(h, 100.0),
            ),
            'flat',
            windows={'year': (1, 12)},
        )

        monthly = np.concatenate([np.asarray(v) for v in result.log_residuals_by_horizon.values()])
        annual = np.asarray(result.log_residuals_by_window['year'])
        assert annual.std() < monthly.std(), 'the alternating months should cancel over a year'

    def test_a_window_the_origin_cannot_see_all_of_is_not_scored(self):
        idx = pd.date_range('2020-01-01', periods=30, freq='MS')
        series = TimeSeries.from_dataframe(pd.DataFrame({'value': [100.0] * 30}, index=idx), freq='MS')
        bt = RollingOriginBacktest(horizon=12, min_train=24, step=1, max_origins=12)
        result = bt.evaluate(
            series,
            lambda train, h: TimeSeries.from_times_and_values(
                pd.date_range(train.end_time() + pd.DateOffset(months=1), periods=h, freq='MS'),
                np.full(h, 100.0),
            ),
            'flat',
            windows={'year': (1, 12)},
        )
        # Origins 24..29 leave 6..1 months to score, never a full twelve, so a
        # part-observed window must not be recorded as a large under-forecast.
        assert result.log_residuals_by_window.get('year', []) == []


class TestCachedResiduals:
    """
    Scoring runs for at most a dozen CNAs a run, so residuals that are not
    persisted leave the other ~128 with no band and rotate which CNAs have one.
    """

    def test_residuals_survive_a_save_and_reload(self, tmp_path):
        path = str(tmp_path / 'sel.json')
        first = ModelSelectionCache(path=path)
        first.put(
            'a',
            'Prophet',
            1.2,
            60,
            {},
            residuals={1: [0.1, -0.2], 2: [0.3]},
            window_residuals={'2027': [0.05, -0.04]},
        )
        first.save()

        entry = ModelSelectionCache(path=path).get('a')
        assert entry['log_residuals'] == {'1': [0.1, -0.2], '2': [0.3]}
        assert entry['log_residuals_by_window'] == {'2027': [0.05, -0.04]}

    def test_scoring_that_found_nothing_still_records_that_it_ran(self, tmp_path):
        """
        Empty, not absent. An entry that predates residual caching jumps the
        refresh queue; one whose backtest simply produced nothing must not, or
        it would be re-scored every run and starve the rest of the cap. An
        absent key is what tells the two apart.
        """
        cache = ModelSelectionCache(path=str(tmp_path / 'sel.json'))
        cache.put('a', 'Prophet', 1.2, 60, {})
        assert cache.get('a')['log_residuals'] == {}
        assert cache.get('a')['log_residuals_by_window'] == {}
        assert cache.plan_refresh({'a': 60}) == []

    def test_an_entry_from_before_residuals_were_cached_jumps_the_queue(self, tmp_path):
        """
        Otherwise the whole population sits unrefreshed until it ages past
        refresh_days: not new, not grown, not old. A release that starts caching
        residuals would publish no intervals at all for a month, then begin
        filling - six weeks before the feature is visible anywhere.
        """
        cache = ModelSelectionCache(path=str(tmp_path / 'sel.json'))
        cache.put('legacy', 'Prophet', 1.2, 60, {})
        del cache.entries['legacy']['log_residuals']
        assert cache.plan_refresh({'legacy': 60}) == ['legacy']

    def test_entries_written_before_residuals_existed_still_load(self, tmp_path):
        import json

        path = tmp_path / 'sel.json'
        path.write_text(
            json.dumps(
                {
                    'version': '1.0',
                    'selections': {
                        'a': {
                            'model': 'Prophet',
                            'mase': 1.2,
                            'n_months': 60,
                            'selected_at': datetime.now(timezone.utc).isoformat(),
                            'all_scores': {},
                        }
                    },
                }
            )
        )
        entry = ModelSelectionCache(path=str(path)).get('a')
        assert entry['model'] == 'Prophet'
        assert entry.get('log_residuals') is None


class TestSpanCoverage:
    """
    A CNA is banded only where its cached residuals cover every span the run
    publishes. Partial cover is how the headline and the chart come to disagree.
    """

    def test_windows_cover_both_the_year_and_every_running_total(self):
        year_spans = windows_for(2026, 9)
        cumulative = cumulative_windows_for(2026, 9)

        # Every running total starts where its year starts, so a year's last
        # running total IS that year's span - the chart closes on the same
        # measurement the headline is made of, rather than one kept in step.
        for year, span in year_spans.items():
            closing = [k for k, v in cumulative.items() if k.startswith(year) and v == span]
            assert closing, f'{year} has no running total covering its whole span'

    def test_a_running_total_never_reaches_past_its_own_year(self):
        cumulative = cumulative_windows_for(2026, 9)
        year_spans = windows_for(2026, 9)
        for name, (first, last) in cumulative.items():
            year = name[:4]
            assert (first, last) >= (year_spans[year][0], first)
            assert last <= year_spans[year][1]

    @pytest.mark.parametrize('month', range(1, 13))
    def test_running_totals_exist_for_every_forecast_month(self, month):
        cumulative = cumulative_windows_for(2026, month)
        assert len(cumulative) == 25 - month


class TestConeWidening:
    """
    Each running total is fitted from its own handful of residuals, so the cone
    drawn through them wobbles. A reader cannot be told that eleven months of a
    year are more certain than four of it.
    """

    @staticmethod
    def band(lo, hi):
        from core.intervals import IntervalBands

        b = IntervalBands()
        b.factors = {1: {'80': (lo, hi)}}
        b.max_horizon = 1
        return b

    def widened(self, bands, forecasts):
        from adapters.cna_adapter import CNAForecaster

        return CNAForecaster._widen_along_the_year(bands, forecasts)

    def test_the_cone_never_narrows_as_the_year_fills_in(self):
        # A wide middle span followed by a tight one, which is what sampling
        # noise across separately fitted spans produces.
        bands = {
            '2027-01': self.band(0.8, 1.2),
            '2027-02': self.band(0.5, 1.5),
            '2027-03': self.band(0.97, 1.03),
            '2027': self.band(0.97, 1.03),
        }
        forecasts = {'2027-01-01': 100.0, '2027-02-01': 100.0, '2027-03-01': 100.0}
        out = self.widened(bands, forecasts)

        widths = []
        total = 0
        for month in ('2027-01', '2027-02', '2027-03'):
            total += 100
            lo, hi = out[month].for_horizon(1)['80']
            widths.append(total * (hi - lo))
        # Tolerant of float noise: the widening lands on the running maximum to
        # within 1e-9, and the published figures are rounded to whole CVEs.
        for earlier, later in zip(widths, widths[1:]):
            assert later >= earlier - 1e-6, widths

    def test_the_year_keeps_the_closing_span_so_the_headline_agrees(self):
        bands = {
            '2027-01': self.band(0.5, 1.5),
            '2027-02': self.band(0.98, 1.02),
            '2027': self.band(0.98, 1.02),
        }
        out = self.widened(bands, {'2027-01-01': 100.0, '2027-02-01': 100.0})
        assert out['2027'].for_horizon(1)['80'] == out['2027-02'].for_horizon(1)['80']

    def test_widening_only_ever_widens(self):
        bands = {
            '2027-01': self.band(0.9, 1.1),
            '2027-02': self.band(0.9, 1.1),
            '2027': self.band(0.9, 1.1),
        }
        forecasts = {'2027-01-01': 100.0, '2027-02-01': 100.0}
        out = self.widened(bands, forecasts)
        for month in ('2027-01', '2027-02'):
            lo, hi = out[month].for_horizon(1)['80']
            before_lo, before_hi = bands[month].for_horizon(1)['80']
            assert lo <= before_lo and hi >= before_hi

    def test_a_year_with_no_running_totals_is_left_alone(self):
        bands = {'2027': self.band(0.9, 1.1)}
        out = self.widened(bands, {'2027-01-01': 100.0})
        assert out['2027'].for_horizon(1)['80'] == (0.9, 1.1)


class TestNowcast:
    """
    Training stops at the last complete month, so the forecast opens on the month
    in progress and predicts all of it - including the part already published.
    Counting both counted that part twice; counting only the forecast threw it
    away, and 46 of 140 CNAs then showed less for this month than was already
    out. The main pipeline has always nowcast instead.
    """

    @staticmethod
    def nowcast(values, now):
        import adapters.cna_adapter as mod
        from adapters.cna_adapter import CNAForecaster

        fc = CNAForecaster.__new__(CNAForecaster)
        real = mod.datetime

        class Frozen:
            @staticmethod
            def now(tz=None):
                return now

        mod.datetime = Frozen
        try:
            return fc._nowcast(values)
        finally:
            mod.datetime = real

    def test_the_month_in_progress_is_cut_to_its_remainder(self):
        now = datetime(2026, 9, 18, tzinfo=timezone.utc)  # 14 of 22 business days
        out = self.nowcast({'2026-09-01': 100, '2026-10-01': 100}, now)
        assert out['2026-09-01'] == 36
        assert out['2026-10-01'] == 100, 'later months are untouched'

    def test_nothing_is_left_of_a_month_that_is_over(self):
        now = datetime(2026, 9, 30, tzinfo=timezone.utc)
        assert self.nowcast({'2026-09-01': 100}, now)['2026-09-01'] == 0

    def test_a_forecast_that_does_not_reach_this_month_is_untouched(self):
        # 22 of 140 CNAs forecast from an earlier month because their history
        # stops early; some end before the current month entirely.
        now = datetime(2026, 9, 18, tzinfo=timezone.utc)
        values = {'2022-04-01': 10, '2022-05-01': 12}
        assert self.nowcast(values, now) == values


class TestSpanDrift:
    """
    Spans move with the calendar: the rest of 2026 is h1-4 in September, h1-3 in
    October. A band measured in September is keyed for a span that does not exist
    a month later, so requiring an exact match emptied every page on the first of
    each month and kept it empty for the twelve runs it takes to re-score the
    population.
    """

    @staticmethod
    def band(width=0.2):
        from core.intervals import IntervalBands

        b = IntervalBands()
        b.factors = {1: {'80': (1 - width, 1 + width)}}
        b.max_horizon = 1
        return b

    def nearest(self, needed, fitted):
        from adapters.cna_adapter import CNAForecaster

        return CNAForecaster._nearest_span(needed, fitted)

    def test_last_months_span_is_reused_for_this_months(self):
        # September measured h1-4; October wants h1-3.
        september = {'h1-4': self.band()}
        assert self.nearest((1, 3), september) is september['h1-4']

    def test_next_years_span_is_reused_when_the_whole_window_shifts(self):
        # h5-16 measured in September; October wants h4-15 - same length, moved.
        september = {'h5-16': self.band()}
        assert self.nearest((4, 15), september) is september['h5-16']

    def test_the_closest_span_wins(self):
        near, far = self.band(0.1), self.band(0.9)
        fitted = {'h1-3': near, 'h1-1': far}
        assert self.nearest((1, 4), fitted) is near

    def test_an_exact_match_is_preferred_over_a_neighbour(self):
        exact, neighbour = self.band(0.1), self.band(0.9)
        fitted = {'h1-4': exact, 'h1-3': neighbour}
        assert self.nearest((1, 4), fitted) is exact

    def test_a_span_too_far_away_is_not_reused(self):
        from adapters.cna_adapter import MAX_SPAN_DRIFT

        # Beyond the tolerance the band describes a materially different
        # quantity, and absent beats wrong.
        stale = {f'h1-{1 + MAX_SPAN_DRIFT + 3}': self.band()}
        assert self.nearest((1, 1), stale) is None

    def test_a_year_label_from_an_older_cache_is_not_mistaken_for_a_span(self):
        # Entries written before spans were named by horizon carry '2026'.
        assert self.nearest((1, 4), {'2026': self.band()}) is None

    def test_nothing_fitted_means_nothing_reused(self):
        assert self.nearest((1, 4), {}) is None
