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
from darts import TimeSeries
from validation.rolling_origin import RollingOriginBacktest


class FakeHorizon:
    """A forecaster stub with only the horizon behaviour publication_windows needs."""

    def __init__(self, start, end):
        self._start, self._end = start, end

    def get_forecast_horizon(self):
        return self._start, self._end


def windows_for(year, month):
    """Publication windows as they would be derived in a given month."""
    from core.base_forecaster import BaseForecaster

    stub = FakeHorizon(
        datetime(year, month, 1, tzinfo=timezone.utc),
        datetime(year + 1, 12, 31, tzinfo=timezone.utc),
    )
    return BaseForecaster.publication_windows(stub)


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
