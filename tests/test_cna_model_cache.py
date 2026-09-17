"""
Tests for cached per-CNA model selection.

The cache exists to bound daily runtime: scoring every CNA by backtest on every
run measured at over an hour. These pin the properties that make that safe -
that the per-run cost really is capped, and that the entries most likely to be
wrong are the ones refreshed first.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
from cna_model_cache import SERIES_GROWTH_TRIGGER, ModelSelectionCache


@pytest.fixture
def cache(tmp_path):
    return ModelSelectionCache(path=str(tmp_path / 'sel.json'), refresh_days=30, max_refresh_per_run=3)


def age(cache, cna_id, days):
    """Backdate an entry's selection timestamp."""
    cache.entries[cna_id]['selected_at'] = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


class TestRefreshPlanning:
    def test_everything_is_new_on_a_cold_cache(self, cache):
        plan = cache.plan_refresh({'a': 30, 'b': 30, 'c': 30})
        assert sorted(plan) == ['a', 'b', 'c']

    def test_per_run_cap_is_respected(self, cache):
        candidates = {f'cna{i}': 30 for i in range(50)}
        assert len(cache.plan_refresh(candidates)) == 3

    def test_fresh_entries_are_left_alone(self, cache):
        for cna_id in ('a', 'b'):
            cache.put(cna_id, 'Prophet', 1.2, 30, {})
        assert cache.plan_refresh({'a': 30, 'b': 30}) == []

    def test_aged_entries_come_back_oldest_first(self, cache):
        for cna_id in ('a', 'b', 'c'):
            cache.put(cna_id, 'Prophet', 1.2, 30, {})
        age(cache, 'a', 40)
        age(cache, 'b', 100)
        age(cache, 'c', 60)

        assert cache.plan_refresh({'a': 30, 'b': 30, 'c': 30}) == ['b', 'c', 'a']

    def test_never_scored_outranks_merely_aged(self, cache):
        cache.put('old', 'Prophet', 1.2, 30, {})
        age(cache, 'old', 365)
        plan = cache.plan_refresh({'old': 30, 'brand_new': 30})
        assert plan[0] == 'brand_new'

    def test_substantial_new_history_forces_a_refresh(self, cache):
        cache.put('grown', 'Prophet', 1.2, 30, {})
        # Still fresh by age, but the series is materially longer now.
        assert cache.plan_refresh({'grown': 30 + SERIES_GROWTH_TRIGGER}) == ['grown']

    def test_small_growth_does_not(self, cache):
        cache.put('steady', 'Prophet', 1.2, 30, {})
        assert cache.plan_refresh({'steady': 31}) == []

    def test_corrupt_timestamp_is_treated_as_unscored(self, cache):
        cache.put('bad', 'Prophet', 1.2, 30, {})
        cache.entries['bad']['selected_at'] = 'not-a-date'
        assert cache.plan_refresh({'bad': 30}) == ['bad']


class TestColdStartContract:
    """
    On a cold cache the refresh cap means most entries are neither refreshed nor
    cached, so get() returns None for them. Callers must handle that - missing it
    crashed the first pipeline run with a TypeError.
    """

    def test_unrefreshed_entries_have_no_cached_selection(self, cache):
        candidates = {f'cna{i}': 30 for i in range(10)}
        plan = set(cache.plan_refresh(candidates))

        assert len(plan) == 3  # capped
        unrefreshed = [cid for cid in candidates if cid not in plan]
        assert len(unrefreshed) == 7
        assert all(cache.get(cid) is None for cid in unrefreshed)

    def test_cache_fills_over_successive_runs(self, cache):
        candidates = {f'cna{i}': 30 for i in range(10)}
        for _ in range(4):
            for cna_id in cache.plan_refresh(candidates):
                cache.put(cna_id, 'Prophet', 1.0, 30, {})

        # 10 CNAs at 3 per run: fully populated by the fourth run, none missed.
        assert len(cache.entries) == 10
        assert cache.plan_refresh(candidates) == []


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / 'sel.json')
        first = ModelSelectionCache(path=path)
        first.put('a', 'LightGBM', 1.42, 36, {'LightGBM': 1.42, 'NaiveDrift': 2.0})
        first.save()

        second = ModelSelectionCache(path=path)
        entry = second.get('a')
        assert entry['model'] == 'LightGBM'
        assert entry['mase'] == pytest.approx(1.42)
        assert entry['n_months'] == 36
        assert entry['all_scores']['NaiveDrift'] == pytest.approx(2.0)

    def test_missing_file_starts_empty(self, tmp_path):
        assert ModelSelectionCache(path=str(tmp_path / 'nope.json')).entries == {}

    def test_corrupt_file_starts_empty_rather_than_raising(self, tmp_path):
        path = tmp_path / 'sel.json'
        path.write_text('{ not json')
        assert ModelSelectionCache(path=str(path)).entries == {}

    def test_unknown_cna_returns_none(self, cache):
        assert cache.get('never-seen') is None

    def test_saved_file_records_its_settings(self, tmp_path):
        path = tmp_path / 'sel.json'
        cache = ModelSelectionCache(path=str(path), refresh_days=14)
        cache.put('a', 'Prophet', 1.0, 30, {})
        cache.save()
        assert json.loads(path.read_text())['refresh_days'] == 14


class TestCostIsBounded:
    def test_a_full_population_never_exceeds_the_cap(self, tmp_path):
        """The whole point: daily cost cannot scale with the number of CNAs."""
        cache = ModelSelectionCache(path=str(tmp_path / 'sel.json'), refresh_days=30, max_refresh_per_run=12)
        candidates = {f'cna{i}': 30 for i in range(144)}

        # Cold start still refreshes at most the cap, then the rest cycle in.
        for _ in range(5):
            plan = cache.plan_refresh(candidates)
            assert len(plan) <= 12
            for cna_id in plan:
                cache.put(cna_id, 'Prophet', 1.0, 30, {})

        assert len(cache.entries) == 60  # 5 runs x 12, no more
