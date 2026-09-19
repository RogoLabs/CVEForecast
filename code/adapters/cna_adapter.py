"""
CNA Forecaster Adapter - CNA-specific implementation of forecasting system.

Extends BaseForecaster to handle per-CNA forecasting with model selection.
"""

import json
import os
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from cna_model_cache import ModelSelectionCache
from core.base_forecaster import BaseForecaster, ForecastResult
from core.forecast_engine import ForecastEngine, ForecastSettings
from core.intervals import IntervalBands, apply_intervals, build_shared_shape, scale_bands, validate_coverage
from core.model_utils import create_model_safe
from darts import TimeSeries
from darts.models import AutoARIMA, ExponentialSmoothing, LightGBMModel, LinearRegressionModel, Prophet, XGBModel
from darts.models.forecasting.baselines import NaiveDrift, NaiveMean, NaiveSeasonal
from forecast_constraints import build_cumulative_band, business_day_share
from validation.rolling_origin import BacktestResult, RollingOriginBacktest, mark_naive_baselines, rank_models

# What a CNA falls back to when no model beats it. NaiveDrift extrapolates the
# recent level, which is the right default for a short, noisy series - and is an
# honest answer where v0.11 reported a 160%-error model as the "best" one.
FALLBACK_MODEL = 'NaiveDrift'

# A forecast month above this multiple of the CNA's trailing 24-month peak is
# treated as a runaway rather than a prediction. Derived from the series
# themselves: across 6,858 CNA-months with a meaningful prior peak, the 99.9th
# percentile of actual growth over that peak is 5.0x, and 8x has been exceeded
# three times - 0.04% of months. Set against real blowups, which are not close:
# TR-CERT was published forecasting 1,018,180 CVEs for a single month against an
# all-time monthly peak of 70, a ratio of 14,545x.
RUNAWAY_CEILING = 8.0

# Months of history the ceiling is measured against. Matches cna_min_train, so
# every CNA eligible to be forecast has at least this much.
RUNAWAY_LOOKBACK = 24

# An annual 80% interval wider than this is not published: past here the band is
# a measurement artefact rather than a wide measurement. A CNA's annual width
# comes from the handful of origins far enough from the end to have seen a whole
# year, so one pathological origin moves it a long way - a model compounding a
# trend in log space can forecast 10^13 times the actual.
#
# Set where the data separates rather than by taste. Sorted, every fitted band
# runs 11.3x, 20.1x, 23.2x, 66.4x, then 179,429x, 15,436,783x, 23,208,181x.
# Nothing real spans twenty-three million fold, and the gap either side of 100x
# is a factor of 2,700 - so a cut there divides wide-but-measured from broken,
# where the 20x this started at divided nothing in particular and dropped three
# bands that were merely wide.
#
# For scale, the widest band currently published is apache's 2027 at 11.2x
# (1,291 to 14,502) against a median of 1.69x, so this is a backstop rather than
# something shaping what a reader usually sees.
MAX_INFORMATIVE_ANNUAL_RATIO = 100.0

# A CNA with nothing published in this long is not forecast. Its series has no
# recent level to extrapolate from, and running a model over the gap produces a
# flat line that looks like a prediction and is really a four-year-old
# extrapolation: pivotal last published in March 2022, Mend in October 2022.
#
# Twelve months separates the three dormant CNAs from every active one cleanly -
# the next quietest, Liferay, is ten months idle but published 92 CVEs inside the
# last year.
DORMANT_AFTER_MONTHS = 12

# How far a cached span may sit from the one a run needs before its band is
# dropped rather than reused, counted in months of start drift plus months of
# length difference.
#
# Spans move every month: the rest of 2026 is h1-4 in September, h1-3 in October,
# h1-2 in November. A band measured in September is therefore keyed for a span
# that does not exist a month later, and requiring an exact match emptied every
# band on the first of each month and left it empty for the twelve runs it takes
# to re-score the population.
#
# Reusing a near neighbour instead costs a little accuracy in a known direction:
# a four-month band carried onto a three-month remainder is about sqrt(4/3), some
# 15%, too wide, because a shorter total has less month-to-month error to cancel.
# Too wide is the safe direction, and selection refreshes every 30 days, so drift
# is normally one month and two is the most a catch-up should produce.
MAX_SPAN_DRIFT = 2


class CNAForecaster(BaseForecaster):
    """
    CNA-specific forecasting implementation.

    Handles per-CNA forecasting with automatic model selection based
    on validation performance for each CNA.
    """

    def __init__(self, config_path: str = 'cna_config.json', cvelist_dir: str = 'cvelistV5', min_cves: int = 100):
        """
        Initialize CNA forecaster.

        Args:
            config_path: Path to CNA configuration
            cvelist_dir: Path to cvelistV5 repository
            min_cves: Minimum CVEs for CNA inclusion
        """
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        super().__init__(config)

        self.cvelist_dir = cvelist_dir
        self.min_cves = min_cves

        # Same forecast path as the CVE pipeline: log space, business-day
        # normalisation, damping. v0.11 fitted CNA models raw, so the two halves
        # of the site were forecasting by different methods.
        self.settings = ForecastSettings.from_config(self.config)
        self.engine = ForecastEngine(self.settings, self.create_model)

        # CNA series are short and spiky, so fewer origins and a shorter minimum
        # than the main pipeline - otherwise most CNAs score nothing at all.
        cv = self.config.get('cross_validation', {})
        self.cv_horizon = cv.get('cna_horizon', 6)
        self.cv_min_train = cv.get('cna_min_train', 24)
        self.cv_max_origins = cv.get('cna_max_origins', 8)

        # Scoring every CNA by backtest on every run measured at over an hour.
        # Model choice is cached and refreshed a few CNAs at a time instead.
        paths = self.config.get('file_paths', {})
        self.selection_cache = ModelSelectionCache(
            path=paths.get('cna_model_selection', 'web/cna_model_selection.json'),
            refresh_days=cv.get('cna_refresh_days', 30),
            max_refresh_per_run=cv.get('cna_max_refresh_per_run', 12),
        )

        # CNA-specific attributes
        self.cna_data = {}  # {cna_id: {historical: series, name: str}}
        self.cna_names = {}  # {org_id: short_name}

        self.logger.info(f'CNA Forecaster initialized (min CVEs: {min_cves})')

    def load_data(self) -> TimeSeries:
        """
        Load and parse CVE data from cvelistV5 for all CNAs.

        Returns:
            Combined TimeSeries (not used for CNA, returns None)
        """
        self.logger.info(f'Scanning CVE data from {self.cvelist_dir}...')

        # Scan cvelist for CNA data
        df, names = self._scan_cvelist_for_cna_counts()
        self.cna_names = names

        # Filter CNAs by minimum CVE count
        cna_counts = df['org_id'].value_counts()
        eligible_cnas = cna_counts[cna_counts >= self.min_cves].index.tolist()

        self.logger.info(f'✓ Found {len(eligible_cnas)} CNAs with ≥{self.min_cves} CVEs')

        # Build time series for each eligible CNA
        for cna_id in eligible_cnas:
            counts, current_partial = self._build_monthly_series(df, cna_id)

            if len(counts) < 12:  # Need minimum history
                continue

            ts = self._series_to_darts(counts)

            self.cna_data[cna_id] = {
                'historical': ts,
                'name': self.cna_names.get(cna_id, cna_id),
                'total_cves': int(counts.sum()),
                'current_month_partial': current_partial,
            }

        self.logger.info(f'✓ Loaded data for {len(self.cna_data)} CNAs')

        # For compatibility, set series to None (CNA doesn't use combined series)
        self.series = None

        return None

    def _scan_cvelist_for_cna_counts(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Scan cvelistV5 and extract CNA publication data."""
        pattern = os.path.join(self.cvelist_dir, 'cves', '*', '*', 'CVE-*.json')
        paths = glob(pattern)

        self.logger.info(f'Scanning {len(paths)} CVE files...')

        rows = []
        names = {}

        for path in paths:
            parsed = self._parse_cve_file(path)
            if parsed:
                published, org_id, short_name = parsed
                rows.append((org_id, pd.to_datetime(published).tz_localize(None)))
                if short_name and org_id not in names:
                    names[org_id] = short_name

        df = pd.DataFrame(rows, columns=['org_id', 'date']) if rows else pd.DataFrame(columns=['org_id', 'date'])

        return df, names

    def _parse_cve_file(self, path: str) -> Optional[Tuple[datetime, str, Optional[str]]]:
        """Parse single CVE file for CNA data."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            meta = data.get('cveMetadata', {})
            date_str = meta.get('datePublished') or meta.get('datePublic')
            if not date_str:
                return None

            try:
                ts = pd.to_datetime(date_str, utc=True).tz_convert(None)
                published = ts.to_pydatetime()
            except Exception:
                return None

            containers = data.get('containers', {})
            cna = containers.get('cna', {})
            provider = cna.get('providerMetadata', {}) if isinstance(cna, dict) else {}

            org_id = provider.get('orgId') or meta.get('assignerOrgId')
            short_name = provider.get('shortName') or meta.get('assignerShortName')

            if not org_id:
                return None

            return published, org_id, short_name

        except Exception:
            return None

    def _build_monthly_series(self, df: pd.DataFrame, org_id: str) -> Tuple[pd.Series, int]:
        """Build monthly time series for a CNA."""
        sub = df[df['org_id'] == org_id].copy()
        if sub.empty:
            return pd.Series(dtype=float), 0

        # Limit to 2017-01-01 onwards
        start_cutoff = pd.Timestamp('2017-01-01')
        sub = sub[sub['date'] >= start_cutoff]

        if sub.empty:
            return pd.Series(dtype=float), 0

        data_start = sub['date'].min().to_period('M').to_timestamp(how='start')
        start = max(start_cutoff, data_start)
        end = sub['date'].max().to_period('M').to_timestamp(how='start')

        counts = sub.set_index('date').resample('MS').size()
        full_index = pd.date_range(start=start, end=end, freq='MS')
        counts = counts.reindex(full_index, fill_value=0).astype(float)
        counts.index.name = 'date'

        # Current month partial
        current_month_start = pd.Timestamp.now().to_period('M').to_timestamp(how='start')
        current_month_data = sub[sub['date'] >= current_month_start]
        current_month_partial = len(current_month_data)

        return counts, current_month_partial

    def _series_to_darts(self, counts: pd.Series) -> TimeSeries:
        """Convert pandas Series to Darts TimeSeries."""
        df = counts.reset_index()
        df.columns = ['date', 'value']
        return TimeSeries.from_dataframe(df, time_col='date', value_cols='value', fill_missing_dates=True, freq='MS')

    def get_forecast_horizon(self) -> Tuple[datetime, datetime]:
        """
        Determine CNA forecast period.
        Start from CURRENT month (even though incomplete) to match CVE adapter behavior.
        Training excludes current month, but model predicts from current month onwards.

        Returns:
            Tuple of (start_date, end_date)
        """
        now = datetime.now(timezone.utc)
        current_year = now.year
        current_month = now.month

        # Start forecast from CURRENT month (match CVE adapter)
        # Even though current month is incomplete, we include its forecast
        # Forecast through end of next year
        start_date = datetime(current_year, current_month, 1, tzinfo=timezone.utc)
        end_date = datetime(current_year + 1, 12, 31, tzinfo=timezone.utc)

        self.logger.info(f'CNA Forecast horizon: {start_date} to {end_date} (current month: {current_month})')
        return start_date, end_date

    def get_model_list(self) -> List[str]:
        """
        Get list of models for CNA forecasting.

        Returns:
            List of fast, CPU-only models
        """
        return ['ExponentialSmoothing', 'LightGBM', 'XGBoost', 'LinearRegression', 'Prophet']

    def create_model(self, model_name: str, hyperparameters: Dict[str, Any]):
        """
        Create CNA forecast model instance.

        Args:
            model_name: Model name
            hyperparameters: Model hyperparameters

        Returns:
            Configured model instance
        """
        model_classes = {
            'Prophet': Prophet,
            'ExponentialSmoothing': ExponentialSmoothing,
            'AutoARIMA': AutoARIMA,
            'LightGBM': LightGBMModel,
            'XGBoost': XGBModel,
            'LinearRegression': LinearRegressionModel,
            # The baseline every other model has to clear.
            'NaiveDrift': NaiveDrift,
            'NaiveSeasonal': NaiveSeasonal,
            'NaiveMean': NaiveMean,
        }

        if model_name not in model_classes:
            raise ValueError(f'Unknown CNA model: {model_name}')

        return create_model_safe(model_classes[model_name], model_name, hyperparameters, self.logger)

    def _complete_months(self, series: TimeSeries) -> Optional[TimeSeries]:
        """
        Drop the month currently in progress.

        Args:
            series: Historical series for one CNA

        Returns:
            Series of complete months, or None if nothing is left
        """
        cutoff = pd.Timestamp.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        frame = series.to_dataframe()
        complete = frame[frame.index < cutoff]
        if complete.empty:
            return None
        if len(complete) == len(frame):
            return series
        return TimeSeries.from_dataframe(complete, freq=series.freq_str, fill_missing_dates=False)

    def select_best_model_for_cna(
        self, cna_id: str, ts: TimeSeries
    ) -> Tuple[str, float, Dict[str, float], Optional[BacktestResult]]:
        """
        Pick a model for one CNA by rolling-origin MASE.

        v0.11 scored each model on a single 6-month holdout by MAPE and took the
        minimum, independently for ~140 CNAs across 5 models - roughly 700
        comparisons on 6 points each. The winner was largely sampling noise, and
        the headline it produced said so: Patchstack's "best" model carried a
        validation MAPE of 160%.

        Scoring now uses the same rolling-origin backtest as the main pipeline,
        and a naive baseline is scored alongside. A CNA whose best model cannot
        beat the baseline gets the baseline, which is the honest outcome for a
        short, spiky series.

        Args:
            cna_id: CNA identifier
            ts: Historical time series for this CNA

        Returns:
            Tuple of (best_model_name, best_mase, all_scores, backtest).

            ``backtest`` is the winning model's full result, carrying the
            out-of-sample residuals the MASE is computed from. Through v0.13 only
            the MASE was read off it and the rest was dropped on the floor, which
            is why the CNA pages had no interval to publish.
        """
        backtest = RollingOriginBacktest(
            horizon=self.cv_horizon,
            min_train=self.cv_min_train,
            step=1,
            max_origins=self.cv_max_origins,
        )

        candidates = list(self.get_model_list())
        if FALLBACK_MODEL not in candidates:
            candidates.append(FALLBACK_MODEL)

        # The year spans the headline needs, and the running totals the chart
        # draws. Both are sums, so both are scored as sums, and both are named by
        # horizon because this CNA's forecast may not start where the horizon does.
        windows, _labels = self._spans_for(ts)

        results = {}
        for model_name in candidates:
            hyperparameters = self.config.get('models', {}).get(model_name, {}).get('hyperparameters', {})

            def forecast_fn(train, horizon, _name=model_name, _hp=hyperparameters):
                attempt = self.engine.forecast(train, _name, _hp, horizon)
                return attempt.forecast if attempt.ok else None

            results[model_name] = backtest.evaluate(ts, forecast_fn, model_name, windows=windows)

        mark_naive_baselines(results)
        scores = {name: (r.mase if r.is_valid else None) for name, r in results.items()}

        ranked = [r for r in rank_models(results) if r.is_valid]
        if not ranked:
            return FALLBACK_MODEL, float('inf'), scores, None

        best = ranked[0]
        baseline = results.get(FALLBACK_MODEL)
        # Prefer the baseline when nothing beats it: a model chosen from a field
        # that all lost is the least-bad noise, not a signal.
        if baseline and baseline.is_valid and best.model_name != FALLBACK_MODEL and best.mase >= baseline.mase:
            self.logger.debug(f'{cna_id}: no model beat {FALLBACK_MODEL}; using the baseline')
            return FALLBACK_MODEL, baseline.mase, scores, baseline

        # The residuals must come from the model actually being published. A band
        # built from the winner's errors and drawn around the baseline's forecast
        # would describe a forecast nobody sees.
        return best.model_name, best.mase, scores, best

    def _is_runaway(self, forecast_values: Dict[str, Any], ts: TimeSeries) -> Optional[str]:
        """
        Whether a forecast has left the range the series could plausibly reach.

        A model fitted in log space can compound a positive trend into a number
        with no relation to the CNA that produced it, and MASE will not catch it:
        selection scores the first months of the path, where the compounding has
        barely begun, and the divergence happens out at the end where nothing was
        measured. TR-CERT was live with 1,018,180 CVEs forecast for December 2027
        against 809 in its entire history, carrying a respectable MASE of 1.79.

        Args:
            forecast_values: ``{date_string: value}``
            ts: The CNA's history

        Returns:
            A reason string when the forecast runs away, None when it is fine
        """
        values = ts.values().flatten()
        peak = float(np.max(values[-RUNAWAY_LOOKBACK:])) if len(values) else 0.0
        if peak <= 0:
            return None

        ceiling = peak * RUNAWAY_CEILING
        worst = max((float(v) for v in forecast_values.values()), default=0.0)
        if worst <= ceiling:
            return None
        return f'peaks at {worst:,.0f} against a {RUNAWAY_LOOKBACK}-month high of {peak:,.0f} ({worst / peak:,.0f}x)'

    def _months_idle(self, ts: TimeSeries, now: datetime) -> int:
        """
        Months since this CNA last published anything.

        Args:
            ts: The CNA's history, complete months only
            now: The moment to measure back from

        Returns:
            Months since the last month with a publication, or a large number
            when the series is empty
        """
        values = ts.values().flatten()
        published = [i for i, v in enumerate(values) if v > 0]
        if not published:
            return 10**6

        last = ts.time_index[published[-1]]
        return (now.year - last.year) * 12 + (now.month - last.month)

    def _nowcast(self, forecast_values: Dict[str, int]) -> Dict[str, int]:
        """
        Replace the in-progress month's full-month forecast with its remainder.

        Training stops at the last complete month, so the forecast opens on the
        month in progress and predicts the whole of it - including the part
        already published. Counting both counted that part twice; counting only
        the forecast threw away real data, and for 46 of 140 CNAs the page then
        showed a smaller figure for this month than had already been published.
        Linux had 1,509 September CVEs out and the page said 639.

        The main pipeline has always nowcast this month rather than choosing
        between the two: what is out stays, and the forecast contributes only
        what is left. This does the same, scaled by business days elapsed,
        because publication happens on working days.

        Args:
            forecast_values: ``{date_string: full-month forecast}``

        Returns:
            The same mapping with the current month cut to its remainder
        """
        now = datetime.now(timezone.utc)
        current = now.strftime('%Y-%m')
        remaining = max(0.0, 1.0 - business_day_share(now))

        return {
            date: (max(0, round(value * remaining)) if str(date)[:7] == current else value)
            for date, value in forecast_values.items()
        }

    def _spans_for(self, ts: TimeSeries) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, str]]:
        """
        The spans this CNA publishes, named by horizon rather than by year.

        A forecast runs from the end of the series it was fitted to, so a CNA
        that has not published for a month or two starts its forecast earlier
        than everyone else and its calendar years sit at different horizons. 22
        of 140 are in that position. Naming a span '2027' would therefore mean a
        different quantity for different CNAs, and pooling them to fit a shared
        shape would be pooling unlike things.

        Named 'h5-16' instead, which is what the residual actually measures, so
        the pool holds one quantity. Each CNA keeps its own map from the year it
        publishes to the span that carries it.

        Args:
            ts: This CNA's history, whose end is where its forecast begins

        Returns:
            (spans, by_label) - ``{'h5-16': (5, 16)}`` and
            ``{'2027': 'h5-16', '2027-03': 'h5-7'}``
        """
        start = (ts.end_time() + pd.DateOffset(months=1)).to_pydatetime().replace(tzinfo=timezone.utc)
        named = {**self.publication_windows(start), **self.cumulative_windows(start)}

        spans: Dict[str, Tuple[int, int]] = {}
        by_label: Dict[str, str] = {}
        for label, (first, last) in named.items():
            key = f'h{first}-{last}'
            spans[key] = (first, last)
            by_label[label] = key
        return spans, by_label

    def _build_interval_bands(self) -> Tuple[Dict[str, IntervalBands], Dict[str, Any]]:
        """
        Build every CNA's prediction-interval band from the cached residuals.

        One band per CNA, but not built from that CNA alone. The shape - how the
        band widens with horizon - is pooled across the population; only its
        width is this CNA's own. Measured on the real residuals, that beats both
        alternatives and is the only one that stays calibrated at long horizons:

            method                    mean |per-CNA coverage - 80%|
            this one                   7.7pp  [5.8, 10.1]
            one pooled band for all   11.6pp  [9.4, 13.7]
            each CNA's own residuals  19.1pp  [15.3, 23.4]

        The reason is in how a rolling-origin backtest runs out of data. Its
        newest origin can only be scored at h=1, the next at h=1..2, so horizon h
        holds at most ``max_origins - h + 1`` residuals however long the series
        is. At the default eight origins that is eight residuals at h=1 and one
        at h=16 - so a per-CNA band has nothing to fit at exactly the horizons
        the next-year column is built from, while a pooled shape has thousands.
        A single pooled band avoids that and gets the width wrong instead:
        measured dispersion varies about fivefold between CNAs.

        Returns:
            ({cna_id: bands}, coverage). Both empty when the cache holds too
            little to fit a shape, which publishes no intervals at all rather
            than intervals nobody measured.
        """
        residuals_by_cna: Dict[str, Dict[int, List[float]]] = {}
        for cna_id, entry in self.selection_cache.entries.items():
            stored = entry.get('log_residuals') or {}
            by_horizon = {int(h): list(v) for h, v in stored.items() if v}
            if by_horizon:
                residuals_by_cna[cna_id] = by_horizon

        if not residuals_by_cna:
            self.logger.info('No cached backtest residuals yet; CNA intervals unavailable this run')
            return {}, {}

        shape, scales = build_shared_shape(residuals_by_cna)
        if not shape.factors:
            return {}, {}

        bands = {cna_id: scale_bands(shape, scale) for cna_id, scale in scales.items()}
        bands = {cna_id: b for cna_id, b in bands.items() if b.factors}

        # Coverage is measured per CNA against that CNA's own band, then pooled -
        # the question a reader has is whether the band on the page they are
        # looking at holds, not whether the population averages out.
        covered: Dict[str, List[int]] = {}
        for cna_id, band in bands.items():
            stats = validate_coverage(residuals_by_cna[cna_id], band)
            for label, row in stats.items():
                acc = covered.setdefault(label, [0, 0])
                acc[0] += int(round(row['empirical'] * row['n']))
                acc[1] += row['n']

        coverage = {
            label: {
                'nominal': float(label) / 100.0,
                'empirical': round(hit / total, 4),
                'n': total,
                'n_cnas': len(bands),
                'calibrated': bool(abs(hit / total - float(label) / 100.0) <= 0.05),
            }
            for label, (hit, total) in covered.items()
            if total
        }

        self.logger.info(
            f'Built interval bands for {len(bands)}/{len(self.cna_data)} CNAs '
            f'to h={shape.max_horizon}'
            + ''.join(f' | {k}% coverage {v["empirical"]:.1%}' for k, v in sorted(coverage.items()))
        )
        return bands, coverage

    def _build_annual_bands(self) -> Dict[str, Dict[str, IntervalBands]]:
        """
        Bands on the totals this site publishes, measured on those totals.

        Same construction as the monthly bands - shape pooled across CNAs, width
        per CNA - but fitted to a different quantity. The error on a total is not
        the sum of the errors on its months: summing monthly bounds assumes the
        model is wrong in the same direction all year, and measured on these
        series the months largely cancel instead. Carrying the summed assumption
        into the published figure gave the median CNA an 80% range spanning 25x.

        Covers the year totals the headline leads with and the running totals the
        chart plots, so the cone and the headline are the same measurement rather
        than two that have to be kept in step.

        Returns:
            ``{cna_id: {'h5-16': bands}}``, keyed by horizon span. A CNA's own
            years map onto those keys through ``_spans_for``, because a forecast
            that starts early puts its calendar years at different horizons.
        """
        by_span: Dict[str, Dict[str, List[float]]] = {}
        for cna_id, entry in self.selection_cache.entries.items():
            stored = entry.get('log_residuals_by_window') or {}
            if stored:
                by_span[cna_id] = stored

        span_names = sorted({name for stored in by_span.values() for name in stored})
        fitted: Dict[str, Dict[str, IntervalBands]] = {}
        for name in span_names:
            contributors = {cna_id: {1: list(stored[name])} for cna_id, stored in by_span.items() if stored.get(name)}
            if not contributors:
                continue
            shape, scales = build_shared_shape(contributors)
            if not shape.factors:
                continue
            for cna_id, scale in scales.items():
                band = scale_bands(shape, scale)
                if band.factors:
                    fitted.setdefault(cna_id, {})[name] = band

        if fitted:
            self.logger.info(f'Fitted bands on {len(span_names)} published spans for {len(fitted)} CNAs')
        return fitted

    @staticmethod
    def _nearest_span(needed: Tuple[int, int], fitted: Dict[str, IntervalBands]) -> Optional[IntervalBands]:
        """
        The closest span this CNA has a band for, when the exact one is missing.

        Distance is months of start drift plus months of length difference,
        which are the two ways a cached span differs from a current one: the
        calendar moving the forecast's first month forward, and a shrinking year
        leaving fewer months in it.

        Args:
            needed: ``(first_horizon, last_horizon)`` the run wants
            fitted: This CNA's bands, keyed 'h<first>-<last>'

        Returns:
            The nearest band within ``MAX_SPAN_DRIFT``, or None
        """
        first, last = needed
        best: Optional[Tuple[int, str]] = None

        for name in fitted:
            try:
                start, end = (int(part) for part in name[1:].split('-'))
            except ValueError:
                continue  # not a horizon span; a year label from an older cache
            drift = abs(start - first) + abs((end - start) - (last - first))
            if drift > MAX_SPAN_DRIFT:
                continue
            if best is None or drift < best[0]:
                best = (drift, name)

        return fitted[best[1]] if best else None

    def _bands_for_cna(
        self, ts: TimeSeries, fitted: Dict[str, IntervalBands], forecast_values: Dict[str, Any]
    ) -> Dict[str, IntervalBands]:
        """
        This CNA's spans, relabelled from horizon to what the page calls them.

        Args:
            ts: The CNA's history, which fixes where its forecast begins
            fitted: Its bands from ``_build_annual_bands``, keyed by horizon span
            forecast_values: The published forecast, needed to turn each span's
                factors into the width a reader actually sees

        A span measured a month or two ago is reused where the exact one is
        missing, because the spans move with the calendar and an exact match
        would leave every page bandless on the first of each month. See
        ``MAX_SPAN_DRIFT``.

        Returns:
            ``{'2027': bands, '2027-03': bands}``. Empty unless every span this
            CNA publishes was matched: partial cover is how the headline and the
            chart come to disagree, since the year would be measured while the
            cone beneath it fell back to accumulating months. Absent is the
            honest state and the page already has somewhere to fall back to.
        """
        spans, by_label = self._spans_for(ts)
        if not by_label:
            return {}

        matched: Dict[str, IntervalBands] = {}
        for label, key in by_label.items():
            band = fitted.get(key) or self._nearest_span(spans[key], fitted)
            if band is None:
                return {}
            matched[label] = band

        bands = matched

        # A year too uncertain to state is too uncertain to draw, so where the
        # guard drops a year's headline range the cone over it goes too. Keeping
        # the cone would leave the chart asserting a range the page had just
        # declined to make.
        bands = self._widen_along_the_year(bands, forecast_values)

        too_wide = set()
        for label, band in bands.items():
            if len(label) != 4:
                continue
            low, high = band.for_horizon(1)['80']
            if high / low > MAX_INFORMATIVE_ANNUAL_RATIO:
                too_wide.add(label)
        return {label: band for label, band in bands.items() if label[:4] not in too_wide}

    @staticmethod
    def _widen_along_the_year(
        bands: Dict[str, IntervalBands], forecast_values: Dict[str, Any]
    ) -> Dict[str, IntervalBands]:
        """
        Stop the cone from narrowing as the year fills in.

        Each running total is fitted from its own handful of residuals, so the
        cone drawn through them wobbles: mitre's 2027 band was 713 CVEs wide in
        April, 386 in November and 1,030 at the close. A reader cannot be told
        that eleven months of a year are more certain than four of it.

        The same argument ``build_intervals`` already makes for horizons, applied
        across spans instead, and in the width a reader sees rather than in the
        factors - the factors legitimately tighten as a span lengthens, because
        that is the month-to-month error cancelling, while the absolute width has
        to grow because there is more forecast underneath it.

        Widened rather than clipped, so nothing is narrowed to fit, and the year
        label takes the closing span's band so the headline and the cone stay the
        same number.

        Args:
            bands: This CNA's spans, labelled 'YYYY' and 'YYYY-MM'
            forecast_values: The published forecast

        Returns:
            The same mapping with each year's spans widened into order
        """
        by_month: Dict[str, float] = {}
        for date_str, value in forecast_values.items():
            if isinstance(value, (int, float)):
                by_month[pd.to_datetime(date_str).strftime('%Y-%m')] = float(value)

        out = dict(bands)
        for year in sorted({label[:4] for label in bands}):
            running = sorted(label for label in bands if len(label) == 7 and label[:4] == year)
            if not running:
                continue

            widest = 0.0
            total = 0.0
            closing = None
            for label in running:
                total += by_month.get(label, 0.0)
                band = out[label]
                factors = band.for_horizon(1)
                low, high = factors['80']
                widest = max(widest, total * (high - low))

                if total > 0 and total * (high - low) < widest:
                    # Spread the shortfall evenly about the point estimate, which
                    # keeps the band centred where the forecast is.
                    grow = (widest / total - (high - low)) / 2.0
                    widened = IntervalBands(levels=band.levels)
                    widened.factors[1] = {lbl: (max(lo - grow, 1e-6), hi + grow) for lbl, (lo, hi) in factors.items()}
                    widened.n_residuals = dict(band.n_residuals)
                    widened.max_horizon = 1
                    out[label] = widened
                closing = label

            # The year is the closing span, so they cannot drift apart.
            if closing and year in out:
                out[year] = out[closing]
        return out

    def _monthly_intervals(self, forecast_values: Dict[str, Any], bands: IntervalBands) -> Dict[str, Dict[str, float]]:
        """
        Attach bounds to each forecast month, stopping where the backtest stops.

        ``IntervalBands.for_horizon`` reuses the longest fitted horizon for
        anything beyond it, silently. That is a reasonable default for a caller
        that knows it is extrapolating, and a trap for one that does not: with a
        6-month backtest behind a 16-month forecast it would draw a
        one-month-ahead band across the whole of next year, flat, on the column
        where uncertainty matters most. Months past the fitted horizon get no
        band at all, and the front end falls back for them.

        Args:
            forecast_values: ``{date_string: point_forecast}``
            bands: This CNA's fitted band

        Returns:
            ``{'YYYY-MM': {'lower_80': x, 'upper_80': y, ...}}``, empty when
            there is no band
        """
        if not bands.factors or not forecast_values:
            return {}

        numeric = {d: float(v) for d, v in forecast_values.items() if isinstance(v, (int, float))}
        raw = apply_intervals(numeric, bands)

        out: Dict[str, Dict[str, float]] = {}
        for step, date_str in enumerate(sorted(numeric), start=1):
            if step > bands.max_horizon:
                break
            band = raw.get(date_str)
            if band:
                out[pd.to_datetime(date_str).strftime('%Y-%m')] = band
        return out

    def _interval_payload(self, forecast_result: ForecastResult, historical_dict: Dict[str, int]) -> Dict[str, Any]:
        """
        Assemble what the CNA page publishes about its uncertainty.

        The annual figures are what the page leads with, and they come from bands
        measured on year totals - not from summing the monthly bounds. Summing
        them would assume the model errs in the same direction every month of the
        year, and on these series the months largely cancel: monthly residual
        spread runs about 3.4x the spread of the same model's error on a
        16-month sum, where 4.0x would mean the months cancel completely.

        A year is published only if a band was fitted for it. A part-covered year
        would understate its own range with nothing on the page to show which
        months were left out.

        Args:
            forecast_result: The published forecast, carrying monthly bands
            historical_dict: This CNA's published months, for the actual YTD

        Returns:
            The ``intervals`` block, or ``{}`` when this CNA has no band
        """
        monthly = forecast_result.confidence_intervals or {}
        annual_bands = forecast_result.metadata.get('annual_bands') or {}
        if not monthly and not annual_bands:
            return {}

        # Both halves count. The month in progress is nowcast - what is published
        # stays in the history and the forecast holds only the remainder - so
        # adding them is exactly right, where adding a full-month forecast to a
        # part-published month would have counted the published part twice.
        actual_by_year: Dict[int, int] = {}
        for date_str, count in historical_dict.items():
            try:
                year = int(str(date_str)[:4])
            except ValueError:
                continue
            actual_by_year[year] = actual_by_year.get(year, 0) + int(count)

        forecast_by_year: Dict[str, float] = {}
        for date_str, value in forecast_result.forecast_values.items():
            if not isinstance(value, (int, float)):
                continue
            year = pd.to_datetime(date_str).strftime('%Y')
            forecast_by_year[year] = forecast_by_year.get(year, 0.0) + float(value)

        annual: Dict[str, Dict[str, int]] = {}
        for year, band in annual_bands.items():
            if len(year) != 4:
                continue  # a running total, not a year
            total = forecast_by_year.get(year)
            factors = band.for_horizon(1)
            if total is None or not factors:
                continue
            # The published year is what has already happened plus what is
            # forecast, and only the forecast half carries model error.
            base = actual_by_year.get(int(year), 0)
            row = {}
            for label, (lo, hi) in factors.items():
                row[f'lower_{label}'] = int(round(base + total * lo))
                row[f'upper_{label}'] = int(round(base + total * hi))
            annual[year] = row

        payload: Dict[str, Any] = {
            'max_horizon': forecast_result.metadata.get('interval_max_horizon'),
        }
        if monthly:
            payload['monthly'] = monthly
        if annual:
            payload['annual'] = annual
        coverage = forecast_result.metadata.get('interval_coverage')
        if coverage:
            payload['coverage'] = coverage
        return payload if (monthly or annual) else {}

    def apply_constraints(self, forecasts: Dict[str, ForecastResult]) -> Dict[str, ForecastResult]:
        """
        Apply CNA-specific constraints (minimal for CNA forecasts).

        Args:
            forecasts: Raw forecasts

        Returns:
            Constrained forecasts (minimal changes for CNA)
        """
        # CNA forecasts typically don't need heavy constraints
        # Just ensure non-negative and round to integers

        constrained = {}

        for model_name, forecast_result in forecasts.items():
            constrained_values = {date: max(0, round(value)) for date, value in forecast_result.forecast_values.items()}

            constrained[model_name] = ForecastResult(
                forecast_values=constrained_values,
                model_name=model_name,
                confidence_intervals=forecast_result.confidence_intervals,
                metrics=forecast_result.metrics,
                metadata={**forecast_result.metadata, 'constraints_applied': True},
            )

        return constrained

    def _generate_historical_cumulative(
        self, historical_dict: Dict[str, int], now: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate per-year cumulative historical data for chart display.
        Each year resets to 0 on January 1st (matches CVE adapter behavior).
        Includes Jan 1 baseline and current month-to-date point.

        Args:
            historical_dict: Dictionary of {date_string: count}

        Returns:
            List of {date, cumulative_total} dictionaries
        """
        from datetime import datetime

        # Sort dates and calculate per-year cumulative totals
        # Match CVE adapter: each month-start shows cumulative BEFORE that month
        sorted_dates = sorted(historical_dict.keys())
        result = []
        current_year = None
        year_cumulative = 0
        # Aware and shared with the forecast timeline: these two series meet at
        # this point, and a naive local clock put them hours apart off UTC.
        current_datetime = now or datetime.now(timezone.utc)

        for date_str in sorted_dates:
            # Parse date
            try:
                if ' ' in date_str:
                    date_obj = datetime.strptime(date_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    date_obj = datetime.strptime(date_str[:10], '%Y-%m-%d')
            except ValueError:
                # Fallback: try to parse just the date part
                date_obj = datetime.strptime(date_str[:10], '%Y-%m-%d')

            # Reset at year boundary
            if current_year is None or date_obj.year != current_year:
                current_year = date_obj.year
                year_cumulative = 0

            # Add month-start entry BEFORE adding this month's count
            # This shows cumulative up to (but not including) this month
            result.append({'date': date_obj.strftime('%Y-%m-%dT%H:%M:%SZ'), 'cumulative_total': year_cumulative})

            # Now add this month's count to the running total
            year_cumulative += historical_dict[date_str]

        # Add current month boundary (Oct 1) if not already present
        # This ensures CNAs with no data this month still show the month start point
        if result and current_year == current_datetime.year:
            last_entry = result[-1]
            last_date = datetime.fromisoformat(last_entry['date'].replace('Z', '+00:00'))

            # If last entry is before current month, add current month start
            if last_date.month < current_datetime.month:
                current_month_start = datetime(current_datetime.year, current_datetime.month, 1)
                result.append(
                    {'date': current_month_start.strftime('%Y-%m-%dT%H:%M:%SZ'), 'cumulative_total': year_cumulative}
                )

        # Add current month-to-date point if we're in the current year
        if result and current_year == current_datetime.year:
            last_entry = result[-1]
            last_date = datetime.fromisoformat(last_entry['date'].replace('Z', '+00:00'))

            # Only add if current date is after the last month boundary
            if current_datetime.month > last_date.month or (
                current_datetime.month == last_date.month and current_datetime.day > 1
            ):
                # Add current date with MTD cumulative
                result.append(
                    {'date': current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'), 'cumulative_total': year_cumulative}
                )

        return result

    def _generate_cna_cumulative_timelines(
        self,
        forecast_dict: Dict[str, int],
        model_name: str,
        actuals_base: int,
        now: Optional[datetime] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate cumulative forecast timelines for chart display.
        Matches CVE adapter logic exactly: uses actuals_base from last complete month.

        Args:
            forecast_dict: Forecast data {date_string: count}
            model_name: Name of the forecasting model
            actuals_base: Cumulative total from last complete month (from historical_cumulative)

        Returns:
            Dictionary with model_cumulative timeline
        """
        from datetime import datetime

        now = now or datetime.now(timezone.utc)
        timeline = []

        if not forecast_dict:
            return {f'{model_name}_cumulative': timeline}

        # Sort forecast dates
        sorted_dates = sorted(forecast_dict.items())

        # Get first forecast year
        first_date_str = sorted_dates[0][0]
        if ' ' in first_date_str:
            first_forecast_date = datetime.strptime(first_date_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
        else:
            first_forecast_date = datetime.strptime(first_date_str[:10], '%Y-%m-%d')

        current_year = first_forecast_date.year
        year_total = actuals_base

        # Add Jan 1 marker for the first forecast year
        timeline.append({'date': f'{current_year}-01-01T00:00:00Z', 'cumulative_total': 0})

        for i, (date_str, cve_count) in enumerate(sorted_dates):
            # Parse forecast date
            try:
                if ' ' in date_str:
                    forecast_date = datetime.strptime(date_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                elif len(date_str) == 7:
                    forecast_date = datetime.strptime(date_str + '-01', '%Y-%m-%d')
                else:
                    forecast_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
            except ValueError:
                continue

            forecast_year = forecast_date.year

            # Handle year boundary
            if forecast_year > current_year:
                # Finalize the previous year with Dec 31 marker
                timeline.append({'date': f'{current_year}-12-31T23:59:59Z', 'cumulative_total': int(round(year_total))})
                # Start the new year
                timeline.append({'date': f'{forecast_year}-01-01T00:00:00Z', 'cumulative_total': 0})
                current_year = forecast_year
                year_total = 0

            # A marker carries the total BEFORE this month's contribution. For
            # the month in progress that state is "published so far", which is
            # true as of now rather than as of the 1st - and the historical
            # series already ends on exactly that point. Dating it to the 1st
            # put the month-to-date total at a position where the actuals line
            # was still showing last month's, so the chart jumped between two
            # values at the same x.
            if forecast_date.year == now.year and forecast_date.month == now.month:
                marker_date = now.strftime('%Y-%m-%dT%H:%M:%SZ')
            else:
                marker_date = f'{forecast_date.year}-{forecast_date.month:02d}-01T00:00:00Z'

            # Check if this date already exists
            existing_entry = next((entry for entry in timeline if entry['date'] == marker_date), None)
            if not existing_entry:
                timeline.append({'date': marker_date, 'cumulative_total': int(round(year_total))})

            # Now add this month's forecast to the running total
            year_total += cve_count

        # Add final Dec 31 marker for the last year in the forecast
        # Only add if the last forecast month is December (to show year-end total)
        if sorted_dates:
            last_date_str = sorted_dates[-1][0]
            if ' ' in last_date_str:
                last_forecast_date = datetime.strptime(last_date_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
            else:
                last_forecast_date = datetime.strptime(last_date_str[:10], '%Y-%m-%d')

            # Only add Dec 31 if last forecast is in December
            if last_forecast_date.month == 12:
                last_year_end = f'{last_forecast_date.year}-12-31T23:59:59Z'
                # Only add if not already present
                if not any(e['date'] == last_year_end for e in timeline):
                    timeline.append({'date': last_year_end, 'cumulative_total': int(round(year_total))})

        return {f'{model_name}_cumulative': timeline}

    def save_results(self, forecasts: Dict[str, ForecastResult]) -> str:
        """
        Save CNA forecast results to web/cna_data.json.

        Args:
            forecasts: Forecast results (organized by CNA)

        Returns:
            Path to saved file
        """
        # The CVE adapter reads file_paths.output_data; this read a top-level
        # output_path, so anything overriding the documented key silently wrote to
        # the real web/cna_data.json instead. Accept both, preferring file_paths.
        paths = self.config.get('file_paths', {})
        output_path = Path(paths.get('cna_output') or self.config.get('output_path') or 'web/cna_data.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f'Saving CNA forecasts to {output_path}...')

        # Note: forecasts dict is organized differently for CNAs
        # It should be {cna_id: ForecastResult} not {model_name: ForecastResult}

        output_data = {}

        for cna_id, forecast_result in forecasts.items():
            cna_info = self.cna_data.get(cna_id, {})

            # Build historical data dict
            historical_dict = {}
            if cna_info.get('historical'):
                for date, value in zip(cna_info['historical'].time_index, cna_info['historical'].values().flatten()):
                    historical_dict[str(date)] = int(value)

            # Calculate actuals_base: cumulative through last COMPLETE month
            # Must calculate from raw historical_dict to handle all CNAs consistently
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)
            current_year = now.year
            last_complete_month = now.month - 1 if now.month > 1 else 12
            current_year if now.month > 1 else current_year - 1

            actuals_base = 0
            for date_str, count in historical_dict.items():
                try:
                    if ' ' in date_str:
                        date_obj = datetime.strptime(date_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                    else:
                        date_obj = datetime.strptime(date_str[:10], '%Y-%m-%d')

                    # Through NOW, not through the last complete month: the
                    # forecast starts from this month's remainder, so the chart
                    # has to start from what is already out - otherwise the line
                    # branches below the actuals and reality overtakes it within
                    # days of the run.
                    if date_obj.year == current_year and date_obj.month <= now.month:
                        actuals_base += count
                except ValueError:
                    continue

            self.logger.debug(
                f'CNA {cna_id}: actuals_base (cumulative through {current_year}-{last_complete_month:02d}): {actuals_base:,} CVEs'
            )

            # Generate historical_cumulative for chart display
            historical_cumulative = self._generate_historical_cumulative(historical_dict, now)

            # Generate cumulative timelines for chart display
            cumulative_timelines = self._generate_cna_cumulative_timelines(
                forecast_result.forecast_values, forecast_result.model_name, actuals_base, now
            )

            record = {
                'id': cna_id,
                'name': cna_info.get('name'),
                'scope': None,
                'historical': historical_dict,
                'historical_cumulative': historical_cumulative,
                'forecasts': {forecast_result.model_name: forecast_result.forecast_values},
                'cumulative_timelines': cumulative_timelines,
                'model_selection': {
                    'selected_model': forecast_result.model_name,
                    'validation_mase': forecast_result.metrics.get('validation_mase'),
                    'is_fallback': forecast_result.metadata.get('is_fallback', False),
                    # When the model was last chosen. Selection is cached and
                    # refreshed periodically, so a reader seeing a model name
                    # should know it was not necessarily picked today.
                    'selected_at': forecast_result.metadata.get('selected_at'),
                    'all_model_scores': forecast_result.metadata.get('all_scores', {}),
                    'awaiting_scoring': forecast_result.metadata.get('awaiting_scoring', False),
                    # Published rather than forecast: the page shows the history
                    # and says when this CNA went quiet.
                    'dormant': forecast_result.metadata.get('dormant', False),
                    'last_published': forecast_result.metadata.get('last_published'),
                    # The chosen model produced a forecast its own history could
                    # not support and was replaced. The page says so rather than
                    # naming the baseline as though it had been selected on merit.
                    'runaway_guarded': forecast_result.metadata.get('runaway_guarded', False),
                },
            }

            # Strictly additive: a CNA with a band gains these keys, one without
            # is byte-for-byte what it was before. The front end leads with a
            # range only where 'intervals' is present and keeps the model-and-
            # MASE line everywhere else, so absence needs no sentinel value.
            intervals = self._interval_payload(forecast_result, historical_dict)
            if intervals:
                record['intervals'] = intervals

                # The shaded cone on the chart, from the same measured spans the
                # headline is built from. Accumulating the monthly bands here
                # instead would be the obvious shortcut and would draw a cone
                # several times too wide, contradicting the figure above it.
                band = build_cumulative_band(
                    cumulative_timelines.get(f'{forecast_result.model_name}_cumulative', []),
                    intervals.get('monthly') or {},
                    forecast_result.metadata.get('annual_bands') or {},
                )
                if band:
                    record['cumulative_band'] = band

            output_data[cna_id] = record

        # Save to file with NaN/inf handling
        import math

        def clean_value(obj):
            """Recursively clean NaN/Infinity values from nested structures."""
            if isinstance(obj, dict):
                return {k: clean_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_value(item) for item in obj]
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            elif isinstance(obj, np.floating):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        cleaned_data = clean_value(output_data)

        with open(output_path, 'w') as f:
            json.dump(cleaned_data, f, indent=2)

        self.logger.info(f'✓ Saved forecasts for {len(output_data)} CNAs')

        return str(output_path)

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete CNA forecasting pipeline.

        Returns:
            Pipeline results
        """
        self.logger.info('=' * 70)
        self.logger.info('CNA FORECASTING PIPELINE - STARTING')
        self.logger.info('=' * 70)

        results = {}

        # 1. Load data
        self.load_data()
        results['cnas_loaded'] = len(self.cna_data)

        # 2. Forecast each CNA
        cna_forecasts = {}

        # Work out the series once, then decide which CNAs need re-scoring before
        # touching a model. Scoring is the expensive step; forecasting is not.
        eligible: Dict[str, TimeSeries] = {}
        for cna_id, cna_info in self.cna_data.items():
            # ForecastEngine does not strip the incomplete current month - the CVE
            # adapter does that before calling it - so do the same here. Training
            # on a part-published month drags the level down and shifts the whole
            # forecast a month late.
            ts = self._complete_months(cna_info['historical'])
            if ts is not None and len(ts) >= 24:
                eligible[cna_id] = ts

        to_refresh = set(self.selection_cache.plan_refresh({cid: len(ts) for cid, ts in eligible.items()}))
        results['cnas_rescored'] = len(to_refresh)
        results['cnas_from_cache'] = sum(
            1 for cid in eligible if cid not in to_refresh and self.selection_cache.get(cid) is not None
        )
        results['cnas_awaiting_scoring'] = len(eligible) - len(to_refresh) - results['cnas_from_cache']

        # Score first, so the cache is as complete as it will get before any band
        # is fitted: the interval shape is pooled across CNAs, so a CNA re-scored
        # this run should contribute to it rather than wait for the next one.
        for cna_id in list(eligible):
            if cna_id not in to_refresh:
                continue
            ts = eligible[cna_id]
            best_model, best_mase, all_scores, backtest = self.select_best_model_for_cna(cna_id, ts)
            self.selection_cache.put(
                cna_id,
                best_model,
                best_mase,
                len(ts),
                all_scores,
                residuals=backtest.log_residuals_by_horizon if backtest else None,
                window_residuals=backtest.log_residuals_by_window if backtest else None,
            )
            mase_text = f'{best_mase:.2f}' if best_mase is not None else 'n/a'
            self.logger.info(f'{self.cna_data[cna_id]["name"]} → {best_model} (MASE: {mase_text}, re-scored)')

        # Persist the scoring before forecasting, so an hour of backtesting is
        # not lost to a failure in a later, cheaper stage.
        self.selection_cache.save()

        interval_bands, interval_coverage = self._build_interval_bands()
        annual_bands = self._build_annual_bands()
        results['cnas_with_intervals'] = len(interval_bands)
        results['interval_coverage'] = interval_coverage

        now = datetime.now(timezone.utc)

        for cna_id, ts in eligible.items():
            cna_info = self.cna_data[cna_id]

            # A CNA that has published nothing for a year has no recent level to
            # extrapolate from. Forecasting it anyway produces a flat line that
            # reads as a prediction and is really an extrapolation from a series
            # that stopped years ago. Its history still publishes - it was a CNA,
            # and dropping it silently loses that - but with no forecast and a
            # note saying when it went quiet.
            idle = self._months_idle(ts, now)
            if idle >= DORMANT_AFTER_MONTHS:
                last = ts.time_index[[i for i, v in enumerate(ts.values().flatten()) if v > 0][-1]]
                self.logger.info(
                    f'{cna_info["name"]}: nothing published since {last:%Y-%m} ({idle} months); not forecast'
                )
                cna_forecasts[cna_id] = ForecastResult(
                    forecast_values={},
                    model_name=FALLBACK_MODEL,
                    metrics={'validation_mase': None},
                    metadata={
                        'all_scores': {},
                        'is_fallback': False,
                        'dormant': True,
                        'last_published': f'{last:%Y-%m}',
                        'months_idle': idle,
                    },
                )
                continue

            if cna_id in to_refresh:
                cached_now = self.selection_cache.get(cna_id) or {}
                best_model = cached_now.get('model', FALLBACK_MODEL)
                best_mase = cached_now.get('mase')
                all_scores = cached_now.get('all_scores', {})
            else:
                cached = self.selection_cache.get(cna_id)
                if cached is None:
                    # Cold start: the refresh cap means most CNAs have no
                    # selection on the first run. Forecast with the baseline
                    # rather than skipping them - it is a defensible forecast,
                    # and each run scores another batch until the cache is full.
                    best_model, best_mase, all_scores = FALLBACK_MODEL, None, {}
                    self.logger.debug(f'{cna_info["name"]} → {FALLBACK_MODEL} (awaiting first scoring)')
                else:
                    best_model = cached['model']
                    best_mase = cached.get('mase')
                    all_scores = cached.get('all_scores', {})
                    self.logger.debug(f'{cna_info["name"]} → {best_model} (cached)')

            # How many months to forecast, counted from where THIS CNA's series
            # ends rather than from the horizon's own start. A forecast runs on
            # from the last month it was fitted to, so a fixed count lands short
            # for any CNA that has been quiet: 19 of 140 were missing between one
            # and nine months of next year, and Liferay published a 2027 total of
            # 17 covering three months where a full year implies about 68.
            _start, end_date = self.get_forecast_horizon()
            first = ts.end_time() + pd.DateOffset(months=1)
            forecast_months = (end_date.year - first.year) * 12 + (end_date.month - first.month) + 1
            if forecast_months < 1:
                self.logger.debug(f'{cna_id}: series already runs past the horizon; nothing to forecast')
                continue

            # Forecast through the shared engine, which handles the incomplete
            # current month, log space and damping consistently with the CVE side.
            try:
                hyperparameters = self.config.get('models', {}).get(best_model, {}).get('hyperparameters', {})
                attempt = self.engine.forecast(ts, best_model, hyperparameters, forecast_months)

                if not attempt.ok:
                    self.logger.debug(f'Skipping {cna_id}: {attempt.error}')
                    continue

                forecast_values = {
                    str(date): max(0, round(float(value)))
                    for date, value in zip(attempt.forecast.time_index, attempt.forecast.values().flatten())
                }
                forecast_values = self._nowcast(forecast_values)

                # Publishing an obviously impossible number costs more than
                # publishing a dull one. Fall back to the baseline, which is what
                # this pipeline already does whenever the chosen model cannot be
                # trusted, and say so rather than quietly clipping the peak.
                runaway = self._is_runaway(forecast_values, ts)
                if runaway and best_model != FALLBACK_MODEL:
                    self.logger.warning(f'{cna_info["name"]}: {best_model} forecast {runaway}; using {FALLBACK_MODEL}')
                    fallback_hp = self.config.get('models', {}).get(FALLBACK_MODEL, {}).get('hyperparameters', {})
                    fallback = self.engine.forecast(ts, FALLBACK_MODEL, fallback_hp, forecast_months)
                    if fallback.ok:
                        best_model, best_mase = FALLBACK_MODEL, None
                        forecast_values = {
                            str(date): max(0, round(float(value)))
                            for date, value in zip(fallback.forecast.time_index, fallback.forecast.values().flatten())
                        }
                        runaway = self._is_runaway(forecast_values, ts)

                if runaway:
                    # Even the baseline ran away, which means the history itself
                    # is pathological. Skip rather than publish it.
                    self.logger.error(f'{cna_info["name"]}: {FALLBACK_MODEL} also {runaway}; skipping this CNA')
                    continue

                entry = self.selection_cache.get(cna_id) or {}
                bands = interval_bands.get(cna_id, IntervalBands())

                # A band belongs to the model it was measured on. If the cached
                # selection and the model actually forecast here disagree - a
                # cold-start CNA falling back to the baseline, say - the band
                # describes a different forecast and must not be drawn around
                # this one.
                if bands.factors and entry.get('model') != best_model:
                    self.logger.debug(
                        f'{cna_id}: band was measured on {entry.get("model")}, '
                        f'publishing {best_model}; dropping the band'
                    )
                    bands = IntervalBands()

                cna_forecasts[cna_id] = ForecastResult(
                    forecast_values=forecast_values,
                    model_name=best_model,
                    confidence_intervals=self._monthly_intervals(forecast_values, bands) or None,
                    metrics={'validation_mase': best_mase},
                    metadata={
                        'all_scores': all_scores,
                        'is_fallback': best_model == FALLBACK_MODEL,
                        'runaway_guarded': best_model == FALLBACK_MODEL
                        and (self.selection_cache.get(cna_id) or {}).get('model') != FALLBACK_MODEL,
                        'awaiting_scoring': best_mase is None and best_model == FALLBACK_MODEL,
                        'selected_at': entry.get('selected_at'),
                        'interval_coverage': interval_coverage,
                        'interval_max_horizon': bands.max_horizon,
                        # Relabelled from horizon span to the year this CNA
                        # publishes, widened into order, and dropped where a year
                        # is too uncertain to state. Passing the raw spans
                        # through leaves the headline keyed 'h1-4' and the chart
                        # with nothing to look up.
                        'annual_bands': (
                            self._bands_for_cna(ts, annual_bands.get(cna_id, {}), forecast_values)
                            if entry.get('model') == best_model
                            else {}
                        ),
                    },
                )

            except ValueError as e:
                # Expected errors for CNAs with insufficient/incompatible data
                error_msg = str(e)
                if any(
                    phrase in error_msg
                    for phrase in [
                        'only contains',
                        'requires at least',
                        'do not share any common times',
                        'output_chunk_shift',
                        'Cannot perform auto-regression',
                    ]
                ):
                    self.logger.debug(f'{cna_id}: {e}')
                else:
                    self.logger.error(f'ValueError for {cna_id}: {e}')
            except Exception as e:
                # Truly unexpected errors
                self.logger.error(f'Unexpected error for {cna_id}: {e}')

        results['forecasts_generated'] = len(cna_forecasts)

        # 3. Save results
        output_path = self.save_results(cna_forecasts)
        results['output_path'] = output_path

        self.logger.info('=' * 70)
        self.logger.info('CNA FORECASTING PIPELINE - COMPLETE')
        self.logger.info('=' * 70)
        self.logger.info(f'✓ CNAs: {results["cnas_loaded"]}')
        self.logger.info(f'✓ Forecasts: {results["forecasts_generated"]}')
        self.logger.info(f'✓ Output: {results["output_path"]}')

        return results
