"""
CVE Forecaster Adapter - CVE-specific implementation of forecasting system.

Extends BaseForecaster with CVE-specific data loading, constraints, and output formatting.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from cna_trend_data import calculate_cna_momentum
from core.base_forecaster import BaseForecaster, ForecastResult
from core.forecast_engine import ForecastEngine, ForecastSettings
from core.intervals import apply_intervals, build_intervals, pooled_residuals, validate_coverage
from core.model_utils import create_model_safe
from core.validation_mixin import ValidationMixin
from darts import TimeSeries
from darts.models import (
    TBATS,
    AutoARIMA,
    CatBoostModel,
    Croston,
    DLinearModel,
    ExponentialSmoothing,
    FourTheta,
    KalmanForecaster,
    LightGBMModel,
    LinearRegressionModel,
    NBEATSModel,
    NHiTSModel,
    Prophet,
    RandomForestModel,
    TCNModel,
    Theta,
    TiDEModel,
    XGBModel,
)
from darts.models.forecasting.baselines import NaiveDrift, NaiveMean, NaiveSeasonal
from data_loader import load_cve_data
from data_vintage import VintageLog
from dateutil.relativedelta import relativedelta
from forecast_constraints import (
    ForecastConstraints,
    build_year_projections,
    combine_model_forecasts,
)
from forecast_tracker import ForecastTracker
from validation.rolling_origin import NAIVE_MODELS, RollingOriginBacktest, mark_naive_baselines, rank_models

# Output key for the combined forecast. Named for what it is - a trimmed mean over
# a curated pool - rather than v0.11's "all models average", which was a median of
# every model including the ones that lost to a naive baseline.
ENSEMBLE_KEY = 'Ensemble'


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and maps NaN/inf to null."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class CVEForecaster(BaseForecaster, ValidationMixin):
    """
    CVE-specific forecasting implementation.

    Handles total CVE forecasting with constraints, tracking, and
    integration with the existing CVE forecasting infrastructure.
    """

    def __init__(self, config_path: str = 'config.json'):
        """
        Initialize CVE forecaster.

        Args:
            config_path: Path to configuration file
        """
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        super().__init__(config)

        # CVE-specific attributes
        self.forecast_tracker = ForecastTracker(
            history_path=config['file_paths'].get('forecast_history', 'web/forecast_history.json')
        )

        self.forecast_constraints = ForecastConstraints(config.get('forecast_constraints', {}), self.logger)
        self.settings = ForecastSettings.from_config(config)
        self.engine = ForecastEngine(self.settings, self.create_model, cna_counts=self._load_cna_counts())
        self.cna_momentum = None

        # Populated by run_full_pipeline
        self.backtest_results: Dict[str, Any] = {}
        self.interval_bands = None
        self.annual_bands: Dict[str, Any] = {}
        self.coverage: Dict[str, Any] = {}
        self.naive_threshold: Optional[float] = None
        self.ensemble_members: List[str] = []

        # Time variables
        self.current_datetime = datetime.now(timezone.utc)
        self.start_of_current_month = self.current_datetime.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        self.start_of_next_month = self.start_of_current_month + relativedelta(months=1)

        self.logger.info('CVE Forecaster initialized')

    def _load_cna_counts(self):
        """
        Monthly active-CNA counts, for the optional exogenous covariate.

        Loaded only when the covariate is enabled - it hits a cached CNA list and
        is pure cost otherwise. Returns None on any failure; the covariate is
        optional and a missing driver must not take the forecast down.

        Returns:
            DataFrame of monthly CNA counts, or None
        """
        if not self.settings.use_cna_covariate:
            return None
        try:
            from cna_trend_data import CNATrendData

            counts = CNATrendData(self.logger).get_monthly_cna_counts()
            self.logger.info(f'Loaded {len(counts)} months of CNA counts for the exogenous covariate')
            return counts
        except (ImportError, OSError, ValueError, KeyError) as e:
            self.logger.warning(f'CNA covariate unavailable: {type(e).__name__}: {e}')
            return None

    def load_data(self) -> TimeSeries:
        """
        Load CVE data from database.

        Returns:
            TimeSeries of monthly CVE counts
        """
        self.logger.info('Loading CVE data...')

        monthly_counts = load_cve_data(self.config)

        # Create time series
        # 'M' is deprecated in pandas 3 and resolves to 'ME'; be explicit.
        self.series = TimeSeries.from_dataframe(
            monthly_counts, freq='ME', fill_missing_dates=True, value_cols='cve_count'
        )

        self.logger.info(f'✓ Loaded {len(self.series)} months of CVE data')

        # Calculate CNA momentum
        momentum_score, momentum_stats = calculate_cna_momentum(self.logger)
        self.cna_momentum = momentum_stats  # Store the stats dict
        self.logger.info(
            f'✓ CNA momentum: {momentum_stats.get("current_cna_count", 0)} CNAs, '
            f'{momentum_stats.get("growth_rate_12m", 0):.1f}% 12m growth'
        )

        return self.series

    def get_forecast_horizon(self) -> Tuple[datetime, datetime]:
        """
        Determine the CVE forecast period.

        Forecasting starts at the first month the data does not already cover.
        The current month is excluded from training because it is incomplete, so
        it is the first month forecast - which is what the dashboard wants, since
        it needs a projection for the month in progress.

        Runs through December of next year.

        Example: on 2026-09-17 the series ends 2026-08 (complete), so the horizon
        is 2026-09 .. 2027-12 = 16 months.

        Returns:
            Tuple of (start_date, end_date), both month-anchored
        """
        start_date = self.start_of_current_month
        end_date = datetime(self.current_datetime.year + 1, 12, 31, tzinfo=timezone.utc)
        return start_date, end_date

    def forecast_months(self) -> int:
        """
        Number of months to forecast.

        v0.11 computed this from a start date one month later than the month
        ``predict()`` actually began at, so the final month of the declared range
        was never produced.

        Returns:
            Month count spanning the forecast horizon inclusive
        """
        start, end = self.get_forecast_horizon()
        return (end.year - start.year) * 12 + (end.month - start.month) + 1

    def complete_series(self) -> TimeSeries:
        """
        The series with the current, still-filling month removed.

        Returns:
            TimeSeries of complete months only
        """
        df = self.series.to_dataframe()
        cutoff = pd.Timestamp(self.start_of_current_month).tz_localize(None)
        complete = df[df.index < cutoff]
        return TimeSeries.from_dataframe(complete, freq='ME', fill_missing_dates=False)

    def monthly_actuals(self, complete_only: bool = True) -> Dict[str, float]:
        """
        Published monthly counts keyed ``YYYY-MM``.

        Args:
            complete_only: Drop the current month, which is still accumulating

        Returns:
            Mapping of month string to count
        """
        series = self.complete_series() if complete_only else self.series
        df = series.to_dataframe()
        return {idx.strftime('%Y-%m'): float(row.iloc[0]) for idx, row in df.iterrows()}

    def current_month_progress(self) -> Tuple[str, float, float]:
        """
        How far through the in-progress month we are, in business days.

        Business days rather than calendar days because publication happens on
        working days - by calendar day 17 of a 30-day month we may be 75% through
        the month's publishing capacity, not 57%.

        Returns:
            Tuple of (month string, published so far, share of business days elapsed)
        """
        month = self.current_datetime.strftime('%Y-%m')
        published = self.monthly_actuals(complete_only=False).get(month, 0.0)

        start = pd.Timestamp(self.start_of_current_month).tz_localize(None)
        today = pd.Timestamp(self.current_datetime).tz_localize(None).normalize()
        total_bdays = len(pd.bdate_range(start, start + pd.offsets.MonthEnd(0)))
        elapsed_bdays = len(pd.bdate_range(start, today))
        share = min(elapsed_bdays / total_bdays, 1.0) if total_bdays else 1.0

        return month, float(published), float(share)

    def nowcast_forecasts(self, monthly: Dict[str, float]) -> Dict[str, float]:
        """
        Replace the in-progress month's full-month forecast with its remainder.

        The chart and the year total should both start from what has actually been
        published as of now, not from the last complete month boundary. Without
        this the forecast line branches below the actuals and is overtaken by
        reality within days of each run.

        Args:
            monthly: ``{'YYYY-MM': full-month forecast}``

        Returns:
            The same mapping with the current month scaled to the days remaining
        """
        month, _published, share = self.current_month_progress()
        if month not in monthly:
            return dict(monthly)

        adjusted = dict(monthly)
        adjusted[month] = monthly[month] * max(0.0, 1.0 - share)
        return adjusted

    def get_model_list(self) -> List[str]:
        """
        Get list of enabled CVE models.

        Returns:
            List of model names to use
        """
        enabled_models = [name for name, config in self.config['models'].items() if config.get('enabled', False)]

        self.logger.info(f'Enabled models: {len(enabled_models)}')
        return enabled_models

    def create_model(self, model_name: str, hyperparameters: Dict[str, Any]):
        """
        Create CVE forecast model instance.

        Args:
            model_name: Model name
            hyperparameters: Model hyperparameters

        Returns:
            Configured model instance
        """
        # Model class mapping
        model_classes = {
            'Prophet': Prophet,
            'ExponentialSmoothing': ExponentialSmoothing,
            'AutoARIMA': AutoARIMA,
            'Theta': Theta,
            'FourTheta': FourTheta,
            'TBATS': TBATS,
            'Croston': Croston,
            'KalmanForecaster': KalmanForecaster,
            'KalmanFilter': KalmanForecaster,  # Alias for backwards compatibility
            'XGBoost': XGBModel,
            'LightGBM': LightGBMModel,
            'CatBoost': CatBoostModel,
            'RandomForest': RandomForestModel,
            'LinearRegression': LinearRegressionModel,
            'TCN': TCNModel,
            'NBEATS': NBEATSModel,
            'NHiTS': NHiTSModel,
            'TiDE': TiDEModel,
            'DLinear': DLinearModel,
            'NaiveMean': NaiveMean,
            'NaiveDrift': NaiveDrift,
            'NaiveSeasonal': NaiveSeasonal,
        }

        if model_name not in model_classes:
            raise ValueError(f'Unknown model: {model_name}')

        return create_model_safe(model_classes[model_name], model_name, hyperparameters, self.logger)

    def _get_previous_year_actuals(self) -> Dict[int, int]:
        """Get actual yearly CVE totals from historical data."""
        if self.series is None:
            return {}
        df = self.series.to_dataframe()
        yearly = {}
        for idx, row in df.iterrows():
            year = idx.year
            yearly[year] = yearly.get(year, 0) + int(row.iloc[0])
        return yearly

    def apply_constraints(self, forecasts: Dict[str, ForecastResult]) -> Dict[str, ForecastResult]:
        """
        Check forecasts for divergence. Does not modify them.

        v0.11 rewrote every model's yearly total to clear a hard-coded growth
        floor, comparing a partial-year remainder against a full prior year. That
        collapsed all twelve models onto one number. Bias is now handled where it
        belongs - in log-space modelling - so this only reports.

        Args:
            forecasts: Raw forecasts from all models

        Returns:
            The same forecasts, unmodified
        """
        recent = list(self.complete_series().values().flatten()[-12:])
        actuals = self.monthly_actuals()
        prev_year_totals = self._get_previous_year_actuals()

        flagged = 0
        for model_name, result in forecasts.items():
            monthly = {pd.to_datetime(d).strftime('%Y-%m'): v for d, v in result.forecast_values.items()}
            warnings = self.forecast_constraints.check_monthly(list(monthly.values()), recent)

            for year, proj in build_year_projections(actuals, monthly).items():
                warnings += self.forecast_constraints.check_annual(proj, prev_year_totals.get(year - 1))

            if warnings:
                flagged += 1
                result.metadata['sanity_warnings'] = warnings
                self.logger.warning(f'{model_name}: {len(warnings)} sanity warning(s)')

        self.logger.info(f'Sanity checks complete: {flagged}/{len(forecasts)} models flagged (none modified)')
        return forecasts

    def _get_actuals_cumulative(self) -> List[Dict[str, Any]]:
        """
        Generate cumulative timeline of actual CVE data for current year.

        Returns:
            List of {"date": timestamp, "cumulative_total": int} entries
        """
        self.logger.info('Generating cumulative timeline for actuals (current year)')

        current_year = self.current_datetime.year

        # Start with zero point at beginning of year
        actuals_cumulative = [{'date': f'{current_year}-01-01T00:00:00Z', 'cumulative_total': 0}]

        # Get historical data for current year
        df = self.series.to_dataframe()
        df['year'] = df.index.year
        current_year_df = df[df['year'] == current_year].copy()

        if not current_year_df.empty:
            current_year_df = current_year_df.sort_index()
            current_year_df['cumulative'] = current_year_df.iloc[:, 0].cumsum()

            # Add entry at beginning of NEXT month for each completed month
            for date, row in current_year_df.iterrows():
                # Get the first day of the NEXT month (not just date + 1 month)
                year = date.year
                month = date.month
                next_year = year if month < 12 else year + 1
                next_month_num = month + 1 if month < 12 else 1

                # Create first day of next month at 00:00:00 UTC
                next_month_first = pd.Timestamp(
                    year=next_year, month=next_month_num, day=1, hour=0, minute=0, second=0, tz='UTC'
                )

                current_aware = pd.Timestamp(self.current_datetime)

                # Only add if next month start is not in the future
                if next_month_first <= current_aware:
                    actuals_cumulative.append(
                        {
                            'date': next_month_first.strftime('%Y-%m-%dT%H:%M:%SZ'),
                            'cumulative_total': int(row['cumulative']),
                        }
                    )

        # Add current date with current cumulative
        if not current_year_df.empty:
            current_cumulative = int(current_year_df['cumulative'].iloc[-1])
            actuals_cumulative.append(
                {'date': self.current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'), 'cumulative_total': current_cumulative}
            )

        self.logger.info(f'Generated {len(actuals_cumulative)} actuals cumulative entries')
        return actuals_cumulative

    def _generate_cumulative_timelines(
        self, forecasts: Dict[str, ForecastResult], actuals_base: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate cumulative forecast timelines with year boundary handling.

        The path starts from the cumulative count *as of now* - published months
        plus the part of the current month already published - and the current
        month contributes only its remaining days. Anchoring on the last complete
        month instead makes the forecast branch below the actuals and get
        overtaken by reality within days of each run.

        Args:
            forecasts: Per-model forecasts
            actuals_base: Cumulative published total as of now

        Returns:
            Per-model cumulative timelines keyed ``<model>_cumulative``
        """
        self.logger.info('Generating cumulative forecast timelines with year boundaries')
        cumulative_timelines = {}
        partial_month, _published, _share = self.current_month_progress()

        for model_name, forecast_result in forecasts.items():
            timeline: List[Dict[str, Any]] = []
            if not forecast_result.forecast_values:
                cumulative_timelines[f'{model_name}_cumulative'] = timeline
                continue

            nowcast = self.nowcast_forecasts(
                {pd.to_datetime(d).strftime('%Y-%m'): v for d, v in forecast_result.forecast_values.items()}
            )
            sorted_dates = [
                (d, nowcast[pd.to_datetime(d).strftime('%Y-%m')]) for d in sorted(forecast_result.forecast_values)
            ]

            year_total = actuals_base
            current_year = pd.to_datetime(sorted_dates[0][0]).year

            # Jan 1 marker for the first forecast year, then an anchor at "now" so
            # the forecast line continues from where the actuals line stops rather
            # than starting at a boundary already in the past.
            timeline.append({'date': f'{current_year}-01-01T00:00:00Z', 'cumulative_total': 0})
            if current_year == self.current_datetime.year and actuals_base:
                timeline.append(
                    {
                        'date': self.current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        'cumulative_total': int(round(actuals_base)),
                    }
                )

            for i, (date_str, cve_count) in enumerate(sorted_dates):
                forecast_date = pd.to_datetime(date_str)
                forecast_year = forecast_date.year

                if forecast_year > current_year:
                    # Finalize the previous year with Dec 31 marker
                    timeline.append(
                        {'date': f'{current_year}-12-31T23:59:59Z', 'cumulative_total': int(round(year_total))}
                    )
                    # Start the new year
                    timeline.append({'date': f'{forecast_year}-01-01T00:00:00Z', 'cumulative_total': 0})
                    current_year = forecast_year
                    year_total = 0

                # Month-start marker, except for the month already in progress -
                # that boundary is in the past and the actuals line already covers it.
                if forecast_date.strftime('%Y-%m') != partial_month:
                    month_start_date = f'{forecast_date.year}-{forecast_date.month:02d}-01T00:00:00Z'
                    if not any(entry['date'] == month_start_date for entry in timeline):
                        timeline.append({'date': month_start_date, 'cumulative_total': int(round(year_total))})

                # Now add this month's forecast to the running total
                year_total += cve_count

            # Add final Dec 31 marker for the last year in the forecast
            if sorted_dates:
                last_forecast_date = pd.to_datetime(sorted_dates[-1][0])
                last_year_end = f'{last_forecast_date.year}-12-31T23:59:59Z'
                # Only add if not already present
                if not any(e['date'] == last_year_end for e in timeline):
                    timeline.append({'date': last_year_end, 'cumulative_total': int(round(year_total))})

            cumulative_timelines[f'{model_name}_cumulative'] = timeline

        # The ensemble carries its own timeline; no synthetic all-model average.
        if cumulative_timelines:
            # The ensemble already has its own timeline; no synthetic average needed.
            pass

        self.logger.info(f'Generated {len(cumulative_timelines)} cumulative timelines')
        return cumulative_timelines

    @staticmethod
    def _generate_cumulative_band(
        timeline: List[Dict[str, Any]],
        step_intervals: Dict[str, Dict[str, float]],
        measured_bands: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Cumulative 80% bounds aligned to the ensemble timeline.

        Computed here rather than in the browser: the front-end would have to
        re-derive which month each cumulative step belongs to, and getting that
        off by one silently mislabels every band on the chart.

        Each point on this chart is a running total for the year, so its band is
        measured on running totals - the same way the year figure is, and using
        the span that ends at that month. Through v0.14 the bounds were instead
        accumulated month by month, which assumes the model errs in the same
        direction every month running. The months largely cancel, so that
        overstated the band, and once the year figure started being measured
        properly the two disagreed: 2027 read 109,546-138,059 in the headline
        and 97,760-215,537 on the chart drawn beneath it.

        A month's span starts where its year starts, so the last point of a year
        carries that year's own band and the chart closes exactly where the
        headline says it should.

        Args:
            timeline: The ensemble cumulative timeline
            step_intervals: Per-month bands on what each month adds, used only
                where no measured span is available
            measured_bands: ``{'YYYY-MM': IntervalBands}`` on the year-to-date
                total at that month, from ``cumulative_windows``

        Returns:
            ``{'lower': [{date, cumulative_total}], 'upper': [...]}``
        """
        measured = measured_bands or {}
        if not timeline or (not measured and not step_intervals):
            return {}

        lower: List[Dict[str, Any]] = []
        upper: List[Dict[str, Any]] = []
        # Retained for the fallback path below, which still accumulates.
        lower_offset = 0.0
        upper_offset = 0.0
        previous: Optional[Dict[str, Any]] = None
        # Where this year's forecasting starts from. Set on the first banded
        # marker of the year, to whatever was already published by then.
        forecast_base: Optional[float] = None

        for entry in timeline:
            if previous is None or entry['cumulative_total'] == 0:
                # Start of the path, or a year reset: no accumulated uncertainty
                # yet, and nothing forecast for this year so far.
                lower_offset = upper_offset = 0.0
                forecast_base = None
                lower.append(dict(entry))
                upper.append(dict(entry))
                previous = entry
                continue

            # A marker shows the total BEFORE its own month, so the span it
            # closes is the one ending at the previous marker's month - the same
            # alignment the accumulating path uses, and the reason it is computed
            # here rather than in the browser.
            band = measured.get(previous['date'][:7])
            factors = band.for_horizon(1).get('80') if band else None

            if factors:
                if forecast_base is None:
                    forecast_base = previous['cumulative_total']
                # Only the forecast part of the year carries model error; what
                # is already published is observed and does not move.
                forecast_so_far = entry['cumulative_total'] - forecast_base
                low = forecast_base + forecast_so_far * factors[0]
                high = forecast_base + forecast_so_far * factors[1]
            else:
                # No measured span for this marker - accumulate, as before.
                step = entry['cumulative_total'] - previous['cumulative_total']
                step_band = step_intervals.get(previous['date'][:7])
                if step_band and step > 0:
                    lower_offset += step_band['lower_80'] - step
                    upper_offset += step_band['upper_80'] - step
                low = entry['cumulative_total'] + lower_offset
                high = entry['cumulative_total'] + upper_offset

            lower.append({'date': entry['date'], 'cumulative_total': int(round(low))})
            upper.append({'date': entry['date'], 'cumulative_total': int(round(high))})
            previous = entry

        return {'lower': lower, 'upper': upper}

    def run_backtest(self) -> Dict[str, Any]:
        """
        Score every enabled model across many forecast origins.

        Runs the same ``ForecastEngine`` that produces the published forecast, so
        the accuracy figures describe the forecast on the dashboard rather than a
        differently-fitted model as they did through v0.11.

        Returns:
            Mapping of model name to BacktestResult
        """
        cv = self.config.get('cross_validation', {})
        # The year spans the headline needs, and the running totals the chart
        # draws. Both are sums, so both are scored as sums.
        windows = {**self.publication_windows(), **self.cumulative_windows()}
        # The horizon must reach the end of what is published, or the year totals
        # cannot be scored and the last months of the band are extrapolated from
        # the longest horizon that was. Through v0.13 it was 12 against a
        # 16-month forecast, so September to December of next year all carried
        # the h=12 band, flat.
        needed = max((last for _first, last in windows.values()), default=0)
        horizon = max(cv.get('horizon', 12), needed)
        backtest = RollingOriginBacktest(
            horizon=horizon,
            min_train=cv.get('min_train', 48),
            step=cv.get('step', 1),
            max_origins=cv.get('max_origins', 24),
        )
        series = self.complete_series()
        self.logger.info(
            f'Rolling-origin backtest: {len(backtest.origins_for(series))} origins, h=1..{backtest.horizon}'
        )

        results = {}
        for model_name in self.get_model_list():
            hyperparameters = self.config['models'][model_name].get('hyperparameters', {})

            def forecast_fn(train, horizon, _name=model_name, _hp=hyperparameters):
                attempt = self.engine.forecast(train, _name, _hp, horizon)
                return attempt.forecast if attempt.ok else None

            results[model_name] = backtest.evaluate(series, forecast_fn, model_name, windows=windows)

        self.naive_threshold = mark_naive_baselines(results)
        self.backtest_results = results
        return results

    def _build_intervals(self, results: Dict[str, Any]) -> None:
        """
        Build prediction intervals from the backtest residuals of the chosen pool.

        Pools residuals over the models that beat the naive baseline, so the band
        reflects the errors of forecasts we would actually publish.

        Args:
            results: Backtest results from run_backtest()
        """
        ranked = [r for r in rank_models(results) if r.is_valid and r.model_name not in NAIVE_MODELS]
        if not ranked:
            self.logger.warning('No valid backtest results; prediction intervals unavailable')
            return

        winners = [r.model_name for r in ranked if r.beats_naive]
        if not winners:
            winners = [ranked[0].model_name]
            self.logger.warning(f'No model beat the naive baseline; intervals based on best available ({winners[0]})')

        self.ensemble_members = winners
        residuals = pooled_residuals({m: r.log_residuals_by_horizon for m, r in results.items()}, winners)
        self.interval_bands = build_intervals(residuals)
        self.coverage = validate_coverage(residuals, self.interval_bands)

        # A year's band, measured on that year's total rather than summed from
        # its months. Summing assumes the model errs in the same direction all
        # year; the months substantially cancel, so summing overstates the range.
        self.annual_bands = {}
        for name in {**self.publication_windows(), **self.cumulative_windows()}:
            pooled = pooled_residuals(
                {m: {1: r.log_residuals_by_window.get(name, [])} for m, r in results.items()}, winners
            )
            band = build_intervals(pooled)
            if band.factors:
                self.annual_bands[name] = band
        if self.annual_bands:
            self.logger.info(
                'Measured band on each published total: '
                + ', '.join(
                    f'{y} 80% [{self.annual_bands[y].for_horizon(1)["80"][0]:.2f}x, '
                    f'{self.annual_bands[y].for_horizon(1)["80"][1]:.2f}x]'
                    for y in sorted(self.publication_windows())
                    if y in self.annual_bands
                )
            )

    def _generate_model_rankings(self) -> List[Dict[str, Any]]:
        """
        Build the dashboard ranking table, ordered by MASE.

        Returns:
            List of ranking entries, best first
        """
        rankings = []
        for result in rank_models(self.backtest_results):
            entry = result.to_dict()
            model_config = self.config.get('models', {}).get(result.model_name, {})

            entry['is_baseline'] = result.model_name in NAIVE_MODELS
            entry['naive_threshold'] = round(self.naive_threshold, 3) if self.naive_threshold else None
            entry['in_ensemble'] = result.model_name in self.ensemble_members

            hyperparameters = model_config.get('hyperparameters', {})
            if hyperparameters and any(v is not None for v in hyperparameters.values()):
                entry['hyperparameters'] = hyperparameters
            tuning = model_config.get('tuning_results', {})
            if tuning.get('tuned_at'):
                entry['tuned_at'] = tuning['tuned_at']

            rankings.append(entry)

        self.logger.info(f'Ranked {len(rankings)} models by MASE')
        return rankings

    def _calculate_yearly_totals(
        self, forecasts: Dict[str, ForecastResult], step_intervals: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Year-end totals as published months plus forecast months.

        The headline number is no longer a pure model output: by September, 9/12
        of it is already known. Splitting it makes the figure honest and makes it
        tighten naturally as the year fills in.

        Args:
            forecasts: Per-model forecasts
            step_intervals: Bounds on what each month still adds (not the published
                monthly figure, whose band for the current month also covers days
                already counted in actual_ytd)

        Returns:
            ``{year_string: {model_name: projection_dict}}`` - string keys, because
            v0.11 built integer keys here and then tested ``str(year) in ...``,
            which was never true and silently disabled the year-end marker.
        """
        # Includes the in-progress month, whose forecast entry is a remainder.
        actuals = self.monthly_actuals(complete_only=False)
        partial_month, _published, _share = self.current_month_progress()
        yearly: Dict[str, Dict[str, Any]] = {}

        for model_name, result in forecasts.items():
            monthly = self.nowcast_forecasts(
                {pd.to_datetime(d).strftime('%Y-%m'): v for d, v in result.forecast_values.items()}
            )
            intervals = step_intervals if model_name == ENSEMBLE_KEY else None
            annual = self.annual_bands if model_name == ENSEMBLE_KEY else None
            projections = build_year_projections(
                actuals, monthly, intervals, partial_month=partial_month, annual_bands=annual
            )
            for year, projection in projections.items():
                yearly.setdefault(str(year), {})[model_name] = projection.to_dict()

        self.logger.info(f'Calculated year projections for {sorted(yearly)}')
        return yearly

    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics.

        Returns:
            Summary dict with data/forecast periods and aggregate stats
        """
        self.logger.info('Generating summary statistics')

        df = self.series.to_dataframe()
        forecast_start, forecast_end = self.get_forecast_horizon()
        summary = {
            'data_period': {'start': df.index.min().strftime('%Y-%m-%d'), 'end': df.index.max().strftime('%Y-%m-%d')},
            'forecast_period': {
                # Derived from the real horizon: v0.11 hard-coded a start one month
                # later than predict() actually began at.
                'start': forecast_start.strftime('%Y-%m-%d'),
                'end': forecast_end.strftime('%Y-%m-%d'),
            },
            'total_historical_cves': int(df.iloc[:, 0].sum()),
            'models_evaluated': len(self.model_results),
            'data_points': len(df),
        }

        # Add current year and previous year totals
        current_year = self.current_datetime.year
        df['year'] = df.index.year

        current_year_total = int(df[df['year'] == current_year].iloc[:, 0].sum())
        previous_year_total = int(df[df['year'] == (current_year - 1)].iloc[:, 0].sum())

        summary[f'cumulative_cves_{current_year}'] = current_year_total
        summary['previous_year_total'] = previous_year_total

        return summary

    def _save_forecast_snapshot(self, forecasts: Dict[str, ForecastResult]):
        """
        Record this run's forecast so accuracy can be measured as months land.

        v0.11 passed ``actuals={}`` with a TODO, and wrote to a default path while
        the tracker's own file used an incompatible schema - so nine months of
        daily vintages were lost to a swallowed KeyError. Actuals are now real and
        the exception handling is narrow enough to surface a repeat.

        Args:
            forecasts: Final forecasts from all models
        """
        forecast_dict: Dict[str, Dict[str, float]] = {}
        for model_name, result in forecasts.items():
            for date_str, count in result.forecast_values.items():
                month = pd.to_datetime(date_str).strftime('%Y-%m')
                forecast_dict.setdefault(month, {})[model_name] = float(count)

        performance = {
            name: {'mase': r.mase, 'mape': r.mape, 'beats_naive': r.beats_naive}
            for name, r in self.backtest_results.items()
            if r.is_valid
        }

        try:
            self.forecast_tracker.add_snapshot(
                forecasts=forecast_dict,
                actuals=self.monthly_actuals(),
                model_performance=performance,
                snapshot_date=self.current_datetime,
                metadata={
                    'data_periods': len(self.series),
                    'forecast_horizon': self.forecast_months(),
                    'settings': {
                        'log_space': self.settings.log_space,
                        'business_day_normalise': self.settings.business_day_normalise,
                        'damping_phi': self.settings.damping_phi,
                    },
                },
            )
            self.logger.info('Saved forecast snapshot to tracker')
        except (KeyError, OSError, TypeError, ValueError) as e:
            # Narrow on purpose: a bare except here is what hid the v0.11 schema bug.
            self.logger.error(f'Could not save forecast snapshot: {type(e).__name__}: {e}', exc_info=True)

    def _calculate_forecast_vs_published(self, model_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Month-by-month comparison of forecast against published counts, this year.

        Runs through ``ForecastEngine`` so the table reflects the shipped pipeline.
        It remains a single-origin view - trained through 31 December, forecasting
        the year - which is a legible story for a reader but far too small a sample
        to rank on. Ranking uses the rolling-origin backtest instead.

        Args:
            model_name: Model to evaluate

        Returns:
            Tuple of (per-month rows, summary stats)
        """
        try:
            series = self.complete_series()
            current_year = self.current_datetime.year
            df = series.to_dataframe()

            train_df = df[df.index.year < current_year]
            actual_df = df[df.index.year == current_year]
            if train_df.empty or actual_df.empty:
                return [], {}

            train_series = TimeSeries.from_dataframe(train_df, freq='ME', fill_missing_dates=False)
            hyperparameters = self.config['models'].get(model_name, {}).get('hyperparameters', {})

            attempt = self.engine.forecast(train_series, model_name, hyperparameters, len(actual_df))
            if not attempt.ok:
                self.logger.warning(f'{model_name}: backtest table unavailable - {attempt.error}')
                return [], {}

            predicted = attempt.forecast.values().flatten()
            table_data = []
            abs_errors = []
            pct_errors = []

            for i, (date, row) in enumerate(actual_df.iterrows()):
                actual = int(row.iloc[0])
                forecast = int(round(predicted[i]))
                error = forecast - actual
                pct = (error / actual * 100) if actual else 0.0
                abs_pct = abs(pct)

                if abs_pct < 5:
                    performance = 'Excellent'
                elif abs_pct < 10:
                    performance = 'Good'
                elif abs_pct < 20:
                    performance = 'Fair'
                else:
                    performance = 'Poor'

                table_data.append(
                    {
                        'MONTH': date.strftime('%Y-%m'),
                        'PUBLISHED': actual,
                        'FORECAST': forecast,
                        'ERROR': error,
                        'PERCENT_ERROR': round(pct, 2),
                        'PERFORMANCE': performance,
                    }
                )
                abs_errors.append(abs(error))
                pct_errors.append(abs_pct)

            summary_stats = {
                'mean_absolute_error': round(float(np.mean(abs_errors)), 2),
                'mean_absolute_percentage_error': round(float(np.mean(pct_errors)), 2),
            }
            self.logger.info(
                f'{model_name} single-origin table: MAE={summary_stats["mean_absolute_error"]}, '
                f'MAPE={summary_stats["mean_absolute_percentage_error"]}%'
            )
            return table_data, summary_stats

        except (KeyError, ValueError, IndexError) as e:
            self.logger.warning(f'Could not build comparison table for {model_name}: {type(e).__name__}: {e}')
            return [], {}

    def save_results(self, forecasts: Dict[str, ForecastResult]) -> str:
        """
        Write web/data.json plus web/validation.json.

        Args:
            forecasts: Final forecasts, including the ``Ensemble`` combination

        Returns:
            Path to the saved forecast data
        """
        paths = self.config['file_paths']
        # v0.11 read a key that config.json never defined and survived on the
        # fallback happening to match. Accept both spellings.
        output_path = Path(paths.get('output_data') or paths.get('output') or 'web/data.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Saving CVE forecast data to {output_path}')

        partial_month, partial_published, _share = self.current_month_progress()
        ensemble = forecasts.get(ENSEMBLE_KEY)

        # Two related but distinct quantities, kept explicitly separate because
        # conflating them is how the September band stopped containing the
        # September forecast:
        #   step_intervals      - bands on what each month still ADDS (the current
        #                         month's remainder). Used for cumulative maths.
        #   monthly_intervals   - bands on the figure we publish for that month
        #                         (the current month's full-month nowcast).
        step_intervals: Dict[str, Dict[str, float]] = {}
        monthly_intervals: Dict[str, Dict[str, float]] = {}
        if ensemble is not None and self.interval_bands is not None:
            remainder_by_date = {
                d: self.nowcast_forecasts({pd.to_datetime(d).strftime('%Y-%m'): v})[pd.to_datetime(d).strftime('%Y-%m')]
                for d, v in ensemble.forecast_values.items()
            }
            raw = apply_intervals(remainder_by_date, self.interval_bands)
            step_intervals = {pd.to_datetime(d).strftime('%Y-%m'): v for d, v in raw.items()}
            monthly_intervals = {m: dict(v) for m, v in step_intervals.items()}
            if partial_month in monthly_intervals:
                # The published September figure is what is already out plus the
                # remainder, so its band shifts by the same amount.
                monthly_intervals[partial_month] = {
                    k: round(v + partial_published, 2) for k, v in step_intervals[partial_month].items()
                }
            ensemble.confidence_intervals = monthly_intervals

        model_rankings = self._generate_model_rankings()
        best_model = next((r['model_name'] for r in model_rankings if not r['is_baseline']), None)

        forecast_vs_published = {}
        for model_name in [r['model_name'] for r in model_rankings[:5]]:
            table_data, summary_stats = self._calculate_forecast_vs_published(model_name)
            forecast_vs_published[model_name] = {'table_data': table_data, 'summary_stats': summary_stats}

        actuals_cumulative = self._get_actuals_cumulative()
        # Anchor the forecast on the count as of now, including the part of the
        # current month already published - not the last complete month boundary.
        actuals_base = actuals_cumulative[-1]['cumulative_total'] if actuals_cumulative else 0
        self.logger.info(f'Forecast anchored at {actuals_base:,} CVEs (as of {self.current_datetime:%Y-%m-%d})')

        cumulative_timelines = self._generate_cumulative_timelines(forecasts, actuals_base)
        cumulative_band = self._generate_cumulative_band(
            cumulative_timelines.get(f'{ENSEMBLE_KEY}_cumulative', []), step_intervals, self.annual_bands
        )
        # Step bands, not published bands: actual_ytd already contains the current
        # month's published portion, so the year band must add only what each
        # month still contributes.
        yearly_forecast_totals = self._calculate_yearly_totals(forecasts, step_intervals)

        # Deliberately NOT appending a Dec 31 projection to actuals_cumulative.
        # v0.11 intended to, but a str/int key mismatch meant the code never ran.
        # Fixing that mismatch switched it on and the chart began drawing a
        # forecast as part of the blue "Actual CVEs" line, which ran to year end.
        # The year-end total already lives on the Ensemble forecast timeline;
        # the actuals series stops at the last real observation.

        # Publish the current month as a full-month nowcast (already published +
        # model expectation for the days left), so the figure and its band describe
        # the same quantity.
        forecasts_simple = {}
        for model_name, result in forecasts.items():
            nowcast = self.nowcast_forecasts(
                {pd.to_datetime(d).strftime('%Y-%m'): v for d, v in result.forecast_values.items()}
            )
            if partial_month in nowcast:
                nowcast[partial_month] += partial_published
            forecasts_simple[model_name] = [
                {'date': month, 'cve_count': int(round(value))} for month, value in sorted(nowcast.items())
            ]

        df = self.series.to_dataframe()
        current_year_df = df[df.index.year == self.current_datetime.year]
        current_month_actual = {
            'date': self.current_datetime.strftime('%Y-%m'),
            'cve_count': int(current_year_df.iloc[-1, 0]) if not current_year_df.empty else 0,
            'cumulative_total': actuals_base,
        }

        self._save_forecast_snapshot(forecasts)

        try:
            vintage = VintageLog(paths.get('data_vintages', 'web/data_vintages.json'))
            vintage.record(self.monthly_actuals(complete_only=False))
            output_vintage_summary = vintage.summary()
        except (OSError, ValueError, KeyError) as e:
            self.logger.error(f'Could not record data vintage: {type(e).__name__}: {e}')
            output_vintage_summary = {}

        output_data = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'version': '0.13',
            'best_model': best_model,
            'model_rankings': model_rankings,
            'yearly_forecast_totals': yearly_forecast_totals,
            'monthly_intervals': monthly_intervals,
            'cumulative_band': cumulative_band,
            'current_month_actual': current_month_actual,
            'actuals_cumulative': actuals_cumulative,
            'cumulative_timelines': cumulative_timelines,
            'forecasts': forecasts_simple,
            'summary': self._generate_summary(),
            'forecast_vs_published': forecast_vs_published,
            'data_vintages': output_vintage_summary,
            'methodology': {
                'ranking_metric': 'MASE',
                'naive_threshold': round(self.naive_threshold, 3) if self.naive_threshold else None,
                'ensemble_members': self.ensemble_members,
                'interval_coverage': self.coverage,
                'settings': {
                    'log_space': self.settings.log_space,
                    'business_day_normalise': self.settings.business_day_normalise,
                    'damping_phi': self.settings.damping_phi,
                    'training_window_months': self.settings.training_window_months,
                    'use_future_covariates': self.settings.use_future_covariates,
                },
            },
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=_NumpyEncoder)

        validation_path = Path(paths.get('validation', 'web/validation.json'))
        validation_payload = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'naive_threshold': round(self.naive_threshold, 3) if self.naive_threshold else None,
            'ensemble_members': self.ensemble_members,
            'coverage': self.coverage,
            'intervals': self.interval_bands.to_dict() if self.interval_bands else {},
            'models': {name: r.to_dict() for name, r in self.backtest_results.items()},
        }
        with open(validation_path, 'w') as f:
            json.dump(validation_payload, f, indent=2, cls=_NumpyEncoder)

        self.logger.info(f'Saved {len(model_rankings)} ranked models, years {sorted(yearly_forecast_totals)}')
        self.logger.info(f'Saved validation detail to {validation_path}')
        return str(output_path)

    def load_optimized_models(self) -> Dict[str, Any]:
        """
        Load pre-optimized model hyperparameters from tuner results.

        Returns:
            Dictionary of model configurations
        """
        self.logger.info('Loading optimized model configurations...')

        models_loaded = {}

        for model_name, model_config in self.config['models'].items():
            if not model_config.get('enabled', False):
                continue

            # Check for tuning results
            if 'tuning_results' not in model_config:
                self.logger.info(f'No tuning results for {model_name}, using defaults')
                hyperparameters = model_config.get('hyperparameters', {})
            else:
                tuning_results = model_config['tuning_results']
                hyperparameters = tuning_results.get('best_hyperparameters', {})

            models_loaded[model_name] = {
                'hyperparameters': hyperparameters,
                'enabled': True,
                'tuning_results': model_config.get('tuning_results', {}),
            }

        self.model_results = {
            name: {'hyperparameters': config['hyperparameters'], 'trained': False}
            for name, config in models_loaded.items()
        }

        self.logger.info(f'✓ Loaded {len(models_loaded)} model configurations')

        return models_loaded

    def run_full_pipeline(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the complete CVE forecasting pipeline.

        Order matters: the backtest runs before the forecast, so model ranking,
        ensemble membership and prediction intervals are all settled before
        anything is published.

        Args:
            **kwargs: Accepted and ignored, for compatibility with v0.11 callers
                that passed train_ratio / run_validation / run_diagnostics

        Returns:
            Pipeline result summary
        """
        for legacy in ('train_ratio', 'run_validation', 'run_diagnostics'):
            if legacy in kwargs:
                self.logger.info(f'Ignoring legacy argument {legacy}; v0.12 always backtests before forecasting')

        self.logger.info('=' * 70)
        self.logger.info('CVE FORECASTING PIPELINE - v0.12')
        self.logger.info('=' * 70)

        results: Dict[str, Any] = {}

        self.load_data()
        series = self.complete_series()
        results['data_periods'] = len(self.series)
        results['complete_periods'] = len(series)

        backtest_results = self.run_backtest()
        results['models_backtested'] = sum(1 for r in backtest_results.values() if r.is_valid)

        self._build_intervals(backtest_results)
        results['ensemble_members'] = self.ensemble_members
        results['interval_coverage'] = self.coverage

        horizon = self.forecast_months()
        self.logger.info(f'Forecasting {horizon} months from {series.end_time().date()}')

        forecasts: Dict[str, ForecastResult] = {}
        for model_name in self.get_model_list():
            hyperparameters = self.config['models'][model_name].get('hyperparameters', {})
            attempt = self.engine.forecast(series, model_name, hyperparameters, horizon)
            if not attempt.ok:
                self.logger.warning(f'{model_name}: excluded from output - {attempt.error}')
                continue

            backtest = backtest_results.get(model_name)
            forecasts[model_name] = ForecastResult(
                forecast_values={
                    str(d.date()): float(v)
                    for d, v in zip(attempt.forecast.time_index, attempt.forecast.values().flatten())
                },
                model_name=model_name,
                metrics={
                    'mase': backtest.mase if backtest else None,
                    'mape': backtest.mape if backtest else None,
                    'beats_naive': backtest.beats_naive if backtest else None,
                },
                metadata={'hyperparameters': hyperparameters, **attempt.metadata},
            )
        results['models_forecast'] = len(forecasts)

        ensemble_monthly = combine_model_forecasts(
            {name: r.forecast_values for name, r in forecasts.items()},
            members=self.ensemble_members or None,
            method=self.config.get('model_evaluation', {}).get('ensemble_method', 'trimmed_mean'),
        )
        if ensemble_monthly:
            forecasts[ENSEMBLE_KEY] = ForecastResult(
                forecast_values=ensemble_monthly,
                model_name=ENSEMBLE_KEY,
                metadata={'members': self.ensemble_members},
            )

        self.apply_constraints(forecasts)
        results['output_path'] = self.save_results(forecasts)

        self.logger.info('=' * 70)
        self.logger.info('CVE FORECASTING PIPELINE - COMPLETE')
        self.logger.info(f'  Backtested: {results["models_backtested"]} models')
        self.logger.info(f'  Ensemble:   {", ".join(self.ensemble_members) or "none"}')
        self.logger.info(f'  Output:     {results["output_path"]}')
        self.logger.info('=' * 70)
        return results
