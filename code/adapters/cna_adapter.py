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
from core.model_utils import create_model_safe
from darts import TimeSeries
from darts.models import AutoARIMA, ExponentialSmoothing, LightGBMModel, LinearRegressionModel, Prophet, XGBModel
from darts.models.forecasting.baselines import NaiveDrift, NaiveMean, NaiveSeasonal
from validation.rolling_origin import RollingOriginBacktest, mark_naive_baselines, rank_models

# What a CNA falls back to when no model beats it. NaiveDrift extrapolates the
# recent level, which is the right default for a short, noisy series - and is an
# honest answer where v0.11 reported a 160%-error model as the "best" one.
FALLBACK_MODEL = 'NaiveDrift'


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

    def select_best_model_for_cna(self, cna_id: str, ts: TimeSeries) -> Tuple[str, float, Dict[str, float]]:
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
            Tuple of (best_model_name, best_mase, all_scores)
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

        results = {}
        for model_name in candidates:
            hyperparameters = self.config.get('models', {}).get(model_name, {}).get('hyperparameters', {})

            def forecast_fn(train, horizon, _name=model_name, _hp=hyperparameters):
                attempt = self.engine.forecast(train, _name, _hp, horizon)
                return attempt.forecast if attempt.ok else None

            results[model_name] = backtest.evaluate(ts, forecast_fn, model_name)

        mark_naive_baselines(results)
        scores = {name: (r.mase if r.is_valid else None) for name, r in results.items()}

        ranked = [r for r in rank_models(results) if r.is_valid]
        if not ranked:
            return FALLBACK_MODEL, float('inf'), scores

        best = ranked[0]
        baseline = results.get(FALLBACK_MODEL)
        # Prefer the baseline when nothing beats it: a model chosen from a field
        # that all lost is the least-bad noise, not a signal.
        if baseline and baseline.is_valid and best.model_name != FALLBACK_MODEL and best.mase >= baseline.mase:
            self.logger.debug(f'{cna_id}: no model beat {FALLBACK_MODEL}; using the baseline')
            return FALLBACK_MODEL, baseline.mase, scores

        return best.model_name, best.mase, scores

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

    def _generate_historical_cumulative(self, historical_dict: Dict[str, int]) -> List[Dict[str, Any]]:
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
        current_datetime = datetime.now()

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
        self, forecast_dict: Dict[str, int], model_name: str, actuals_base: int
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

            # Add month-start entry BEFORE adding this month's forecast
            month_start_date = f'{forecast_date.year}-{forecast_date.month:02d}-01T00:00:00Z'

            # Check if this date already exists
            existing_entry = next((entry for entry in timeline if entry['date'] == month_start_date), None)
            if not existing_entry:
                timeline.append({'date': month_start_date, 'cumulative_total': int(round(year_total))})

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
        output_path = Path(self.config.get('output_path', 'web/cna_data.json'))
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

                    # Sum all months in current year through last complete month
                    if date_obj.year == current_year:
                        if date_obj.month <= last_complete_month:
                            actuals_base += count
                except ValueError:
                    continue

            self.logger.debug(
                f'CNA {cna_id}: actuals_base (cumulative through {current_year}-{last_complete_month:02d}): {actuals_base:,} CVEs'
            )

            # Generate historical_cumulative for chart display
            historical_cumulative = self._generate_historical_cumulative(historical_dict)

            # Generate cumulative timelines for chart display
            cumulative_timelines = self._generate_cna_cumulative_timelines(
                forecast_result.forecast_values, forecast_result.model_name, actuals_base
            )

            output_data[cna_id] = {
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
                },
            }

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

        for cna_id, ts in eligible.items():
            cna_info = self.cna_data[cna_id]

            if cna_id in to_refresh:
                best_model, best_mase, all_scores = self.select_best_model_for_cna(cna_id, ts)
                self.selection_cache.put(cna_id, best_model, best_mase, len(ts), all_scores)
                self.logger.info(f'{cna_info["name"]} → {best_model} (MASE: {best_mase:.2f}, re-scored)')
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

            # Get forecast horizon
            start_date, end_date = self.get_forecast_horizon()
            forecast_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

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

                cna_forecasts[cna_id] = ForecastResult(
                    forecast_values=forecast_values,
                    model_name=best_model,
                    metrics={'validation_mase': best_mase},
                    metadata={
                        'all_scores': all_scores,
                        'is_fallback': best_model == FALLBACK_MODEL,
                        'awaiting_scoring': best_mase is None and best_model == FALLBACK_MODEL,
                        'selected_at': (self.selection_cache.get(cna_id) or {}).get('selected_at'),
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

        # Persist after the loop so a mid-run failure does not lose the
        # selections already paid for.
        self.selection_cache.save()

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
