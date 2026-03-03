"""
CVE Forecaster Adapter - CVE-specific implementation of forecasting system.

Extends BaseForecaster with CVE-specific data loading, constraints, and output formatting.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from cna_trend_data import calculate_cna_momentum
from core.base_forecaster import BaseForecaster, ForecastResult
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
from dateutil.relativedelta import relativedelta
from forecast_constraints import ForecastConstraints
from forecast_tracker import ForecastTracker


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

        self.forecast_constraints = None  # Initialized after data load
        self.cna_momentum = None

        # Time variables
        self.current_datetime = datetime.now(timezone.utc)
        self.start_of_current_month = self.current_datetime.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        self.start_of_next_month = self.start_of_current_month + relativedelta(months=1)

        self.logger.info('CVE Forecaster initialized')

    def load_data(self) -> TimeSeries:
        """
        Load CVE data from database.

        Returns:
            TimeSeries of monthly CVE counts
        """
        self.logger.info('Loading CVE data...')

        monthly_counts = load_cve_data(self.config)

        # Create time series
        self.series = TimeSeries.from_dataframe(
            monthly_counts, freq='M', fill_missing_dates=True, value_cols='cve_count'
        )

        self.logger.info(f'✓ Loaded {len(self.series)} months of CVE data')

        # Initialize constraints after data load
        self.forecast_constraints = ForecastConstraints(config=self.config, logger=self.logger)

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
        Determine CVE forecast period.

        Forecasts from next complete month through January of year+2.
        - Remaining months of current year
        - All 12 months of next year
        - January of year+2 (needed for Dec 31 year-end marker)

        Example: In Oct 2025 → forecast Nov 2025 through Jan 2027
        Example: In Jan 2026 → forecast Feb 2026 through Jan 2028

        Returns:
            Tuple of (start_date, end_date)
        """
        # Start from next complete month
        start_date = self.start_of_next_month

        # End at January of year+2 (needed for Dec 31 year-end marker calculation)
        # This gives us: remaining current year + full next year + January of year+2
        current_year = self.current_datetime.year
        year_plus_2 = current_year + 2
        end_date = datetime(year_plus_2, 1, 31, tzinfo=timezone.utc)

        return start_date, end_date

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
        df = self.series.pd_dataframe()
        yearly = {}
        for idx, row in df.iterrows():
            year = idx.year
            yearly[year] = yearly.get(year, 0) + int(row.iloc[0])
        return yearly

    def apply_constraints(self, forecasts: Dict[str, ForecastResult]) -> Dict[str, ForecastResult]:
        """
        Apply CVE-specific forecast constraints.

        Converts monthly ForecastResult values into yearly totals,
        applies growth floor / trend constraints via ForecastConstraints,
        then scales monthly values proportionally so yearly totals match.

        Args:
            forecasts: Raw forecasts from all models

        Returns:
            Constrained forecasts with adjusted monthly values
        """
        if not self.forecast_constraints:
            self.logger.warning('Forecast constraints not initialized, passing through')
            return forecasts

        # 1. Build yearly totals by summing monthly forecast values per model per year
        yearly_totals: Dict[int, Dict[str, int]] = {}
        for model_name, result in forecasts.items():
            if model_name == 'Ensemble':
                continue
            for date_str, value in result.forecast_values.items():
                year = pd.to_datetime(date_str).year
                if year not in yearly_totals:
                    yearly_totals[year] = {}
                yearly_totals[year][model_name] = yearly_totals[year].get(model_name, 0) + int(round(value))

        if not yearly_totals:
            self.logger.info('No yearly totals to constrain, passing through')
            return forecasts

        # 2. Get previous year actuals from historical series data
        prev_year_actuals = self._get_previous_year_actuals()

        # 3. Apply constraints
        self.logger.info(
            f'Applying constraints to {len(yearly_totals)} forecast years, '
            f'{sum(len(m) for m in yearly_totals.values())} model-year entries'
        )
        constrained_totals = self.forecast_constraints.apply_constraints(
            yearly_totals, previous_year_actuals=prev_year_actuals
        )

        # 4. Calculate scaling factors and apply back to monthly forecasts
        adjusted_count = 0
        for model_name, result in forecasts.items():
            if model_name == 'Ensemble':
                continue

            # Group monthly values by year for this model
            year_months: Dict[int, List[str]] = {}
            for date_str in result.forecast_values:
                year = pd.to_datetime(date_str).year
                year_months.setdefault(year, []).append(date_str)

            for year, date_strs in year_months.items():
                original_total = yearly_totals.get(year, {}).get(model_name, 0)
                constrained_total = constrained_totals.get(year, {}).get(model_name, original_total)

                if original_total == 0 or constrained_total == original_total:
                    continue

                # Scale each month proportionally
                scale_factor = constrained_total / original_total
                for date_str in date_strs:
                    old_val = result.forecast_values[date_str]
                    result.forecast_values[date_str] = round(old_val * scale_factor, 2)

                adjusted_count += 1
                self.logger.info(
                    f'  {model_name} {year}: {original_total:,} -> {constrained_total:,} (scale {scale_factor:.4f})'
                )

        self.logger.info(f'Constraints applied to {len(forecasts)} models ({adjusted_count} model-year adjustments)')
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
        df = self.series.pd_dataframe() if hasattr(self.series, 'pd_dataframe') else self.series.to_dataframe()
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
        """
        self.logger.info('Generating cumulative forecast timelines with year boundaries')
        cumulative_timelines = {}

        for model_name, forecast_result in forecasts.items():
            if model_name == 'Ensemble':
                continue

            timeline: List[Dict[str, Any]] = []
            if not forecast_result.forecast_values:
                cumulative_timelines[f'{model_name}_cumulative'] = timeline
                continue

            sorted_dates = sorted(forecast_result.forecast_values.items())

            year_total = actuals_base
            current_year = pd.to_datetime(sorted_dates[0][0]).year

            # Add Jan 1 marker for the first forecast year
            timeline.append({'date': f'{current_year}-01-01T00:00:00Z', 'cumulative_total': 0})

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

                # Add month-start entry BEFORE adding this month's forecast
                month_start_date = f'{forecast_date.year}-{forecast_date.month:02d}-01T00:00:00Z'

                # Check if this date already exists (e.g., Jan 1 from year boundary)
                existing_entry = next((entry for entry in timeline if entry['date'] == month_start_date), None)
                if not existing_entry:
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

        # Generate all_models_cumulative (average)
        if cumulative_timelines:
            all_model_timelines = [tl for name, tl in cumulative_timelines.items() if name != 'Ensemble_cumulative']
            if all_model_timelines:
                all_dates = sorted(list(set(item['date'] for tl in all_model_timelines for item in tl)))
                avg_timeline = []
                for date_str in all_dates:
                    totals = [
                        item['cumulative_total']
                        for tl in all_model_timelines
                        for item in tl
                        if item['date'] == date_str
                    ]
                    if totals:
                        avg_timeline.append(
                            {'date': date_str, 'cumulative_total': int(round(sum(totals) / len(totals)))}
                        )
                cumulative_timelines['all_models_cumulative'] = avg_timeline

        self.logger.info(f'Generated {len(cumulative_timelines)} cumulative timelines')
        return cumulative_timelines

    def _generate_model_rankings_with_backtest(
        self, forecasts: Dict[str, ForecastResult], backtest_metrics: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Generate model rankings sorted by performance using backtest metrics.

        Args:
            forecasts: Model forecasts
            backtest_metrics: Backtest MAE/MAPE for each model

        Returns:
            List of ranking entries sorted by MAPE
        """
        self.logger.info('Generating model rankings with backtest metrics')

        rankings = []

        for model_name, forecast_result in forecasts.items():
            if model_name == 'Ensemble':
                continue  # Skip ensemble

            # Get model config
            model_config = self.config.get('models', {}).get(model_name, {})
            hyperparameters = model_config.get('hyperparameters', {})
            tuning_results = model_config.get('tuning_results', {})

            # Use backtest metrics if available, otherwise use forecast_result metrics
            metrics = backtest_metrics.get(model_name, {})

            ranking_entry = {
                'model_name': model_name,
                'mape': metrics.get('mean_absolute_percentage_error'),
                'mae': metrics.get('mean_absolute_error'),
                'rmse': None,  # Not calculated in backtest
            }

            # Add hyperparameters if meaningful
            if hyperparameters and any(v for v in hyperparameters.values() if v is not None):
                ranking_entry['hyperparameters'] = hyperparameters

            # Add tuning metadata
            if tuning_results:
                if 'tuned_at' in tuning_results:
                    ranking_entry['tuned_at'] = tuning_results['tuned_at']
                if 'method' in tuning_results:
                    ranking_entry['tuning_method'] = tuning_results['method']

            rankings.append(ranking_entry)

        # Sort by MAPE (lower is better)
        rankings.sort(key=lambda x: x.get('mape') if x.get('mape') is not None else float('inf'))

        self.logger.info(f'Generated rankings for {len(rankings)} models')
        return rankings

    def _calculate_yearly_totals(
        self, cumulative_timelines: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[int, Dict[str, int]]:
        """
        Calculate year-end totals from cumulative timelines.

        Args:
            cumulative_timelines: Generated cumulative timelines

        Returns:
            Dict of {year: {model_name: total}}
        """
        self.logger.info('Calculating yearly forecast totals')

        yearly_totals = {}

        for model_key, timeline in cumulative_timelines.items():
            if not timeline:
                continue

            model_name = model_key.replace('_cumulative', '')

            # Find last entry for each year (year-end total)
            # Group entries by year and take the last (highest cumulative) for each
            year_entries = {}
            for entry in timeline:
                if entry['cumulative_total'] == 0:  # Skip Jan 1 reset markers
                    continue
                year = int(entry['date'][:4])
                if year not in year_entries or entry['date'] > year_entries[year]['date']:
                    year_entries[year] = entry

            # Store year-end totals
            for year, entry in year_entries.items():
                if year not in yearly_totals:
                    yearly_totals[year] = {}
                yearly_totals[year][model_name] = entry['cumulative_total']

        self.logger.info(f'Calculated totals for {len(yearly_totals)} years')
        return yearly_totals

    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics.

        Returns:
            Summary dict with data/forecast periods and aggregate stats
        """
        self.logger.info('Generating summary statistics')

        df = self.series.pd_dataframe() if hasattr(self.series, 'pd_dataframe') else self.series.to_dataframe()

        summary = {
            'data_period': {'start': df.index.min().strftime('%Y-%m-%d'), 'end': df.index.max().strftime('%Y-%m-%d')},
            'forecast_period': {
                'start': self.start_of_next_month.strftime('%Y-%m-%d'),
                'end': datetime(self.current_datetime.year + 1, 12, 31).strftime('%Y-%m-%d'),
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

    def _save_forecast_snapshot(self, forecasts: Dict[str, ForecastResult], actuals_cumulative: List[Dict]):
        """
        Save current forecast snapshot to ForecastTracker for future comparison.

        Args:
            forecasts: Current forecasts from all models
            actuals_cumulative: Current actual CVE counts
        """
        try:
            # Initialize tracker
            tracker = ForecastTracker()

            # Prepare forecasts in tracker format: {"2025-10": {"Prophet": 4049, "LightGBM": 4100, ...}}
            forecast_dict = {}
            for model_name, forecast_result in forecasts.items():
                if model_name == 'Ensemble':
                    continue
                for date_str, cve_count in forecast_result.forecast_values.items():
                    # Convert "2025-10-31" to "2025-10"
                    month_str = pd.to_datetime(date_str).strftime('%Y-%m')
                    if month_str not in forecast_dict:
                        forecast_dict[month_str] = {}
                    forecast_dict[month_str][model_name] = float(cve_count)

            # Prepare actuals in tracker format: {"2025-01": 4274, "2025-02": 3676, ...}
            for entry in actuals_cumulative:
                date_str = entry['date']
                if 'T00:00:00Z' in date_str and date_str.endswith('-01T00:00:00Z'):
                    # This is a month boundary entry
                    month_str = pd.to_datetime(date_str).strftime('%Y-%m')
                    # Get the actual count for this month (not cumulative)
                    # We'll need to calculate month-over-month difference
                    # For now, skip this as it requires more complex logic
                    pass

            # Prepare model performance metrics
            model_performance = {}
            for model_name in self.model_results:
                if model_name in forecasts:
                    metrics = self.model_results[model_name].get('metrics', {})
                    model_performance[model_name] = {'mape': metrics.get('mape'), 'mae': metrics.get('mae')}

            # Add snapshot
            tracker.add_snapshot(
                forecasts=forecast_dict,
                actuals={},  # Will be populated in future runs when months complete
                model_performance=model_performance,
                snapshot_date=self.current_datetime,
                metadata={
                    'data_periods': len(self.series),
                    'forecast_horizon': len(list(forecasts.values())[0].forecast_values) if forecasts else 0,
                },
            )

            self.logger.info('✓ Saved forecast snapshot to tracker')

        except Exception as e:
            self.logger.warning(f'Could not save forecast snapshot: {e}')

    def _calculate_forecast_vs_published(self, model_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Calculate validation data using historical backtest for current year.

        This performs a backtest by:
        1. Training the model on data through end of previous year
        2. Forecasting all months of current year
        3. Comparing forecasts against actual published CVE counts for completed months

        Args:
            model_name: Name of model to validate

        Returns:
            Tuple of (table_data, summary_stats)
        """
        # Check if we have trained model results
        if model_name not in self.model_results:
            return [], {}

        model_result = self.model_results[model_name]
        if not model_result.get('trained', False):
            return [], {}

        try:
            # Get full dataset
            df = self.series.pd_dataframe() if hasattr(self.series, 'pd_dataframe') else self.series.to_dataframe()
            current_year = self.current_datetime.year

            # Split: train on data through end of previous year
            df['year'] = df.index.year
            train_df = df[df['year'] < current_year]
            actual_df = df[df['year'] == current_year].copy()

            if train_df.empty or actual_df.empty:
                return [], {}

            # Only compare completed months (exclude current incomplete month)
            import pandas as pd

            current_month_start = pd.Timestamp.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            completed_months_df = actual_df[actual_df.index < current_month_start]

            if completed_months_df.empty:
                return [], {}

            # Create training series
            from darts import TimeSeries

            train_series = TimeSeries.from_dataframe(train_df.drop('year', axis=1), freq='M', fill_missing_dates=False)

            # Train a fresh model on historical data only
            hyperparameters = model_result.get('hyperparameters', {})
            backtest_model = self.create_model(model_name, hyperparameters)
            backtest_model.fit(train_series)

            # Forecast for all months in current year
            num_months = len(completed_months_df)
            forecast_series = backtest_model.predict(num_months)

            # Compare forecasts to actuals
            table_data = []
            errors = []
            percent_errors = []

            for i, (date, row) in enumerate(completed_months_df.iterrows()):
                actual_count = int(row.iloc[0])
                forecast_count = int(round(forecast_series.values()[i][0]))
                month_str = date.strftime('%Y-%m')  # e.g., "2025-01"

                error = forecast_count - actual_count
                percent_error = (error / actual_count * 100) if actual_count != 0 else 0

                # Determine performance rating
                abs_pct_error = abs(percent_error)
                if abs_pct_error < 5:
                    performance = 'Excellent'
                elif abs_pct_error < 10:
                    performance = 'Good'
                elif abs_pct_error < 20:
                    performance = 'Fair'
                else:
                    performance = 'Poor'

                table_data.append(
                    {
                        'MONTH': month_str,
                        'PUBLISHED': actual_count,
                        'FORECAST': forecast_count,
                        'ERROR': error,
                        'PERCENT_ERROR': round(percent_error, 2),
                        'PERFORMANCE': performance,
                    }
                )

                errors.append(abs(error))
                percent_errors.append(abs_pct_error)

            # Calculate summary stats
            summary_stats = {
                'mean_absolute_error': round(float(np.mean(errors)), 2),
                'mean_absolute_percentage_error': round(float(np.mean(percent_errors)), 2),
            }

            self.logger.info(
                f'{model_name} backtest: MAE={summary_stats["mean_absolute_error"]}, MAPE={summary_stats["mean_absolute_percentage_error"]}%'
            )

            return table_data, summary_stats

        except Exception as e:
            self.logger.warning(f'Could not generate backtest for {model_name}: {e}')
            return [], {}

    def save_results(self, forecasts: Dict[str, ForecastResult]) -> str:
        """
        Save CVE forecast results to web/data.json with complete structure.

        Generates all required data structures including:
        - Model rankings (sorted by MAPE)
        - Yearly forecast totals (2025, 2026)
        - Cumulative timelines (historical + forecast with year boundaries)
        - Actuals cumulative (current year progress)
        - Forecast vs published validation data
        - Summary statistics

        Args:
            forecasts: Final forecasts

        Returns:
            Path to saved file
        """
        output_path = Path(self.config['file_paths'].get('output', 'web/data.json'))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f'Saving complete CVE forecast data to {output_path}...')

        # 1. Generate forecast_vs_published backtest data FIRST (to get metrics)
        self.logger.info('Generating backtest validation data...')
        forecast_vs_published = {}
        backtest_metrics = {}  # Store metrics for model rankings
        for model_name in [m for m in forecasts.keys() if m != 'Ensemble']:
            table_data, summary_stats = self._calculate_forecast_vs_published(model_name)
            forecast_vs_published[model_name] = {'table_data': table_data, 'summary_stats': summary_stats}
            # Store metrics for rankings
            if summary_stats:
                backtest_metrics[model_name] = summary_stats

        # 2. Generate model rankings using backtest metrics
        model_rankings = self._generate_model_rankings_with_backtest(forecasts, backtest_metrics)

        # 3. Generate actuals cumulative for current year
        actuals_cumulative = self._get_actuals_cumulative()

        # Get base cumulative value for forecasts (last COMPLETE month, not current MTD)
        # Find the last entry that's on the 1st of a month (complete month boundary)
        actuals_base = 0
        for entry in reversed(actuals_cumulative):
            if 'T00:00:00Z' in entry['date'] and entry['date'].endswith('-01T00:00:00Z'):
                actuals_base = entry['cumulative_total']
                self.logger.info(f'Using {entry["date"]} as forecast base: {actuals_base:,} CVEs')
                break

        if actuals_base == 0 and actuals_cumulative:
            # Fallback to last entry if no month boundary found
            actuals_base = actuals_cumulative[-1]['cumulative_total']

        # 3. Generate cumulative forecast timelines with year boundaries
        cumulative_timelines = self._generate_cumulative_timelines(forecasts, actuals_base)

        # 4. Calculate yearly forecast totals from cumulative timelines
        yearly_forecast_totals = self._calculate_yearly_totals(cumulative_timelines)

        # 4.5. Add Dec 31 projection to actuals_cumulative for current year
        current_year = self.current_datetime.year
        if str(current_year) in yearly_forecast_totals and 'all_models' in yearly_forecast_totals[str(current_year)]:
            # Use all_models average for Dec 31 projection
            dec_31_total = yearly_forecast_totals[str(current_year)]['all_models']
            actuals_cumulative.append({'date': f'{current_year}-12-31T23:59:59Z', 'cumulative_total': dec_31_total})
            self.logger.info(f'Added Dec 31 {current_year} projection: {dec_31_total:,} CVEs')

        # 5. Generate summary statistics
        summary = self._generate_summary()

        # 6. Convert forecasts to simple month format for backwards compatibility
        forecasts_simple = {}
        for model_name, forecast_result in forecasts.items():
            if model_name == 'Ensemble':
                continue

            forecast_list = []
            for date_str, cve_count in sorted(forecast_result.forecast_values.items()):
                date_obj = pd.to_datetime(date_str)
                forecast_list.append({'date': date_obj.strftime('%Y-%m'), 'cve_count': int(cve_count)})
            forecasts_simple[model_name] = forecast_list

        # 7. Add weighted ensemble forecast (average of all models)
        all_forecast_dates = set()
        for forecast_result in forecasts.values():
            if forecast_result.model_name == 'Ensemble':
                continue
            all_forecast_dates.update(forecast_result.forecast_values.keys())

        ensemble_list = []
        for date_str in sorted(all_forecast_dates):
            values = [
                forecast_result.forecast_values.get(date_str, 0)
                for forecast_result in forecasts.values()
                if forecast_result.model_name != 'Ensemble' and date_str in forecast_result.forecast_values
            ]
            if values:
                date_obj = pd.to_datetime(date_str)
                ensemble_list.append({'date': date_obj.strftime('%Y-%m'), 'cve_count': int(np.median(values))})

        if ensemble_list:
            forecasts_simple['all_models_avg'] = ensemble_list

        # 8. Generate current month actual data
        current_year = self.current_datetime.year
        df = self.series.pd_dataframe() if hasattr(self.series, 'pd_dataframe') else self.series.to_dataframe()
        df['year'] = df.index.year
        current_year_df = df[df['year'] == current_year]

        current_month_actual = {
            'date': self.current_datetime.strftime('%Y-%m'),
            'cve_count': int(current_year_df.iloc[-1, 0]) if not current_year_df.empty else 0,
            'cumulative_total': actuals_base,
        }

        # 9. Save forecast snapshot to tracker for future comparison
        self._save_forecast_snapshot(forecasts, actuals_cumulative)

        # 10. Assemble complete output structure
        output_data = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'model_rankings': model_rankings,
            'yearly_forecast_totals': yearly_forecast_totals,
            'current_month_actual': current_month_actual,
            'actuals_cumulative': actuals_cumulative,
            'cumulative_timelines': cumulative_timelines,
            'forecasts': forecasts_simple,
            'summary': summary,
            'forecast_vs_published': forecast_vs_published,
        }

        # Save to file with NaN/inf handling
        with open(output_path, 'w') as f:

            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        if np.isnan(obj) or np.isinf(obj):
                            return None
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)

            json.dump(output_data, f, indent=2, cls=NumpyEncoder)

        self.logger.info('✓ Saved complete CVE forecast data:')
        self.logger.info(f'  - Models: {len(model_rankings)} ranked')
        self.logger.info(f'  - Years: {list(yearly_forecast_totals.keys())}')
        self.logger.info(f'  - Cumulative timelines: {len(cumulative_timelines)}')
        self.logger.info(f'  - Actuals data points: {len(actuals_cumulative)}')

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

    def run_full_pipeline(
        self, train_ratio: float = 0.8, run_validation: bool = True, run_diagnostics: bool = False
    ) -> Dict[str, Any]:
        """
        Execute complete CVE forecasting pipeline.

        Args:
            train_ratio: Train/test split ratio
            run_validation: Whether to run cross-validation
            run_diagnostics: Whether to run diagnostics

        Returns:
            Pipeline results
        """
        self.logger.info('=' * 70)
        self.logger.info('CVE FORECASTING PIPELINE - STARTING')
        self.logger.info('=' * 70)

        results = {}

        # 1. Load data
        self.load_data()
        results['data_loaded'] = True
        results['data_periods'] = len(self.series)

        # 2. Load optimized models
        self.load_optimized_models()
        results['models_loaded'] = len(self.model_results)

        # 3. Train models
        self.train_all_models(train_ratio=train_ratio)
        trained_count = len([m for m in self.model_results.values() if m.get('trained', False)])
        results['models_trained'] = trained_count

        # 4. Generate forecasts
        start_date, end_date = self.get_forecast_horizon()
        forecast_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

        raw_forecasts = self.generate_forecasts(forecast_horizon=forecast_months)
        results['raw_forecasts_generated'] = len(raw_forecasts)

        # 5. Apply constraints
        constrained_forecasts = self.apply_constraints(raw_forecasts)
        results['constrained_forecasts'] = len(constrained_forecasts)

        # 6. Save results
        output_path = self.save_results(constrained_forecasts)
        results['output_path'] = output_path

        # 7. Optional validation
        if run_validation:
            self.logger.info('\nRunning cross-validation...')
            cv_results = self.perform_cross_validation(n_splits=5, forecast_horizon=12)
            results['cv_completed'] = True
            results['cv_models'] = len(cv_results)

            # Statistical tests
            if len(cv_results) >= 2:
                self.perform_statistical_tests(cv_results)
                results['statistical_tests'] = True

        # 8. Optional diagnostics
        if run_diagnostics:
            self.logger.info('\nRunning diagnostics...')
            diag_results = self.run_residual_diagnostics()
            results['diagnostics_completed'] = True
            results['diagnostics_models'] = len(diag_results)

        self.logger.info('=' * 70)
        self.logger.info('CVE FORECASTING PIPELINE - COMPLETE')
        self.logger.info('=' * 70)
        self.logger.info(f'✓ Data: {results["data_periods"]} periods')
        self.logger.info(f'✓ Models trained: {results["models_trained"]}')
        self.logger.info(f'✓ Forecasts: {results["constrained_forecasts"]} models')
        self.logger.info(f'✓ Output: {results["output_path"]}')

        return results
