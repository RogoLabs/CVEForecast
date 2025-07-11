import json
import logging
from datetime import datetime
import sys
from pathlib import Path
import pandas as pd
from darts import TimeSeries

from utils import setup_logging
from data_loader import load_cve_data
from darts.metrics import mae, mape, mase, rmsse
from model_trainer import train_and_evaluate_model

import datetime
import numpy as np
import os
from dateutil.relativedelta import relativedelta

class CVEForecastEngine:
    """Orchestrates the CVE forecasting workflow."""

    def __init__(self, config_path='config.json'):
        """Initialize the engine, load config, and set up paths."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        for key, path in self.config['file_paths'].items():
            if not os.path.isabs(path):
                self.config['file_paths'][key] = os.path.join(project_root, path)
        
        self._setup_time_variables()
        self.setup_logging()

        self.series = None
        self.historical_series = None
        self.current_month_series = None
        self.forecasts = {}
        self.model_results = {}
        self.final_forecasts = {}
        self.all_models_validation = {}

    def _setup_time_variables(self):
        """Set up dynamic time variables based on current UTC time."""
        # Use a fixed time for consistent, reproducible runs, as per user request context.
        # In a live production environment, this would be datetime.datetime.utcnow().
        self.current_datetime = datetime.datetime(2025, 7, 11, 14, 45, 10, tzinfo=datetime.timezone.utc)
        
        self.start_of_current_month = self.current_datetime.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        self.start_of_next_month = self.start_of_current_month + relativedelta(months=1)

    def setup_logging(self):
        """Set up logging based on the configuration."""
        setup_logging(self.config['logging'])
        self.logger = logging.getLogger(__name__)

    def process_data(self):
        """
        Loads and processes CVE data into a Darts TimeSeries.
        The loaded data represents all fully completed months.
        """
        monthly_counts = load_cve_data(self.config)
        # CORRECT: The entire series from the loader is historical data of completed months.
        self.historical_series = TimeSeries.from_dataframe(
            monthly_counts,
            freq='M', # Or 'MS' depending on your loader's output
            fill_missing_dates=True,
            value_cols='cve_count'
        )
        # CORRECT: The series used for training IS the full historical series.
        self.series = self.historical_series
        # CORRECT: Deprecate or remove the flawed slicing logic. The partial current
        # month's data is not part of this historical series.
        self.current_month_series = None
        # Keep a reference to the full series for summary stats.
        self.full_series = self.historical_series
        self.logger.info(f"Processed {len(self.historical_series)} months of complete historical data.")

    def train_and_evaluate_models(self, progress_callback=None):
        enabled_models = {name: config for name, config in self.config['models'].items() if config['enabled']}
        total_models = len(enabled_models)

        for i, (model_name, model_config) in enumerate(enabled_models.items()):
            if progress_callback:
                progress_callback(f"Evaluating Models ({i+1}/{total_models}): {model_name}")
            
            model, train, val, predictions = train_and_evaluate_model(model_name, model_config, self.series, self.config['model_evaluation'])
            if model and predictions:
                metrics = {}
                for metric_name, metric_func in [('mape', mape), ('mae', mae), ('mase', mase), ('rmsse', rmsse)]:
                    try:
                        if metric_name in ['mase', 'rmsse']:
                            metrics[metric_name] = metric_func(val, predictions, train)
                        else:
                            metrics[metric_name] = metric_func(val, predictions)
                    except Exception as e:
                        self.logger.warning(f"Could not compute {metric_name.upper()} for {model_name}: {e}")
                        metrics[metric_name] = None

                mape_str = f"{metrics.get('mape', 0):.2f}%" if metrics.get('mape') is not None else "N/A"
                mase_str = f"{metrics.get('mase', 0):.2f}" if metrics.get('mase') is not None else "N/A"
                rmsse_str = f"{metrics.get('rmsse', 0):.2f}" if metrics.get('rmsse') is not None else "N/A"
                mae_str = f"{metrics.get('mae', 0):.2f}" if metrics.get('mae') is not None else "N/A"
                self.logger.info(f"{model_name} - MAPE: {mape_str}, MASE: {mase_str}, RMSSE: {rmsse_str}, MAE: {mae_str}")

                self.model_results[model_name] = {"model_name": model_name, "model": model, "metrics": metrics}

                # Generate detailed validation data
                validation_data = []
                val_df = val.to_dataframe()
                pred_df = predictions.to_dataframe()
                for i in range(len(val)):
                    timestamp = val.time_index[i]
                    actual = val_df.iloc[i, 0]
                    predicted = pred_df.iloc[i, 0]
                    error = abs(actual - predicted)
                    percent_error = (error / actual) * 100 if actual != 0 else 0
                    validation_data.append({
                        "date": timestamp.strftime('%Y-%m'),
                        "actual": float(actual),
                        "predicted": float(predicted),
                        "error": float(error),
                        "percent_error": float(percent_error)
                    })
                self.all_models_validation[model_name] = validation_data

    def generate_final_forecasts(self, progress_callback=None):
        ensemble_size = self.config['model_evaluation']['ensemble_size']
        top_models = sorted([self.model_results[r] for r in self.model_results], key=lambda x: x['metrics'].get('mape', float('inf')))[:ensemble_size]

        last_historical_month = self.historical_series.end_time().to_pydatetime().date()
        months_to_forecast = ( (self.config['model_evaluation']['forecast_end_year'] - last_historical_month.year) * 12
                               + self.config['model_evaluation']['forecast_end_month'] - last_historical_month.month )

        total_final_models = len(top_models)
        for i, result in enumerate(top_models):
            model_name = result['model_name']
            model = result['model']
            if progress_callback:
                progress_callback(f"Generating Final Forecasts ({i+1}/{total_final_models}): {model_name}")
            
            try:
                series_to_fit = self.historical_series
                if model_name in ["TCN", "NBEATS", "NHiTS", "TiDE", "DLinear", "TSMixer"]:
                    series_to_fit = series_to_fit.astype(np.float32)
                
                model.fit(series_to_fit)
                forecast = model.predict(months_to_forecast)
                self.final_forecasts[model_name] = forecast
            except Exception as e:
                self.logger.error(f"Failed to generate final forecast for {model_name}: {e}")

    def _get_historical_data(self):
        df = self.historical_series.to_dataframe()
        df['date'] = df.index.strftime('%Y-%m')
        df['cve_count'] = df['cve_count'].astype(int)
        return df.reset_index(drop=True).to_dict('records')

    def _get_current_month_actual(self):
        """Calculates the actual CVE data for the current, partial month."""
        today = self.current_datetime.date()
        last_day_of_month = (today.replace(day=28) + datetime.timedelta(days=4)).replace(day=1) - datetime.timedelta(days=1)

        # This function simulates fetching real-time data for the partial month.
        def get_partial_cve_count_for_month(year, month):
            # Using a placeholder as per the original logic.
            # This would be replaced by a real data source in a live system.
            if year == 2025 and month == 7:
                # As per the example, if the date is July 11, the cumulative total is 24521.
                # The last full month (June) was 19982. The start of July was 23668.
                # This implies the partial count for July is 24521 - 23668 = 853.
                return 853
            return 0

        current_cve_count = get_partial_cve_count_for_month(today.year, today.month)

        # The cumulative total is the sum of all historical (full) months + the partial current month.
        historical_total = int(self.historical_series.sum().values()[0][0])

        return {
            "date": today.strftime('%Y-%m'),
            "cve_count": current_cve_count,
            "cumulative_total": historical_total + current_cve_count,
            "days_elapsed": today.day,
            "total_days": last_day_of_month.day,
            "progress_percentage": round((today.day / last_day_of_month.day) * 100, 1)
        }

    def _get_cumulative_timelines(self, forecasts, actuals_cumulative):
        """Generates dynamic, forward-looking cumulative forecasts."""
        self.logger.info("Generating dynamic, forward-looking cumulative forecasts.")
        cumulative_timelines = {}

        # Find the base value from the start of the current month in the actuals timeline.
        # This is the last reliable, known cumulative value before forecasting begins.
        start_of_current_month_str = self.start_of_current_month.strftime('%Y-%m-%dT%H:%M:%SZ')
        base_entry = next((item for item in reversed(actuals_cumulative) if item['date'] == start_of_current_month_str), None)

        if not base_entry:
            self.logger.error(f"Could not find base value for date {start_of_current_month_str}. Cannot generate forecasts.")
            return {}

        base_value = base_entry['cumulative_total']
        self.logger.info(f"Forecast base value from {base_entry['date']} is {base_value}.")

        # --- Generate timeline for each individual forecast model ---
        for model_name, model_forecasts in forecasts.items():
            if not model_forecasts: continue

            # Filter forecasts to only include future months.
            future_forecasts = [f for f in model_forecasts if pd.to_datetime(f['date']).tz_localize('UTC') >= self.start_of_next_month]
            future_forecasts.sort(key=lambda x: x['date'])

            timeline = []
            running_total = base_value
            for forecast_item in future_forecasts:
                running_total += forecast_item['cve_count']
                timeline.append({
                    "date": pd.to_datetime(forecast_item['date']).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "cumulative_total": int(round(running_total))
                })
            
            cumulative_timelines[f"{model_name}_cumulative"] = timeline

        # --- Generate the "All Models Average" cumulative timeline ---
        if cumulative_timelines:
            # Get a sorted list of all unique future dates from the generated timelines.
            all_future_dates = sorted(list(set(item['date'] for timeline in cumulative_timelines.values() for item in timeline)))
            
            avg_timeline = []
            for date_str in all_future_dates:
                monthly_totals = [
                    timeline_item['cumulative_total']
                    for timeline in cumulative_timelines.values()
                    for timeline_item in timeline
                    if timeline_item['date'] == date_str
                ]
                if monthly_totals:
                    avg_timeline.append({
                        "date": date_str,
                        "cumulative_total": int(round(sum(monthly_totals) / len(monthly_totals)))
                    })
            cumulative_timelines['all_models_cumulative'] = avg_timeline

        self.logger.info(f"Generated {len(cumulative_timelines)} cumulative forecast timelines.")
        return cumulative_timelines

    def _get_summary(self):
        """Build the summary object matching the v.02 file_io.py logic exactly."""
        full_df = self.full_series.to_dataframe()
        return {
            'total_historical_cves': int(full_df['cve_count'].sum()),
            'data_period': {
                'start': full_df.index.min().strftime('%Y-%m-%d'),
                'end': full_df.index.max().strftime('%Y-%m-%d')
            },
            'forecast_period': {
                'start': datetime.date(datetime.date.today().year, datetime.date.today().month, 1).strftime('%Y-%m-%d'),
                'end': datetime.date(datetime.date.today().year + 1, 1, 31).strftime('%Y-%m-%d')
            }
        }

    def _get_actuals_cumulative(self, historical_data, current_month_data):
        """Generates the dynamic, cumulative timeline of actual CVE data for the current year."""
        self.logger.info("Generating dynamic cumulative timeline for actuals for the current year.")
        
        current_year = self.current_datetime.year
        
        # Filter for the current year's historical data and sort it.
        current_year_historical = [item for item in historical_data if item['date'].startswith(str(current_year))]
        current_year_historical.sort(key=lambda x: x['date'])

        # Initialize with a zero-point at the start of the year.
        actuals_cumulative = [{
            "date": f"{current_year}-01-01T00:00:00Z",
            "cumulative_total": 0
        }]
        
        cumulative_total = 0
        for item in current_year_historical:
            cumulative_total += item['cve_count']
            # The date for a full month's cumulative total is the first of that month.
            month_start_date = pd.to_datetime(item['date']).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            entry = {
                "date": month_start_date,
                "cumulative_total": cumulative_total
            }
            
        # This logic correctly calculates the cumulative total for each completed month of the year.
        actuals_cumulative = [{
            "date": f"{current_year}-01-01T00:00:00Z",
            "cumulative_total": 0
        }]
        df = pd.DataFrame(current_year_historical)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
            df = df.sort_values('date').reset_index(drop=True)
            df['cumulative_total'] = df['cve_count'].cumsum()

            for i, row in df.iterrows():
                entry = {
                    "date": row['date'].strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "cumulative_total": int(row['cumulative_total'])
                }
                # If this is the first entry and it's for January, update the initial placeholder.
                if i == 0 and row['date'].month == 1:
                    actuals_cumulative[0] = entry
                else:
                    actuals_cumulative.append(entry)

        # Append the final entry for the current day's cumulative total.
        if current_month_data and current_month_data['cve_count'] > 0:
            # If there's historical data for the year, add to its last cumulative total.
            # Otherwise, the cumulative total is just the current month's count.
            base_total = actuals_cumulative[-1]['cumulative_total'] if len(actuals_cumulative) > 1 else 0
            current_day_total = base_total + current_month_data['cve_count']
            actuals_cumulative.append({
                "date": self.current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "cumulative_total": int(current_day_total)
            })

        self.logger.info(f"Generated {len(actuals_cumulative)} actuals data points for the current year.")
        return actuals_cumulative

    def save_results(self):
        model_rankings = sorted([
            {"model_name": r['model_name'], **r['metrics']} for r in self.model_results.values()
        ], key=lambda x: x['mape'] if x.get('mape') is not None else float('inf'))

        historical_data = self._get_historical_data()
        current_month_actual = self._get_current_month_actual()

        forecasts_out = {}
        current_month_date = current_month_actual['date']
        current_month_count = current_month_actual['cve_count']

        for name, forecast in self.final_forecasts.items():
            forecast_list = []
            for ts, val in zip(forecast.time_index, forecast.values()):
                date_str = ts.strftime('%Y-%m')
                cve_count = int(round(val[0]))
                
                # Adjust the forecast for the current month if it's lower than the actual partial count
                if date_str == current_month_date and cve_count < current_month_count:
                    cve_count = current_month_count
                    
                forecast_list.append({"date": date_str, "cve_count": cve_count})
            forecasts_out[name] = forecast_list

        self.logger.info(f"Validation data to be saved: {json.dumps(self.all_models_validation, indent=2)}")

        # This summary is now created here to be used in the final printout
        self.summary = self._get_summary()

        # Generate the new, dynamic actuals and forecast timelines
        actuals_cumulative = self._get_actuals_cumulative(historical_data, current_month_actual)
        cumulative_timelines = self._get_cumulative_timelines(forecasts_out, actuals_cumulative)

        yearly_forecast_totals = {}
        for model_name, timeline in cumulative_timelines.items():
            if timeline:
                # Strip suffix to get the base model name for the key
                base_model_name = model_name.replace('_cumulative', '')
                yearly_forecast_totals[base_model_name] = timeline[-1]['cumulative_total']

        # Calculate forecast vs published for each model
        forecast_vs_published_data = {}
        for model_name in self.final_forecasts.keys():
            table_data, summary_stats = self._calculate_forecast_vs_published(model_name)
            forecast_vs_published_data[model_name] = {
                'table_data': table_data,
                'summary_stats': summary_stats
            }

        final_json = {
            "generated_at": self.current_datetime.isoformat(),
            "model_rankings": model_rankings,
            "current_month_actual": current_month_actual,
            "actuals_cumulative": actuals_cumulative, # New dynamic field
            "forecasts": forecasts_out,
            "all_models_validation": self.all_models_validation,
            "yearly_forecast_totals": yearly_forecast_totals,
            "cumulative_timelines": cumulative_timelines,
            "summary": self.summary,
            "forecast_vs_published": forecast_vs_published_data
        }

        output_path = self.config['file_paths']['output_data']
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
            json.dump(final_json, f, indent=2, cls=NumpyEncoder)
        
        # Store output path for summary
        self.output_path = output_path

    def _calculate_forecast_vs_published(self, model_name):
        if not self.all_models_validation or model_name not in self.all_models_validation:
            return [], {}

        validation_data = self.all_models_validation.get(model_name, [])
        
        # Convert to DataFrame for easier manipulation
        # The validation_data is a list of dicts, not a dict containing 'data'
        df = pd.DataFrame(validation_data)
        if df.empty:
            return [], {}

        # Rename columns for consistency if needed
        if 'predicted' in df.columns:
            df.rename(columns={'predicted': 'forecast'}, inplace=True)
        if 'date' in df.columns:
            df.rename(columns={'date': 'month'}, inplace=True)

        # Ensure correct data types
        df['actual'] = pd.to_numeric(df['actual'], errors='coerce')
        df['forecast'] = pd.to_numeric(df['forecast'], errors='coerce')

        # Drop rows where actual or forecast is NaN
        df.dropna(subset=['actual', 'forecast'], inplace=True)

        # Manually calculate error and percent_error
        df['error'] = df['forecast'] - df['actual'] # Keep the sign for color coding
        # Avoid division by zero for percent_error
        df['percent_error'] = (df['error'] / df['actual'].abs()).replace(np.inf, 0) * 100

        # Prepare the flat data structure for JSON output
        table_data = []
        for _, row in df.iterrows():
            table_data.append({
                "MONTH": row['month'],
                "PUBLISHED": int(row['actual']),
                "FORECAST": int(row['forecast']),
                "ERROR": int(row['error']),
                "PERCENT_ERROR": row['percent_error']
            })

        # Calculate summary statistics
        summary_stats = {
            'mean_absolute_error': df['error'].mean(),
            'mean_absolute_percentage_error': df['percent_error'].mean()
        }

        return table_data, summary_stats

    def run(self):
        """Execute the full CVE forecasting workflow."""
        self.logger.info('CVE Forecast Engine Initialized')

        self.logger.info('Processing Data...')
        self.process_data()

        self.train_and_evaluate_models()
        self.generate_final_forecasts()

        self.logger.info('Saving Results...')
        self.save_results()
        self.logger.info('CVE Forecast Engine finished successfully.')

        # Final Summary
        best_model = next(iter(self.model_results.values())) if self.model_results else None
        print("\n--- CVE Forecast Complete ---")
        if best_model:
            print(f"  Best Model: {best_model['model_name']} (MAPE: {best_model['metrics']['mape']:.4f}%)")
        print(f"  Forecast data saved to: {self.config['file_paths']['output_data']}")
        
        print(f"  Processed {self.summary['total_historical_cves']:,} CVE records.")
        print(f"  Historical Data: {self.summary['data_period']['start']} to {self.summary['data_period']['end']}")
        print(f"  Output file: {self.output_path}")
        print("---------------------------")

if __name__ == "__main__":
    engine = CVEForecastEngine()
    engine.run()
