import json
import logging
import sys
from pathlib import Path
import pandas as pd
from darts import TimeSeries

from utils import setup_logging
from data_loader import load_cve_data
from model_trainer import train_and_evaluate_model

import datetime
import numpy as np

class CVEForecastEngine:
    """Orchestrates the CVE forecasting workflow."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        self.series = None
        self.historical_series = None
        self.current_month_series = None
        self.model_results = []
        self.final_forecasts = {}
        self.all_models_validation = {}

    def load_configuration(self):
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

    def setup_logging(self):
        setup_logging(self.config['logging'])
        self.logger = logging.getLogger(__name__)

    def process_data(self):
        monthly_counts = load_cve_data(self.config)
        full_series = TimeSeries.from_dataframe(monthly_counts, freq='M', fill_missing_dates=True, value_cols='cve_count')
        
        # The last entry is the current, partial month.
        self.historical_series = full_series[:-1]
        self.current_month_series = full_series[-1:]
        self.series = self.historical_series # Use historical for training
        self.full_series = full_series # Keep for summary stats

    def train_and_evaluate_models(self, progress_callback=None):
        enabled_models = {name: config for name, config in self.config['models'].items() if config['enabled']}
        total_models = len(enabled_models)

        for i, (model_name, model_config) in enumerate(enabled_models.items()):
            if progress_callback:
                progress_callback(f"Evaluating Models ({i+1}/{total_models}): {model_name}")
            
            model, metrics, validation_data = train_and_evaluate_model(model_name, model_config, self.series, self.config['model_evaluation'])
            if model and metrics:
                self.model_results.append({"model_name": model_name, "model": model, "metrics": metrics})
                if validation_data:
                    self.all_models_validation[model_name] = validation_data

    def generate_final_forecasts(self, progress_callback=None):
        ensemble_size = self.config['model_evaluation']['ensemble_size']
        top_models = sorted(self.model_results, key=lambda x: x['metrics'].get('mape', float('inf')))[:ensemble_size]

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
        today = datetime.date.today()
        last_day_of_month = (today.replace(day=28) + datetime.timedelta(days=4)).replace(day=1) - datetime.timedelta(days=1)
        
        current_cve_count = int(self.current_month_series.values()[0][0])
        historical_total = int(self.historical_series.to_dataframe()['cve_count'].sum())

        return {
            "date": today.strftime('%Y-%m'),
            "cve_count": current_cve_count,
            "cumulative_total": historical_total + current_cve_count,
            "days_elapsed": today.day,
            "total_days": last_day_of_month.day,
            "progress_percentage": round((today.day / last_day_of_month.day) * 100, 1)
        }

    def _get_cumulative_timelines(self, historical_data, forecasts, current_month_data):
        """Generate cumulative timelines matching the v.02 file_io.py logic exactly."""
        cumulative_timelines = {}
        historical_2025_data = [item for item in historical_data if item['date'].startswith('2025')]

        # Generate timeline for each model
        for model_name, model_forecasts in forecasts.items():
            if not model_forecasts: continue
            timeline, running_total = [], 0
            timeline.append({"date": "2025-01", "cumulative_total": 0})

            for item in historical_2025_data:
                timeline.append({"date": item['date'], "cumulative_total": running_total})
                running_total += item['cve_count']
            
            if current_month_data:
                current_month_forecast_val = next((f['cve_count'] for f in model_forecasts if f['date'] == current_month_data['date']), None)
                timeline.append({"date": current_month_data['date'], "cumulative_total": running_total})
                if current_month_forecast_val is not None:
                    running_total += current_month_forecast_val

            # Add forecast months, ensuring not to double-count the current month
            for forecast in model_forecasts:
                if current_month_data and forecast['date'] == current_month_data['date']:
                    continue
                timeline.append({"date": forecast['date'], "cumulative_total": running_total})
                running_total += forecast['cve_count']
            
            timeline.append({"date": "2026-01", "cumulative_total": running_total})
            cumulative_timelines[f"{model_name}_cumulative"] = timeline

        # Add all_models_cumulative average
        if forecasts:
            all_models_timeline, running_total = [], 0
            all_models_timeline.append({"date": "2025-01", "cumulative_total": 0})

            for item in historical_2025_data:
                all_models_timeline.append({"date": item['date'], "cumulative_total": running_total})
                running_total += item['cve_count']
            
            if current_month_data:
                all_models_timeline.append({"date": current_month_data['date'], "cumulative_total": running_total})
                current_month_forecasts = [f['cve_count'] for f_list in forecasts.values() for f in f_list if f['date'] == current_month_data['date']]
                if current_month_forecasts:
                    running_total += np.mean(current_month_forecasts)

            # Determine all unique forecast dates, excluding the current month if already handled
            all_forecast_dates = sorted(list(set(f['date'] for f_list in forecasts.values() for f in f_list)))
            for date in all_forecast_dates:
                if current_month_data and date == current_month_data['date']:
                    continue # Already processed the current month's forecast average
                
                avg_forecast = np.mean([f['cve_count'] for f_list in forecasts.values() for f in f_list if f['date'] == date])
                # Only add if there are valid forecasts for this date
                if not np.isnan(avg_forecast):
                    all_models_timeline.append({"date": date, "cumulative_total": running_total})
                    running_total += avg_forecast

            all_models_timeline.append({"date": "2026-01", "cumulative_total": running_total})
            cumulative_timelines['all_models_cumulative'] = all_models_timeline

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

    def save_results(self):
        model_rankings = sorted([
            {"model_name": r['model_name'], **r['metrics']} for r in self.model_results
        ], key=lambda x: x['mape'] if x.get('mape') is not None else float('inf'))

        historical_data = self._get_historical_data()
        current_month_actual = self._get_current_month_actual()

        forecasts_out = {name: [{"date": ts.strftime('%Y-%m'), "cve_count": int(round(val[0]))} 
                                for ts, val in zip(forecast.time_index, forecast.values())] 
                         for name, forecast in self.final_forecasts.items()}

        self.logger.info(f"Validation data to be saved: {json.dumps(self.all_models_validation, indent=2)}")

        # This summary is now created here to be used in the final printout
        self.summary = self._get_summary()

        final_json = {
            "generated_at": datetime.datetime.now().isoformat(),
            "model_rankings": model_rankings,
            "historical_data": historical_data,
            "current_month_actual": current_month_actual,
            "forecasts": forecasts_out,
            "all_models_validation": self.all_models_validation,
            "yearly_forecast_totals": {name: int(v.sum().values()[0][0]) for name, v in self.final_forecasts.items()},
            "cumulative_timelines": self._get_cumulative_timelines(historical_data, forecasts_out, current_month_actual),
            "summary": self.summary
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

    def run(self):
        def progress_indicator(message):
            # Ensure the message doesn't exceed terminal width by truncating it
            max_len = 100
            message = (message[:max_len-3] + '...') if len(message) > max_len else message
            sys.stdout.write(f'\r\033[K{message}')
            sys.stdout.flush()

        progress_indicator("Initializing Forecast Engine...")
        self.load_configuration()
        self.setup_logging()

        progress_indicator("Processing Data...")
        self.process_data()

        self.train_and_evaluate_models(progress_indicator)
        self.generate_final_forecasts(progress_indicator)

        progress_indicator("Saving Results...")
        self.save_results()
        sys.stdout.write('\r\033[K') # Clear the progress line

        # Final Summary
        best_model = self.model_results[0] if self.model_results else None
        print("\n--- CVE Forecast Complete ---")
        if best_model:
            print(f"  Best Model: {best_model['model_name']} (MAPE: {best_model['metrics']['mape']:.4f}%)")
        else:
            print("  No models were successfully trained.")
        
        print(f"  Processed {self.summary['total_historical_cves']:,} CVE records.")
        print(f"  Historical Data: {self.summary['data_period']['start']} to {self.summary['data_period']['end']}")
        print(f"  Output file: {self.output_path}")
        print("---------------------------")

if __name__ == "__main__":
    # Construct path to config.json relative to the script's location
    config_path = Path(__file__).parent / 'config.json'
    engine = CVEForecastEngine(config_path)
    engine.run()
