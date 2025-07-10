import json
import logging
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
        
        self.setup_logging()

        self.series = None
        self.historical_series = None
        self.current_month_series = None
        self.forecasts = {}
        self.model_results = {}
        self.final_forecasts = {}
        self.all_models_validation = {}

    def setup_logging(self):
        """Set up logging based on the configuration."""
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
        """Generate cumulative timelines for all models and the ensemble average."""
        cumulative_timelines = {}
        historical_2025_data = {item['date']: item['cve_count'] for item in historical_data if item['date'].startswith('2025')}

        # --- Generate timeline for each individual model ---
        for model_name, model_forecasts in forecasts.items():
            if not model_forecasts:
                continue
            
            model_forecast_map = {fc['date']: fc['cve_count'] for fc in model_forecasts}
            timeline, running_total = [], 0

            for month_num in range(1, 13):
                date_str = f"2025-{month_num:02d}"
                timeline.append({"date": date_str, "cumulative_total": running_total})
                
                if date_str in historical_2025_data:
                    running_total += historical_2025_data[date_str]
                elif current_month_data and date_str == current_month_data['date']:
                    running_total += current_month_data['cve_count']
                elif date_str in model_forecast_map:
                    running_total += model_forecast_map[date_str]

            timeline.append({"date": "2026-01", "cumulative_total": running_total})
            cumulative_timelines[f"{model_name}_cumulative"] = timeline

        # --- Generate timeline for the average of all models ---
        if forecasts:
            all_models_timeline, running_total = [], 0
            all_forecast_dates = sorted(list(set(f['date'] for f_list in forecasts.values() for f in f_list)))
            
            for month_num in range(1, 13):
                date_str = f"2025-{month_num:02d}"
                all_models_timeline.append({"date": date_str, "cumulative_total": running_total})

                if date_str in historical_2025_data:
                    running_total += historical_2025_data[date_str]
                elif current_month_data and date_str == current_month_data['date']:
                    running_total += current_month_data['cve_count']
                elif date_str in all_forecast_dates:
                    avg_forecast = np.mean([f['cve_count'] for f_list in forecasts.values() for f in f_list if f['date'] == date_str])
                    if not np.isnan(avg_forecast):
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
            {"model_name": r['model_name'], **r['metrics']} for r in self.model_results.values()
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
