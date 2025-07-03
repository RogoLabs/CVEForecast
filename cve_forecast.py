#!/usr/bin/env python3
"""
CVE Forecast Script
This script processes CVE data, trains forecasting models, and generates predictions.
"""

import json
import os
import shutil
import tempfile
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import (
    ARIMA, ExponentialSmoothing, Prophet, Theta,
    LinearRegressionModel, RandomForest, XGBModel
)
from darts.models.forecasting.sf_auto_arima import StatsForecastAutoARIMA
from darts.models.forecasting.sf_auto_ets import StatsForecastAutoETS
from darts.models.forecasting.sf_auto_theta import StatsForecastAutoTheta
from darts.metrics import mape
from darts.utils.utils import ModelMode, SeasonalityMode


class CVEForecastEngine:
    """Main engine for CVE forecasting"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.historical_data = None
        self.forecasts = {}
        self.model_rankings = []
        
    def parse_cve_data(self, repo_path: str = None) -> pd.DataFrame:
        """Parse CVE JSON files and extract publication dates"""
        if repo_path is None:
            repo_path = self.data_path
            
        if repo_path is None:
            # Default to cvelistV5 directory in project root
            script_dir = Path(__file__).parent
            repo_path = str(script_dir / "cvelistV5")
            
        # Check if the CVE data directory exists
        cves_path = Path(repo_path) / "cves"
        if not cves_path.exists():
            print(f"❌ CVE data not found at: {repo_path}")
            print("\nPlease run the following command to download CVE data:")
            print("    python download_data.py")
            print("\nThis will download the CVE data repository to the project root.")
            raise FileNotFoundError(f"CVE data directory not found: {cves_path}")
            
        print(f"Parsing CVE data from {repo_path}...")
        cve_dates = []
        
        cves_path = Path(repo_path) / "cves"
        # Recursively find all JSON files
        json_files = list(cves_path.rglob("*.json"))
        print(f"Found {len(json_files)} CVE JSON files")
        
        for i, json_file in enumerate(json_files):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(json_files)} files")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    cve_data = json.load(f)
                
                # Try multiple ways to extract publication date
                published_date = None
                
                # Method 1: cveMetadata.datePublished (primary field in schema)
                if 'cveMetadata' in cve_data and 'datePublished' in cve_data['cveMetadata']:
                    published_date = cve_data['cveMetadata']['datePublished']
                
                # Method 2: cveMetadata.dateReserved (fallback - reservation date)
                elif 'cveMetadata' in cve_data and 'dateReserved' in cve_data['cveMetadata']:
                    published_date = cve_data['cveMetadata']['dateReserved']
                
                # Method 3: containers.cna.datePublic (alternative publication date)
                elif 'containers' in cve_data and 'cna' in cve_data['containers'] and 'datePublic' in cve_data['containers']['cna']:
                    published_date = cve_data['containers']['cna']['datePublic']
                
                # Method 4: Legacy formats (for older CVE files)
                elif 'publishedDate' in cve_data:
                    published_date = cve_data['publishedDate']
                elif 'Published_Date' in cve_data:
                    published_date = cve_data['Published_Date']
                
                if published_date:
                    try:
                        # Handle different date formats
                        if 'T' in published_date:
                            # ISO format with time (RFC3339/ISO8601)
                            date_obj = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                        else:
                            # Date only format
                            date_obj = datetime.strptime(published_date, '%Y-%m-%d')
                        
                        cve_dates.append(date_obj.date())
                    except ValueError:
                        # Try other common date formats
                        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                            try:
                                date_obj = datetime.strptime(published_date, fmt)
                                cve_dates.append(date_obj.date())
                                break
                            except ValueError:
                                continue
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Skip malformed or incomplete CVE entries
                continue
        
        print(f"Successfully parsed {len(cve_dates)} CVE publication dates")
        
        # Check if we found any valid dates
        if len(cve_dates) == 0:
            print("❌ No valid CVE publication dates found!")
            print("This could be due to:")
            print("1. CVE files don't contain 'cveMetadata.datePublished' fields")
            print("2. Date format is different than expected")
            print("3. CVE files are malformed or empty")
            
            # Let's examine a few files to understand the structure
            print("\nExamining first few CVE files for debugging...")
            for i, json_file in enumerate(json_files[:3]):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        cve_data = json.load(f)
                    print(f"\nFile {i+1}: {json_file.name}")
                    print(f"  Keys: {list(cve_data.keys())}")
                    if 'cveMetadata' in cve_data:
                        print(f"  cveMetadata keys: {list(cve_data['cveMetadata'].keys())}")
                        # Check for date fields
                        date_fields = ['datePublished', 'dateReserved', 'dateUpdated']
                        for field in date_fields:
                            if field in cve_data['cveMetadata']:
                                print(f"    {field}: {cve_data['cveMetadata'][field]}")
                    else:
                        print("  No 'cveMetadata' found")
                        
                    # Check for containers
                    if 'containers' in cve_data:
                        print(f"  containers keys: {list(cve_data['containers'].keys())}")
                        if 'cna' in cve_data['containers']:
                            cna_keys = list(cve_data['containers']['cna'].keys())
                            print(f"    cna keys: {cna_keys}")
                            if 'datePublic' in cve_data['containers']['cna']:
                                print(f"      datePublic: {cve_data['containers']['cna']['datePublic']}")
                            if 'dateAssigned' in cve_data['containers']['cna']:
                                print(f"      dateAssigned: {cve_data['containers']['cna']['dateAssigned']}")
                except Exception as e:
                    print(f"  Error reading file: {e}")
            
            raise ValueError("No valid CVE publication dates found in the dataset. Please check the CVE data format.")
        
        # Create time series DataFrame
        df = pd.DataFrame({'date': cve_dates})
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by date and count CVEs per day
        daily_counts = df.groupby('date').size().reset_index(name='cve_count')
        
        print(f"Date range: {daily_counts['date'].min()} to {daily_counts['date'].max()}")
        print(f"Total days with CVE data: {len(daily_counts)}")
        
        # Fill missing dates with zero counts
        date_range = pd.date_range(
            start=daily_counts['date'].min(),
            end=daily_counts['date'].max(),
            freq='D'
        )
        
        complete_df = pd.DataFrame({'date': date_range})
        complete_df = complete_df.merge(daily_counts, on='date', how='left')
        complete_df['cve_count'] = complete_df['cve_count'].fillna(0)
        
        return complete_df
    
    def prepare_models(self) -> List[Tuple[str, Any]]:
        """Prepare list of the 5 best available statistical forecasting models for time series"""
        models = []
        
        # Top 5 Statistical Models for Time Series Forecasting
        try:
            # AutoARIMA - Automatically selects the best ARIMA parameters
            models.append(("AutoARIMA", StatsForecastAutoARIMA()))
            print("Added AutoARIMA model")
        except Exception as e:
            print(f"Failed to add AutoARIMA: {e}")
        
        try:
            # AutoETS - Error, Trend, Seasonality model with automatic parameter selection
            models.append(("AutoETS", StatsForecastAutoETS()))
            print("Added AutoETS model")
        except Exception as e:
            print(f"Failed to add AutoETS: {e}")
        
        try:
            # ExponentialSmoothing - Good for trends and seasonality patterns
            models.append(("ExponentialSmoothing", ExponentialSmoothing()))
            print("Added ExponentialSmoothing model")
        except Exception as e:
            print(f"Failed to add ExponentialSmoothing: {e}")
        
        try:
            # Prophet - Facebook's robust forecasting method, handles seasonality well
            models.append(("Prophet", Prophet()))
            print("Added Prophet model")
        except Exception as e:
            print(f"Failed to add Prophet: {e}")
        
        try:
            # AutoTheta - Theta method with automatic parameter selection
            models.append(("AutoTheta", StatsForecastAutoTheta()))
            print("Added AutoTheta model")
        except Exception as e:
            print(f"Failed to add AutoTheta: {e}")
        
        # Fallback models if any of the above fail
        if len(models) < 5:
            try:
                # ARIMA - Classic time series model, excellent for trends
                models.append(("ARIMA", ARIMA()))
                print("Added ARIMA as fallback model")
            except Exception as e:
                print(f"Failed to add ARIMA: {e}")
        
        if len(models) < 5:
            try:
                # Theta - Simple but effective method for irregular time series
                models.append(("Theta", Theta()))
                print("Added Theta as fallback model")
            except Exception as e:
                print(f"Failed to add Theta: {e}")
        
        if len(models) < 5:
            try:
                # LinearRegression - Good baseline for trend-based forecasting
                models.append(("LinearRegression", LinearRegressionModel(lags=30)))
                print("Added LinearRegression as fallback model")
            except Exception as e:
                print(f"Failed to add LinearRegression: {e}")
        
        print(f"Prepared {len(models)} forecasting models")
        return models
    
    def evaluate_models(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Train and evaluate all models"""
        print("Converting data to Darts TimeSeries...")
        
        # Convert to Darts TimeSeries
        ts = TimeSeries.from_dataframe(
            data,
            time_col='date',
            value_cols='cve_count',
            freq='D'
        )
        
        # Split data for validation (last 30 days for validation)
        train_ts, val_ts = ts[:-30], ts[-30:]
        
        print(f"Training set size: {len(train_ts)}")
        print(f"Validation set size: {len(val_ts)}")
        
        models = self.prepare_models()
        results = []
        
        for model_name, model in models:
            print(f"\nTraining {model_name}...")
            try:
                # Train the model
                model.fit(train_ts)
                
                # Generate forecast
                forecast = model.predict(len(val_ts))
                
                # Calculate MAPE
                mape_score = mape(val_ts, forecast)
                
                results.append({
                    'model_name': model_name,
                    'mape': mape_score,
                    'model': model
                })
                
                print(f"{model_name} MAPE: {mape_score:.4f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # Sort by MAPE (lower is better)
        results.sort(key=lambda x: x['mape'])
        
        print("\nModel Rankings:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['model_name']}: MAPE = {result['mape']:.4f}")
        
        return results
    
    def generate_forecasts(self, data: pd.DataFrame, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate forecasts for the remainder of the current year and all of next year"""
        print("Generating forecasts for the remainder of current year and all of next year...")
        
        # Convert to Darts TimeSeries
        ts = TimeSeries.from_dataframe(
            data,
            time_col='date',
            value_cols='cve_count',
            freq='D'
        )
        
        # Calculate days until end of next year (2026)
        current_date = datetime.now().date()
        end_of_next_year = datetime(current_date.year + 1, 12, 31).date()
        days_to_forecast = (end_of_next_year - current_date).days
        
        print(f"Forecasting {days_to_forecast} days into the future (through {end_of_next_year})")
        
        # Use top 5 models
        top_models = model_results[:5]
        forecasts = {}
        
        for result in top_models:
            model_name = result['model_name']
            model = result['model']
            
            try:
                # Retrain on full dataset
                model.fit(ts)
                
                # Generate forecast
                forecast = model.predict(days_to_forecast)
                
                # Convert to list of dictionaries
                forecast_df = forecast.pd_dataframe().reset_index()
                forecast_data = []
                
                for _, row in forecast_df.iterrows():
                    forecast_data.append({
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'cve_count': max(0, round(row['cve_count']))  # Ensure non-negative integers
                    })
                
                forecasts[model_name] = forecast_data
                
                print(f"Generated forecast for {model_name}")
                
            except Exception as e:
                print(f"Error generating forecast for {model_name}: {e}")
                continue
        
        return forecasts
    
    def save_data_file(self, data: pd.DataFrame, model_results: List[Dict[str, Any]], 
                      forecasts: Dict[str, Any], output_path: str):
        """Save data file for web interface"""
        print("Saving data file...")
        
        # Prepare historical data
        historical_data = []
        for _, row in data.iterrows():
            historical_data.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'cve_count': int(row['cve_count'])
            })
        
        # Prepare model rankings
        rankings = []
        for result in model_results:
            rankings.append({
                'model_name': result['model_name'],
                'mape': round(result['mape'], 4)
            })
        
        # Prepare output data
        output_data = {
            'generated_at': datetime.now().isoformat(),
            'model_rankings': rankings,
            'historical_data': historical_data,
            'forecasts': forecasts,
            'summary': {
                'total_historical_cves': int(data['cve_count'].sum()),
                'data_period': {
                    'start': data['date'].min().strftime('%Y-%m-%d'),
                    'end': data['date'].max().strftime('%Y-%m-%d')
                },
                'forecast_period': {
                    'start': datetime.now().strftime('%Y-%m-%d'),
                    'end': datetime(datetime.now().year + 1, 12, 31).strftime('%Y-%m-%d')
                }
            }
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save data file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Data file saved to {output_path}")
    
    def run(self, output_path: str = "web/data.json"):
        """Main execution method"""
        try:
            # Step 1: Parse CVE data
            data = self.parse_cve_data()
            
            # Step 2: Evaluate models
            model_results = self.evaluate_models(data)
            
            # Step 3: Generate forecasts
            forecasts = self.generate_forecasts(data, model_results)
            
            # Step 4: Save data file
            self.save_data_file(data, model_results, forecasts, output_path)
            
            print("\nCVE Forecast generation completed successfully!")
            
        except Exception as e:
            print(f"Error in CVE forecast generation: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate CVE forecasts from data')
    parser.add_argument('--data-path', 
                      help='Path to the CVE data directory (default: cvelistV5 in project root)')
    parser.add_argument('--output', default='web/data.json',
                      help='Output path for the generated data file (default: web/data.json)')
    
    args = parser.parse_args()
    
    engine = CVEForecastEngine(data_path=args.data_path)
    engine.run(output_path=args.output)


if __name__ == "__main__":
    main()
