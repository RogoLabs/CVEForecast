#!/usr/bin/env python3
"""
File I/O module for CVE Forecast application.
Handles reading and writing data files, including the main data.json output.
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

from config import DEFAULT_OUTPUT_PATH, DEFAULT_PERFORMANCE_HISTORY_PATH
from utils import setup_logging, ensure_directory_exists

logger = setup_logging()


class FileIOManager:
    """Manages file input/output operations for the CVE forecast application."""
    
    def __init__(self):
        """Initialize the FileIO manager."""
        pass
    
    def _clean_nan_values(self, obj: Any) -> Any:
        """Recursively clean NaN values from nested data structures for valid JSON serialization.
        
        Args:
            obj: Object to clean (can be dict, list, or primitive type)
            
        Returns:
            Cleaned object with NaN values replaced by None
        """
        if isinstance(obj, dict):
            return {key: self._clean_nan_values(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_nan_values(item) for item in obj]
        elif isinstance(obj, (int, str, bool)) or obj is None:
            return obj
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
        elif hasattr(obj, 'item'):  # numpy scalar
            val = obj.item()
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            return val
        else:
            # For any other type, try to convert or return as-is
            try:
                if hasattr(obj, '__float__'):
                    val = float(obj)
                    if math.isnan(val) or math.isinf(val):
                        return None
                    return val
            except (ValueError, TypeError):
                pass
            return obj
    
    def load_performance_history(self, history_path: str = DEFAULT_PERFORMANCE_HISTORY_PATH) -> List[Dict[str, Any]]:
        """
        Load existing performance history from JSON file.
        
        Args:
            history_path: Path to the performance history file
            
        Returns:
            List of performance history records
        """
        try:
            if Path(history_path).exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                logger.info(f"Loaded {len(history)} performance history records from {history_path}")
                return history
            else:
                logger.info(f"Performance history file {history_path} not found, starting with empty history")
                return []
        except Exception as e:
            logger.error(f"Error loading performance history from {history_path}: {e}")
            return []
    
    def save_performance_history(self, history: List[Dict[str, Any]], 
                               history_path: str = DEFAULT_PERFORMANCE_HISTORY_PATH) -> None:
        """
        Save performance history to JSON file.
        
        Args:
            history: List of performance history records
            history_path: Path to save the performance history file
        """
        try:
            ensure_directory_exists(history_path)
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Saved {len(history)} performance history records to {history_path}")
        except Exception as e:
            logger.error(f"Error saving performance history to {history_path}: {e}")
            raise
    
    def save_data_file(self, data: pd.DataFrame, model_results: List[Dict[str, Any]], 
                      forecasts: Dict[str, Any], output_path: str = DEFAULT_OUTPUT_PATH) -> None:
        """
        Save data file for web interface.
        
        Args:
            data: Historical CVE data DataFrame
            model_results: List of model evaluation results
            forecasts: Dictionary of forecasts by model
            output_path: Path to save the output data file
        """
        logger.info("Preparing data file for web interface...")
        
        # Prepare historical data (exclude current month since it's being forecasted)
        historical_data = self._prepare_historical_data(data)
        
        # Prepare model rankings with comprehensive metrics
        rankings = self._prepare_model_rankings(model_results)
        
        # Prepare validation data for all models
        all_models_validation = self._prepare_validation_data(model_results)
        best_model_validation = model_results[0]['validation_data'] if model_results else []
        
        # Prepare current month progress data FIRST (workflow order fix)
        current_month_raw_data = self._prepare_current_month_data(data)
        
        # 🚀 CRITICAL FIX: Enhance validation data with current month info after forecasts are available
        all_models_validation = self._enhance_validation_with_current_month(
            all_models_validation, forecasts, current_month_raw_data
        )
        
        # Calculate yearly forecast totals for each model
        yearly_forecast_totals = self._calculate_yearly_forecast_totals(
            historical_data, forecasts, rankings
        )
        
        # Generate cumulative timelines for visualization
        cumulative_timelines = self._generate_cumulative_timelines(
            historical_data, forecasts, current_month_raw_data
        )
        
        # Build the complete output data structure
        output_data = self._build_output_data_structure(
            data, historical_data, rankings, forecasts, yearly_forecast_totals,
            cumulative_timelines, best_model_validation, all_models_validation
        )
        
        # Save the data file
        self._write_data_file(output_data, output_path)
        logger.info(f"Data file saved to {output_path}")
    
    def _prepare_historical_data(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Prepare historical data excluding the current month.
        
        Args:
            data: Historical CVE data DataFrame
            
        Returns:
            List of historical data records
        """
        current_date = datetime.now().date()
        current_month = datetime(current_date.year, current_date.month, 1).date()
        
        historical_data = []
        for _, row in data.iterrows():
            row_date = row['date'].date() if hasattr(row['date'], 'date') else pd.to_datetime(row['date']).date()
            if row_date < current_month:  # Only include complete months in historical data
                historical_data.append({
                    'date': row['date'].strftime('%Y-%m'),
                    'cve_count': int(row['cve_count'])
                })
        
        return historical_data
    
    def _prepare_model_rankings(self, model_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare model rankings with comprehensive metrics.
        
        Args:
            model_results: List of model evaluation results
            
        Returns:
            List of model ranking data
        """
        rankings = []
        for result in model_results:
            # Handle NaN values by converting to None for valid JSON
            mape_val = result['mape']
            mae_val = result['mae']
            mase_val = result['mase']
            rmsse_val = result['rmsse']
            
            # Convert NaN to None for valid JSON serialization
            import math
            rankings.append({
                'model_name': result['model_name'],
                'mape': round(mape_val, 4) if not math.isnan(mape_val) else None,
                'mase': round(mase_val, 4) if mase_val is not None and not math.isnan(mase_val) else None,
                'rmsse': round(rmsse_val, 4) if rmsse_val is not None and not math.isnan(rmsse_val) else None,
                'mae': round(mae_val, 4) if not math.isnan(mae_val) else None
            })
        
        return rankings
    
    def _enhance_validation_with_current_month(self, all_models_validation: Dict[str, Any], 
                                              forecasts: Dict[str, Any], 
                                              current_month_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance validation data with current month information after forecasts are available.
        This fixes the workflow timing issue and matches production website behavior.
        
        Args:
            all_models_validation: Existing validation data for all models
            forecasts: Dictionary of forecasts by model (now available)
            current_month_data: Current month progress data
            
        Returns:
            Enhanced validation data with current month information
        """
        if not current_month_data or not forecasts:
            return all_models_validation
            
        current_month_str = current_month_data['date']  # e.g., '2025-07'
        current_month_actual = current_month_data['cve_count']  # e.g., 550
        
        logger.info(f"Enhancing validation data with current month ({current_month_str}) information")
        
        # Enhance each model's validation data
        for model_name, validation_list in all_models_validation.items():
            if model_name in forecasts and forecasts[model_name]:
                # Find current month forecast for this model
                current_month_forecast = None
                for forecast in forecasts[model_name]:
                    if forecast.get('date') == current_month_str:
                        current_month_forecast = forecast['cve_count']
                        break
                
                if current_month_forecast:
                    # Remove any existing current month entry (from earlier processing)
                    validation_list[:] = [item for item in validation_list if item.get('date') != current_month_str]
                    
                    # Add current month with FULL forecast vs partial actual
                    validation_list.append({
                        'date': current_month_str,
                        'actual': int(current_month_actual),  # Partial actual (e.g., 550 CVEs for 8 days)
                        'predicted': int(current_month_forecast),  # FULL month forecast (e.g., 931 CVEs)
                        'is_current_month': True  # Flag for frontend special handling
                    })
                    
                    logger.debug(f"Enhanced {model_name}: {current_month_str} actual={current_month_actual}, predicted={current_month_forecast}")
        
        return all_models_validation
    
    def _prepare_validation_data(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare validation data for all models.
        
        Args:
            model_results: List of model evaluation results
            
        Returns:
            Dictionary of validation data by model
        """
        all_models_validation = {}
        for result in model_results:
            all_models_validation[result['model_name']] = result['validation_data']
        
        return all_models_validation
    
    def _prepare_current_month_data(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Prepare current month progress data.
        
        Args:
            data: Historical CVE data DataFrame
            
        Returns:
            Current month data dictionary or None if not found
        """
        current_date = datetime.now().date()
        current_month = datetime(current_date.year, current_date.month, 1).date()
        
        # Find current month's raw data
        for _, row in data.iterrows():
            row_date = row['date'].date() if hasattr(row['date'], 'date') else pd.to_datetime(row['date']).date()
            if row_date.year == current_month.year and row_date.month == current_month.month:
                # Calculate days in month
                if current_date.month == 12:
                    next_month = datetime(current_date.year + 1, 1, 1)
                else:
                    next_month = datetime(current_date.year, current_date.month + 1, 1)
                
                days_in_month = (next_month - datetime(current_date.year, current_date.month, 1)).days
                
                return {
                    'date': row['date'].strftime('%Y-%m'),
                    'cve_count': int(row['cve_count']),
                    'days_elapsed': current_date.day,
                    'total_days': days_in_month,
                    'progress_percentage': round((current_date.day / days_in_month) * 100, 1)
                }
        
        return None
    
    def _prepare_current_month_data_with_cumulative(self, data: pd.DataFrame, 
                                                   historical_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Prepare current month data with cumulative total calculation (matching original script).
        
        Args:
            data: Historical CVE data DataFrame
            historical_data: List of historical data records
            
        Returns:
            Current month data with cumulative total or None if not found
        """
        current_date = datetime.now().date()
        current_month = datetime(current_date.year, current_date.month, 1).date()
        current_month_str = current_month.strftime('%Y-%m')
        
        # Find current month's raw data
        current_month_raw_data = None
        for _, row in data.iterrows():
            row_date = row['date'].date() if hasattr(row['date'], 'date') else pd.to_datetime(row['date']).date()
            if row_date.year == current_month.year and row_date.month == current_month.month:
                # Calculate days in month
                if current_date.month == 12:
                    next_month = datetime(current_date.year + 1, 1, 1)
                else:
                    next_month = datetime(current_date.year, current_date.month + 1, 1)
                
                days_in_month = (next_month - datetime(current_date.year, current_date.month, 1)).days
                
                current_month_raw_data = {
                    'date': row['date'].strftime('%Y-%m'),
                    'cve_count': int(row['cve_count']),
                    'days_elapsed': current_date.day,
                    'total_days': days_in_month,
                    'progress_percentage': round((current_date.day / days_in_month) * 100, 1)
                }
                break
        
        if current_month_raw_data:
            # Calculate cumulative total (matching original script logic)
            current_month_progress = current_month_raw_data['cve_count']
            
            # Calculate cumulative total at start of current month using ONLY data published BEFORE current month
            cumulative_at_month_start = 0
            current_month_num = int(current_month_str.split('-')[1])  # Extract month number
            
            for item in historical_data:
                item_month_num = int(item['date'].split('-')[1])  # Extract month number
                # Only include months that are strictly before the current month
                if item['date'].startswith('2025') and item_month_num < current_month_num:
                    cumulative_at_month_start += item['cve_count']
                    logger.debug(f"Including {item['date']}: {item['cve_count']:,} CVEs")
            
            # Final cumulative = cumulative at month start + current month progress
            cumulative_total = cumulative_at_month_start + current_month_progress
            
            logger.debug(f"Cumulative at month start: {cumulative_at_month_start:,}")
            logger.debug(f"Current month progress: {current_month_progress:,}")
            logger.debug(f"Final cumulative total: {cumulative_total:,}")
            
            # Create final current_month_data with cumulative total (matching original)
            return {
                'date': current_month_str,
                'cve_count': current_month_progress,  # Monthly progress count (for display)
                'cumulative_total': cumulative_total,  # Cumulative total (for chart)
                'days_elapsed': current_month_raw_data['days_elapsed'],
                'total_days': current_month_raw_data['total_days'],
                'progress_percentage': current_month_raw_data['progress_percentage']
            }
        
        return None
    
    def _calculate_yearly_forecast_totals(self, historical_data: List[Dict[str, Any]], 
                                        forecasts: Dict[str, Any], 
                                        rankings: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate yearly forecast totals for each model.
        
        Args:
            historical_data: List of historical data records
            forecasts: Dictionary of forecasts by model
            rankings: List of model rankings
            
        Returns:
            Dictionary of yearly forecast totals by model
        """
        current_date = datetime.now().date()
        yearly_forecast_totals = {}
        
        # Sum only 2025 historical data for yearly forecast totals
        historical_total = sum(item['cve_count'] for item in historical_data 
                             if item['date'].startswith('2025'))
        
        # Calculate individual model totals
        for model_name, model_forecasts in forecasts.items():
            model_forecast_total = 0
            for forecast in model_forecasts:
                # Include current month and future months within 2025 only
                forecast_date = datetime.strptime(forecast['date'], '%Y-%m')
                if forecast_date.year == 2025 and forecast_date.month >= current_date.month:
                    model_forecast_total += forecast['cve_count']
                    logger.debug(f"Including {forecast['date']}: {forecast['cve_count']} CVEs for {model_name}")
            
            # Store complete yearly total: historical + future forecasts
            yearly_forecast_totals[model_name] = round(historical_total + model_forecast_total)
        
        # Calculate "All Models" average
        if yearly_forecast_totals:
            all_models_average = sum(yearly_forecast_totals.values()) / len(yearly_forecast_totals)
            yearly_forecast_totals['all_models_average'] = round(all_models_average)
        
        # Store best model total
        if rankings:
            best_model_name = rankings[0]['model_name']
            yearly_forecast_totals['best_model'] = yearly_forecast_totals.get(best_model_name, 0)
        
        return yearly_forecast_totals
    
    def _generate_cumulative_timelines(self, historical_data: List[Dict[str, Any]], 
                                     forecasts: Dict[str, Any], 
                                     current_month_data: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate cumulative timelines for visualization matching original format exactly.
        
        Args:
            historical_data: List of historical data records  
            forecasts: Dictionary of forecasts by model
            current_month_data: Current month progress data
            
        Returns:
            Dictionary of cumulative timelines by model (matching original format)
        """
        cumulative_timelines = {}
        
        # Calculate historical cumulative total up to 2025
        historical_2025_data = [item for item in historical_data if item['date'].startswith('2025')]
        
        # Generate timeline for each model with forecasts
        for model_name, model_forecasts in forecasts.items():
            if not model_forecasts:  # Skip models with no forecasts
                continue
                
            timeline = []
            
            # Start with cumulative total at beginning of 2025 (should be 0)
            timeline.append({
                "date": "2025-01",
                "cumulative_total": 0
            })
            
            running_total = 0
            
            # Add historical 2025 months first
            for item in historical_2025_data:
                timeline.append({
                    "date": item['date'],
                    "cumulative_total": running_total
                })
                running_total += item['cve_count']
            
            # CRITICAL FIX: For current month, use FORECAST value (not actual) to match yearly_forecast_totals logic
            # This matches original working solution: past months use actual, current/future use forecast
            if current_month_data:
                # Find current month's forecast value for this model
                current_month_forecast = None
                for forecast in model_forecasts:
                    if forecast['date'] == current_month_data['date']:
                        current_month_forecast = forecast
                        break
                
                timeline.append({
                    "date": current_month_data['date'],
                    "cumulative_total": running_total  # Cumulative total at START of current month
                })
                
                # Use forecast value for current month (not actual) to match yearly_forecast_totals
                if current_month_forecast:
                    running_total += current_month_forecast['cve_count']
                    logger.debug(f"Using forecast value for {current_month_data['date']}: {current_month_forecast['cve_count']} CVEs")
            
            # Add forecast months
            for forecast in model_forecasts:
                # Add data point at beginning of month (before month's CVEs)
                timeline.append({
                    "date": forecast['date'],
                    "cumulative_total": running_total
                })
                # Then add month's CVEs to running total
                running_total += forecast['cve_count']
            
            # Add final Jan 1 2026 data point to show complete 2025 total
            # This represents the cumulative total at the END of 2025 (beginning of 2026)
            timeline.append({
                "date": "2026-01",
                "cumulative_total": running_total  # Final total for all of 2025
            })
            
            # Use model_name_cumulative format to match original
            cumulative_timelines[f"{model_name}_cumulative"] = timeline
        
        # Add all_models_cumulative average
        if forecasts:
            all_models_timeline = []
            running_total = 0
            
            # Start with 0
            all_models_timeline.append({
                "date": "2025-01",
                "cumulative_total": 0
            })
            
            # Historical 2025 data
            for item in historical_2025_data:
                running_total += item['cve_count']
                all_models_timeline.append({
                    "date": item['date'],
                    "cumulative_total": running_total
                })
            
            # CRITICAL FIX: For current month, use average FORECAST value (not actual) to match yearly_forecast_totals logic
            # This matches original working solution: past months use actual, current/future use forecast
            if current_month_data:
                all_models_timeline.append({
                    "date": current_month_data['date'],
                    "cumulative_total": running_total  # Cumulative total at START of current month
                })
                
                # Calculate average forecast value for current month across all models
                current_month_forecasts = []
                for model_name, model_forecasts in forecasts.items():
                    if model_forecasts:
                        for forecast in model_forecasts:
                            if forecast['date'] == current_month_data['date']:
                                current_month_forecasts.append(forecast['cve_count'])
                                break
                
                if current_month_forecasts:
                    avg_current_month_forecast = sum(current_month_forecasts) / len(current_month_forecasts)
                    running_total += avg_current_month_forecast
                    logger.debug(f"Using average forecast value for {current_month_data['date']}: {avg_current_month_forecast:.1f} CVEs")
            
            # Average forecast months
            forecast_months = {}
            for model_name, model_forecasts in forecasts.items():
                if model_forecasts:
                    for forecast in model_forecasts:
                        month = forecast['date']
                        if month not in forecast_months:
                            forecast_months[month] = []
                        forecast_months[month].append(forecast['cve_count'])
            
            # Add averaged forecast data
            for month in sorted(forecast_months.keys()):
                avg_forecast = sum(forecast_months[month]) / len(forecast_months[month])
                running_total += avg_forecast
                all_models_timeline.append({
                    "date": month,
                    "cumulative_total": int(running_total)
                })
            
            # Add final Jan 1 2026 data point to show complete 2025 total
            all_models_timeline.append({
                "date": "2026-01",
                "cumulative_total": int(running_total)  # Final total for all of 2025
            })
            
            cumulative_timelines["all_models_cumulative"] = all_models_timeline
        
        return cumulative_timelines
    
    def _build_output_data_structure(self, data: pd.DataFrame, historical_data: List[Dict[str, Any]], 
                                   rankings: List[Dict[str, Any]], forecasts: Dict[str, Any],
                                   yearly_forecast_totals: Dict[str, int], 
                                   cumulative_timelines: Dict[str, List[Dict[str, Any]]], 
                                   best_model_validation: List[Dict[str, Any]], 
                                   all_models_validation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the complete output data structure matching original script format exactly.
        
        Args:
            data: Historical CVE data DataFrame
            historical_data: Prepared historical data
            rankings: Model rankings
            forecasts: Model forecasts
            yearly_forecast_totals: Yearly forecast totals
            cumulative_timelines: Cumulative timelines
            best_model_validation: Best model validation data
            all_models_validation: All models validation data
            
        Returns:
            Complete output data structure matching original format
        """
        # Prepare current month data with cumulative_total (matching original)
        current_month_data = self._prepare_current_month_data_with_cumulative(data, historical_data)
        
        # Build output structure matching original script exactly
        return {
            'generated_at': datetime.now().isoformat(),
            'model_rankings': rankings,
            'historical_data': historical_data,
            'current_month_actual': current_month_data,
            'forecasts': forecasts,
            'yearly_forecast_totals': yearly_forecast_totals,
            'cumulative_timelines': cumulative_timelines,
            'validation_against_actuals': best_model_validation,
            'all_models_validation': all_models_validation,
            'summary': {
                'total_historical_cves': int(data['cve_count'].sum()),
                'data_period': {
                    'start': data['date'].min().strftime('%Y-%m-%d'),
                    'end': data['date'].max().strftime('%Y-%m-%d')
                },
                'forecast_period': {
                    'start': datetime(datetime.now().year, datetime.now().month, 1).strftime('%Y-%m-%d'),
                    'end': datetime(datetime.now().year + 1, 1, 31).strftime('%Y-%m-%d')
                }
            }
        }
    
    def _write_data_file(self, output_data: Dict[str, Any], output_path: str) -> None:
        """
        Write the output data to a JSON file with comprehensive NaN cleaning.
        
        Args:
            output_data: Complete output data structure
            output_path: Path to save the output file
        """
        ensure_directory_exists(output_path)
        
        # Clean all NaN values for valid JSON serialization
        cleaned_data = self._clean_nan_values(output_data)
        
        with open(output_path, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        
        logger.info(f"Successfully saved data file with {len(output_data)} top-level sections")
    
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Loaded JSON data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded JSON data from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"JSON file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise
    
    def save_json_file(self, data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
        """
        Save data to a JSON file.
        
        Args:
            data: Data to save
            file_path: Path to save the JSON file
            indent: Number of spaces for indentation
        """
        try:
            ensure_directory_exists(file_path)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=indent)
            logger.info(f"Successfully saved JSON data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving JSON file {file_path}: {e}")
            raise
    
    def load_existing_data_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load existing data.json file for fresh forecast generation.
        
        Args:
            file_path: Path to the existing data.json file
            
        Returns:
            Loaded data dictionary or None if file doesn't exist
        """
        try:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Successfully loaded existing data file from {file_path}")
                logger.info(f"Found {len(data.get('model_rankings', []))} model rankings")
                return data
            else:
                logger.warning(f"Existing data file not found: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading existing data file {file_path}: {e}")
            return None
    
    def save_updated_data_file(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Save updated data.json file with fresh forecast results.
        
        Args:
            data: Updated data dictionary including new_forecast_runs
            file_path: Path to save the updated data file
        """
        try:
            ensure_directory_exists(file_path)
            
            # Clean NaN values before saving
            cleaned_data = self._clean_nan_values(data)
            
            # Update generation timestamp
            cleaned_data['generated_at'] = datetime.now().isoformat()
            
            with open(file_path, 'w') as f:
                json.dump(cleaned_data, f, indent=2)
            
            logger.info(f"Successfully saved updated data file to {file_path}")
            
            # Log summary of new forecast runs
            if 'new_forecast_runs' in cleaned_data:
                fresh_data = cleaned_data['new_forecast_runs']
                yearly_totals = fresh_data.get('yearly_forecast_totals', {})
                timelines = fresh_data.get('cumulative_timelines', {})
                
                logger.info(f"Fresh forecast summary:")
                logger.info(f"  - Yearly totals for {len(yearly_totals)} models")
                logger.info(f"  - Cumulative timelines for {len(timelines)} models")
            
        except Exception as e:
            logger.error(f"Error saving updated data file {file_path}: {e}")
            raise
