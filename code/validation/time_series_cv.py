"""
Time Series Cross-Validation for CVE Forecasting

Implements robust expanding window cross-validation to provide
reliable performance estimates with proper uncertainty quantification.

Scientific Justification:
- Single train/test splits can be unrepresentative
- Multiple validation origins provide stable estimates
- Expanding window respects temporal ordering
- Reports mean ± std across folds for transparency
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from sklearn.model_selection import TimeSeriesSplit
from darts import TimeSeries
from darts.metrics import mape as darts_mape


class RobustTimeSeriesValidator:
    """
    Implements proper time series cross-validation with expanding windows.
    
    This addresses the critical scientific gap where a single 80/20 split
    may provide unreliable performance estimates. Multiple validation origins
    give a more accurate picture of model performance.
    
    Example:
        validator = RobustTimeSeriesValidator(n_splits=5)
        results = validator.validate_model(data, model, forecast_horizon=12)
        print(f"MAPE = {results['mean_error']:.2f}% ± {results['std_error']:.2f}%")
    """
    
    def __init__(self, n_splits: int = 5, min_train_size: int = 24, 
                 gap: int = 0, test_size: Optional[int] = None):
        """
        Initialize validator.
        
        Args:
            n_splits: Number of cross-validation folds (default: 5)
            min_train_size: Minimum training observations required (default: 24 months)
            gap: Gap between train and test (default: 0)
            test_size: Fixed test size or None for expanding window
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.gap = gap
        self.test_size = test_size
        self.logger = logging.getLogger(__name__)
    
    def validate_model(self, data: TimeSeries, model, forecast_horizon: int = 12,
                      metric_name: str = 'MAPE') -> Dict[str, Any]:
        """
        Perform expanding window cross-validation.
        
        This method trains the model on multiple expanding training sets
        and evaluates on subsequent test periods, providing a robust estimate
        of forecast performance.
        
        Args:
            data: Full historical time series
            model: Darts model instance (will be re-fitted for each fold)
            forecast_horizon: Number of periods to forecast (default: 12)
            metric_name: Metric to use ('MAPE', 'MAE', 'RMSE')
        
        Returns:
            dict: {
                'mean_error': float,           # Average across folds
                'std_error': float,            # Std deviation across folds
                'median_error': float,         # Median error
                'min_error': float,            # Best fold
                'max_error': float,            # Worst fold
                'errors_by_fold': list,        # Individual fold errors
                'fold_details': list,          # Detailed results per fold
                'is_valid': bool,              # At least 3 successful folds
                'n_successful_folds': int      # Number of successful folds
            }
        """
        self.logger.info(f"Starting {self.n_splits}-fold time series cross-validation")
        
        # Validate inputs
        if len(data) < self.min_train_size + forecast_horizon:
            raise ValueError(
                f"Data length ({len(data)}) insufficient for CV. "
                f"Need at least {self.min_train_size + forecast_horizon} observations"
            )
        
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            gap=self.gap,
            test_size=self.test_size or forecast_horizon
        )
        
        errors = []
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(data.values())):
            # Check minimum training size
            if len(train_idx) < self.min_train_size:
                self.logger.debug(
                    f"Fold {fold_idx}: Skipping (train size {len(train_idx)} < {self.min_train_size})"
                )
                continue
            
            # Limit test size to forecast horizon
            test_idx = test_idx[:forecast_horizon]
            
            try:
                # Extract train and test data
                train = data[train_idx]
                test = data[test_idx]
                
                self.logger.debug(
                    f"Fold {fold_idx}: Train size={len(train)}, Test size={len(test)}, "
                    f"Origin={train.end_time()}"
                )
                
                # Train model on this fold
                model.fit(train)
                
                # Generate forecast
                forecast = model.predict(len(test))
                
                # Calculate error
                if metric_name == 'MAPE':
                    error = self._calculate_mape(test, forecast)
                elif metric_name == 'MAE':
                    error = self._calculate_mae(test, forecast)
                elif metric_name == 'RMSE':
                    error = self._calculate_rmse(test, forecast)
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")
                
                errors.append(error)
                
                fold_results.append({
                    'fold': fold_idx,
                    'train_size': len(train),
                    'test_size': len(test),
                    'train_end': str(train.end_time()),
                    'test_start': str(test.start_time()),
                    'test_end': str(test.end_time()),
                    'error': error,
                    'metric': metric_name,
                    'forecast_mean': float(forecast.values().mean()),
                    'actual_mean': float(test.values().mean())
                })
                
                self.logger.debug(f"Fold {fold_idx}: {metric_name} = {error:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Fold {fold_idx} failed: {e}")
                continue
        
        # Compute summary statistics
        if len(errors) == 0:
            self.logger.error("All CV folds failed!")
            return {
                'mean_error': None,
                'std_error': None,
                'errors_by_fold': [],
                'fold_details': [],
                'is_valid': False,
                'n_successful_folds': 0
            }
        
        results = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'median_error': float(np.median(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'errors_by_fold': errors,
            'fold_details': fold_results,
            'is_valid': len(errors) >= 3,  # Need at least 3 successful folds
            'n_successful_folds': len(errors),
            'metric_name': metric_name
        }
        
        self.logger.info(
            f"CV Complete: {metric_name} = {results['mean_error']:.4f} ± "
            f"{results['std_error']:.4f} (n={results['n_successful_folds']} folds)"
        )
        
        return results
    
    @staticmethod
    def _calculate_mape(actual: TimeSeries, forecast: TimeSeries) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Handles zeros in actual values by masking them out.
        """
        actual_vals = actual.values().flatten()
        forecast_vals = forecast.values().flatten()
        
        # Mask zeros to avoid division by zero
        mask = actual_vals != 0
        
        if not np.any(mask):
            return np.inf
        
        ape = np.abs((actual_vals[mask] - forecast_vals[mask]) / actual_vals[mask])
        return float(np.mean(ape) * 100)
    
    @staticmethod
    def _calculate_mae(actual: TimeSeries, forecast: TimeSeries) -> float:
        """Calculate Mean Absolute Error."""
        actual_vals = actual.values().flatten()
        forecast_vals = forecast.values().flatten()
        return float(np.mean(np.abs(actual_vals - forecast_vals)))
    
    @staticmethod
    def _calculate_rmse(actual: TimeSeries, forecast: TimeSeries) -> float:
        """Calculate Root Mean Squared Error."""
        actual_vals = actual.values().flatten()
        forecast_vals = forecast.values().flatten()
        return float(np.sqrt(np.mean((actual_vals - forecast_vals) ** 2)))
    
    def compare_models(self, data: TimeSeries, models: Dict[str, Any], 
                      forecast_horizon: int = 12) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.
        
        Args:
            data: Historical time series
            models: Dictionary of {model_name: model_instance}
            forecast_horizon: Forecast horizon
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name, model in models.items():
            self.logger.info(f"Validating {model_name}...")
            
            try:
                cv_results = self.validate_model(data, model, forecast_horizon)
                
                results.append({
                    'model': model_name,
                    'mean_mape': cv_results['mean_error'],
                    'std_mape': cv_results['std_error'],
                    'median_mape': cv_results['median_error'],
                    'min_mape': cv_results['min_error'],
                    'max_mape': cv_results['max_error'],
                    'n_folds': cv_results['n_successful_folds'],
                    'is_valid': cv_results['is_valid']
                })
            except Exception as e:
                self.logger.error(f"{model_name} validation failed: {e}")
                results.append({
                    'model': model_name,
                    'mean_mape': None,
                    'std_mape': None,
                    'median_mape': None,
                    'min_mape': None,
                    'max_mape': None,
                    'n_folds': 0,
                    'is_valid': False
                })
        
        df = pd.DataFrame(results)
        
        # Sort by mean MAPE (excluding invalid)
        df_valid = df[df['is_valid']].copy()
        df_valid = df_valid.sort_values('mean_mape')
        
        return df_valid


def format_cv_results(results: Dict[str, Any]) -> str:
    """
    Format CV results for display.
    
    Args:
        results: Results from validate_model()
    
    Returns:
        Formatted string
    """
    if not results['is_valid']:
        return "❌ Cross-validation failed (insufficient successful folds)"
    
    metric = results.get('metric_name', 'MAPE')
    
    output = f"""
📊 Cross-Validation Results ({results['n_successful_folds']} folds):
   {metric}: {results['mean_error']:.2f}% ± {results['std_error']:.2f}%
   Median: {results['median_error']:.2f}%
   Range: [{results['min_error']:.2f}%, {results['max_error']:.2f}%]
   
✅ Validation passed with {results['n_successful_folds']} successful folds
"""
    
    return output.strip()
