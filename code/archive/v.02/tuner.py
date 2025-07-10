#!/usr/bin/env python3
"""
Systematic hyperparameter tuning module for CVE Forecast application.
Proactively searches for optimal hyperparameters using grid search and random search.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from pathlib import Path
from collections import defaultdict
import statistics
import numpy as np
import pandas as pd
from datetime import datetime
import itertools
import random
from copy import deepcopy
import time

from darts import TimeSeries
from config import (
    ARIMA, ExponentialSmoothing, Prophet, Theta, FourTheta,
    LinearRegressionModel, RandomForestModel, XGBModel,
    KalmanForecaster, NaiveSeasonal, NaiveDrift, NaiveMean, NaiveMovingAverage,
    DartsAutoARIMA, NaiveEnsembleModel, RegressionEnsembleModel, 
    LightGBMModel, CatBoostModel, FFT, TBATS, Croston,
    TCNModel, TFTModel, NBEATSModel, NHiTSModel,
    TransformerModel, RNNModel, BlockRNNModel,
    TiDEModel, DLinearModel, NLinearModel, TSMixerModel,
    STATSFORECAST_ARIMA_AVAILABLE, StatsForecastAutoARIMA,
    STATSFORECAST_ETS_AVAILABLE, StatsForecastAutoETS,
    STATSFORECAST_THETA_AVAILABLE, StatsForecastAutoTheta,
    STATSFORECAST_CES_AVAILABLE, StatsForecastAutoCES,
    STATSFORECAST_MFLES_AVAILABLE, StatsForecastAutoMFLES,
    STATSFORECAST_TBATS_AVAILABLE, StatsForecastAutoTBATS,
    mape, rmse, mae, mase, rmsse, ModelMode, SeasonalityMode,
    MODEL_EVALUATION_SPLIT, FORECAST_HORIZON_MONTHS, ENSEMBLE_SIZE
)
from utils import setup_logging, get_model_category

logger = setup_logging()


class HyperparameterTuner:
    """Systematic hyperparameter optimization using grid search and random search."""
    
    def __init__(self, 
                 hyperparameters_config_path: str = "hyperparameters.json",
                 performance_history_path: str = "../web/performance_history.json",
                 tuning_results_path: str = "../web/tuning_results.json"):
        """
        Initialize the systematic hyperparameter tuner.
        
        Args:
            hyperparameters_config_path: Path to the hyperparameters configuration JSON file
            performance_history_path: Path to the performance history file
            tuning_results_path: Path to save tuning results
        """
        self.config_path = hyperparameters_config_path
        self.history_path = performance_history_path
        self.results_path = tuning_results_path
        
        # Configuration and results storage
        self.hyperparameters_config = {}
        self.tuning_config = {}
        self.tuning_results = {}
        self.performance_history = []
        
        # Load configuration
        self._load_hyperparameters_config()
        
        # Search state
        self.current_model = None
        self.current_trial = 0
        self.total_trials = 0
        
    def _load_hyperparameters_config(self) -> bool:
        """
        Load hyperparameters configuration from JSON file.
        
        Returns:
            True if configuration was loaded successfully, False otherwise
        """
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.error(f"Hyperparameters configuration file {self.config_path} not found")
                return False
                
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
            self.hyperparameters_config = config_data.get('models', {})
            self.tuning_config = config_data.get('tuning_config', {})
            
            logger.info(f"Loaded hyperparameter configurations for {len(self.hyperparameters_config)} models")
            logger.info(f"Tuning configuration: {self.tuning_config}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading hyperparameters configuration: {e}")
            return False
    
    def load_performance_history(self) -> bool:
        """
        Load performance history from JSON file.
        
        Returns:
            True if history was loaded successfully, False otherwise
        """
        try:
            if Path(self.history_path).exists():
                with open(self.history_path, 'r') as f:
                    self.performance_history = json.load(f)
                logger.info(f"Loaded {len(self.performance_history)} performance history records")
                return True
            else:
                logger.warning(f"Performance history file {self.history_path} not found")
                return False
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
            return False
    
    def _generate_parameter_combinations(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for systematic search.
        
        Args:
            model_name: Name of the model to generate combinations for
            
        Returns:
            List of parameter combinations to test
        """
        if model_name not in self.hyperparameters_config:
            logger.warning(f"No configuration found for model {model_name}")
            return []
        
        model_config = self.hyperparameters_config[model_name]
        search_space = model_config.get('search_space', {})
        search_method = model_config.get('search_method', 'grid')
        default_params = model_config.get('default_params', {})
        
        if not search_space:
            logger.info(f"No search space defined for {model_name}, using default parameters only")
            return [default_params]
        
        logger.info(f"Generating parameter combinations for {model_name} using {search_method} search")
        
        if search_method == 'grid':
            return self._generate_grid_combinations(search_space, default_params)
        elif search_method == 'random':
            n_trials = model_config.get('n_trials', 10)
            return self._generate_random_combinations(search_space, default_params, n_trials)
        else:
            logger.info(f"Search method 'none' for {model_name}, using default parameters only")
            return [default_params]
    
    def _generate_grid_combinations(self, search_space: Dict[str, List], default_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate all combinations using grid search.
        
        Args:
            search_space: Dictionary of parameter names to lists of values
            default_params: Default parameter values
            
        Returns:
            List of all parameter combinations
        """
        if not search_space:
            return [default_params]
        
        # Create parameter combinations
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        
        combinations = []
        for combination in itertools.product(*param_values):
            params = default_params.copy()
            for i, param_name in enumerate(param_names):
                params[param_name] = combination[i]
            combinations.append(params)
        
        logger.info(f"Generated {len(combinations)} grid search combinations")
        return combinations
    
    def _generate_random_combinations(self, search_space: Dict[str, List], default_params: Dict[str, Any], n_trials: int) -> List[Dict[str, Any]]:
        """
        Generate random combinations using random search.
        
        Args:
            search_space: Dictionary of parameter names to lists of values
            default_params: Default parameter values
            n_trials: Number of random combinations to generate
            
        Returns:
            List of random parameter combinations
        """
        if not search_space:
            return [default_params]
        
        random.seed(self.tuning_config.get('random_seed', 42))
        
        combinations = []
        # Always include default parameters as first trial
        combinations.append(default_params)
        
        # Generate random combinations
        for _ in range(n_trials - 1):
            params = default_params.copy()
            for param_name, param_values in search_space.items():
                params[param_name] = random.choice(param_values)
            combinations.append(params)
        
        logger.info(f"Generated {len(combinations)} random search combinations")
        return combinations
    
    def _create_model_instance(self, model_name: str, hyperparameters: Dict[str, Any]):
        """
        Create a model instance with given hyperparameters.
        
        Args:
            model_name: Name of the model
            hyperparameters: Hyperparameters to use
            
        Returns:
            Model instance or None if creation fails
        """
        try:
            # Deep learning models common trainer kwargs
            mps_trainer_kwargs = {
                "enable_progress_bar": False,
                "accelerator": "cpu",
                "precision": "64-true"
            }
            
            # Filter out None values and prepare parameters
            clean_params = {k: v for k, v in hyperparameters.items() if v is not None}
            
            # Statistical models
            if model_name == 'Prophet':
                return Prophet(**clean_params)
            elif model_name == 'ExponentialSmoothing':
                return ExponentialSmoothing(**clean_params)
            elif model_name == 'TBATS':
                return TBATS(**clean_params)
            elif model_name == 'AutoARIMA':
                if STATSFORECAST_ARIMA_AVAILABLE and StatsForecastAutoARIMA is not None:
                    return StatsForecastAutoARIMA()
                else:
                    return DartsAutoARIMA(**clean_params)
            elif model_name == 'Theta':
                season_mode = clean_params.get('season_mode', 'ADDITIVE')
                if season_mode == 'ADDITIVE':
                    return Theta(season_mode=SeasonalityMode.ADDITIVE)
                elif season_mode == 'MULTIPLICATIVE':
                    return Theta(season_mode=SeasonalityMode.MULTIPLICATIVE)
                else:
                    return Theta(season_mode=SeasonalityMode.NONE)
            elif model_name == 'FourTheta':
                season_mode = clean_params.get('season_mode', 'ADDITIVE')
                if season_mode == 'ADDITIVE':
                    return FourTheta(season_mode=SeasonalityMode.ADDITIVE)
                elif season_mode == 'MULTIPLICATIVE':
                    return FourTheta(season_mode=SeasonalityMode.MULTIPLICATIVE)
                else:
                    return FourTheta(season_mode=SeasonalityMode.NONE)
            
            # Tree-based models
            elif model_name == 'XGBoost':
                return XGBModel(**clean_params)
            elif model_name == 'LightGBM':
                return LightGBMModel(**clean_params)
            elif model_name == 'CatBoost':
                return CatBoostModel(**clean_params)
            elif model_name == 'RandomForest':
                return RandomForestModel(**clean_params)
            
            # Deep learning models
            elif model_name == 'TCN':
                params = clean_params.copy()
                params['pl_trainer_kwargs'] = mps_trainer_kwargs
                return TCNModel(**params)
            elif model_name == 'NBEATS':
                params = clean_params.copy()
                params['pl_trainer_kwargs'] = mps_trainer_kwargs
                return NBEATSModel(**params)
            elif model_name == 'NHiTS':
                params = clean_params.copy()
                params['pl_trainer_kwargs'] = mps_trainer_kwargs
                return NHiTSModel(**params)
            elif model_name == 'TiDE':
                params = clean_params.copy()
                params['pl_trainer_kwargs'] = mps_trainer_kwargs
                return TiDEModel(**params)
            elif model_name == 'DLinear':
                params = clean_params.copy()
                params['pl_trainer_kwargs'] = mps_trainer_kwargs
                return DLinearModel(**params)
            elif model_name == 'NLinear':
                params = clean_params.copy()
                params['pl_trainer_kwargs'] = mps_trainer_kwargs
                return NLinearModel(**params)
            elif model_name == 'TSMixer':
                params = clean_params.copy()
                params['pl_trainer_kwargs'] = mps_trainer_kwargs
                return TSMixerModel(**params)
            
            # Baseline models
            elif model_name == 'LinearRegression':
                return LinearRegressionModel(**clean_params)
            elif model_name == 'NaiveSeasonal':
                return NaiveSeasonal(**clean_params)
            elif model_name == 'NaiveMean':
                return NaiveMean()
            elif model_name == 'NaiveDrift':
                return NaiveDrift()
            
            # Auto models (no hyperparameters)
            elif model_name == 'AutoETS' and STATSFORECAST_ETS_AVAILABLE:
                return StatsForecastAutoETS()
            elif model_name == 'AutoTheta' and STATSFORECAST_THETA_AVAILABLE:
                return StatsForecastAutoTheta()
            elif model_name == 'KalmanFilter':
                return KalmanForecaster()
            elif model_name == 'Croston':
                return Croston()
            
            else:
                logger.warning(f"Unknown model: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create {model_name} model: {e}")
            return None
    
    def _evaluate_model_configuration(self, model_name: str, hyperparameters: Dict[str, Any], 
                                    train_ts: TimeSeries, val_ts: TimeSeries) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single model configuration.
        
        Args:
            model_name: Name of the model
            hyperparameters: Hyperparameters to test
            train_ts: Training time series
            val_ts: Validation time series
            
        Returns:
            Dictionary with evaluation results or None if evaluation failed
        """
        try:
            # Create model instance
            model = self._create_model_instance(model_name, hyperparameters)
            if model is None:
                return None
            
            # Validate TimeSeries input format
            if not isinstance(train_ts, TimeSeries) or not isinstance(val_ts, TimeSeries):
                logger.warning(f"Invalid TimeSeries format for {model_name}")
                return None
            
            # Ensure TimeSeries has proper structure
            if len(train_ts) < 2 or len(val_ts) < 1:
                logger.warning(f"Insufficient data for {model_name}: train={len(train_ts)}, val={len(val_ts)}")
                return None
            
            start_time = time.time()
            
            # Train model with proper error handling
            try:
                model.fit(train_ts)
            except Exception as fit_error:
                logger.warning(f"Model {model_name} fit failed: {fit_error}")
                return None
            
            # Generate predictions with proper error handling
            try:
                predictions = model.predict(n=len(val_ts))
            except Exception as pred_error:
                logger.warning(f"Model {model_name} prediction failed: {pred_error}")
                return None
            
            # Validate predictions
            if predictions is None or len(predictions) == 0:
                logger.warning(f"Model {model_name} returned empty predictions")
                return None
            
            # Apply log transformation reverse to get original scale for evaluation
            # BUT use TimeSeries objects directly for Darts metrics (they expect TimeSeries input)
            
            # Calculate metrics using Darts TimeSeries-based functions
            try:
                # Darts metrics expect TimeSeries objects, not numpy arrays
                mape_score = float(mape(val_ts, predictions))
                mae_score = float(mae(val_ts, predictions))
                rmse_score = float(rmse(val_ts, predictions))
                
                logger.debug(f"Basic metrics calculated: MAPE={mape_score:.4f}, MAE={mae_score:.4f}, RMSE={rmse_score:.4f}")
                
            except Exception as metric_error:
                logger.warning(f"Darts TimeSeries metric calculation failed: {metric_error}")
                # Fallback to manual metric calculation with numpy arrays
                try:
                    # Extract values manually for fallback calculation
                    if hasattr(predictions, 'values'):
                        pred_vals = predictions.values().flatten()
                    else:
                        pred_vals = predictions.pd_dataframe().values.flatten()
                        
                    if hasattr(val_ts, 'values'):
                        val_vals = val_ts.values().flatten()
                    else:
                        val_vals = val_ts.pd_dataframe().values.flatten()
                    
                    # Manual MAPE calculation
                    mape_score = float(np.mean(np.abs((val_vals - pred_vals) / np.maximum(np.abs(val_vals), 1e-8)) * 100))
                    mae_score = float(np.mean(np.abs(val_vals - pred_vals)))
                    rmse_score = float(np.sqrt(np.mean((val_vals - pred_vals) ** 2)))
                    
                    logger.debug(f"Fallback metrics calculated: MAPE={mape_score:.4f}, MAE={mae_score:.4f}, RMSE={rmse_score:.4f}")
                    
                except Exception as fallback_error:
                    logger.warning(f"Fallback metric calculation also failed: {fallback_error}")
                    return None
            
            # Calculate advanced metrics (MASE, RMSSE) with proper error handling
            try:
                mase_score = float(mase(val_ts, predictions, train_ts))
                rmsse_score = float(rmsse(val_ts, predictions, train_ts))
                logger.debug(f"Advanced metrics calculated: MASE={mase_score:.4f}, RMSSE={rmsse_score:.4f}")
            except Exception as advanced_metric_error:
                logger.debug(f"Advanced metrics calculation failed (normal): {advanced_metric_error}")
                mase_score = None
                rmsse_score = None
            
            training_time = time.time() - start_time
            
            return {
                'model_name': model_name,
                'hyperparameters': hyperparameters,
                'metrics': {
                    'mape': mape_score,
                    'mae': mae_score,
                    'rmse': rmse_score,
                    'mase': mase_score,
                    'rmsse': rmsse_score
                },
                'training_time': training_time,
                'status': 'success'
            }
            
        except Exception as e:
            logger.warning(f"Model {model_name} failed evaluation: {e}")
            return {
                'model_name': model_name,
                'hyperparameters': hyperparameters,
                'metrics': {},
                'training_time': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    def tune_model_hyperparameters(self, model_name: str, train_ts: TimeSeries, val_ts: TimeSeries, 
                                  primary_metric: str = 'mape') -> Optional[Dict[str, Any]]:
        """
        Systematically tune hyperparameters for a single model.
        
        Args:
            model_name: Name of the model to tune
            train_ts: Training time series data
            val_ts: Validation time series data
            primary_metric: Primary metric to optimize for
            
        Returns:
            Dictionary with best hyperparameters and performance or None if tuning failed
        """
        if model_name not in self.hyperparameters_config:
            logger.warning(f"No configuration found for model {model_name}")
            return None
            
        logger.info(f"🔧 Starting systematic hyperparameter tuning for {model_name}...")
        
        # Generate parameter combinations to test
        combinations = self._generate_parameter_combinations(model_name)
        
        if not combinations:
            logger.warning(f"No parameter combinations generated for {model_name}")
            return None
        
        logger.info(f"Testing {len(combinations)} parameter combinations for {model_name}")
        
        # Evaluate each combination
        evaluation_results = []
        for i, combination in enumerate(combinations):
            logger.info(f"  [{i+1}/{len(combinations)}] Testing configuration: {combination}")
            
            result = self._evaluate_model_configuration(model_name, combination, train_ts, val_ts)
            if result is not None and result['status'] == 'success':
                evaluation_results.append(result)
                mape_score = result['metrics'].get('mape', float('inf'))
                logger.info(f"    ✓ Success: MAPE = {mape_score:.4f}")
            else:
                logger.warning(f"    ✗ Failed: {result.get('error', 'Unknown error') if result else 'Model creation failed'}")
        
        if not evaluation_results:
            logger.error(f"All parameter combinations failed for {model_name}")
            return None
        
        # Find best performing configuration
        best_result = min(evaluation_results, key=lambda x: x['metrics'].get(primary_metric, float('inf')))
        
        logger.info(f"🎯 Best configuration for {model_name}:")
        logger.info(f"   MAPE: {best_result['metrics'].get('mape', 'N/A'):.4f}")
        logger.info(f"   Hyperparameters: {best_result['hyperparameters']}")
        
        return {
            'model_name': model_name,
            'best_hyperparameters': best_result['hyperparameters'],
            'best_performance': best_result['metrics'],
            'total_combinations_tested': len(combinations),
            'successful_evaluations': len(evaluation_results),
            'tuning_timestamp': datetime.now().isoformat()
        }
    
    def get_tuned_hyperparameters(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get optimized hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Optimized hyperparameters dict or None if not available
        """
        if model_name in self.optimal_hyperparameters:
            return self.optimal_hyperparameters[model_name]['hyperparameters']
        else:
            logger.debug(f"No optimized hyperparameters available for {model_name}")
            return None
    
    def get_expected_performance(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Get expected performance metrics for a model with optimal hyperparameters.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Expected performance metrics dict or None if not available
        """
        if model_name in self.optimal_hyperparameters:
            return self.optimal_hyperparameters[model_name]['expected_performance']
        else:
            return None
    
    def should_retune(self, model_name: str, runs_threshold: int = 5) -> bool:
        """
        Determine if a model should be retuned based on the number of runs.
        
        Args:
            model_name: Name of the model
            runs_threshold: Minimum number of runs before considering retuning
            
        Returns:
            True if model should be retuned, False otherwise
        """
        if model_name not in self.model_performance_db:
            return False
            
        run_count = len(self.model_performance_db[model_name])
        return run_count >= runs_threshold
    
    def export_optimization_report(self, output_path: str = "../web/optimization_report.json") -> bool:
        """
        Export detailed optimization report to JSON file.
        
        Args:
            output_path: Path to save the optimization report
            
        Returns:
            True if report was exported successfully, False otherwise
        """
        try:
            report = {
                'generation_timestamp': self.performance_history[-1].get('timestamp') if self.performance_history else None,
                'total_runs_analyzed': len(self.performance_history),
                'models_optimized': len(self.optimal_hyperparameters),
                'optimization_details': self.optimal_hyperparameters,
                'model_performance_summary': {}
            }
            
            # Add performance summary for each model
            for model_name, records in self.model_performance_db.items():
                if records:
                    mape_values = [r['metrics'].get('mape') for r in records 
                                 if r['metrics'].get('mape') is not None]
                    if mape_values:
                        report['model_performance_summary'][model_name] = {
                            'runs_count': len(records),
                            'best_mape': min(mape_values),
                            'worst_mape': max(mape_values),
                            'average_mape': statistics.mean(mape_values),
                            'performance_trend': 'improving' if len(mape_values) > 1 and mape_values[-1] < mape_values[0] else 'stable'
                        }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Optimization report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export optimization report: {e}")
            return False
    
    def tune_models(self) -> bool:
        """
        Complete hyperparameter tuning workflow.
        
        Returns:
            True if tuning completed successfully, False otherwise
        """
        logger.info("🔧 Starting hyperparameter tuning workflow...")
        
        # Load performance history
        if not self.load_performance_history():
            logger.error("Cannot proceed without performance history")
            return False
        
        if not self.performance_history:
            logger.warning("No performance history available for tuning")
            return False
        
        # Analyze performances and find optimal parameters
        self.analyze_model_performances()
        self.find_optimal_hyperparameters()
        
        if not self.optimal_hyperparameters:
            logger.warning("No optimal hyperparameters found")
            return False
        
        # Export detailed report
        self.export_optimization_report()
        
        logger.info(f"🎯 Hyperparameter tuning completed successfully!")
        logger.info(f"   Optimized {len(self.optimal_hyperparameters)} models")
        logger.info(f"   Based on {len(self.performance_history)} historical runs")
        
        return True


def main():
    """Main function for testing the tuner module."""
    tuner = HyperparameterTuner()
    success = tuner.tune_models()
    
    if success:
        print("✅ Hyperparameter tuning completed successfully!")
        
        # Display some results
        print("\nOptimized models:")
        for model_name in tuner.optimal_hyperparameters:
            expected_perf = tuner.get_expected_performance(model_name)
            mape = expected_perf.get('mape') if expected_perf else None
            print(f"  - {model_name}: Expected MAPE = {mape:.4f}" if mape else f"  - {model_name}")
    else:
        print("❌ Hyperparameter tuning failed!")


if __name__ == "__main__":
    main()
