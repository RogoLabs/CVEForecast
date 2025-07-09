#!/usr/bin/env python3
"""
Comprehensive hyperparameter tuning script for CVEForecast system.
Runs systematic hyperparameter optimization across all models.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from darts import TimeSeries
from tuner import HyperparameterTuner
from analysis import CVEForecastAnalyzer
from data_processor import CVEDataProcessor
from utils import setup_logging

logger = setup_logging()

class FullHyperparameterTuningSession:
    """Orchestrates comprehensive hyperparameter tuning across all models."""
    
    def __init__(self, data_file_path: str = None):
        """
        Initialize full tuning session.
        
        Args:
            data_file_path: Path to your CVE data directory (optional)
        """
        self.data_file_path = data_file_path
        # Fix file path issue for hyperparameters.json
        config_path = Path(__file__).parent / "hyperparameters.json"
        self.tuner = HyperparameterTuner(hyperparameters_config_path=str(config_path))
        self.results = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"🚀 Starting hyperparameter tuning session: {self.session_id}")
    
    def prepare_data(self) -> tuple[TimeSeries, TimeSeries]:
        """Prepare training and validation data from your CVE dataset."""
        logger.info("📊 Preparing CVE data for hyperparameter tuning...")
        
        try:
            # Load and process your actual CVE data
            processor = CVEDataProcessor(self.data_file_path)
            df = processor.parse_cve_data()
            
            # Apply log transformation (as done in your main system)
            df['cve_count_log_transformed'] = np.log(df['cve_count'] + 1)
            
            # Create time series
            ts_data = TimeSeries.from_dataframe(
                df,
                time_col='date',
                value_cols='cve_count_log_transformed'
            )
            
            # Split into train/validation (80/20)
            split_point = int(len(ts_data) * 0.8)
            train_ts = ts_data[:split_point]
            val_ts = ts_data[split_point:]
            
            logger.info(f"✅ Data prepared: {len(train_ts)} train, {len(val_ts)} validation periods")
            logger.info(f"Training: {train_ts.time_index[0]} to {train_ts.time_index[-1]}")
            logger.info(f"Validation: {val_ts.time_index[0]} to {val_ts.time_index[-1]}")
            
            return train_ts, val_ts
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare data: {e}")
            raise
    
    def get_models_for_tuning(self, priority_models: List[str] = None) -> List[str]:
        """
        Get list of models to tune.
        
        Args:
            priority_models: Optional list of priority models to tune first
            
        Returns:
            List of model names to tune
        """
        available_models = list(self.tuner.hyperparameters_config.keys())
        
        if priority_models:
            # Tune priority models first, then others
            priority_set = set(priority_models)
            other_models = [m for m in available_models if m not in priority_set]
            valid_priority = [m for m in priority_models if m in available_models]
            return valid_priority + other_models
        
        return available_models
    
    def tune_single_model(self, model_name: str, train_ts: TimeSeries, 
                         val_ts: TimeSeries, max_combinations: int = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for a single model.
        
        Args:
            model_name: Name of model to tune
            train_ts: Training time series
            val_ts: Validation time series  
            max_combinations: Optional limit on combinations to test
            
        Returns:
            Tuning results dictionary
        """
        logger.info(f"🔧 Tuning {model_name}...")
        
        start_time = datetime.now()
        
        try:
            result = self.tuner.tune_model_hyperparameters(
                model_name=model_name,
                train_ts=train_ts,
                val_ts=val_ts,
                primary_metric='mape'
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result and result.get('successful_evaluations', 0) > 0:
                best_mape = result['best_performance'].get('mape', float('inf'))
                success_rate = result['successful_evaluations'] / result['total_combinations_tested'] * 100
                
                logger.info(f"✅ {model_name} tuning complete:")
                logger.info(f"   Best MAPE: {best_mape:.4f}")
                logger.info(f"   Success rate: {success_rate:.1f}%")
                logger.info(f"   Duration: {duration:.1f}s")
                
                result['tuning_duration_seconds'] = duration
                result['tuning_timestamp'] = end_time.isoformat()
                return result
            else:
                logger.warning(f"⚠️ {model_name} tuning failed - no successful evaluations")
                return {
                    'model_name': model_name,
                    'status': 'failed',
                    'tuning_duration_seconds': duration,
                    'tuning_timestamp': end_time.isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ {model_name} tuning failed: {e}")
            return {
                'model_name': model_name,
                'status': 'error',
                'error': str(e),
                'tuning_timestamp': datetime.now().isoformat()
            }
    
    def run_full_tuning_session(self, 
                               priority_models: List[str] = None,
                               max_models: int = None,
                               save_results: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive hyperparameter tuning across all models.
        
        Args:
            priority_models: Models to tune first (e.g., ['Prophet', 'XGBoost', 'TCN'])
            max_models: Maximum number of models to tune (for testing)
            save_results: Whether to save results to JSON file
            
        Returns:
            Complete tuning session results
        """
        session_start = datetime.now()
        logger.info(f"🎯 Starting full hyperparameter tuning session: {self.session_id}")
        
        # Prepare data
        train_ts, val_ts = self.prepare_data()
        
        # Get models to tune
        models_to_tune = self.get_models_for_tuning(priority_models)
        if max_models:
            models_to_tune = models_to_tune[:max_models]
        
        logger.info(f"📋 Models to tune: {len(models_to_tune)}")
        if priority_models:
            logger.info(f"🎯 Priority models: {priority_models}")
        
        # Tune each model
        session_results = {
            'session_id': self.session_id,
            'start_time': session_start.isoformat(),
            'models_tuned': {},
            'summary': {}
        }
        
        successful_tuning = 0
        failed_tuning = 0
        
        for i, model_name in enumerate(models_to_tune, 1):
            logger.info(f"\n--- [{i}/{len(models_to_tune)}] {model_name} ---")
            
            model_result = self.tune_single_model(model_name, train_ts, val_ts)
            session_results['models_tuned'][model_name] = model_result
            
            if model_result.get('successful_evaluations', 0) > 0:
                successful_tuning += 1
            else:
                failed_tuning += 1
            
            # Progress update
            progress = (i / len(models_to_tune)) * 100
            logger.info(f"📈 Session progress: {progress:.1f}% ({i}/{len(models_to_tune)})")
        
        # Session summary
        session_end = datetime.now()
        total_duration = (session_end - session_start).total_seconds()
        
        session_results['end_time'] = session_end.isoformat()
        session_results['total_duration_seconds'] = total_duration
        session_results['summary'] = {
            'total_models': len(models_to_tune),
            'successful_tuning': successful_tuning,
            'failed_tuning': failed_tuning,
            'success_rate_percent': (successful_tuning / len(models_to_tune)) * 100,
            'average_duration_per_model': total_duration / len(models_to_tune)
        }
        
        # Find best performing models
        best_models = []
        for model_name, result in session_results['models_tuned'].items():
            if result.get('successful_evaluations', 0) > 0:
                mape = result['best_performance'].get('mape', float('inf'))
                best_models.append((model_name, mape))
        
        best_models.sort(key=lambda x: x[1])
        session_results['summary']['top_3_models'] = best_models[:3]
        
        # Log session summary
        logger.info(f"\n🏁 HYPERPARAMETER TUNING SESSION COMPLETE")
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Duration: {total_duration/60:.1f} minutes")
        logger.info(f"Success rate: {session_results['summary']['success_rate_percent']:.1f}%")
        logger.info(f"Models successfully tuned: {successful_tuning}/{len(models_to_tune)}")
        
        if best_models:
            logger.info("🏆 Top 3 performing models:")
            for i, (model_name, mape) in enumerate(best_models[:3], 1):
                logger.info(f"  {i}. {model_name}: MAPE={mape:.4f}")
        
        # Save results
        if save_results:
            self.save_tuning_results(session_results)
        
        return session_results
    
    def save_tuning_results(self, results: Dict[str, Any]):
        """Save tuning results to JSON file in organized directory structure."""
        # Create results directory if it doesn't exist
        results_dir = Path(__file__).parent / "hyperparameter_results"
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"hyperparameter_tuning_results_{self.session_id}.json"
        
        try:
            import json
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"💾 Tuning results saved to: {results_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

def main():
    """Example usage of full hyperparameter tuning."""
    
    # Option 1: Quick tuning of priority models
    logger.info("=== OPTION 1: Priority Models Tuning ===")
    session = FullHyperparameterTuningSession()
    
    priority_models = ['Prophet', 'XGBoost', 'TCN', 'DLinear', 'LinearRegression']
    results = session.run_full_tuning_session(
        priority_models=priority_models,
        max_models=5  # Limit for testing
    )
    
    print(f"\n🎯 Quick tuning session completed!")
    print(f"Session ID: {results['session_id']}")
    print(f"Success rate: {results['summary']['success_rate_percent']:.1f}%")
    
    # Option 2: Full comprehensive tuning (uncomment to run)
    # logger.info("=== OPTION 2: Full Comprehensive Tuning ===")
    # full_results = session.run_full_tuning_session()

if __name__ == "__main__":
    main()
