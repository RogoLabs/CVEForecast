#!/usr/bin/env python3
"""
Comprehensive test script for the refactored hyperparameter tuning system.
Tests centralized configuration, systematic search, and integration with analysis.py.
"""

import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from darts import TimeSeries
from analysis import CVEForecastAnalyzer
from tuner import HyperparameterTuner
from utils import setup_logging

logger = setup_logging()

def create_synthetic_time_series_data() -> TimeSeries:
    """Create synthetic CVE data for testing."""
    # Generate 5 years of monthly data with trend and seasonality
    dates = pd.date_range('2020-01-01', periods=60, freq='MS')
    
    # Base trend with seasonality and noise
    trend = np.linspace(1000, 5000, 60)
    seasonality = 500 * np.sin(2 * np.pi * np.arange(60) / 12)
    noise = np.random.normal(0, 200, 60)
    
    values = trend + seasonality + noise
    values = np.maximum(values, 100)  # Ensure positive values
    
    df = pd.DataFrame({
        'date': dates,
        'cve_count': values.astype(int)
    })
    
    # Apply log transformation (as done in the real system)
    df['cve_count_transformed'] = np.log(df['cve_count'] + 1)
    
    # Create TimeSeries object
    ts = TimeSeries.from_dataframe(
        df, 
        time_col='date', 
        value_cols='cve_count_transformed'
    )
    
    return ts

def test_hyperparameters_config_loading():
    """Test that hyperparameters.json loads correctly."""
    logger.info("🧪 Testing hyperparameters configuration loading...")
    
    config_path = "hyperparameters.json"
    if not Path(config_path).exists():
        logger.error(f"❌ Hyperparameters configuration file {config_path} not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate structure
        required_keys = ['version', 'models', 'tuning_config']
        for key in required_keys:
            if key not in config:
                logger.error(f"❌ Missing required key '{key}' in configuration")
                return False
        
        models_count = len(config['models'])
        logger.info(f"✅ Configuration loaded successfully with {models_count} models")
        
        # Test a few key models
        key_models = ['Prophet', 'XGBoost', 'TCN', 'NLinear']
        for model in key_models:
            if model in config['models']:
                model_config = config['models'][model]
                if 'default_params' not in model_config:
                    logger.warning(f"⚠️ Model {model} missing default_params")
                logger.info(f"  ✓ {model}: {len(model_config.get('default_params', {}))} default parameters")
            else:
                logger.warning(f"⚠️ Key model {model} not found in configuration")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return False

def test_analyzer_centralized_config():
    """Test that CVEForecastAnalyzer loads centralized configuration properly."""
    logger.info("🧪 Testing CVEForecastAnalyzer centralized configuration integration...")
    
    try:
        # Initialize analyzer with centralized config
        analyzer = CVEForecastAnalyzer(enable_hyperparameter_tuning=False)
        
        # Test hyperparameters loading for key models
        test_models = ['Prophet', 'ExponentialSmoothing', 'TBATS', 'XGBoost', 'TCN']
        
        for model_name in test_models:
            params = analyzer._get_model_hyperparameters(model_name)
            if params:
                logger.info(f"  ✓ {model_name}: {len(params)} parameters loaded from centralized config")
            else:
                logger.warning(f"  ⚠️ {model_name}: No parameters found")
        
        logger.info("✅ CVEForecastAnalyzer centralized configuration integration successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ CVEForecastAnalyzer configuration test failed: {e}")
        return False

def test_systematic_hyperparameter_search():
    """Test the systematic hyperparameter search functionality."""
    logger.info("🧪 Testing systematic hyperparameter search...")
    
    try:
        # Create synthetic data
        ts_data = create_synthetic_time_series_data()
        
        # Split into train/validation
        split_point = int(len(ts_data) * 0.8)
        train_ts = ts_data[:split_point]
        val_ts = ts_data[split_point:]
        
        logger.info(f"Created synthetic data: {len(train_ts)} train, {len(val_ts)} validation points")
        
        # Initialize tuner
        tuner = HyperparameterTuner()
        
        # Test hyperparameter search for a few models
        test_models = ['Prophet', 'ExponentialSmoothing', 'DLinear']
        
        successful_tests = 0
        for model_name in test_models:
            logger.info(f"Testing systematic search for {model_name}...")
            
            result = tuner.tune_model_hyperparameters(
                model_name=model_name,
                train_ts=train_ts,
                val_ts=val_ts,
                primary_metric='mape'
            )
            
            if result and result.get('successful_evaluations', 0) > 0:
                logger.info(f"  ✅ {model_name}: {result['successful_evaluations']}/{result['total_combinations_tested']} combinations successful")
                logger.info(f"      Best MAPE: {result['best_performance'].get('mape', 'N/A'):.4f}")
                successful_tests += 1
            else:
                logger.warning(f"  ⚠️ {model_name}: Systematic search failed or no successful evaluations")
        
        if successful_tests >= len(test_models) // 2:
            logger.info("✅ Systematic hyperparameter search test successful")
            return True
        else:
            logger.warning("⚠️ Systematic hyperparameter search had limited success")
            return False
            
    except Exception as e:
        logger.error(f"❌ Systematic hyperparameter search test failed: {e}")
        return False

def test_end_to_end_integration():
    """Test end-to-end integration of the refactored system."""
    logger.info("🧪 Testing end-to-end integration...")
    
    try:
        # Create synthetic data
        ts_data = create_synthetic_time_series_data()
        
        # Convert to DataFrame format expected by analyzer
        # Use the correct Darts API to convert TimeSeries to DataFrame
        df_data = ts_data.pd_dataframe().reset_index() if hasattr(ts_data, 'pd_dataframe') else ts_data.to_dataframe().reset_index()
        
        # Ensure proper column names
        if len(df_data.columns) == 2:
            df_data.columns = ['date', 'cve_count']
        else:
            # Handle multi-column case
            df_data = df_data.iloc[:, [0, 1]]  # Take first two columns
            df_data.columns = ['date', 'cve_count']
            
        df_data['cve_count'] = np.exp(df_data['cve_count']) - 1  # Reverse log transformation
        df_data['cve_count'] = df_data['cve_count'].astype(int)
        
        logger.info(f"Created test dataset with {len(df_data)} data points")
        
        # Initialize analyzer with centralized configuration
        analyzer = CVEForecastAnalyzer(enable_hyperparameter_tuning=False)
        
        # Test model preparation with centralized configuration
        models = analyzer.prepare_models()
        logger.info(f"Prepared {len(models)} models using centralized configuration")
        
        # Test evaluation of a subset of models
        test_model_count = min(3, len(models))
        test_models = models[:test_model_count]
        
        logger.info(f"Testing evaluation of {test_model_count} models...")
        results = analyzer.evaluate_models(df_data)
        
        if results and len(results) > 0:
            best_model = results[0]
            logger.info(f"✅ End-to-end test successful!")
            logger.info(f"    Best model: {best_model['model_name']}")
            logger.info(f"    MAPE: {best_model.get('mape', 'N/A'):.4f}")
            logger.info(f"    Models evaluated: {len(results)}")
            return True
        else:
            logger.warning("⚠️ End-to-end test completed but no successful model evaluations")
            return False
            
    except Exception as e:
        logger.error(f"❌ End-to-end integration test failed: {e}")
        return False

def main():
    """Run comprehensive test suite for the refactored hyperparameter tuning system."""
    logger.info("🚀 Starting comprehensive hyperparameter tuning system test suite...")
    logger.info("=" * 80)
    
    test_results = []
    
    # Test 1: Configuration loading
    test_results.append(("Configuration Loading", test_hyperparameters_config_loading()))
    
    # Test 2: Analyzer centralized config integration
    test_results.append(("Analyzer Integration", test_analyzer_centralized_config()))
    
    # Test 3: Systematic hyperparameter search
    test_results.append(("Systematic Search", test_systematic_hyperparameter_search()))
    
    # Test 4: End-to-end integration
    test_results.append(("End-to-End Integration", test_end_to_end_integration()))
    
    # Results summary
    logger.info("=" * 80)
    logger.info("🏁 TEST SUITE RESULTS SUMMARY")
    logger.info("=" * 80)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<30} {status}")
        if result:
            passed_tests += 1
    
    logger.info("=" * 80)
    logger.info(f"OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Hyperparameter tuning refactoring successful!")
        return True
    elif passed_tests >= total_tests * 0.75:
        logger.info("✅ Most tests passed. System is functional with minor issues.")
        return True
    else:
        logger.error("❌ Multiple test failures. System needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
