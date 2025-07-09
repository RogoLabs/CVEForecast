#!/usr/bin/env python3
"""
End-to-end pipeline test script for CVE Forecast system.
Tests the complete workflow from hyperparameter loading to data.json generation.
"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from main import main
from analysis import CVEForecastAnalyzer
from utils import setup_logging

logger = setup_logging()

def test_hyperparameter_loading():
    """Test that optimized hyperparameters are properly loaded."""
    logger.info("🔬 Testing hyperparameter loading...")
    
    analyzer = CVEForecastAnalyzer(enable_hyperparameter_tuning=True)
    
    # Check if tuned hyperparameters were loaded
    if analyzer.tuned_hyperparameters:
        logger.info(f"✅ Successfully loaded {len(analyzer.tuned_hyperparameters)} optimized models:")
        for model_name, result in analyzer.tuned_hyperparameters.items():
            mape = result.get('best_performance', {}).get('mape', 'N/A')
            logger.info(f"  - {model_name}: MAPE = {mape}")
        return True
    else:
        logger.error("❌ Failed to load optimized hyperparameters")
        return False

def test_model_hyperparameter_usage():
    """Test that models actually use the optimized hyperparameters."""
    logger.info("🔬 Testing model hyperparameter usage...")
    
    analyzer = CVEForecastAnalyzer(enable_hyperparameter_tuning=True)
    
    # Test specific models that should have optimized parameters
    test_models = ['Prophet', 'XGBoost', 'LightGBM', 'CatBoost']
    success_count = 0
    
    for model_name in test_models:
        params = analyzer._get_model_hyperparameters(model_name)
        if params and model_name in analyzer.tuned_hyperparameters:
            expected_params = analyzer.tuned_hyperparameters[model_name]['hyperparameters']
            if params == expected_params:
                logger.info(f"✅ {model_name}: Using optimized hyperparameters")
                success_count += 1
            else:
                logger.warning(f"⚠️ {model_name}: Parameter mismatch detected")
        else:
            logger.warning(f"⚠️ {model_name}: No optimized parameters found")
    
    return success_count > 0

def test_data_json_format():
    """Test that data.json has the correct format for website compatibility."""
    logger.info("🔬 Testing data.json format...")
    
    # Look for data.json in web directory
    possible_paths = [
        Path('../web/data.json'),
        Path('web/data.json'),
        Path(__file__).parent.parent / 'web' / 'data.json'
    ]
    
    data_json_path = None
    for path in possible_paths:
        if path.exists():
            data_json_path = path
            break
    
    if not data_json_path:
        logger.warning("⚠️ data.json not found - run main pipeline first")
        return False
    
    try:
        with open(data_json_path, 'r') as f:
            data = json.load(f)
        
        # Check for required top-level keys
        required_keys = [
            'generated_at', 'model_rankings', 'historical_data', 
            'current_month_actual', 'forecasts', 'yearly_forecast_totals',
            'cumulative_timelines', 'validation_against_actuals'
        ]
        
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            logger.error(f"❌ Missing required keys in data.json: {missing_keys}")
            return False
        
        # Check cumulative_timelines format (critical for website compatibility)
        cumulative_timelines = data.get('cumulative_timelines', {})
        if not cumulative_timelines:
            logger.error("❌ No cumulative_timelines found in data.json")
            return False
        
        # Check for proper model naming (should end with _cumulative)
        timeline_models = list(cumulative_timelines.keys())
        cumulative_models = [m for m in timeline_models if m.endswith('_cumulative')]
        
        if len(cumulative_models) == 0:
            logger.error("❌ No models with _cumulative suffix found in cumulative_timelines")
            return False
        
        # Check timeline data structure
        sample_timeline = list(cumulative_timelines.values())[0]
        if not isinstance(sample_timeline, list) or len(sample_timeline) == 0:
            logger.error("❌ Invalid timeline data structure")
            return False
        
        # Check for required fields in timeline entries
        sample_entry = sample_timeline[0]
        if 'date' not in sample_entry or 'cumulative_total' not in sample_entry:
            logger.error("❌ Timeline entries missing required fields (date, cumulative_total)")
            return False
        
        logger.info(f"✅ data.json format validation passed:")
        logger.info(f"  - File size: {data_json_path.stat().st_size:,} bytes")
        logger.info(f"  - Model rankings: {len(data.get('model_rankings', []))}")
        logger.info(f"  - Historical data points: {len(data.get('historical_data', []))}")
        logger.info(f"  - Forecast models: {len(data.get('forecasts', {}))}")
        logger.info(f"  - Cumulative timelines: {len(cumulative_timelines)} ({len(cumulative_models)} with _cumulative suffix)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error reading data.json: {e}")
        return False

def test_model_performance_regression():
    """Test that model performance has not regressed (should be <10% MAPE for top models)."""
    logger.info("🔬 Testing model performance regression...")
    
    # Look for data.json to check current model performance
    possible_paths = [
        Path('../web/data.json'),
        Path('web/data.json'),
        Path(__file__).parent.parent / 'web' / 'data.json'
    ]
    
    data_json_path = None
    for path in possible_paths:
        if path.exists():
            data_json_path = path
            break
    
    if not data_json_path:
        logger.warning("⚠️ data.json not found - run main pipeline first")
        return False
    
    try:
        with open(data_json_path, 'r') as f:
            data = json.load(f)
        
        model_rankings = data.get('model_rankings', [])
        if not model_rankings:
            logger.error("❌ No model rankings found in data.json")
            return False
        
        # Check performance of top models
        performance_ok = True
        for i, model in enumerate(model_rankings[:5]):  # Check top 5 models
            model_name = model.get('model_name', 'Unknown')
            mape = model.get('mape')
            
            if mape is None:
                logger.warning(f"⚠️ {model_name}: No MAPE value found")
                continue
            
            # Performance thresholds based on expected optimized results
            if i == 0 and mape > 5.0:  # Best model should be <5% MAPE
                logger.error(f"❌ Best model {model_name} has high MAPE: {mape:.2f}% (expected <5%)")
                performance_ok = False
            elif mape > 15.0:  # Other top models should be <15% MAPE
                logger.error(f"❌ Model {model_name} has high MAPE: {mape:.2f}% (expected <15%)")
                performance_ok = False
            else:
                logger.info(f"✅ {model_name}: MAPE = {mape:.2f}%")
        
        return performance_ok
        
    except Exception as e:
        logger.error(f"❌ Error checking model performance: {e}")
        return False

def run_full_pipeline_test():
    """Run the complete pipeline and test all components."""
    logger.info("🚀 Starting full pipeline test...")
    
    test_results = {
        'hyperparameter_loading': False,
        'hyperparameter_usage': False,
        'data_json_format': False,
        'performance_regression': False
    }
    
    # Test 1: Hyperparameter loading
    test_results['hyperparameter_loading'] = test_hyperparameter_loading()
    
    # Test 2: Hyperparameter usage
    test_results['hyperparameter_usage'] = test_model_hyperparameter_usage()
    
    # Test 3: Run main pipeline (if hyperparameters are working)
    if test_results['hyperparameter_loading']:
        logger.info("🔥 Running main CVE forecast pipeline...")
        try:
            main()
            logger.info("✅ Main pipeline completed successfully")
        except Exception as e:
            logger.error(f"❌ Main pipeline failed: {e}")
            return test_results
    
    # Test 4: Data.json format validation
    test_results['data_json_format'] = test_data_json_format()
    
    # Test 5: Performance regression test
    test_results['performance_regression'] = test_model_performance_regression()
    
    # Summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info(f"\n🎯 TEST SUMMARY:")
    logger.info(f"{'='*50}")
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"{'='*50}")
    logger.info(f"Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! CVE Forecast system is working correctly.")
        return True
    else:
        logger.error(f"⚠️ {total_tests - passed_tests} test(s) failed. System needs attention.")
        return False

if __name__ == "__main__":
    print(f"CVE Forecast Pipeline Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    success = run_full_pipeline_test()
    
    exit_code = 0 if success else 1
    print(f"\nExiting with code: {exit_code}")
    sys.exit(exit_code)
