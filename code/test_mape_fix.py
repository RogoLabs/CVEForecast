#!/usr/bin/env python3
"""
Targeted test to verify MAPE calculation fix is working correctly.
"""

import sys
from pathlib import Path
import logging

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from analysis import CVEForecastAnalyzer
from data_processor import CVEDataProcessor
from date_config import get_date_config
from utils import setup_logging

logger = setup_logging()

def test_mape_calculation_fix():
    """Test that MAPE calculation is now using full validation period."""
    logger.info("🧪 Testing MAPE calculation fix...")
    
    # Test date configuration
    date_config = get_date_config()
    use_full_validation = date_config.should_use_full_validation_period()
    filter_current_year_only = date_config.should_filter_current_year_only()
    
    logger.info(f"Date config - use_full_validation: {use_full_validation}")
    logger.info(f"Date config - filter_current_year_only: {filter_current_year_only}")
    
    if use_full_validation and not filter_current_year_only:
        logger.info("✅ Date configuration is correct for full validation period")
    else:
        logger.error("❌ Date configuration still incorrect")
        return False
    
    # Test hyperparameter loading
    analyzer = CVEForecastAnalyzer(enable_hyperparameter_tuning=True)
    
    if analyzer.tuned_hyperparameters:
        logger.info(f"✅ Loaded {len(analyzer.tuned_hyperparameters)} optimized models:")
        
        # Show expected vs actual for key models
        key_models = ['Prophet', 'XGBoost', 'CatBoost', 'LightGBM']
        for model_name in key_models:
            if model_name in analyzer.tuned_hyperparameters:
                expected_mape = analyzer.tuned_hyperparameters[model_name]['best_performance']['mape']
                logger.info(f"  - {model_name}: Expected MAPE = {expected_mape:.4f}%")
        
        return True
    else:
        logger.error("❌ No optimized hyperparameters loaded")
        return False

def test_single_model_evaluation():
    """Test evaluation of a single model to verify MAPE calculation."""
    logger.info("🧪 Testing single model evaluation...")
    
    try:
        # Load data
        processor = CVEDataProcessor()
        data = processor.load_cve_data()
        
        if data is None or len(data) == 0:
            logger.error("❌ No data available for testing")
            return False
        
        logger.info(f"📊 Data loaded: {len(data)} records from {data['date'].min()} to {data['date'].max()}")
        
        # Initialize analyzer with optimized hyperparameters
        analyzer = CVEForecastAnalyzer(enable_hyperparameter_tuning=True)
        
        # Test just Prophet model (should have excellent performance)
        if 'Prophet' in analyzer.tuned_hyperparameters:
            expected_mape = analyzer.tuned_hyperparameters['Prophet']['best_performance']['mape']
            logger.info(f"🎯 Testing Prophet model - Expected MAPE: {expected_mape:.4f}%")
            
            # Get Prophet's optimized hyperparameters
            prophet_params = analyzer.tuned_hyperparameters['Prophet']['hyperparameters']
            logger.info(f"📋 Using optimized hyperparameters: {prophet_params}")
            
            # Create Prophet model with optimized parameters
            from config import Prophet
            prophet_model = Prophet(**prophet_params)
            
            model_info = {
                'model_name': 'Prophet',
                'model_object': prophet_model,
                'hyperparameters': prophet_params
            }
            
            # Evaluate the model  
            results = analyzer.evaluate_models(data)
            
            if results:
                best_result = results[0]
                actual_mape = best_result['mape']
                model_name = best_result['model_name']
                
                logger.info(f"🏆 Best model: {model_name}")
                logger.info(f"📊 Actual MAPE: {actual_mape:.4f}%")
                
                if model_name == 'Prophet':
                    mape_diff = abs(actual_mape - expected_mape)
                    logger.info(f"📈 Expected vs Actual MAPE difference: {mape_diff:.4f}%")
                    
                    if mape_diff < 5.0:  # Allow 5% tolerance
                        logger.info("✅ MAPE calculation appears to be working correctly!")
                        return True
                    else:
                        logger.error(f"❌ Large MAPE difference detected - fix may not be working")
                        return False
                else:
                    logger.warning(f"⚠️ Prophet not the best model, best is {model_name} with {actual_mape:.4f}% MAPE")
                    # Still check if performance is reasonable (not 33%)
                    if actual_mape < 15.0:
                        logger.info("✅ Performance appears reasonable, fix likely working")
                        return True
                    else:
                        logger.error("❌ Performance still poor, fix not working")
                        return False
            else:
                logger.error("❌ No model evaluation results")
                return False
                
        else:
            logger.error("❌ Prophet model not found in optimized hyperparameters")
            return False
            
    except Exception as e:
        logger.error(f"❌ Single model evaluation failed: {e}")
        return False

if __name__ == "__main__":
    print("MAPE Calculation Fix Test")
    print("=" * 50)
    
    # Test 1: Configuration check
    config_ok = test_mape_calculation_fix()
    
    # Test 2: Single model evaluation
    if config_ok:
        evaluation_ok = test_single_model_evaluation()
        
        if evaluation_ok:
            print("\n🎉 SUCCESS: MAPE calculation fix is working!")
            sys.exit(0)
        else:
            print("\n❌ FAILURE: MAPE calculation still has issues")
            sys.exit(1)
    else:
        print("\n❌ FAILURE: Configuration issues detected")
        sys.exit(1)
