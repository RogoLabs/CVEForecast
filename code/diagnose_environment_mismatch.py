#!/usr/bin/env python3
"""
Diagnostic script to identify environment mismatch between hyperparameter tuning and production evaluation.
Compares the exact setup used in tuning vs production to find the root cause of 33.23% vs 1.84% MAPE discrepancy.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from analysis import CVEForecastAnalyzer
from data_processor import CVEDataProcessor
from config import Prophet
from utils import setup_logging

logger = setup_logging()

def load_tuning_environment_info():
    """Load the exact environment and parameters used during hyperparameter tuning."""
    try:
        # Load the latest hyperparameter tuning results
        results_path = Path('hyperparameter_results/hyperparameter_tuning_results_20250708_204606.json')
        with open(results_path, 'r') as f:
            tuning_results = json.load(f)
        
        # Extract Prophet's tuning environment
        prophet_tuning = tuning_results['models_tuned']['Prophet']
        
        return {
            'tuning_timestamp': prophet_tuning['tuning_timestamp'],
            'tuning_duration': prophet_tuning['tuning_duration_seconds'],
            'best_hyperparameters': prophet_tuning['best_hyperparameters'],
            'best_performance': prophet_tuning['best_performance'],
            'combinations_tested': prophet_tuning['total_combinations_tested'],
            'successful_evaluations': prophet_tuning['successful_evaluations']
        }
    except Exception as e:
        logger.error(f"Failed to load tuning environment info: {e}")
        return None

def reproduce_tuning_environment():
    """Attempt to reproduce the exact environment used during hyperparameter tuning."""
    logger.info("🔬 Reproducing tuning environment...")
    
    # Load tuning info
    tuning_info = load_tuning_environment_info()
    if not tuning_info:
        return None
    
    tuned_params = tuning_info['best_hyperparameters']
    expected_mape = tuning_info['best_performance']['mape']
    
    logger.info(f"📊 Expected MAPE from tuning: {expected_mape:.4f}%")
    logger.info(f"📋 Tuned hyperparameters: {tuned_params}")
    
    try:
        # Load data using same processor as production
        processor = CVEDataProcessor()
        data = processor.parse_cve_data()
        
        logger.info(f"📅 Data loaded: {len(data)} records from {data['date'].min()} to {data['date'].max()}")
        
        # Create analyzer to get same data preprocessing as production
        analyzer = CVEForecastAnalyzer(enable_hyperparameter_tuning=True)
        
        # Get the data preprocessing that would happen in production evaluation
        # We need to simulate the exact same data preprocessing pipeline
        clean_data = data.copy()
        clean_data['cve_count'] = clean_data['cve_count'].fillna(0)
        clean_data = clean_data.dropna()
        
        # Create complete monthly date range (same as production)
        start_date = clean_data['date'].min()
        end_date = clean_data['date'].max()
        complete_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        complete_df = pd.DataFrame({'date': complete_dates})
        clean_data = complete_df.merge(clean_data, on='date', how='left')
        clean_data['cve_count'] = clean_data['cve_count'].fillna(0)
        
        # Apply the same data transformation as production
        original_cv = clean_data['cve_count'].std() / clean_data['cve_count'].mean()
        logger.info(f"📊 Original data CV: {original_cv:.3f}")
        
        # Apply log transformation if needed (same logic as production)
        if original_cv > 0.8:
            logger.info("Applying log transformation (same as production)")
            clean_data['cve_count_original'] = clean_data['cve_count'].copy()
            clean_data['cve_count'] = np.log1p(clean_data['cve_count'])
            transformed_cv = clean_data['cve_count'].std() / clean_data['cve_count'].mean()
            logger.info(f"📊 Transformed data CV: {transformed_cv:.3f}")
        
        return {
            'data': clean_data,
            'tuned_params': tuned_params,
            'expected_mape': expected_mape,
            'data_transformation_applied': original_cv > 0.8
        }
        
    except Exception as e:
        logger.error(f"Failed to reproduce tuning environment: {e}")
        return None

def test_direct_model_evaluation():
    """Test Prophet model directly with tuned hyperparameters using same data as production."""
    logger.info("🧪 Testing direct model evaluation...")
    
    # Reproduce tuning environment
    env_info = reproduce_tuning_environment()
    if not env_info:
        return False
    
    data = env_info['data']
    tuned_params = env_info['tuned_params']
    expected_mape = env_info['expected_mape']
    
    try:
        from darts import TimeSeries
        
        # Create TimeSeries (same as production)
        ts = TimeSeries.from_dataframe(
            data,
            time_col='date',
            value_cols='cve_count', 
            freq='MS',
            fill_missing_dates=False
        )
        
        logger.info(f"📈 TimeSeries created: {len(ts)} data points")
        
        # Use same train/validation split as production (80/20)
        split_point = int(len(ts) * 0.8)
        train_ts = ts[:split_point]
        val_ts = ts[split_point:]
        
        logger.info(f"🔄 Train/validation split: {len(train_ts)}/{len(val_ts)}")
        
        # Create Prophet model with EXACT tuned hyperparameters
        logger.info("🔮 Creating Prophet model with exact tuned hyperparameters...")
        prophet_model = Prophet(**tuned_params)
        
        # Train and predict (same as production)
        logger.info("🏋️ Training Prophet model...")
        prophet_model.fit(train_ts)
        
        logger.info("🔮 Generating predictions...")
        prediction = prophet_model.predict(len(val_ts))
        
        # Calculate MAPE manually (same logic as production)
        val_values = val_ts.values().flatten()
        pred_values = prediction.values().flatten()
        
        # Apply reverse transformation if needed
        if env_info['data_transformation_applied']:
            logger.info("🔄 Applying reverse log transformation...")
            val_original = np.expm1(val_values)
            pred_original = np.expm1(pred_values) 
        else:
            val_original = val_values
            pred_original = pred_values
        
        # Calculate MAPE
        mape = np.mean(np.abs((val_original - pred_original) / val_original)) * 100
        
        logger.info(f"📊 DIRECT EVALUATION RESULTS:")
        logger.info(f"  Expected MAPE (from tuning): {expected_mape:.4f}%")
        logger.info(f"  Actual MAPE (direct test): {mape:.4f}%")
        logger.info(f"  Difference: {abs(mape - expected_mape):.4f}%")
        
        # Check if results match
        if abs(mape - expected_mape) < 5.0:  # 5% tolerance
            logger.info("✅ DIRECT TEST MATCHES TUNING RESULTS!")
            logger.info("🔍 This suggests production evaluation has a different issue")
            return True
        else:
            logger.error("❌ DIRECT TEST DOESN'T MATCH TUNING RESULTS")
            logger.error("🔍 This confirms environment mismatch")
            
            # Additional diagnostics
            logger.info("🔬 Additional diagnostics:")
            logger.info(f"  Validation data range: {val_original.min():.0f} - {val_original.max():.0f}")
            logger.info(f"  Prediction data range: {pred_original.min():.0f} - {pred_original.max():.0f}")
            logger.info(f"  Mean validation value: {val_original.mean():.0f}")
            logger.info(f"  Mean prediction value: {pred_original.mean():.0f}")
            
            return False
        
    except Exception as e:
        logger.error(f"Direct model evaluation failed: {e}")
        return False

def compare_production_vs_tuning():
    """Compare production evaluation setup vs tuning setup to identify differences."""
    logger.info("🔬 Comparing production vs tuning setup...")
    
    try:
        # Test 1: Direct model evaluation (reproducing tuning environment)
        direct_test_ok = test_direct_model_evaluation()
        
        # Test 2: Production evaluation (using current analysis.py)
        logger.info("🏭 Testing production evaluation setup...")
        analyzer = CVEForecastAnalyzer(enable_hyperparameter_tuning=True)
        processor = CVEDataProcessor()
        data = processor.parse_cve_data()
        
        # Run production evaluation (should show 33.23% MAPE)
        results = analyzer.evaluate_models(data)
        
        if results:
            production_mape = results[0]['mape']
            production_model = results[0]['model_name']
            logger.info(f"🏭 Production evaluation: {production_model} = {production_mape:.4f}% MAPE")
        else:
            logger.error("❌ Production evaluation failed")
            return False
        
        # Compare results
        tuning_info = load_tuning_environment_info()
        if tuning_info:
            expected_mape = tuning_info['best_performance']['mape']
            
            logger.info(f"\n📊 COMPARISON SUMMARY:")
            logger.info(f"  Expected (tuning): {expected_mape:.4f}%")
            logger.info(f"  Production: {production_mape:.4f}%")
            logger.info(f"  Direct test: {'PASSED' if direct_test_ok else 'FAILED'}")
            
            if direct_test_ok and abs(production_mape - expected_mape) > 5.0:
                logger.info("🎯 ROOT CAUSE IDENTIFIED:")
                logger.info("  - Direct reproduction of tuning environment works correctly")
                logger.info("  - Production evaluation pipeline has different setup")
                logger.info("  - Issue is in production evaluation logic, not hyperparameters or data")
                return True
            elif not direct_test_ok:
                logger.info("🎯 ROOT CAUSE IDENTIFIED:")
                logger.info("  - Cannot reproduce tuning environment results")
                logger.info("  - Issue is in data preprocessing or model setup")
                return False
        
        return False
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return False

if __name__ == "__main__":
    print("CVE Forecast Environment Mismatch Diagnostic")
    print("=" * 60)
    
    success = compare_production_vs_tuning()
    
    if success:
        print("\n🎉 ROOT CAUSE IDENTIFIED! Environment mismatch found.")
        sys.exit(0)
    else:
        print("\n❌ Unable to identify root cause. Further investigation needed.")
        sys.exit(1)
