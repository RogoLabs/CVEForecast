#!/usr/bin/env python3
"""
2017 Structural Break Analysis
Analyzes existing validation data to test if post-2017 regime performs better than full historical data.
"""

import json
import pandas as pd
from datetime import datetime

def analyze_structural_break():
    """Compare validation data between filtered (post-2016) and full historical datasets."""
    
    print("🔬 DATASET COMPARISON ANALYSIS")
    print("="*50)
    
    # Load both data files for comparison
    print("📁 Loading data files...")
    
    try:
        with open('data.json', 'r') as f:
            data_filtered = json.load(f)
        print("✅ Loaded data.json (filtered/post-2016 data)")
    except FileNotFoundError:
        print("❌ data.json not found")
        return
    
    try:
        with open('data_baseline_full_historical.json', 'r') as f:
            data_full = json.load(f)
        print("✅ Loaded data_baseline_full_historical.json (full historical baseline)")
    except FileNotFoundError:
        print("❌ data_baseline_full_historical.json not found")
        return
    
    # Check both files have validation data
    if 'all_models_validation' not in data_filtered:
        print("❌ No validation data found in data.json (filtered dataset)")
        return
    if 'all_models_validation' not in data_full:
        print("❌ No validation data found in data_allyears.json (full dataset)")
        return
    
    # Show data range comparison
    print("\n🔍 VALIDATION DATA COMPARISON:")
    for file_name, file_data in [("data.json (post-2016 filtered)", data_filtered), ("data_baseline_full_historical.json (full baseline)", data_full)]:
        if 'all_models_validation' in file_data:
            first_model = list(file_data['all_models_validation'].keys())[0]
            validation_data = file_data['all_models_validation'][first_model]
            if validation_data:
                dates = [item['date'] for item in validation_data]
                print(f"  {file_name}: {min(dates)} to {max(dates)} ({len(dates)} entries)")
            else:
                print(f"  {file_name}: No validation data")
    
    results = {}
    
    # Get common models between both datasets
    filtered_models = set(data_filtered['all_models_validation'].keys())
    full_models = set(data_full['all_models_validation'].keys())
    common_models = filtered_models.intersection(full_models)
    
    print(f"\n📊 Found {len(common_models)} models in both datasets for comparison")
    
    # Analyze each model's performance between datasets
    for model_name in sorted(common_models):
        filtered_validation = data_filtered['all_models_validation'][model_name]
        full_validation = data_full['all_models_validation'][model_name]
        
        if not filtered_validation or not full_validation:
            continue
            
        print(f"\n📈 {model_name} DATASET COMPARISON:")
        
        # Create dictionaries for easy lookup by date
        filtered_data = {item['date']: item for item in filtered_validation if not item.get('is_current_month', False)}
        full_data = {item['date']: item for item in full_validation if not item.get('is_current_month', False)}
        
        # Find overlapping dates
        common_dates = set(filtered_data.keys()).intersection(set(full_data.keys()))
        
        if len(common_dates) < 5:  # Need at least 5 data points for meaningful comparison
            print(f"  ⚠️  Insufficient overlapping data ({len(common_dates)} months)")
            continue
        
        # Calculate MAPE for overlapping period
        filtered_errors = []
        full_errors = []
        
        for date in sorted(common_dates):
            filtered_item = filtered_data[date]
            full_item = full_data[date]
            
            # Calculate percentage errors
            if filtered_item['actual'] > 0:
                filtered_error = abs((filtered_item['actual'] - filtered_item['predicted']) / filtered_item['actual']) * 100
                filtered_errors.append(filtered_error)
            
            if full_item['actual'] > 0:
                full_error = abs((full_item['actual'] - full_item['predicted']) / full_item['actual']) * 100
                full_errors.append(full_error)
        
        if len(filtered_errors) > 0 and len(full_errors) > 0:
            filtered_mape = sum(filtered_errors) / len(filtered_errors)
            full_mape = sum(full_errors) / len(full_errors)
            
            improvement = full_mape - filtered_mape  # Positive means filtered dataset is better
            
            print(f"  Filtered dataset (post-2016): {len(filtered_errors)} months, MAPE: {filtered_mape:.2f}%")
            print(f"  Full dataset (all years): {len(full_errors)} months, MAPE: {full_mape:.2f}%")
            print(f"  Improvement with filtering: {improvement:.2f}% (positive = filtering helps)")
            
            # Store results
            results[model_name] = {
                'filtered_mape': filtered_mape,
                'full_mape': full_mape,
                'improvement': improvement,
                'overlapping_months': len(common_dates)
            }
            
            # Show verdict
            if improvement > 2:
                print(f"  ✅ FILTERING HELPS: Post-2016 data performs better")
            elif improvement < -2:
                print(f"  ❌ FULL DATA BETTER: Historical data improves performance")
            else:
                print(f"  🤔 SIMILAR PERFORMANCE: No significant difference")
        else:
            print(f"  ⚠️  Error calculating metrics for overlapping data")
    
    # Overall analysis
    if results:
        print(f"\n🎯 OVERALL DATASET COMPARISON SUMMARY:")
        print("="*40)
        
        filtering_helps = [name for name, res in results.items() if res['improvement'] > 2]
        full_data_better = [name for name, res in results.items() if res['improvement'] < -2]
        similar_performance = [name for name, res in results.items() if -2 <= res['improvement'] <= 2]
        
        print(f"✅ Filtering HELPS ({len(filtering_helps)} models): {', '.join(filtering_helps[:3])}{'...' if len(filtering_helps) > 3 else ''}")
        print(f"❌ Full data BETTER ({len(full_data_better)} models): {', '.join(full_data_better[:3])}{'...' if len(full_data_better) > 3 else ''}")
        print(f"🤔 Similar performance ({len(similar_performance)} models): {', '.join(similar_performance[:3])}{'...' if len(similar_performance) > 3 else ''}")
        
        # Calculate average improvement
        avg_improvement = sum(res['improvement'] for res in results.values()) / len(results)
        print(f"\n📈 Average improvement with post-2016 filtering: {avg_improvement:.2f}%")
        
        if avg_improvement > 1:
            print("🎉 CONCLUSION: Post-2016 filtering improves forecast accuracy!")
            print("   Recommendation: Continue using filtered post-2016 training data")
        elif avg_improvement < -1:
            print("📚 CONCLUSION: Full historical data provides better accuracy")
            print("   Recommendation: Revert to using complete historical dataset")
        else:
            print("⚖️  CONCLUSION: Minimal difference between approaches")
            print("   Recommendation: Either approach is acceptable")
    
    return results

if __name__ == "__main__":
    results = analyze_structural_break()
