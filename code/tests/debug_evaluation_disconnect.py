#!/usr/bin/env python3
"""
Debug script to investigate why same evaluation method produces different results
Direct evaluation: TiDE 12.5% MAPE vs Production pipeline: TiDE 33.8% MAPE
"""

import sys
import pandas as pd
from data_processor import CVEDataProcessor
from analysis import CVEForecastAnalyzer

def debug_evaluation_disconnect():
    print("=== DEBUGGING EVALUATION DISCONNECT ===")
    
    # Load data same as production
    print("1. Loading CVE data...")
    processor = CVEDataProcessor()
    data = processor.parse_cve_data()
    
    print(f"   Data shape: {data.shape}")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    print(f"   Records: {len(data)}")
    
    # Create analyzer same as production
    print("\n2. Creating analyzer...")
    analyzer = CVEForecastAnalyzer()
    
    # Run evaluation same as production
    print("\n3. Running evaluation (same as production)...")
    try:
        results = analyzer.evaluate_models(data)
        
        print(f"\n4. Results summary:")
        print(f"   Models evaluated: {len(results)}")
        
        if results:
            print("\n   Top 5 models:")
            for i, result in enumerate(results[:5], 1):
                name = result['model_name']
                mape = result['mape']
                print(f"     {i}. {name}: {mape:.4f}% MAPE")
                
            # Check if we're getting the good results (12-18% range) or bad results (33%+ range)
            best_mape = results[0]['mape']
            if best_mape < 20:
                print(f"\n   ✅ GOOD RESULTS: Best MAPE {best_mape:.4f}% (expected ~12-18% range)")
            else:
                print(f"\n   ❌ BAD RESULTS: Best MAPE {best_mape:.4f}% (should be ~12-18% range)")
                print("   🔍 This matches the broken production pipeline results!")
        else:
            print("   ❌ No results returned!")
            
    except Exception as e:
        print(f"   ❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_evaluation_disconnect()
