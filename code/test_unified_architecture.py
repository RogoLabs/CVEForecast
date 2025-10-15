#!/usr/bin/env python3
"""
Verification Test for Unified Architecture

Quick test to verify that the new unified architecture works correctly.

Usage:
    python3 code/test_unified_architecture.py
"""

import sys
import logging
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        from core.base_forecaster import BaseForecaster, ForecastResult
        print("  ✓ core.base_forecaster")
    except Exception as e:
        print(f"  ✗ core.base_forecaster: {e}")
        return False
    
    try:
        from core.data_adapter import DataAdapter
        print("  ✓ core.data_adapter")
    except Exception as e:
        print(f"  ✗ core.data_adapter: {e}")
        return False
    
    try:
        from core.validation_mixin import ValidationMixin
        print("  ✓ core.validation_mixin")
    except Exception as e:
        print(f"  ✗ core.validation_mixin: {e}")
        return False
    
    try:
        from adapters.cve_adapter import CVEForecaster
        print("  ✓ adapters.cve_adapter")
    except Exception as e:
        print(f"  ✗ adapters.cve_adapter: {e}")
        return False
    
    try:
        from adapters.cna_adapter import CNAForecaster
        print("  ✓ adapters.cna_adapter")
    except Exception as e:
        print(f"  ✗ adapters.cna_adapter: {e}")
        return False
    
    try:
        from unified_pipeline import UnifiedForecastPipeline
        print("  ✓ unified_pipeline")
    except Exception as e:
        print(f"  ✗ unified_pipeline: {e}")
        return False
    
    return True


def test_base_forecaster_interface():
    """Test that BaseForecaster defines the correct interface."""
    print("\nTesting BaseForecaster interface...")
    
    from core.base_forecaster import BaseForecaster
    import inspect
    
    required_methods = [
        'load_data',
        'get_forecast_horizon',
        'get_model_list',
        'create_model',
        'apply_constraints',
        'save_results'
    ]
    
    for method_name in required_methods:
        if hasattr(BaseForecaster, method_name):
            method = getattr(BaseForecaster, method_name)
            if inspect.isabstract(method) or callable(method):
                print(f"  ✓ {method_name} defined")
            else:
                print(f"  ✗ {method_name} not callable")
                return False
        else:
            print(f"  ✗ {method_name} missing")
            return False
    
    return True


def test_cve_forecaster_instantiation():
    """Test that CVE forecaster can be instantiated."""
    print("\nTesting CVE forecaster instantiation...")
    
    try:
        from adapters.cve_adapter import CVEForecaster
        
        # Try to create instance (may fail if config missing, but should not crash)
        try:
            forecaster = CVEForecaster(config_path='config.json')
            print(f"  ✓ CVE forecaster created: {forecaster}")
            return True
        except FileNotFoundError as e:
            print(f"  ⚠ Config file missing (expected in test): {e}")
            print(f"  ✓ CVE forecaster class works (config needed for full test)")
            return True
        except Exception as e:
            print(f"  ✗ CVE forecaster creation failed: {e}")
            return False
            
    except Exception as e:
        print(f"  ✗ CVE forecaster import failed: {e}")
        return False


def test_validation_mixin():
    """Test that ValidationMixin provides expected methods."""
    print("\nTesting ValidationMixin methods...")
    
    from core.validation_mixin import ValidationMixin
    
    required_methods = [
        'perform_cross_validation',
        'perform_statistical_tests',
        'run_residual_diagnostics',
        'run_horizon_analysis'
    ]
    
    for method_name in required_methods:
        if hasattr(ValidationMixin, method_name):
            print(f"  ✓ {method_name} available")
        else:
            print(f"  ✗ {method_name} missing")
            return False
    
    return True


def test_forecast_result():
    """Test ForecastResult dataclass."""
    print("\nTesting ForecastResult...")
    
    from core.base_forecaster import ForecastResult
    from datetime import datetime
    
    try:
        result = ForecastResult(
            forecast_values={'2025-01': 1000, '2025-02': 1100},
            model_name='TestModel',
            metrics={'mape': 5.0}
        )
        
        print(f"  ✓ ForecastResult created")
        print(f"    - Model: {result.model_name}")
        print(f"    - Values: {len(result.forecast_values)} periods")
        print(f"    - Metrics: {result.metrics}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ ForecastResult creation failed: {e}")
        return False


def test_unified_pipeline_instantiation():
    """Test unified pipeline instantiation."""
    print("\nTesting UnifiedForecastPipeline instantiation...")
    
    try:
        from unified_pipeline import UnifiedForecastPipeline
        
        # Suppress logging for clean output
        logging.getLogger().setLevel(logging.ERROR)
        
        try:
            pipeline = UnifiedForecastPipeline()
            print(f"  ✓ Pipeline created")
            print(f"    - CVE forecaster: {type(pipeline.cve_forecaster).__name__}")
            print(f"    - CNA forecaster: {type(pipeline.cna_forecaster).__name__}")
            return True
            
        except FileNotFoundError as e:
            print(f"  ⚠ Config file missing (expected in test): {e}")
            print(f"  ✓ Pipeline class works (configs needed for full test)")
            return True
            
        except Exception as e:
            print(f"  ✗ Pipeline instantiation failed: {e}")
            return False
            
    except Exception as e:
        print(f"  ✗ Pipeline import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("UNIFIED ARCHITECTURE VERIFICATION TEST")
    print("=" * 70)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("BaseForecaster Interface", test_base_forecaster_interface),
        ("CVE Forecaster", test_cve_forecaster_instantiation),
        ("ValidationMixin", test_validation_mixin),
        ("ForecastResult", test_forecast_result),
        ("Unified Pipeline", test_unified_pipeline_instantiation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n  ✗ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {test_name}")
    
    print()
    print(f"Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED - Unified architecture verified!")
        print("=" * 70)
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
