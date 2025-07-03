#!/usr/bin/env python3
"""
Test script for new statistical models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_new_models():
    """Test the new statistical models"""
    print("Testing new statistical models...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=50, freq='MS')
    values = np.random.randint(1000, 5000, 50)
    
    # Test individual model imports
    models_to_test = []
    
    try:
        from darts.models import VARIMA
        models_to_test.append(('VARIMA', VARIMA))
        print("✓ VARIMA imported")
    except Exception as e:
        print(f"✗ VARIMA failed: {e}")
    
    try:
        from darts.models import FFT
        models_to_test.append(('FFT', FFT))
        print("✓ FFT imported")
    except Exception as e:
        print(f"✗ FFT failed: {e}")
    
    try:
        from darts.models import TBATS
        models_to_test.append(('TBATS', TBATS))
        print("✓ TBATS imported")
    except Exception as e:
        print(f"✗ TBATS failed: {e}")
    
    try:
        from darts.models import Croston
        models_to_test.append(('Croston', Croston))
        print("✓ Croston imported")
    except Exception as e:
        print(f"✗ Croston failed: {e}")
    
    try:
        from darts.models import FourTheta
        models_to_test.append(('FourTheta', FourTheta))
        print("✓ FourTheta imported")
    except Exception as e:
        print(f"✗ FourTheta failed: {e}")
    
    try:
        from darts.models import NaiveMean
        models_to_test.append(('NaiveMean', NaiveMean))
        print("✓ NaiveMean imported")
    except Exception as e:
        print(f"✗ NaiveMean failed: {e}")
    
    try:
        from darts.models import NaiveMovingAverage
        models_to_test.append(('NaiveMovingAverage', lambda: NaiveMovingAverage(input_chunk_length=12)))
        print("✓ NaiveMovingAverage imported")
    except Exception as e:
        print(f"✗ NaiveMovingAverage failed: {e}")
    
    print(f"\nSuccessfully imported {len(models_to_test)} new models!")
    
    # Test model initialization
    print("\nTesting model initialization...")
    for name, model_class in models_to_test:
        try:
            if name == 'NaiveMovingAverage':
                model = model_class()
            else:
                model = model_class()
            print(f"✓ {name} initialized successfully")
        except Exception as e:
            print(f"✗ {name} initialization failed: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_new_models()
