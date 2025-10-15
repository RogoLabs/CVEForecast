"""
Centralized warning suppression for production runs.

Suppresses known non-critical warnings from:
- sklearn/LightGBM feature name mismatches
- scipy numerical warnings in Prophet
- Other library warnings that don't affect results
"""

import warnings
import logging

def suppress_production_warnings():
    """Suppress known non-critical warnings for production runs."""
    
    # Suppress sklearn feature name warnings
    # These come from Darts internally and don't affect results
    warnings.filterwarnings('ignore', 
                          message='X does not have valid feature names',
                          category=UserWarning,
                          module='sklearn')
    
    # Suppress scipy numerical warnings from Prophet and Kalman filter
    # These are handled internally and don't affect results
    warnings.filterwarnings('ignore',
                          message='overflow encountered in matmul',
                          category=RuntimeWarning)
    
    warnings.filterwarnings('ignore',
                          message='invalid value encountered in matmul',
                          category=RuntimeWarning)
    
    warnings.filterwarnings('ignore',
                          message='divide by zero encountered in matmul',
                          category=RuntimeWarning)
    
    # Suppress nfoursid (Kalman filter) numerical warnings
    warnings.filterwarnings('ignore',
                          category=RuntimeWarning,
                          module='nfoursid')
    
    # Suppress pandas FutureWarnings about downcasting
    warnings.filterwarnings('ignore',
                          message='.*downcasting.*',
                          category=FutureWarning,
                          module='pandas')
    
    # Suppress darts internal warnings
    warnings.filterwarnings('ignore',
                          category=FutureWarning,
                          module='darts')
    
    # Suppress library logging (set to CRITICAL to hide ERROR messages)
    logging.getLogger('darts').setLevel(logging.CRITICAL)
    logging.getLogger('prophet').setLevel(logging.CRITICAL)
    logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
    logging.getLogger('statsmodels').setLevel(logging.CRITICAL)
    logging.getLogger('numba').setLevel(logging.CRITICAL)
    
    # Suppress all warnings from these modules entirely
    import sys
    for module_name in ['darts', 'prophet', 'statsmodels', 'numba']:
        if module_name in sys.modules:
            sys.modules[module_name].logger = logging.getLogger('null')
            sys.modules[module_name].logger.addHandler(logging.NullHandler())


def enable_all_warnings():
    """Re-enable all warnings (for debugging)."""
    warnings.resetwarnings()
