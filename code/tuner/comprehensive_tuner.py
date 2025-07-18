#!/usr/bin/env python3
"""
Comprehensive CVE Forecast Model Hyperparameter Tuner
Performs exhaustive exploration of hyperparameter space with minimal output noise.
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
import signal
import multiprocessing
import contextlib
import threading
import atexit
from datetime import datetime
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
from itertools import product
import random

def load_config(config_path="tuner_config.json"):
    """
    Load configuration from a JSON file (default: tuner_config.json in code/tuner directory)
    Note: The tuner now uses code/tuner/tuner_config.json by default to avoid conflict with the main app config.json.
    """
    # If using default config path, resolve it relative to this script's directory
    if config_path == "tuner_config.json":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, "r") as f:
        return json.load(f)

# Removed argparse in favor of config.json
# All options are now loaded from config.json

try:
    from sklearn.model_selection import ParameterGrid
except ImportError:
    print("sklearn not available, using basic parameter grid")
    ParameterGrid = None
try:
    import pandas as pd
except ImportError:
    print("pandas not available")
    pd = None

# Suppress warnings to reduce output noise
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress LightGBM verbose logging (conditional import to prevent blocking)
try:
    import lightgbm as lgb
    # Set LightGBM environment variables for quiet operation
    os.environ['LIGHTGBM_EXEC'] = '1'
    os.environ['LIGHTGBM_VERBOSITY'] = '-1'
except (ImportError, OSError) as e:
    # LightGBM not available or has missing dependencies - continue without it
    pass

# Add parent directory to path to import CVE forecast modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_cve_data
from model_trainer import train_and_evaluate_model
from utils import setup_logging
from darts import TimeSeries
from darts.metrics import mae, mape, mase, rmsse


def cleanup_multiprocessing():
    """Clean up multiprocessing resources to prevent ResourceTracker errors"""
    try:
        # Force cleanup of any remaining multiprocessing resources
        for p in multiprocessing.active_children():
            try:
                p.terminate()
                p.join(timeout=1)
            except:
                pass
    except Exception:
        pass  # Ignore cleanup errors


# Set multiprocessing start method to avoid conflicts
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set


# Register cleanup function
atexit.register(cleanup_multiprocessing)


class TimeoutException(Exception):
    """Exception raised when model training times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Model training timed out")


class DynamicTimeoutManager:
    """Manages dynamic timeout redistribution across active models"""
    
    def __init__(self, total_budget_minutes: float, active_models: List[str]):
        self.total_budget_seconds = total_budget_minutes * 60
        self.start_time = time.time()
        self.active_models = set(active_models)
        self.completed_models = set()
        self.failed_models = set()
        
    def get_current_timeout(self, model_name: str) -> float:
        """Calculate current timeout for a model based on remaining time and active models"""
        elapsed = time.time() - self.start_time
        remaining_time = max(0, self.total_budget_seconds - elapsed)
        remaining_models = len(self.active_models) - len(self.completed_models) - len(self.failed_models)
        
        if remaining_models <= 0:
            return 0
            
        # Distribute remaining time equally among active models
        timeout_per_model = remaining_time / remaining_models
        return max(30, timeout_per_model)  # Minimum 30 seconds per model
    
    def model_completed(self, model_name: str, success: bool = True):
        """Mark model as completed and redistribute time"""
        if success:
            self.completed_models.add(model_name)
        else:
            self.failed_models.add(model_name)
        
        # Log the redistribution
        remaining_models = len(self.active_models) - len(self.completed_models) - len(self.failed_models)
        if remaining_models > 0:
            new_timeout = self.get_current_timeout(model_name)
            print(f"🔄 Dynamic timeout: {new_timeout/60:.1f} minutes per remaining model ({remaining_models} models left)")
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time in seconds"""
        return time.time() - self.start_time
    
    def get_remaining_time(self) -> float:
        """Get remaining time in seconds"""
        return max(0, self.total_budget_seconds - self.get_elapsed_time())
    
    def is_budget_exhausted(self) -> bool:
        """Check if the total time budget is exhausted"""
        return self.get_remaining_time() <= 0


class ThreadedModelRunner:
    """Run model training in a separate thread with timeout and better resource management"""
    
    def __init__(self):
        self.result = None
        self.exception = None
        self.completed = False
        self.thread = None
        
    def run_model(self, model_name, temp_model_config, series, eval_config):
        """Run model training in separate thread with resource cleanup"""
        try:
            # Force cleanup before starting
            cleanup_multiprocessing()
            
            self.result = train_and_evaluate_model(
                model_name, temp_model_config, series, eval_config
            )
            self.completed = True
            
            # Force cleanup after completion
            cleanup_multiprocessing()
            
        except Exception as e:
            self.exception = e
            self.completed = True
            # Force cleanup on error
            cleanup_multiprocessing()
    
    def run_with_timeout(self, model_name, temp_model_config, series, eval_config, timeout_seconds):
        """Run model training with timeout and improved resource management"""
        try:
            self.thread = threading.Thread(
                target=self.run_model,
                args=(model_name, temp_model_config, series, eval_config),
                daemon=True
            )
            self.thread.start()
            self.thread.join(timeout=timeout_seconds)
            
            if self.thread.is_alive():
                # Thread is still running, meaning timeout occurred
                # Force cleanup
                cleanup_multiprocessing()
                return None, None, None, None
            elif self.exception:
                # Thread completed with exception
                cleanup_multiprocessing()
                return None, None, None, None
            elif self.completed and self.result:
                # Thread completed successfully
                if len(self.result) == 4:
                    return self.result
                else:
                    return self.result + (None,)
            else:
                # Thread completed but no result
                cleanup_multiprocessing()
                return None, None, None, None
                
        except Exception as e:
            # Cleanup on any error
            cleanup_multiprocessing()
            return None, None, None, None
        finally:
            # Final cleanup
            cleanup_multiprocessing()


@dataclass
class HyperparameterResult:
    """Container for hyperparameter tuning results"""
    model_name: str
    split_ratio: float
    hyperparameters: Dict[str, Any]
    mape: float
    mae: float
    mase: float
    rmsse: float
    training_time: float
    success: bool
    error_message: Optional[str] = None
    trial_number: int = 0


@dataclass
class SearchSpaceConfig:
    """Configuration for search space limits and timeouts"""
    # Default limits for different model types
    DEFAULT_LIMITS = {
        'simple': 1000,      # ExponentialSmoothing, KalmanFilter, Croston
        'medium': 2000,      # Prophet, AutoARIMA, TBATS, Theta, FourTheta (reduced from 10000)
        'complex': 50000,    # XGBoost, LightGBM, CatBoost, RandomForest
        'minimal': 100       # Reserved for future models with no hyperparameters
    }
    
    # Default timeout limits by model complexity (in seconds)
    DEFAULT_TIMEOUTS = {
        'simple': 120,       # 2 minutes - ExponentialSmoothing, KalmanFilter, Croston (increased from 60s)
        'medium': 300,       # 5 minutes - Prophet, AutoARIMA, TBATS, Theta, FourTheta (increased from 120s)
        'complex': 900,      # 15 minutes - XGBoost, LightGBM, CatBoost, RandomForest (increased from 600s)
        'minimal': 60        # 1 minute - Reserved for future models with no hyperparameters
    }
    
    # Model-specific timeout overrides for problematic models
    MODEL_SPECIFIC_TIMEOUTS = {
        'Prophet': 300,      # 5 minutes for Prophet (increased from 30s)
        'AutoARIMA': 180,    # 3 minutes for AutoARIMA (increased from 90s)
        'TBATS': 240,        # 4 minutes for TBATS
        'ExponentialSmoothing': 120,  # 2 minutes for ExponentialSmoothing
    }
    
    # Problematic models that need special handling
    PROBLEMATIC_MODELS = ['XGBoost', 'LightGBM', 'CatBoost']
    
    # Models that are known to hang and should be automatically disabled
    DISABLED_MODELS = []  # Re-enabled all models to work through them systematically
    
    # Model complexity classification
    MODEL_COMPLEXITY = {
        'LinearRegression': 'medium',
        'ExponentialSmoothing': 'simple',
        'Prophet': 'medium',
        'AutoARIMA': 'medium',
        'TBATS': 'medium',
        'Theta': 'medium',
        'FourTheta': 'medium',
        'XGBoost': 'complex',
        'LightGBM': 'complex',
        'CatBoost': 'complex',
        'RandomForest': 'complex',
        'KalmanFilter': 'simple',
        'Croston': 'simple'
    }
    
    def __init__(self, max_combinations: Optional[int] = None, unlimited: bool = False, 
                 timeout_minutes: Optional[float] = None, unlimited_time: bool = False):
        self.max_combinations = max_combinations
        self.unlimited = unlimited
        self.timeout_minutes = timeout_minutes
        self.unlimited_time = unlimited_time
        
    def get_limit_for_model(self, model_name: str) -> Optional[int]:
        """Get the combination limit for a specific model"""
        if self.unlimited:
            return None
            
        if self.max_combinations is not None:
            return self.max_combinations
            
        # Use default limits based on model complexity
        complexity = self.MODEL_COMPLEXITY.get(model_name, 'medium')
        return self.DEFAULT_LIMITS[complexity]
    
    def get_timeout_for_model(self, model_name: str) -> Optional[float]:
        """Get the timeout limit for a specific model in seconds"""
        if self.unlimited_time:
            return None
            
        if self.timeout_minutes is not None:
            return self.timeout_minutes * 60
        
        # Check for model-specific overrides first
        if model_name in self.MODEL_SPECIFIC_TIMEOUTS:
            return self.MODEL_SPECIFIC_TIMEOUTS[model_name]
            
        # Use default timeouts based on model complexity
        complexity = self.MODEL_COMPLEXITY.get(model_name, 'medium')
        return self.DEFAULT_TIMEOUTS[complexity]


@dataclass
class SearchSpaceInfo:
    """Information about search space for a model"""
    model_name: str
    total_combinations: int
    limited_combinations: int
    search_type: str
    is_limited: bool
    timeout_seconds: Optional[float] = None
    
    @property
    def limitation_ratio(self) -> float:
        """Ratio of limited to total combinations"""
        if self.total_combinations == 0:
            return 1.0
        return self.limited_combinations / self.total_combinations
    
    @property
    def timeout_minutes(self) -> Optional[float]:
        """Timeout in minutes for display"""
        if self.timeout_seconds is None:
            return None
        return self.timeout_seconds / 60


class ComprehensiveHyperparameterTuner:
    """
    Comprehensive hyperparameter tuner for full space exploration with intelligent timeout management.
    
    This tuner provides comprehensive hyperparameter optimization for time series forecasting models,
    featuring intelligent timeout controls, background process management, and comprehensive parameter 
    space exploration based on official Darts documentation.
    """

    def test_all_models_viability(self, timeout_seconds: float = 30):
        """
        Test all enabled models individually for trainability using test_model_viability.
        Loads data once, iterates through all enabled models, runs viability test, and prints/logs results.
        """
        print("\n🔬 Testing trainability of all enabled models...\n")
        series = self.load_data()
        enabled_models = [name for name, cfg in self.config['models'].items() if cfg.get('enabled', False)]
        results = {}
        for model_name in enabled_models:
            model_config = self.config['models'][model_name]
            try:
                is_viable = self.test_model_viability(model_name, model_config, series, timeout_seconds=timeout_seconds)
                results[model_name] = is_viable
            except Exception as e:
                print(f"  ❌ {model_name} test raised exception: {e}")
                results[model_name] = False
        print("\nSummary of model trainability:")
        for model_name, status in results.items():
            print(f"  {'✅' if status else '❌'} {model_name}")
        print("\nDone.")

    def __init__(self, config_path: str = 'tuner_config.json'):
        """
        Initialize the comprehensive tuner with advanced timeout and training controls.
        Uses code/tuner/tuner_config.json by default.
        Args:
            config_path: Path to configuration file containing model settings
        """
        if not os.path.isabs(config_path):
            tuner_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(tuner_dir, config_path)
        self.config_path = os.path.abspath(config_path)
        
        # Set up failed models persistence file
        self.failed_models_file = os.path.join(os.path.dirname(self.config_path), 'failed_models.json')
        
        self.load_config()
        self.setup_logging()
        self.results: List[HyperparameterResult] = []
        self.best_configs: Dict[str, Dict] = {}
        self.failed_models: set = self.load_failed_models()  # Load persistent failed models
        
        # Define expanded hyperparameter search spaces for comprehensive exploration
        self.hyperparameter_grids = self._define_comprehensive_hyperparameter_grids()
        
    def _define_comprehensive_hyperparameter_grids(self) -> Dict[str, Dict]:
        """
        Define comprehensive hyperparameter grids for full space exploration.
        
        Based on official Darts documentation and extensive parameter research.
        Each model includes comprehensive parameter coverage with intelligent defaults.
        
        Returns:
            Dict mapping model names to their comprehensive hyperparameter grids
        """
        return {
            "Prophet": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],
                "hyperparameters": {
                    "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
                    "seasonality_prior_scale": [0.1, 1.0, 10.0],
                    "n_changepoints": [15, 25],
                    "seasonality_mode": ["additive"],
                    "growth": ["linear"],
                    "yearly_seasonality": [True, "auto"],
                    "weekly_seasonality": [False],
                    "daily_seasonality": [False],
                    "mcmc_samples": [0],
                    "interval_width": [0.80, 0.95]
                }
            },
            "XGBoost": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],  # Keep expanded splits
                "hyperparameters": {
                    # Core high-impact parameters optimized for 30-minute constraint
                    "lags": [12, 18, 24, 36, 48],  # Reduced from 14 to 5 key values
                    "n_estimators": [100, 200, 300, 500],  # Reduced from 13 to 4 key values
                    "max_depth": [4, 6, 8, 10],  # Reduced from 10 to 4 key values
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Reduced from 12 to 4 key values
                    "subsample": [0.8, 0.9, 1.0],  # Reduced from 8 to 3 key values
                    "colsample_bytree": [0.8, 0.9, 1.0],  # Reduced from 8 to 3 key values
                    "reg_alpha": [0, 0.1, 1.0],  # Reduced from 9 to 3 key values
                    "reg_lambda": [0, 0.1, 1.0]  # Reduced from 9 to 3 key values
                }
            },
            "LightGBM": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],  # Keep expanded splits
                "hyperparameters": {
                    # Core high-impact parameters optimized for 30-minute constraint
                    "lags": [12, 18, 24, 36, 48],  # Reduced from 14 to 5 key values
                    "n_estimators": [100, 200, 300, 500],  # Reduced from 13 to 4 key values
                    "max_depth": [4, 6, 8, 10],  # Reduced from 10 to 4 key values
                    "num_leaves": [31, 50, 100, 150],  # Reduced from 13 to 4 key values
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Reduced from 12 to 4 key values
                    "min_child_samples": [5, 10, 20],  # Reduced from 11 to 3 key values
                    "subsample": [0.8, 0.9, 1.0],  # Reduced from 8 to 3 key values
                    "colsample_bytree": [0.8, 0.9, 1.0],  # Reduced from 8 to 3 key values
                    "reg_alpha": [0, 0.1, 1.0],  # Reduced from 9 to 3 key values
                    "reg_lambda": [0, 0.1, 1.0]  # Reduced from 9 to 3 key values
                }
            },
            "CatBoost": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],  # Keep expanded splits
                "hyperparameters": {
                    # Core high-impact parameters optimized for 30-minute constraint
                    "lags": [12, 18, 24, 36, 48],  # Reduced from 14 to 5 key values
                    "iterations": [100, 200, 300, 500],  # Reduced from 13 to 4 key values
                    "depth": [4, 6, 8, 10],  # Reduced from 10 to 4 key values
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Reduced from 12 to 4 key values
                    "l2_leaf_reg": [1, 3, 10],  # Reduced from 12 to 3 key values
                    "border_count": [64, 128, 192]  # Reduced from 8 to 3 key values
                }
            },
            "RandomForest": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],  # Keep expanded splits
                "hyperparameters": {
                    # Core high-impact parameters
                    "lags": [12, 18, 24, 36, 48],  # Reduced from 14 to 5 key values
                    "n_estimators": [100, 200, 300, 500],  # Reduced from 13 to 4 key values
                    "max_depth": [5, 10, 15, None],  # Reduced from 10 to 4 key values
                    "min_samples_split": [2, 5, 10],  # Reduced from 9 to 3 key values
                    "min_samples_leaf": [1, 2, 5],  # Reduced from 9 to 3 key values
                    "max_features": ["sqrt", "log2", 0.5],  # Reduced from 7 to 3 key values
                    "bootstrap": [True],  # Fixed to True (standard practice)
                    "random_state": [42],  # Fixed for reproducibility
                    # Removed low-impact parameters for 30-minute constraint
                }
            },
            "LinearRegression": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],  # Simplified split ratios
                "hyperparameters": {
                    "lags": [6, 12, 18, 24, 30],  # Simplified lag values
                    "output_chunk_length": [1, 3, 6],  # Simplified output chunk lengths
                    "output_chunk_shift": [0, 1],  # Simplified shifts
                    "multi_models": [True, False],
                    "likelihood": [None, "quantile"],
                    "quantiles": [None, [0.1, 0.5, 0.9]],  # Simplified quantiles
                    "fit_intercept": [True, False],
                    "positive": [False, True],
                    "n_jobs": [1, -1]
                }
            },
            "AutoARIMA": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],
                "hyperparameters": {
                    "season_length": [12, 24],  # Use specific values instead of None
                    "quantiles": [None, [0.1, 0.5, 0.9]],  # Valid quantiles
                    "random_state": [None, 42]  # Valid random states
                    # Only valid AutoARIMA parameters based on Darts documentation
                }
            },
            "ExponentialSmoothing": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],
                "hyperparameters": {
                    "trend": [None, "add"],  # Removed "mul" to avoid convergence issues
                    "seasonal": [None, "add"],  # Removed "mul" to avoid convergence issues
                    "damped_trend": [True, False],
                    "seasonal_periods": [None, 12],  # Simplified - only monthly seasonality
                    "initialization_method": ["estimated", "heuristic"],  # Removed legacy method
                    "use_boxcox": [None, False],  # Removed True to avoid transformation issues
                    "missing": ["none"]  # Simplified - only default handling
                }
            },
            "TBATS": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],
                "hyperparameters": {
                    "season_length": [12, 24],  # Required parameter - must be integer, not None
                    "quantiles": [None, [0.1, 0.5, 0.9]],  # Valid quantiles
                    "random_state": [None, 42]  # Valid random states
                    # Only valid TBATS parameters based on Darts documentation
                }
            },
            "Theta": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],
                "hyperparameters": {
                    "season_mode": ["ADDITIVE", "MULTIPLICATIVE", "NONE"],
                    "theta": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8]
                }
            },
            "FourTheta": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],
                "hyperparameters": {
                    "season_mode": ["ADDITIVE", "MULTIPLICATIVE", "NONE"],
                    "theta": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8]
                }
            },
            "KalmanFilter": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],
                "hyperparameters": {
                    "dim_x": [2, 3, 4, 5, 6, 7, 8]
                }
            },
            "Croston": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],
                "hyperparameters": {
                    "version": ["classic", "optimized"],  # Valid versions
                    "alpha_d": [None, 0.1, 0.2, 0.3],  # Valid alpha_d values
                    "alpha_p": [None, 0.1, 0.2, 0.3],  # Valid alpha_p values
                    "quantiles": [None, [0.1, 0.5, 0.9]],  # Valid quantiles
                    "random_state": [None, 42]  # Valid random states
                    # Only valid Croston parameters: version, alpha_d, alpha_p, quantiles, random_state
                }
            },
            "NaiveDrift": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],
                "hyperparameters": {
                    # NaiveDrift is a simple baseline model with minimal hyperparameters
                    "random_state": [None, 42],
                    "quantiles": [None, [0.1, 0.5, 0.9]]
                }
            },
            "TCN": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],  # Keep expanded splits
                "hyperparameters": {
                    # Core architecture parameters optimized for 30-minute constraint
                    "input_chunk_length": [12, 18, 24],  # Reduced from 5 to 3 key values
                    "output_chunk_length": [1, 6],  # Reduced from 4 to 2 key values
                    "kernel_size": [3, 5],  # Reduced from 5 to 2 key values
                    "num_filters": [16, 32],  # Reduced from 4 to 2 key values
                    "dilation_base": [2],  # Reduced from 3 to 1 key value
                    "weight_norm": [True],  # Fixed to True (best practice)
                    "dropout": [0.1, 0.2],  # Reduced from 4 to 2 key values
                    "n_epochs": [50],  # Fixed to balanced value
                    "batch_size": [32],  # Fixed to standard value
                    "optimizer_kwargs": [{"lr": 0.001}, {"lr": 0.01}],  # Reduced from 3 to 2
                    "random_state": [42]
                }
            },
            "NBEATS": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],  # Keep expanded splits
                "hyperparameters": {
                    # Core architecture parameters optimized for 30-minute constraint
                    "input_chunk_length": [12, 18, 24],  # Reduced from 5 to 3 key values
                    "output_chunk_length": [1, 6],  # Reduced from 4 to 2 key values
                    "num_blocks": [2, 3],  # Reduced from 4 to 2 key values
                    "num_layers": [2, 4],  # Reduced from 3 to 2 key values
                    "layer_widths": [128, 256],  # Reduced from 3 to 2 key values
                    "expansion_coefficient_dim": [5, 10],  # Reduced from 3 to 2 key values
                    "trend_polynomial_degree": [2, 3],  # Reduced from 3 to 2 key values
                    "dropout": [0.1],  # Fixed to standard value
                    "activation": ["ReLU"],  # Fixed to standard value
                    "n_epochs": [50],  # Fixed to balanced value
                    "batch_size": [32],  # Fixed to standard value
                    "optimizer_kwargs": [{"lr": 0.001}],  # Fixed to best practice
                    "random_state": [42]
                }
            },
            "NHiTS": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],  # Keep expanded splits
                "hyperparameters": {
                    # Core architecture parameters optimized for 30-minute constraint
                    "input_chunk_length": [12, 18, 24],  # Reduced from 5 to 3 key values
                    "output_chunk_length": [1, 6],  # Reduced from 4 to 2 key values
                    "num_blocks": [2, 3],  # Reduced from 3 to 2 key values
                    "num_layers": [2, 4],  # Reduced from 3 to 2 key values
                    "layer_widths": [128, 256],  # Reduced from 3 to 2 key values
                    "pooling_kernel_sizes": [None, [2, 2, 2]],  # Reduced from 3 to 2 key values
                    "n_pool_kernel_sizes": [None, [2, 2, 2]],  # Reduced from 3 to 2 key values
                    "dropout": [0.1],  # Fixed to standard value
                    "activation": ["ReLU"],  # Fixed to standard value
                    "MaxPool1d": [True],  # Fixed to best practice
                    "n_epochs": [50],  # Fixed to balanced value
                    "batch_size": [32],  # Fixed to standard value
                    "optimizer_kwargs": [{"lr": 0.001}],  # Fixed to best practice
                    "random_state": [42]
                }
            },
            "TiDE": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],  # Keep expanded splits
                "hyperparameters": {
                    # Core architecture parameters optimized for 30-minute constraint
                    "input_chunk_length": [12, 18, 24],  # Reduced from 5 to 3 key values
                    "output_chunk_length": [1, 6],  # Reduced from 4 to 2 key values
                    "num_encoder_layers": [1, 2],  # Reduced from 3 to 2 key values
                    "num_decoder_layers": [1, 2],  # Reduced from 3 to 2 key values
                    "decoder_output_dim": [8, 16],  # Reduced from 3 to 2 key values
                    "hidden_size": [64, 128],  # Reduced from 3 to 2 key values
                    "temporal_width_past": [4, 8],  # Reduced from 3 to 2 key values
                    "temporal_width_future": [4, 8],  # Reduced from 3 to 2 key values
                    "temporal_decoder_hidden": [32, 64],  # Reduced from 3 to 2 key values
                    "dropout": [0.1],  # Fixed to standard value
                    "n_epochs": [50],  # Fixed to balanced value
                    "batch_size": [32],  # Fixed to standard value
                    "optimizer_kwargs": [{"lr": 0.001}],  # Fixed to best practice
                    "random_state": [42]
                }
            },
            "DLinear": {
                "split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],  # Keep expanded splits
                "hyperparameters": {
                    # Core architecture parameters optimized for 30-minute constraint
                    "input_chunk_length": [12, 18, 24],  # Reduced from 5 to 3 key values
                    "output_chunk_length": [1, 6],  # Reduced from 4 to 2 key values
                    "kernel_size": [13, 25],  # Reduced from 4 to 2 key values
                    "const_init": [True],  # Fixed to best practice
                    "use_static_covariates": [True],  # Fixed to best practice
                    "n_epochs": [50],  # Fixed to balanced value
                    "batch_size": [32],  # Fixed to standard value
                    "optimizer_kwargs": [{"lr": 0.001}],  # Fixed to best practice
                    "random_state": [42]
                }
            }
        }
        
    def load_config(self):
        """Load configuration from JSON file"""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Update file paths to be absolute
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for key, path in self.config['file_paths'].items():
            if not os.path.isabs(path):
                self.config['file_paths'][key] = os.path.join(project_root, path)
                
    def setup_logging(self):
        """Set up minimal logging to reduce output noise"""
        log_dir = os.path.join(os.path.dirname(self.config_path), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'comprehensive_tuner_{timestamp}.log')
        
        # Set up logging with minimal console output
        logging.basicConfig(
            level=logging.ERROR,  # Only show errors on console
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),  # Full logging to file
                logging.StreamHandler(sys.stdout)  # Minimal console output
            ]
        )
        
        # Configure specific loggers to be quiet
        logging.getLogger('prophet').setLevel(logging.CRITICAL)
        logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
        logging.getLogger('darts').setLevel(logging.CRITICAL)
        logging.getLogger('sklearn').setLevel(logging.CRITICAL)
        logging.getLogger('xgboost').setLevel(logging.CRITICAL)
        logging.getLogger('lightgbm').setLevel(logging.CRITICAL)
        logging.getLogger('catboost').setLevel(logging.CRITICAL)
        logging.getLogger('statsmodels').setLevel(logging.CRITICAL)
        logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
        logging.getLogger('numpy').setLevel(logging.CRITICAL)
        logging.getLogger('pandas').setLevel(logging.CRITICAL)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Comprehensive tuner initialized. Log file: {log_file}")
    
    def load_failed_models(self) -> set:
        """Load persistent failed models from file"""
        try:
            if os.path.exists(self.failed_models_file):
                with open(self.failed_models_file, 'r') as f:
                    failed_list = json.load(f)
                    return set(failed_list)
        except Exception as e:
            self.logger.warning(f"Could not load failed models: {e}")
        return set()
    
    def save_failed_models(self):
        """Save failed models to persistent file"""
        try:
            os.makedirs(os.path.dirname(self.failed_models_file), exist_ok=True)
            with open(self.failed_models_file, 'w') as f:
                json.dump(list(self.failed_models), f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save failed models: {e}")
        
    def load_data(self) -> TimeSeries:
        """Load and process CVE data"""
        monthly_counts = load_cve_data(self.config)
        
        series = TimeSeries.from_dataframe(
            monthly_counts,
            freq='M',
            fill_missing_dates=True,
            value_cols='cve_count'
        )
        
        return series
        
    def get_total_combinations(self, model_name: str) -> int:
        """Calculate total number of hyperparameter combinations"""
        if model_name not in self.hyperparameter_grids:
            return 0
            
        grid_config = self.hyperparameter_grids[model_name]
        split_ratios = grid_config["split_ratios"]
        hyperparams = grid_config["hyperparameters"]
        
        # Calculate total combinations
        total_param_combinations = 1
        for param_values in hyperparams.values():
            total_param_combinations *= len(param_values)
            
        total_combinations = len(split_ratios) * total_param_combinations
        return total_combinations
        
    def generate_parameter_combinations(self, model_name: str, search_type: str = "grid", 
                                      max_combinations: Optional[int] = None) -> List[Dict]:
        """Generate parameter combinations for comprehensive search"""
        
        if model_name not in self.hyperparameter_grids:
            return []
            
        grid_config = self.hyperparameter_grids[model_name]
        split_ratios = grid_config["split_ratios"]
        hyperparams = grid_config["hyperparameters"]
        
        # Special handling for LinearRegression quantiles parameter
        if model_name == "LinearRegression" and "quantiles" in hyperparams and "likelihood" in hyperparams:
            # Generate combinations with proper quantiles-likelihood pairing
            filtered_combinations = []
            
            # Get all other parameters except quantiles and likelihood
            other_params = {k: v for k, v in hyperparams.items() if k not in ["quantiles", "likelihood"]}
            
            # Generate base combinations for other parameters
            if other_params:
                if ParameterGrid is None:
                    # Fallback manual parameter grid generation
                    param_names = list(other_params.keys())
                    param_values = list(other_params.values())
                    base_combinations = []
                    for combination in product(*param_values):
                        param_combo = dict(zip(param_names, combination))
                        base_combinations.append(param_combo)
                else:
                    param_grid = ParameterGrid(other_params)
                    base_combinations = list(param_grid)
            else:
                base_combinations = [{}]
            
            # Add likelihood-quantiles combinations
            for base_combo in base_combinations:
                for likelihood in hyperparams["likelihood"]:
                    combo = base_combo.copy()
                    combo["likelihood"] = likelihood
                    
                    if likelihood == "quantile":
                        # Only add quantiles when likelihood is "quantile"
                        for quantiles in hyperparams["quantiles"]:
                            if quantiles is not None:  # Skip None quantiles for quantile likelihood
                                combo_with_quantiles = combo.copy()
                                combo_with_quantiles["quantiles"] = quantiles
                                filtered_combinations.append(combo_with_quantiles)
                    else:
                        # For None or "poisson" likelihood, don't use quantiles
                        combo["quantiles"] = None
                        filtered_combinations.append(combo)
            
            param_combinations = filtered_combinations
        else:
            # Standard parameter grid generation for other models
            if ParameterGrid is None:
                # Fallback manual parameter grid generation
                param_combinations = []
                param_names = list(hyperparams.keys())
                param_values = list(hyperparams.values())
                
                # Generate all combinations manually
                for combination in product(*param_values):
                    param_combo = dict(zip(param_names, combination))
                    param_combinations.append(param_combo)
            else:
                param_grid = ParameterGrid(hyperparams)
                param_combinations = list(param_grid)
        
        # Combine with split ratios
        full_combinations = []
        for split_ratio in split_ratios:
            for param_combo in param_combinations:
                full_combinations.append({
                    "split_ratio": split_ratio,
                    "hyperparameters": param_combo
                })
        
        total_combinations = len(full_combinations)
        
        # Apply search strategy
        if search_type == "random" and max_combinations and total_combinations > max_combinations:
            selected_combinations = random.sample(full_combinations, max_combinations)
        elif search_type == "grid" and max_combinations and total_combinations > max_combinations:
            selected_combinations = random.sample(full_combinations, max_combinations)
        else:
            selected_combinations = full_combinations
            
        return selected_combinations
        
    def test_model_viability(self, model_name: str, model_config: Dict, series: TimeSeries, timeout_seconds: float = 30) -> bool:
        """
        Test if a model can be trained successfully with a simple configuration.
        
        This viability test helps identify models that may have persistent issues
        before attempting comprehensive hyperparameter tuning. Uses a short timeout
        to quickly identify problematic models.
        
        Args:
            model_name: Name of the model to test
            model_config: Model configuration dictionary
            series: Time series data for training
            timeout_seconds: Timeout limit for the viability test
            
        Returns:
            bool: True if model can be trained successfully, False otherwise
        """
        print(f"  🧪 Testing {model_name} viability (timeout: {timeout_seconds}s)")
        
        # Use a simple configuration for testing
        test_config = self.config.copy()
        test_config['model_evaluation']['split_ratio'] = 0.8
        
        # Create a minimal test model config
        test_model_config = model_config.copy()
        
        # Use very conservative hyperparameters for testing
        if model_name == "XGBoost":
            test_model_config['hyperparameters'] = {
                'lags': 6,
                'n_estimators': 10,
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1
            }
        elif model_name == "LightGBM":
            test_model_config['hyperparameters'] = {
                'lags': 6,
                'n_estimators': 10,
                'max_depth': 3,
                'num_leaves': 10,
                'learning_rate': 0.1,
                'min_child_samples': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1
            }
        elif model_name == "CatBoost":
            test_model_config['hyperparameters'] = {
                'lags': 6,
                'iterations': 10,
                'depth': 3,
                'learning_rate': 0.1,
                'l2_leaf_reg': 1,
                'border_count': 32
            }
        elif model_name == "RandomForest":
            test_model_config['hyperparameters'] = {
                'lags': 6,
                'n_estimators': 10,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt'
            }
        elif model_name == "ExponentialSmoothing":
            test_model_config['hyperparameters'] = {
                'trend': None,
                'seasonal': None,
                'damped_trend': False,
                'seasonal_periods': None,
                'initialization_method': 'estimated',
                'use_boxcox': False,
                'missing': 'none'
            }
        elif model_name == "TBATS":
            test_model_config['hyperparameters'] = {
                'season_length': 12  # Must be integer, not None
            }
        elif model_name == "AutoARIMA":
            test_model_config['hyperparameters'] = {
                'season_length': 12  # Use specific value instead of None
            }
        elif model_name == "Croston":
            test_model_config['hyperparameters'] = {
                'version': 'classic'
                # Removed invalid parameters: alpha_d, alpha_p, alpha, seasonality, bootstrap, h
            }
        elif model_name == "LinearRegression":
            test_model_config['hyperparameters'] = {
                'lags': 12,
                'output_chunk_length': 1,
                'output_chunk_shift': 0,
                'multi_models': True,
                'likelihood': None,
                'quantiles': None,
                'fit_intercept': True,
                'positive': False,
                'n_jobs': 1
            }
        
        try:
            # Create null device to suppress all output
            import os
            devnull = open(os.devnull, 'w')
            
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                # Temporarily disable all logging
                logging.disable(logging.CRITICAL)
                
                # Use threaded model runner with short timeout
                runner = ThreadedModelRunner()
                model, train, val, predictions = runner.run_with_timeout(
                    model_name, test_model_config, series, test_config['model_evaluation'], timeout_seconds
                )
            
            # Re-enable logging
            logging.disable(logging.NOTSET)
            devnull.close()
            
            if model is not None:
                print(f"  ✅ {model_name} is viable")
                return True
            else:
                print(f"  ❌ {model_name} failed viability test (timeout or error)")
                return False
                
        except Exception as e:
            import traceback
            print(f"  ❌ {model_name} failed viability test: {str(e)}")
            if model_name in ["AutoETS", "NaiveEnsemble"]:
                print(f"  🔍 Full error details for {model_name}:")
                traceback.print_exc()
            return False
            
    def tune_model_comprehensive(self, model_name: str, model_config: Dict, series: TimeSeries,
                               search_type: str = "grid", max_combinations: Optional[int] = None,
                               timeout_seconds: Optional[float] = None) -> List[HyperparameterResult]:
        """
        Perform comprehensive hyperparameter tuning for a model with intelligent timeout management.
        
        This method executes comprehensive parameter space exploration with advanced timeout controls,
        background process management, and comprehensive error handling. Includes automatic model
        viability testing and intelligent early stopping mechanisms.
        
        Args:
            model_name: Name of the model to tune
            model_config: Model configuration dictionary
            series: Time series data for training
            search_type: Type of search ("grid" or "random")
            max_combinations: Maximum number of parameter combinations to test
            timeout_seconds: Overall timeout for the model tuning process
            
        Returns:
            List of HyperparameterResult objects containing tuning results
        """
        
        # Check if this model is disabled due to hanging issues
        if model_name in SearchSpaceConfig.DISABLED_MODELS:
            print(f"  ⚠️  {model_name} is disabled due to hanging issues - skipping")
            return []
        
        # Check if this model is in the failed models list
        if model_name in self.failed_models:
            print(f"  ❌ {model_name} is in failed models list - skipping")
            return []
        
        # First, test if the model is viable (increased timeout for complex models)
        viability_timeout = 60 if model_name in ["TBATS", "AutoARIMA", "ExponentialSmoothing", "Croston"] else 30
        if not self.test_model_viability(model_name, model_config, series, timeout_seconds=viability_timeout):
            print(f"  ❌ {model_name} failed viability test - adding to failed models list")
            self.failed_models.add(model_name)
            self.save_failed_models()  # Persist the failed model
            return []
        
        # Generate parameter combinations
        combinations = self.generate_parameter_combinations(model_name, search_type, max_combinations)
        
        if not combinations:
            return []
        
        results = []
        total_combinations = len(combinations)
        
        print(f"📊 {model_name}: Testing {total_combinations:,} combinations")
        if timeout_seconds:
            print(f"⏱️  Timeout: {timeout_seconds}s ({timeout_seconds/60:.1f} minutes)")
        
        # Set up timeout for the entire model tuning process
        model_start_time = time.time()
        timeout_triggered = False
        
        # Progress tracking
        progress_interval = max(1, total_combinations // 20)  # Show progress every 5%
        consecutive_failures = 0
        max_consecutive_failures = 50 if model_name in ["LinearRegression", "AutoARIMA", "ExponentialSmoothing", "Prophet"] else 25  # Increased for problematic models
        
        for trial_num, combo in enumerate(combinations, 1):
            # Check if we've exceeded the timeout for this model (check more frequently)
            elapsed_model_time = time.time() - model_start_time
            if timeout_seconds and elapsed_model_time >= timeout_seconds:
                timeout_triggered = True
                print(f"  ⏱️ {model_name} timed out after {elapsed_model_time:.1f}s - stopping trials")
                break
            
            # Additional aggressive timeout check - if we're close to timeout, stop
            if timeout_seconds and elapsed_model_time >= (timeout_seconds * 0.95):
                timeout_triggered = True
                print(f"  ⏱️ {model_name} approaching timeout ({elapsed_model_time:.1f}s/{timeout_seconds:.1f}s) - stopping trials")
                break
                
            # Show progress
            if trial_num % progress_interval == 0 or trial_num == total_combinations:
                progress = (trial_num / total_combinations) * 100
                elapsed = time.time() - model_start_time
                if timeout_seconds:
                    remaining = timeout_seconds - elapsed
                    if remaining <= 0:
                        timeout_triggered = True
                        print(f"  ⏱️ {model_name} timed out after {elapsed:.1f}s - stopping trials")
                        break
                    print(f"  ⚡ Progress: {progress:.1f}% ({trial_num:,}/{total_combinations:,}) - {remaining:.0f}s remaining", end='\r', flush=True)
                else:
                    print(f"  ⚡ Progress: {progress:.1f}% ({trial_num:,}/{total_combinations:,}) - {elapsed:.0f}s elapsed", end='\r', flush=True)
            
            # Final timeout check before starting trial
            elapsed_model_time = time.time() - model_start_time
            if timeout_seconds and elapsed_model_time >= timeout_seconds:
                timeout_triggered = True
                print(f"  ⏱️ {model_name} timed out before trial {trial_num} - stopping")
                break
            
            split_ratio = combo["split_ratio"]
            hyperparams = combo["hyperparameters"]
            
            # Create temporary config
            temp_config = self.config.copy()
            temp_config['model_evaluation']['split_ratio'] = split_ratio
            
            # Merge hyperparameters
            temp_model_config = model_config.copy()
            temp_model_config['hyperparameters'].update(hyperparams)
            
            start_time = time.time()
            
            try:
                # Calculate remaining time for this trial
                if timeout_seconds:
                    elapsed_model_time = time.time() - model_start_time
                    remaining_model_time = timeout_seconds - elapsed_model_time
                    
                    # If we have very little time left, skip this trial
                    if remaining_model_time <= 10:  # Less than 10 seconds remaining
                        timeout_triggered = True
                        print(f"  ⏱️ {model_name} timed out - insufficient time for trial {trial_num}")
                        break
                    
                    # Set generous per-trial timeout for different model types
                    if model_name in ["XGBoost", "LightGBM", "CatBoost"]:
                        trial_timeout = min(remaining_model_time, 60)  # 1 minute max for tree models
                    elif model_name in ["Prophet", "AutoARIMA", "TBATS"]:
                        trial_timeout = min(remaining_model_time, 120)  # 2 minutes for complex statistical models
                    else:
                        trial_timeout = min(remaining_model_time, 90)  # 1.5 minutes for other models
                else:
                    # Default timeouts based on model complexity
                    if model_name in ["XGBoost", "LightGBM", "CatBoost"]:
                        trial_timeout = 60  # 1 minute for tree models
                    elif model_name in ["Prophet", "AutoARIMA", "TBATS"]:
                        trial_timeout = 120  # 2 minutes for complex statistical models
                    else:
                        trial_timeout = 90  # 1.5 minutes for other models
                
                print(f"  🔄 Trial {trial_num}: timeout={trial_timeout:.0f}s", end='\r', flush=True)
                
                # Final timeout check before starting the actual trial
                if timeout_seconds and (time.time() - model_start_time) >= timeout_seconds:
                    timeout_triggered = True
                    print(f"  ⏱️ {model_name} timed out just before trial {trial_num} execution")
                    break
                
                # Suppress all output for clean experience
                import io
                import contextlib
                
                # Create null device to suppress all output
                devnull = open(os.devnull, 'w')
                
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    # Temporarily disable all logging
                    logging.disable(logging.CRITICAL)
                    
                    # Use threaded model runner with timeout
                    runner = ThreadedModelRunner()
                    model, train, val, predictions = runner.run_with_timeout(
                        model_name, temp_model_config, series, temp_config['model_evaluation'], trial_timeout
                    )
                
                # Re-enable logging
                logging.disable(logging.NOTSET)
                devnull.close()
                
                training_time = time.time() - start_time
                
                if model and predictions:
                    # Calculate comprehensive metrics for model performance evaluation
                    metrics = {}
                    metric_funcs = [('mape', mape), ('mae', mae), ('mase', mase), ('rmsse', rmsse)]
                    
                    for metric_name, metric_func in metric_funcs:
                        try:
                            if metric_name in ['mase', 'rmsse']:
                                metrics[metric_name] = metric_func(val, predictions, train)
                            else:
                                metrics[metric_name] = metric_func(val, predictions)
                        except Exception:
                            metrics[metric_name] = float('inf')
                    
                    result = HyperparameterResult(
                        model_name=model_name,
                        split_ratio=split_ratio,
                        hyperparameters=hyperparams,
                        mape=metrics.get('mape', float('inf')),
                        mae=metrics.get('mae', float('inf')),
                        mase=metrics.get('mase', float('inf')),
                        rmsse=metrics.get('rmsse', float('inf')),
                        training_time=training_time,
                        success=True,
                        trial_number=trial_num
                    )
                    
                    consecutive_failures = 0  # Reset failure counter on success
                    
                else:
                    consecutive_failures += 1
                    result = HyperparameterResult(
                        model_name=model_name,
                        split_ratio=split_ratio,
                        hyperparameters=hyperparams,
                        mape=float('inf'),
                        mae=float('inf'),
                        mase=float('inf'),
                        rmsse=float('inf'),
                        training_time=time.time() - start_time,
                        success=False,
                        error_message="Model training failed",
                        trial_number=trial_num
                    )
                    
                    # Check for consecutive failures
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"\r  ❌ {model_name} has {consecutive_failures} consecutive failures - stopping early" + " " * 20)
                        self.failed_models.add(model_name)
                        self.save_failed_models()  # Persist the failed model
                        break
                    
            except Exception as e:
                # Re-enable logging in case of error
                logging.disable(logging.NOTSET)
                consecutive_failures += 1
                training_time = time.time() - start_time
                result = HyperparameterResult(
                    model_name=model_name,
                    split_ratio=split_ratio,
                    hyperparameters=hyperparams,
                    mape=float('inf'),
                    mae=float('inf'),
                    mase=float('inf'),
                    rmsse=float('inf'),
                    training_time=training_time,
                    success=False,
                    error_message=str(e),
                    trial_number=trial_num
                )
                
                # Check for consecutive failures
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\r  ❌ {model_name} has {consecutive_failures} consecutive failures - stopping early" + " " * 20)
                    self.failed_models.add(model_name)
                    self.save_failed_models()  # Persist the failed model
                    break
                
            results.append(result)
            
            # Check timeout immediately after each trial
            if timeout_seconds and (time.time() - model_start_time) >= timeout_seconds:
                timeout_triggered = True
                print(f"  ⏱️ {model_name} timed out after trial {trial_num} completed")
                break
            
        # Add timeout summary if applicable
        if timeout_triggered:
            elapsed_time = time.time() - model_start_time
            print(f"  ⏱️ {model_name} stopped due to timeout after {elapsed_time:.1f}s")
            print(f"  📊 Completed {len(results)}/{total_combinations} trials ({len(results)/total_combinations*100:.1f}%)")
            
        return results
        
    def clear_failed_models(self, model_names: Optional[List[str]] = None):
        """
        Clear the failed models list to allow retrying models that previously failed.
        
        Args:
            model_names: Optional list of specific model names to remove from failed list.
                        If None, clears all failed models.
        """
        if model_names is None:
            # Clear all failed models
            cleared_models = list(self.failed_models)
            self.failed_models.clear()
            self.save_failed_models()  # Persist the cleared state
            if cleared_models:
                print(f"🔄 Cleared all failed models: {', '.join(cleared_models)}")
            else:
                print("ℹ️  No failed models to clear")
        else:
            # Clear specific models
            cleared_models = []
            for model_name in model_names:
                if model_name in self.failed_models:
                    self.failed_models.remove(model_name)
                    cleared_models.append(model_name)
            
            if cleared_models:
                self.save_failed_models()  # Persist the changes
                print(f"🔄 Cleared failed models: {', '.join(cleared_models)}")
            else:
                print("ℹ️  None of the specified models were in the failed list")
        
        # Show remaining failed models if any
        if self.failed_models:
            print(f"⚠️  Still failed: {', '.join(self.failed_models)}")
        else:
            print("✅ No models currently in failed list")
    
    def show_failed_models(self):
        """Display the current list of failed models"""
        if self.failed_models:
            print(f"❌ Failed models: {', '.join(sorted(self.failed_models))}")
        else:
            print("✅ No models currently in failed list")
        
    def find_best_configurations(self, results: List[HyperparameterResult], top_n: int = 5) -> List[HyperparameterResult]:
        """Find the top N best configurations from results"""
        successful_results = [r for r in results if r.success and not np.isinf(r.mape)]
        
        if not successful_results:
            return []
            
        # Sort by MAPE (lower is better) and return top N
        best_results = sorted(successful_results, key=lambda x: x.mape)[:top_n]
        return best_results
        
    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization"""
        import numpy as np
        
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle PyTorch tensors if present
        elif hasattr(obj, 'detach') and hasattr(obj, 'cpu'):
            return float(obj.detach().cpu().numpy())
        # Handle generic numeric types that might not be caught above
        elif hasattr(obj, 'item'):
            return obj.item()
        # Handle Python numeric types that might still cause issues
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if 'float' in str(type(obj)) else int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            # Final fallback - try to convert to basic Python type
            try:
                if isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                elif hasattr(obj, '__float__'):
                    return float(obj)
                elif hasattr(obj, '__int__'):
                    return int(obj)
                else:
                    return str(obj)
            except:
                return str(obj)

    def save_results(self, filename: Optional[str] = None):
        """Save detailed results to file with proper numpy type conversion"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comprehensive_tuning_results_{timestamp}.json"
        
        # Create results directory - fix nested tuner folder issue
        results_dir = os.path.join(os.path.dirname(self.config_path), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, filename)
        
        # Convert results to serializable format with numpy type conversion
        results_data = []
        for result in self.results:
            result_dict = {
                'model_name': result.model_name,
                'split_ratio': self._convert_numpy_types(result.split_ratio),
                'hyperparameters': self._convert_numpy_types(result.hyperparameters),
                'mape': self._convert_numpy_types(result.mape),
                'mae': self._convert_numpy_types(result.mae),
                'mase': self._convert_numpy_types(result.mase),
                'rmsse': self._convert_numpy_types(result.rmsse),
                'training_time': self._convert_numpy_types(result.training_time),
                'success': result.success,
                'error_message': result.error_message,
                'trial_number': result.trial_number
            }
            results_data.append(result_dict)
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        return results_file
        
    def update_main_config(self):
        """Update the main production config.json with optimized configurations only if better results are found"""
        if not self.best_configs:
            print("⚠️  No best configurations found. Config file not updated.")
            return
        
        # Load the MAIN production config.json (not tuner config) for comparison and update
        main_config_path = os.path.join(os.path.dirname(os.path.dirname(self.config_path)), 'config.json')
        print(f"🔍 Comparing against main production config: {main_config_path}")
        
        try:
            with open(main_config_path, 'r') as f:
                main_config = json.load(f)
        except FileNotFoundError:
            print(f"❌ Main config file not found: {main_config_path}")
            return
        
        current_config = main_config  # Use main config for comparison
        
        # Check if we have better results than existing ones
        updates_made = False
        models_improved = []
        models_unchanged = []
        
        for model_name, best_config in self.best_configs.items():
            should_update = False
            
            if model_name in current_config['models']:
                # Check if this model has previous tuning results
                existing_results = current_config['models'][model_name].get('tuning_results', {})
                existing_mape = existing_results.get('mape', float('inf'))
                
                # Only update if we found better results
                if best_config['mape'] < existing_mape:
                    should_update = True
                    improvement = existing_mape - best_config['mape']
                    models_improved.append((model_name, best_config['mape'], existing_mape, improvement))
                else:
                    models_unchanged.append((model_name, best_config['mape'], existing_mape))
            else:
                # New model, always update
                should_update = True
                models_improved.append((model_name, best_config['mape'], float('inf'), float('inf')))
            
            if should_update:
                updates_made = True
        
        # Print comparison results
        if models_improved:
            print("🚀 IMPROVEMENTS FOUND:")
            print("-" * 80)
            print(f"{'Model':<15} {'New MAPE':<10} {'Old MAPE':<10} {'Improvement':<12}")
            print("-" * 80)
            for model_name, new_mape, old_mape, improvement in models_improved:
                if old_mape == float('inf'):
                    print(f"{model_name:<15} {new_mape:<10.3f} {'N/A':<10} {'New model':<12}")
                else:
                    print(f"{model_name:<15} {new_mape:<10.3f} {old_mape:<10.3f} {improvement:<12.3f}")
            print("-" * 80)
        
        if models_unchanged:
            print("📊 NO IMPROVEMENT:")
            print("-" * 60)
            print(f"{'Model':<15} {'New MAPE':<10} {'Current MAPE':<12}")
            print("-" * 60)
            for model_name, new_mape, current_mape in models_unchanged:
                print(f"{model_name:<15} {new_mape:<10.3f} {current_mape:<12.3f}")
            print("-" * 60)
        
        # Only proceed with update if we have improvements
        if not updates_made:
            print("⚠️  No improvements found. Config file not updated.")
            print("💡 Current configurations are already optimal or better.")
            return
        
        # Create a backup of the original MAIN config in a dedicated backup folder
        backup_dir = os.path.join(os.path.dirname(self.config_path), 'tuner', 'config_backups')
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"main_config_backup_{timestamp}.json"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Backup the MAIN production config, not the tuner config
        with open(main_config_path, 'r') as f:
            original_config = f.read()
        with open(backup_path, 'w') as f:
            f.write(original_config)
        print(f"💾 Main production config backed up to: {backup_path}")
        
        # Update the config with model-specific optimal configurations (only improved ones)
        updated_config = current_config.copy()
        
        # Add model-specific optimal configurations to each model (only if improved)
        for model_name, best_config in self.best_configs.items():
            if model_name in updated_config['models']:
                # Check if this model should be updated
                existing_results = updated_config['models'][model_name].get('tuning_results', {})
                existing_mape = existing_results.get('mape', float('inf'))
                
                if best_config['mape'] < existing_mape:
                    # Update split ratio
                    updated_config['models'][model_name]['optimal_split_ratio'] = best_config['split_ratio']
                    
                    # Update hyperparameters with optimal values
                    updated_config['models'][model_name]['hyperparameters'].update(best_config['hyperparameters'])
                    
                    # Add tuning results metadata
                    updated_config['models'][model_name]['tuning_results'] = {
                        'mape': best_config['mape'],
                        'mae': best_config['mae'],
                        'mase': best_config.get('mase', 0),
                        'rmsse': best_config.get('rmsse', 0),
                        'training_time': best_config['training_time'],
                        'trial_number': best_config['trial_number'],
                        'tuned_at': datetime.now().isoformat(),
                        'tuning_method': 'comprehensive_hyperparameter_tuning',
                        'previous_mape': existing_mape if existing_mape != float('inf') else None,
                        'improvement': existing_mape - best_config['mape'] if existing_mape != float('inf') else None
                    }
                
        # Update the global split ratio to the best performing model's split ratio (only if improved)
        best_model_name = None
        best_split_ratio = None
        
        if self.best_configs:
            # Find the best overall model from the improved models only
            improved_models = {name: config for name, config in self.best_configs.items() 
                             if any(name == m[0] for m in models_improved)}
            
            if improved_models:
                best_model_name = min(improved_models.items(), key=lambda x: x[1]['mape'])[0]
                best_split_ratio = improved_models[best_model_name]['split_ratio']
                
                # Check if global split ratio should be updated
                current_global_mape = float('inf')
                if 'comprehensive_tuning' in updated_config:
                    current_global_mape = updated_config['comprehensive_tuning'].get('best_mape', float('inf'))
                
                if improved_models[best_model_name]['mape'] < current_global_mape:
                    updated_config['model_evaluation']['split_ratio'] = best_split_ratio
                    
                    # Add comprehensive tuning metadata
                    updated_config['comprehensive_tuning'] = {
                        'last_tuned_at': datetime.now().isoformat(),
                        'best_model': best_model_name,
                        'best_mape': improved_models[best_model_name]['mape'],
                        'optimal_split_ratio': best_split_ratio,
                        'models_tuned': list(improved_models.keys()),
                        'previous_best_mape': current_global_mape if current_global_mape != float('inf') else None,
                        'global_improvement': current_global_mape - improved_models[best_model_name]['mape'] if current_global_mape != float('inf') else None
                    }
                
        # Save updated config to the MAIN production config.json
        # Convert all numpy/float32 types before saving to prevent JSON serialization errors
        config_to_save = self._convert_numpy_types(updated_config)
        with open(main_config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
            
        print(f"✅ Main production config.json updated with {len(models_improved)} improved configurations")
        print(f"🎯 Updated file: {main_config_path}")
        if best_split_ratio is not None and best_model_name is not None:
            print(f"🎯 Global split ratio updated to: {best_split_ratio:.3f}")
            print(f"🏆 Best model: {best_model_name} (MAPE: {self.best_configs[best_model_name]['mape']:.3f}%)")
        return updated_config
    
    def update_tuner_config(self):
        """Update the tuner's own config file with best found configurations for self-improvement"""
        if not self.best_configs:
            print("⚠️  No best configurations found. Tuner config not updated.")
            return
        
        print(f"🔧 Updating tuner config: {self.config_path}")
        
        # Create backup of tuner config
        backup_dir = os.path.join(os.path.dirname(self.config_path), 'config_backups')
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"tuner_config_backup_{timestamp}.json"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        with open(self.config_path, 'r') as f:
            original_config = f.read()
        with open(backup_path, 'w') as f:
            f.write(original_config)
        print(f"💾 Tuner config backed up to: {backup_path}")
        
        # Load current tuner config
        updated_tuner_config = self.config.copy()
        
        updates_made = False
        models_improved = []
        
        # Update each model with better configurations
        for model_name, best_config in self.best_configs.items():
            if model_name in updated_tuner_config['models']:
                # Check if this is an improvement
                current_hyperparams = updated_tuner_config['models'][model_name].get('hyperparameters', {})
                current_results = updated_tuner_config['models'][model_name].get('best_results', {})
                current_mape = current_results.get('mape', float('inf'))
                
                if best_config['mape'] < current_mape:
                    # Update hyperparameters with best found values
                    updated_tuner_config['models'][model_name]['hyperparameters'].update(best_config['hyperparameters'])
                    
                    # Store best results for future comparison
                    updated_tuner_config['models'][model_name]['best_results'] = {
                        'mape': best_config['mape'],
                        'mae': best_config['mae'],
                        'mase': best_config.get('mase', 0),
                        'rmsse': best_config.get('rmsse', 0),
                        'split_ratio': best_config['split_ratio'],
                        'training_time': best_config['training_time'],
                        'found_at': datetime.now().isoformat()
                    }
                    
                    updates_made = True
                    improvement = current_mape - best_config['mape']
                    models_improved.append((model_name, best_config['mape'], current_mape, improvement))
        
        if updates_made:
            # Update global tuner settings with best overall results
            if self.best_configs:
                best_model_name = min(self.best_configs.items(), key=lambda x: x[1]['mape'])[0]
                best_split_ratio = self.best_configs[best_model_name]['split_ratio']
                
                # Update default split ratio if significantly better
                current_split = updated_tuner_config['model_evaluation'].get('split_ratio', 0.75)
                updated_tuner_config['model_evaluation']['split_ratio'] = best_split_ratio
                
                # Add tuning history metadata
                if 'tuning_history' not in updated_tuner_config:
                    updated_tuner_config['tuning_history'] = []
                    
                updated_tuner_config['tuning_history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'best_model': best_model_name,
                    'best_mape': self.best_configs[best_model_name]['mape'],
                    'models_improved': len(models_improved),
                    'total_models_tested': len(self.best_configs)
                })
            
            # Save updated tuner config
            # Convert all numpy/float32 types before saving to prevent JSON serialization errors
            config_to_save = self._convert_numpy_types(updated_tuner_config)
            with open(self.config_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            
            print("🚀 TUNER CONFIG IMPROVEMENTS:")
            print("-" * 70)
            print(f"{'Model':<15} {'New MAPE':<10} {'Old MAPE':<10} {'Improvement':<12}")
            print("-" * 70)
            for model_name, new_mape, old_mape, improvement in models_improved:
                if old_mape == float('inf'):
                    print(f"{model_name:<15} {new_mape:<10.3f} {'N/A':<10} {'New model':<12}")
                else:
                    print(f"{model_name:<15} {new_mape:<10.3f} {old_mape:<10.3f} {improvement:<12.3f}")
            print("-" * 70)
            print(f"✅ Tuner config updated with {len(models_improved)} improved configurations")
            print(f"🎯 Updated file: {self.config_path}")
        else:
            print("⚠️  No improvements found for tuner config.")
            print("💡 Current tuner configurations are already optimal or better.")
        
        return updated_tuner_config if updates_made else None
        
    def show_search_space_overview(self, target_models: Optional[List[str]] = None, 
                                  search_config: Optional[SearchSpaceConfig] = None, 
                                  search_type: str = "random"):
        """Display comprehensive search space overview with intelligent limits"""
        
        if search_config is None:
            search_config = SearchSpaceConfig()
        
        # Get active models (excluding known problematic/disabled models)
        active_models = {
            name: config for name, config in self.config['models'].items()
            if config.get('enabled', False) and name not in SearchSpaceConfig.DISABLED_MODELS
        }
        
        # Filter to target models if specified
        if target_models:
            active_models = {k: v for k, v in active_models.items() if k in target_models}
        
        # Calculate search space information
        search_spaces = []
        total_all_combinations = 0
        total_limited_combinations = 0
        
        for model_name in active_models:
            total_combinations = self.get_total_combinations(model_name)
            limit = search_config.get_limit_for_model(model_name)
            timeout_seconds = search_config.get_timeout_for_model(model_name)
            limited_combinations = min(total_combinations, limit) if limit else total_combinations
            
            is_limited = limit is not None and total_combinations > limit
            
            search_space = SearchSpaceInfo(
                model_name=model_name,
                total_combinations=total_combinations,
                limited_combinations=limited_combinations,
                search_type=search_type,
                is_limited=is_limited,
                timeout_seconds=timeout_seconds
            )
            search_spaces.append(search_space)
            total_all_combinations += total_combinations
            total_limited_combinations += limited_combinations
        
        # Display formatted overview
        print("\n📊 Comprehensive Search Space Overview:")
        print("-" * 95)
        print(f"{'Model':<20} {'Total':<12} {'Limited':<12} {'Type':<8} {'Timeout':<10} {'Status':<15}")
        print("-" * 95)
        
        for space in search_spaces:
            status = "LIMITED" if space.is_limited else "FULL"
            if space.is_limited:
                status += f" ({space.limitation_ratio:.1%})"
            
            # Format timeout display
            timeout_str = "UNLIMITED" if space.timeout_seconds is None else f"{space.timeout_minutes:.0f}min"
            
            print(f"{space.model_name:<20} {space.total_combinations:<12,} "
                  f"{space.limited_combinations:<12,} {space.search_type:<8} {timeout_str:<10} {status:<15}")
        
        print("-" * 95)
        print(f"{'TOTAL':<20} {total_all_combinations:<12,} {total_limited_combinations:<12,}")
        
        # Show summary
        limited_count = sum(1 for space in search_spaces if space.is_limited)
        timeout_count = sum(1 for space in search_spaces if space.timeout_seconds is not None)
        
        if limited_count > 0:
            print(f"\n🎯 {limited_count} models are limited for practical runtime")
            print("💡 Use --unlimited to explore full space (may take very long)")
            print(f"🎲 Random search is recommended for limited space exploration")
        else:
            print("\n🔓 All models will use full search space")
            print("📊 Grid search can be used for complete exploration")
        
        if timeout_count > 0:
            print(f"⏱️  {timeout_count} models have time limits")
            print("💡 Use --unlimited-time to remove all time limits")
        
        if search_config.unlimited:
            print("\n⚠️  UNLIMITED MODE: This may take VERY long time!")
            
        if search_config.unlimited_time:
            print("\n⚠️  UNLIMITED TIME: Models may run indefinitely!")
        
        print()
        
    def run_comprehensive_tuning(self, target_models: Optional[List[str]] = None, 
                               search_type: str = "grid", search_config: Optional[SearchSpaceConfig] = None):
        """
        Run comprehensive hyperparameter tuning with full space exploration.
        
        This is the main entry point for comprehensive tuning, featuring intelligent
        model ordering, timeout management, and comprehensive result tracking.
        Processes models in order of complexity for optimal training efficiency.
        
        Args:
            target_models: List of specific models to tune (None for all enabled models)
            search_type: Type of search to perform ("grid" or "random")
            search_config: Configuration for search space limits and timeouts
        """
        
        if search_config is None:
            search_config = SearchSpaceConfig()
        
        print("🔬 Starting comprehensive hyperparameter space exploration...")
        
        # Load data for training
        series = self.load_data()
        print(f"📈 Loaded {len(series)} months of CVE data")
        
        # Show search space overview
        self.show_search_space_overview(target_models, search_config, search_type)
        
        # Get active models
        active_models = {
            name: config for name, config in self.config['models'].items() 
            if config.get('enabled', False) and name not in SearchSpaceConfig.DISABLED_MODELS
        }
        
        # Filter to target models if specified
        if target_models:
            active_models = {k: v for k, v in active_models.items() if k in target_models}
        
        # Show comprehensive search space information
        print("\n📊 Comprehensive Search Space Overview:")
        print("-" * 70)
        print(f"{'Model':<20} {'Total Combinations':<20} {'Search Type':<15}")
        print("-" * 70)
        
        total_all_combinations = 0
        for model_name in active_models:
            total_combinations = self.get_total_combinations(model_name)
            total_all_combinations += total_combinations
            limit = search_config.get_limit_for_model(model_name)
            actual_combinations = min(total_combinations, limit) if limit else total_combinations
            print(f"{model_name:<20} {total_combinations:<20,} {search_type:<15}")
        
        print("-" * 70)
        print(f"{'TOTAL':<20} {total_all_combinations:<20,}")
        print()
        
        # Sort models by search space size (smallest first for faster feedback)
        print("\n🔄 Sorting models by search space size (smallest first for optimal user experience)...")
        model_sizes = []
        for model_name, config in active_models.items():
            total_combinations = self.get_total_combinations(model_name)
            limit = search_config.get_limit_for_model(model_name)
            # Use actual search space (limited if applicable)
            actual_combinations = min(total_combinations, limit) if limit else total_combinations
            model_sizes.append((model_name, config, actual_combinations, total_combinations))
        
        # Sort by actual search space size (smallest first)
        model_sizes.sort(key=lambda x: x[2])  # Sort by actual_combinations
        sorted_models = [(name, config) for name, config, _, _ in model_sizes]
        
        # Print the execution order for user reference
        print("\n📋 Model Execution Order (by search space size):")
        print("-" * 65)
        print(f"{'#':<3} {'Model':<20} {'Search Space':<15} {'Status':<15}")
        print("-" * 65)
        for i, (name, _, actual, total) in enumerate(model_sizes, 1):
            status = "FULL" if actual == total else f"LIMITED ({100*actual/total:.1f}%)"
            print(f"{i:<3} {name:<20} {actual:<15,} {status:<15}")
        print("-" * 65)
        print()
        
        # Initialize dynamic timeout manager if timeout_minutes is specified
        dynamic_timeout_manager = None
        if search_config.timeout_minutes is not None:
            active_model_names = [name for name, _ in sorted_models]
            dynamic_timeout_manager = DynamicTimeoutManager(
                total_budget_minutes=search_config.timeout_minutes,
                active_models=active_model_names
            )
            print(f"🔄 Dynamic timeout enabled: {search_config.timeout_minutes:.1f} minutes total budget across {len(active_model_names)} models")
        
        # Tune each model with comprehensive parameter exploration
        all_results = []
        total_start_time = time.time()
        
        for i, (model_name, model_config) in enumerate(sorted_models):
            model_start_time = time.time()
            print(f"\n{'='*60}")
            print(f"🎯 Model {i+1}/{len(sorted_models)}: {model_name}")
            print(f"{'='*60}")
            
            # Get intelligent limits and timeout settings for this model
            limit = search_config.get_limit_for_model(model_name)
            
            # Use dynamic timeout if enabled, otherwise use static timeout
            if dynamic_timeout_manager:
                timeout_seconds = dynamic_timeout_manager.get_current_timeout(model_name)
                if dynamic_timeout_manager.is_budget_exhausted():
                    print(f"⏰ Time budget exhausted, skipping remaining models")
                    break
                print(f"🕐 Dynamic timeout for {model_name}: {timeout_seconds/60:.1f} minutes")
            else:
                timeout_seconds = search_config.get_timeout_for_model(model_name)
            
            # Perform comprehensive model tuning with better error handling
            try:
                # Force cleanup before starting model
                cleanup_multiprocessing()
                
                model_results = self.tune_model_comprehensive(
                    model_name, model_config, series, search_type, limit, timeout_seconds
                )
                
                all_results.extend(model_results)
                
                # Force cleanup after model completion
                cleanup_multiprocessing()
                
            except Exception as e:
                print(f"❌ Error during tuning: {e}")
                cleanup_multiprocessing()
                model_results = []
                
                # Add to failed models to prevent further issues
                self.failed_models.add(model_name)
                self.save_failed_models()  # Persist the failed model
                
                # Update dynamic timeout manager for failed model
                if dynamic_timeout_manager:
                    dynamic_timeout_manager.model_completed(model_name, success=False)
                continue
            
            # Find and store best configurations for this model
            best_results = self.find_best_configurations(model_results, top_n=5)
            
            # Enhanced timeout handling: always evaluate partial results against deployed models
            if best_results:
                best_result = best_results[0]
                
                # Get currently deployed model performance for comparison
                current_deployed_mape = float('inf')  # Default if no current deployment
                
                # Load main production config to check for previous tuning results
                main_config_path = os.path.join(os.path.dirname(os.path.dirname(self.config_path)), 'config.json')
                try:
                    with open(main_config_path, 'r') as f:
                        main_config = json.load(f)
                    
                    if model_name in main_config.get('models', {}):
                        existing_results = main_config['models'][model_name].get('tuning_results', {})
                        current_deployed_mape = existing_results.get('mape', float('inf'))
                        if current_deployed_mape != float('inf'):
                            print(f"   📋 Found previous result for {model_name}: {current_deployed_mape:.3f}% MAPE")
                except Exception as e:
                    print(f"   ⚠️ Could not load main config for comparison: {e}")
                
                # Always store the best result, but mark if it's from partial/timeout results
                is_partial_result = len(model_results) < self.get_total_combinations(model_name)
                
                self.best_configs[model_name] = {
                    'split_ratio': best_result.split_ratio,
                    'hyperparameters': best_result.hyperparameters,
                    'mape': best_result.mape,
                    'mae': best_result.mae,
                    'training_time': best_result.training_time,
                    'trial_number': best_result.trial_number,
                    'is_partial_result': is_partial_result,
                    'trials_completed': len(model_results),
                    'current_deployed_mape': current_deployed_mape
                }
                
                # Show comparison against currently deployed model
                if current_deployed_mape != float('inf'):
                    improvement = current_deployed_mape - best_result.mape
                    if improvement > 0:
                        status = "🚀 IMPROVEMENT" if not is_partial_result else "🔥 PARTIAL IMPROVEMENT"
                        print(f"{status}: {improvement:.3f}% better than deployed ({current_deployed_mape:.3f}% → {best_result.mape:.3f}%)")
                    else:
                        status = "📊 NO IMPROVEMENT" if not is_partial_result else "⏱️ PARTIAL NO IMPROVEMENT"
                        print(f"{status}: Current deployed is better ({current_deployed_mape:.3f}% vs {best_result.mape:.3f}%)")
                else:
                    status = "🆕 NEW MODEL" if not is_partial_result else "🆕 PARTIAL NEW MODEL"
                    print(f"{status}: No previous deployment to compare against")
            elif model_results:
                # We have some results but none were good enough - still check if better than deployed
                print(f"⚠️ {model_name}: {len(model_results)} trials completed but no viable configurations found")
                print(f"   Checking if any result is better than currently deployed...")
                
                # Find the best result even if it's not "good enough" by normal standards
                if model_results:
                    best_available = min(model_results, key=lambda x: x.mape if x.success else float('inf'))
                    if best_available.success:
                        current_deployed_mape = float('inf')
                        if model_name in self.config['models']:
                            existing_results = self.config['models'][model_name].get('tuning_results', {})
                            current_deployed_mape = existing_results.get('mape', float('inf'))
                        
                        if best_available.mape < current_deployed_mape:
                            print(f"   ✅ Found better result despite timeout: {best_available.mape:.3f}% vs {current_deployed_mape:.3f}%")
                            self.best_configs[model_name] = {
                                'split_ratio': best_available.split_ratio,
                                'hyperparameters': best_available.hyperparameters,
                                'mape': best_available.mape,
                                'mae': best_available.mae,
                                'training_time': best_available.training_time,
                                'trial_number': best_available.trial_number,
                                'is_partial_result': True,
                                'trials_completed': len(model_results),
                                'current_deployed_mape': current_deployed_mape
                            }
                        else:
                            print(f"   📊 No improvement found: {best_available.mape:.3f}% vs {current_deployed_mape:.3f}%")
        
            if model_name in self.best_configs:
                model_time = time.time() - model_start_time
                print(f"✅ {model_name} completed in {model_time:.1f}s")
                print(f"🏆 Best MAPE: {self.best_configs[model_name]['mape']:.3f}% (split: {self.best_configs[model_name]['split_ratio']:.3f})")
                
                # Update dynamic timeout manager for successful model
                if dynamic_timeout_manager:
                    dynamic_timeout_manager.model_completed(model_name, success=True)
                
                # Show top 3 results with comprehensive parameter details
                print("🔝 Top 3 configurations:")
                for j, result in enumerate(best_results[:3], 1):
                    # Format key hyperparameters for display
                    key_params = []
                    hyperparams = result.hyperparameters
                    
                    # Show most important parameters based on model type
                    if 'lags' in hyperparams:
                        key_params.append(f"lags={hyperparams['lags']}")
                    if 'n_estimators' in hyperparams:
                        key_params.append(f"n_estimators={hyperparams['n_estimators']}")
                    if 'max_depth' in hyperparams:
                        key_params.append(f"max_depth={hyperparams['max_depth']}")
                    if 'learning_rate' in hyperparams:
                        key_params.append(f"lr={hyperparams['learning_rate']}")
                    if 'changepoint_prior_scale' in hyperparams:
                        key_params.append(f"changepoint={hyperparams['changepoint_prior_scale']}")
                    if 'seasonality_prior_scale' in hyperparams:
                        key_params.append(f"seasonality={hyperparams['seasonality_prior_scale']}")
                    if 'season_length' in hyperparams:
                        key_params.append(f"season_length={hyperparams['season_length']}")
                    if 'trend' in hyperparams:
                        key_params.append(f"trend={hyperparams['trend']}")
                    if 'seasonal' in hyperparams:
                        key_params.append(f"seasonal={hyperparams['seasonal']}")
                    if 'version' in hyperparams:
                        key_params.append(f"version={hyperparams['version']}")
                    
                    # Limit to top 4 most important parameters for readability
                    key_params_str = ", ".join(key_params[:4])
                    if len(key_params) > 4:
                        key_params_str += f", +{len(key_params)-4} more"
                    
                    print(f"  {j}. MAPE: {result.mape:.3f}%, Split: {result.split_ratio:.3f}")
                    print(f"     MAE: {result.mae:.1f}, Trial: #{result.trial_number}, Time: {result.training_time:.1f}s")
                    print(f"     Key params: {key_params_str}")
                    if j < len(best_results[:3]):  # Add separator except for last item
                        print()
                
                successful_count = len([r for r in model_results if r.success])
                training_success_rate = successful_count/len(model_results)*100 if model_results else 0
                print(f"📊 Training success rate: {successful_count}/{len(model_results)} ({training_success_rate:.1f}%)")
            else:
                print(f"❌ {model_name} - No successful training configurations found")
                
        # Store all training results
        self.results = all_results
        
        # Final comprehensive tuning summary
        total_time = time.time() - total_start_time
        print(f"\n{'='*80}")
        print("🎉 COMPREHENSIVE HYPERPARAMETER TUNING COMPLETE")
        print(f"{'='*80}")
        print(f"⏱️  Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"🔬 Total training trials: {len(self.results):,}")
        print(f"✅ Successful training trials: {len([r for r in self.results if r.success]):,}")
        print(f"📊 Models successfully tuned: {len(self.best_configs)}")
        
        if self.best_configs:
            print("\n🏆 BEST TRAINING CONFIGURATIONS:")
            print("-" * 120)
            print(f"{'Rank':<5} {'Model':<15} {'Split':<7} {'MAPE':<8} {'MAE':<8} {'Trial':<8} {'Time':<8} {'Key Parameters':<50}")
            print("-" * 120)
            
            # Sort by MAPE (training performance)
            sorted_configs = sorted(self.best_configs.items(), key=lambda x: x[1]['mape'])
            
            for rank, (model_name, config) in enumerate(sorted_configs, 1):
                # Format key parameters for display
                key_params = []
                hyperparams = config['hyperparameters']
                
                # Show most important parameters based on model type
                if 'lags' in hyperparams:
                    key_params.append(f"lags={hyperparams['lags']}")
                if 'n_estimators' in hyperparams:
                    key_params.append(f"n_est={hyperparams['n_estimators']}")
                if 'max_depth' in hyperparams:
                    key_params.append(f"depth={hyperparams['max_depth']}")
                if 'learning_rate' in hyperparams:
                    key_params.append(f"lr={hyperparams['learning_rate']}")
                if 'changepoint_prior_scale' in hyperparams:
                    key_params.append(f"chg={hyperparams['changepoint_prior_scale']}")
                if 'seasonality_prior_scale' in hyperparams:
                    key_params.append(f"seas={hyperparams['seasonality_prior_scale']}")
                if 'season_length' in hyperparams:
                    key_params.append(f"season_len={hyperparams['season_length']}")
                if 'trend' in hyperparams:
                    key_params.append(f"trend={hyperparams['trend']}")
                if 'seasonal' in hyperparams:
                    key_params.append(f"seasonal={hyperparams['seasonal']}")
                if 'version' in hyperparams:
                    key_params.append(f"ver={hyperparams['version']}")
                
                # Limit to top 3 most important parameters for table readability
                key_params_str = ", ".join(key_params[:3])
                if len(key_params) > 3:
                    key_params_str += f", +{len(key_params)-3}"
                
                # Truncate if too long
                if len(key_params_str) > 48:
                    key_params_str = key_params_str[:45] + "..."
                
                print(f"{rank:<5} {model_name:<15} {config['split_ratio']:<7.3f} "
                      f"{config['mape']:<8.3f} {config['mae']:<8.0f} "
                      f"{config['trial_number']:<8} {config['training_time']:<8.2f} {key_params_str:<50}")
            
            print("-" * 120)
            best_model, best_config = sorted_configs[0]
            print(f"🥇 Training Champion: {best_model} (MAPE: {best_config['mape']:.3f}%)")
            
            # Show detailed optimal configuration
            print(f"🎯 Optimal Configuration Details:")
            print(f"   Split Ratio: {best_config['split_ratio']:.3f}")
            print(f"   Trial Number: #{best_config['trial_number']}")
            print(f"   Training Time: {best_config['training_time']:.2f}s")
            print(f"   Full Parameters: {best_config['hyperparameters']}")
        else:
            print("\n⚠️  No successful training configurations found across all models!")
        
        print("="*80)


def main():
    """
    Main entrypoint for the comprehensive tuner.
    Loads all options from config.json instead of command-line arguments.
    Suitable for CI/CD and GitHub Actions.
    """
    config = load_config()

    # NEW: Run test for all models if requested in config
    if config.get('test_all_models', False):
        tuner = ComprehensiveHyperparameterTuner()
        tuner.test_all_models_viability()
        return

    # Extract configuration values
    parser = argparse.ArgumentParser(
        description='Comprehensive CVE Forecast Hyperparameter Explorer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comprehensive_tuner.py                           # Use default limits with random search
  python comprehensive_tuner.py --max-combinations 10000  # Cap all models at 10,000 combinations
  python comprehensive_tuner.py --unlimited               # No limits (full exploration)
  python comprehensive_tuner.py --models Prophet,XGBoost  # Only tune specific models
  python comprehensive_tuner.py --search-type grid        # Use grid search instead of random
  python comprehensive_tuner.py --timeout-minutes 10      # Set 10 minute timeout for all models
  python comprehensive_tuner.py --unlimited-time          # Remove all time limits
  python comprehensive_tuner.py --no-update-config        # Don't update config.json
        """
    )
    
    parser.add_argument('--models', '-m', 
                       help='Comma-separated list of models to tune (default: all enabled)')
    parser.add_argument('--search-type', '-s', 
                       choices=['grid', 'random'], 
                       help='Search strategy (default: intelligent selection based on space limitations)')
    parser.add_argument('--max-combinations', '-c', 
                       type=int,
                       help='Maximum combinations per model (overrides default limits)')
    parser.add_argument('--unlimited', '-u', 
                       action='store_true',
                       help='Remove all combination limits (full exploration)')
    parser.add_argument('--timeout-minutes', '-t', 
                       type=float,
                       help='Maximum time per model in minutes (overrides default timeouts)')
    parser.add_argument('--unlimited-time', 
                       action='store_true',
                       help='Remove all time limits (unlimited training time)')
    parser.add_argument('--no-update-config', 
                       action='store_true',
                       help='Don\'t update config.json with results')
    parser.add_argument('--show-failed', 
                       action='store_true',
                       help='Show failed models list and exit')
    parser.add_argument('--clear-failed', 
                       action='store_true',
                       help='Clear failed models list')
    parser.add_argument('--show-space', 
                       action='store_true',
                       help='Show search space overview and exit')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract values from config or command line args
    target_models = args.models.split(',') if args.models else None
    search_type = args.search_type
    max_combinations = config.get('max_combinations') or args.max_combinations
    unlimited = config.get('unlimited', False) or args.unlimited
    timeout_minutes = config.get('timeout_minutes') or args.timeout_minutes
    unlimited_time = config.get('unlimited_time', False) or args.unlimited_time
    no_update_config = config.get('no_update_config', False) or args.no_update_config
    show_failed = args.show_failed
    clear_failed = args.clear_failed
    show_space = args.show_space
    
    # Configure search space
    search_config = SearchSpaceConfig(
        max_combinations=max_combinations,
        unlimited=unlimited,
        timeout_minutes=timeout_minutes,
        unlimited_time=unlimited_time
    )

    # Initialize comprehensive tuner
    tuner = ComprehensiveHyperparameterTuner()

    # Handle failed models commands
    if show_failed:
        tuner.show_failed_models()
        return

    if clear_failed:
        tuner.clear_failed_models()
        print()

    # Intelligent search type selection if not specified
    if search_type is None:
        # Check if any model will be limited
        active_models = {
            name: cfg for name, cfg in tuner.config['models'].items()
            if cfg.get('enabled', False) and name not in SearchSpaceConfig.DISABLED_MODELS
        }
        if target_models:
            active_models = {k: v for k, v in active_models.items() if k in target_models}
        has_limited = False
        for model_name in active_models:
            total_combinations = tuner.get_total_combinations(model_name)
            limit = search_config.get_limit_for_model(model_name)
            if limit is not None and total_combinations > limit:
                has_limited = True
                break
        search_type = 'random' if has_limited else 'grid'
        print(f"🎯 Intelligent search type selection: {search_type}")
        if has_limited:
            print("   Random search chosen for efficient limited space exploration")
        else:
            print("   Grid search chosen for complete space exploration")

    # Show search space overview
    if show_space or unlimited or max_combinations:
        tuner.show_search_space_overview(target_models, search_config, search_type)
        if show_space:
            return

    # Configure output
    if unlimited:
        print("🔓 UNLIMITED MODE: Full exploration of all parameter combinations")
        print("⚠️  This may take VERY long time for complex models!")
    elif max_combinations:
        print(f"🎲 Custom limit: {max_combinations:,} combinations per model")
    else:
        print("🎯 Using intelligent limits based on model complexity")
        print("   Simple models: 1,000 combinations (ExponentialSmoothing, KalmanFilter, Croston)")
        print("   Medium models: 10,000 combinations (Prophet, AutoARIMA, TBATS, Theta, FourTheta)")
        print("   Complex models: 50,000 combinations (XGBoost, LightGBM, CatBoost, RandomForest)")

    print(f"🔍 Search strategy: {search_type}")

    if no_update_config:
        print("⚠️  Config update disabled")
    else:
        print("🔧 Will update config.json with optimal results")

    try:
        # Run comprehensive tuning
        tuner.run_comprehensive_tuning(target_models, search_type, search_config)
        # Save results
        tuner.save_results()
        # Update config.json with optimal configurations if requested
        if not no_update_config:
            tuner.update_main_config()
            # Also update tuner's own config for self-improvement
            tuner.update_tuner_config()
        else:
            print("⚠️  Skipping config.json update (no_update_config is true)")
            print("⚠️  Skipping tuner_config.json update (no_update_config is true)")
    except KeyboardInterrupt:
        print("\n⚠️  Tuning interrupted by user.")
        if tuner.results:
            print("💾 Saving partial results...")
            tuner.save_results(filename="partial_comprehensive_results.json")
    except Exception as e:
        print(f"❌ Error during tuning: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Alias for backward compatibility
ComprehensiveTuner = ComprehensiveHyperparameterTuner
