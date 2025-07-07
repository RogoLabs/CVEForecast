#!/usr/bin/env python3
"""
CVE Forecast Script
This script processes CVE data, trains forecasting models, and generates predictions.
"""

import json
import os
import shutil
import tempfile
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import (
    ARIMA, ExponentialSmoothing, Prophet, Theta, FourTheta,
    LinearRegressionModel, RandomForestModel, XGBModel,
    KalmanForecaster, NaiveSeasonal, NaiveDrift, NaiveMean, NaiveMovingAverage,
    AutoARIMA as DartsAutoARIMA, NaiveEnsembleModel, 
    RegressionEnsembleModel, LightGBMModel, CatBoostModel,
    # Additional statistical models
    FFT, TBATS, Croston,
    # PyTorch Lightning-based models
    TCNModel, TFTModel, NBEATSModel, NHiTSModel,
    TransformerModel, RNNModel, BlockRNNModel,
    TiDEModel, DLinearModel, NLinearModel, TSMixerModel
)
# Try to import StatsForecast models, fallback if not available
STATSFORECAST_ARIMA_AVAILABLE = False
STATSFORECAST_ETS_AVAILABLE = False
STATSFORECAST_THETA_AVAILABLE = False
STATSFORECAST_CES_AVAILABLE = False
STATSFORECAST_MFLES_AVAILABLE = False
STATSFORECAST_TBATS_AVAILABLE = False

try:
    from darts.models.forecasting.sf_auto_arima import StatsForecastAutoARIMA
    STATSFORECAST_ARIMA_AVAILABLE = True
except ImportError:
    try:
        from darts.models import StatsForecastAutoARIMA
        STATSFORECAST_ARIMA_AVAILABLE = True
    except ImportError:
        StatsForecastAutoARIMA = None

try:
    from darts.models.forecasting.sf_auto_ets import StatsForecastAutoETS
    STATSFORECAST_ETS_AVAILABLE = True
except ImportError:
    try:
        from darts.models import StatsForecastAutoETS
        STATSFORECAST_ETS_AVAILABLE = True
    except ImportError:
        StatsForecastAutoETS = None

try:
    from darts.models.forecasting.sf_auto_theta import StatsForecastAutoTheta
    STATSFORECAST_THETA_AVAILABLE = True
except ImportError:
    try:
        from darts.models import StatsForecastAutoTheta
        STATSFORECAST_THETA_AVAILABLE = True
    except ImportError:
        StatsForecastAutoTheta = None

try:
    from darts.models.forecasting.sf_auto_ces import StatsForecastAutoCES
    STATSFORECAST_CES_AVAILABLE = True
except ImportError:
    try:
        from darts.models import StatsForecastAutoCES
        STATSFORECAST_CES_AVAILABLE = True
    except ImportError:
        StatsForecastAutoCES = None

try:
    from darts.models.forecasting.sf_auto_mfles import StatsForecastAutoMFLES
    STATSFORECAST_MFLES_AVAILABLE = True
except ImportError:
    try:
        from darts.models import StatsForecastAutoMFLES
        STATSFORECAST_MFLES_AVAILABLE = True
    except ImportError:
        StatsForecastAutoMFLES = None

try:
    from darts.models.forecasting.sf_auto_tbats import StatsForecastAutoTBATS
    STATSFORECAST_TBATS_AVAILABLE = True
except ImportError:
    try:
        from darts.models import StatsForecastAutoTBATS
        STATSFORECAST_TBATS_AVAILABLE = True
    except ImportError:
        StatsForecastAutoTBATS = None

from darts.metrics import mape, rmse, mae, mase, rmsse
from darts.utils.utils import ModelMode, SeasonalityMode


class CVEForecastEngine:
    """Main engine for CVE forecasting"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.historical_data = None
        self.forecasts = {}
        self.model_rankings = []
        
    def parse_cve_data(self, repo_path: str = None) -> pd.DataFrame:
        """Parse CVE JSON files and extract publication dates"""
        if repo_path is None:
            repo_path = self.data_path
            
        if repo_path is None:
            # Default to cvelistV5 directory in project root
            script_dir = Path(__file__).parent
            repo_path = str(script_dir / "cvelistV5")
            
        # Check if the CVE data directory exists
        cves_path = Path(repo_path) / "cves"
        if not cves_path.exists():
            print(f"❌ CVE data not found at: {repo_path}")
            print("\nPlease run the following command to download CVE data:")
            print("    python download_data.py")
            print("\nThis will download the CVE data repository to the project root.")
            raise FileNotFoundError(f"CVE data directory not found: {cves_path}")
            
        print(f"Parsing CVE data from {repo_path}...")
        cve_dates = []
        
        cves_path = Path(repo_path) / "cves"
        # Recursively find all JSON files
        json_files = list(cves_path.rglob("*.json"))
        print(f"Found {len(json_files)} CVE JSON files")
        
        for i, json_file in enumerate(json_files):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(json_files)} files")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    cve_data = json.load(f)
                
                # Filter out rejected CVEs - check state array for "REJECTED"
                if 'cveMetadata' in cve_data and 'state' in cve_data['cveMetadata']:
                    state = cve_data['cveMetadata']['state']
                    if isinstance(state, list) and 'REJECTED' in state:
                        continue  # Skip rejected CVEs
                    elif isinstance(state, str) and state == 'REJECTED':
                        continue  # Skip rejected CVEs
                
                # Try multiple ways to extract publication date
                published_date = None
                
                # Method 1: cveMetadata.datePublished (primary field in schema)
                if 'cveMetadata' in cve_data and 'datePublished' in cve_data['cveMetadata']:
                    published_date = cve_data['cveMetadata']['datePublished']
                
                # Method 2: cveMetadata.dateReserved (fallback - reservation date)
                elif 'cveMetadata' in cve_data and 'dateReserved' in cve_data['cveMetadata']:
                    published_date = cve_data['cveMetadata']['dateReserved']
                
                # Method 3: containers.cna.datePublic (alternative publication date)
                elif 'containers' in cve_data and 'cna' in cve_data['containers'] and 'datePublic' in cve_data['containers']['cna']:
                    published_date = cve_data['containers']['cna']['datePublic']
                
                # Method 4: Legacy formats (for older CVE files)
                elif 'publishedDate' in cve_data:
                    published_date = cve_data['publishedDate']
                elif 'Published_Date' in cve_data:
                    published_date = cve_data['Published_Date']
                
                if published_date:
                    try:
                        # Handle different date formats
                        if 'T' in published_date:
                            # ISO format with time (RFC3339/ISO8601)
                            date_obj = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                        else:
                            # Date only format
                            date_obj = datetime.strptime(published_date, '%Y-%m-%d')
                        
                        cve_dates.append(date_obj.date())
                    except ValueError:
                        # Try other common date formats
                        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                            try:
                                date_obj = datetime.strptime(published_date, fmt)
                                cve_dates.append(date_obj.date())
                                break
                            except ValueError:
                                continue
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Skip malformed or incomplete CVE entries
                continue
        
        print(f"Successfully parsed {len(cve_dates)} CVE publication dates")
        
        # Check if we found any valid dates
        if len(cve_dates) == 0:
            print("❌ No valid CVE publication dates found!")
            print("This could be due to:")
            print("1. CVE files don't contain 'cveMetadata.datePublished' fields")
            print("2. Date format is different than expected")
            print("3. CVE files are malformed or empty")
            
            # Let's examine a few files to understand the structure
            print("\nExamining first few CVE files for debugging...")
            for i, json_file in enumerate(json_files[:3]):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        cve_data = json.load(f)
                    print(f"\nFile {i+1}: {json_file.name}")
                    print(f"  Keys: {list(cve_data.keys())}")
                    if 'cveMetadata' in cve_data:
                        print(f"  cveMetadata keys: {list(cve_data['cveMetadata'].keys())}")
                        # Check for date fields
                        date_fields = ['datePublished', 'dateReserved', 'dateUpdated']
                        for field in date_fields:
                            if field in cve_data['cveMetadata']:
                                print(f"    {field}: {cve_data['cveMetadata'][field]}")
                    else:
                        print("  No 'cveMetadata' found")
                        
                    # Check for containers
                    if 'containers' in cve_data:
                        print(f"  containers keys: {list(cve_data['containers'].keys())}")
                        if 'cna' in cve_data['containers']:
                            cna_keys = list(cve_data['containers']['cna'].keys())
                            print(f"    cna keys: {cna_keys}")
                            if 'datePublic' in cve_data['containers']['cna']:
                                print(f"      datePublic: {cve_data['containers']['cna']['datePublic']}")
                            if 'dateAssigned' in cve_data['containers']['cna']:
                                print(f"      dateAssigned: {cve_data['containers']['cna']['dateAssigned']}")
                except Exception as e:
                    print(f"  Error reading file: {e}")
            
            raise ValueError("No valid CVE publication dates found in the dataset. Please check the CVE data format.")
        
        # Create time series DataFrame
        df = pd.DataFrame({'date': cve_dates})
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by month and count CVEs per month
        df['month'] = df['date'].dt.to_period('M')
        monthly_counts = df.groupby('month').size().reset_index(name='cve_count')
        monthly_counts['date'] = monthly_counts['month'].dt.start_time
        monthly_counts = monthly_counts.drop('month', axis=1)
        
        print(f"Date range: {monthly_counts['date'].min()} to {monthly_counts['date'].max()}")
        print(f"Total months with CVE data: {len(monthly_counts)}")
        
        # Fill missing months with zero counts
        date_range = pd.date_range(
            start=monthly_counts['date'].min(),
            end=monthly_counts['date'].max(),
            freq='MS'  # Month Start frequency
        )
        
        complete_df = pd.DataFrame({'date': date_range})
        complete_df = complete_df.merge(monthly_counts, on='date', how='left')
        complete_df['cve_count'] = complete_df['cve_count'].fillna(0)
        
        # Ensure the date column has proper frequency information
        complete_df = complete_df.sort_values('date').reset_index(drop=True)
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        return complete_df
    
    def prepare_models(self) -> List[Tuple[str, Any]]:
        """Prepare and return comprehensive forecasting models for evaluation (25+ models total)"""
        models = []
        
        # Statistical Models - Proven performers for time series
        try:
            # AutoARIMA - Automatically selects the best ARIMA parameters
            if STATSFORECAST_ARIMA_AVAILABLE and StatsForecastAutoARIMA is not None:
                models.append(("AutoARIMA", StatsForecastAutoARIMA()))
                print("Added AutoARIMA model")
            else:
                print("StatsForecastAutoARIMA not available, using DartsAutoARIMA instead")
                models.append(("AutoARIMA", DartsAutoARIMA(
                    start_p=0, start_q=0,      # Starting values for p and q
                    max_p=3, max_q=3,          # Maximum values for p and q  
                    seasonal=True,             # Enable seasonal component
                    stepwise=True,             # Use stepwise algorithm for speed
                    suppress_warnings=True,    # Suppress warnings
                    error_action='ignore',     # Ignore errors and continue
                    random_state=42
                )))
        except Exception as e:
            print(f"Failed to add AutoARIMA: {e}")
        
        try:
            # AutoETS - Error, Trend, Seasonality model with automatic parameter selection
            if STATSFORECAST_ETS_AVAILABLE and StatsForecastAutoETS is not None:
                models.append(("AutoETS", StatsForecastAutoETS()))
                print("Added AutoETS model")
            else:
                print("StatsForecastAutoETS not available, skipping")
        except Exception as e:
            print(f"Failed to add AutoETS: {e}")
        
        try:
            # ExponentialSmoothing - Good for trends and seasonality patterns - Enhanced
            models.append(("ExponentialSmoothing", ExponentialSmoothing(
                trend='add',               # Additive trend
                seasonal='add',            # Additive seasonality
                seasonal_periods=12,       # 12 month seasonality
                damped_trend=True,         # Damped trend for long-term stability
            )))
            print("Added ExponentialSmoothing model")
        except Exception as e:
            print(f"Failed to add ExponentialSmoothing: {e}")
        
        try:
            # Prophet - Facebook's robust forecasting method, handles seasonality well - Enhanced
            models.append(("Prophet", Prophet(
                yearly_seasonality=True,    # Enable yearly patterns
                weekly_seasonality=False,   # Disable weekly (not relevant for CVE data)
                daily_seasonality=False,    # Disable daily (not relevant for monthly data)
                seasonality_mode='additive',        # Use additive instead of multiplicative for zero values
                changepoint_prior_scale=0.1,        # Slightly more flexible changepoints
                seasonality_prior_scale=1.0,        # More conservative seasonality
                n_changepoints=15,                   # Fewer changepoints to reduce overfitting
                mcmc_samples=0,                      # Disable MCMC for faster training
                interval_width=0.8,                  # Confidence intervals
            )))
            print("Added Prophet model")
        except Exception as e:
            print(f"Failed to add Prophet: {e}")
        
        try:
            # AutoTheta - Theta method with automatic parameter selection
            if STATSFORECAST_THETA_AVAILABLE and StatsForecastAutoTheta is not None:
                models.append(("AutoTheta", StatsForecastAutoTheta()))
                print("Added AutoTheta model")
            else:
                print("StatsForecastAutoTheta not available, using standard Theta instead")
                models.append(("Theta", Theta(season_mode=SeasonalityMode.ADDITIVE)))
        except Exception as e:
            print(f"Failed to add AutoTheta: {e}")
        
        try:
            # Kalman Filter - Excellent for noisy time series with trends
            models.append(("KalmanFilter", KalmanForecaster()))
            print("Added KalmanFilter model")
        except Exception as e:
            print(f"Failed to add KalmanFilter: {e}")
        
        # Additional Statistical Models
        # Note: FFT removed as it's not suitable for this type of time series data
        
        try:
            # TBATS - Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components
            models.append(("TBATS", TBATS(season_length=12)))  # 12 for monthly seasonality
            print("Added TBATS model")
        except Exception as e:
            print(f"Failed to add TBATS: {e}")
        
        try:
            # Croston - Intermittent demand forecasting
            models.append(("Croston", Croston()))
            print("Added Croston model")
        except Exception as e:
            print(f"Failed to add Croston: {e}")
        
        try:
            # FourTheta - Four Theta method
            models.append(("FourTheta", FourTheta(season_mode=SeasonalityMode.ADDITIVE)))
            print("Added FourTheta model")
        except Exception as e:
            print(f"Failed to add FourTheta: {e}")
        
        # Additional StatsForecast Models
        try:
            # AutoCES - Complex Exponential Smoothing with automatic parameter selection
            if STATSFORECAST_CES_AVAILABLE and StatsForecastAutoCES is not None:
                models.append(("AutoCES", StatsForecastAutoCES()))
                print("Added AutoCES model")
            else:
                print("StatsForecastAutoCES not available")
        except Exception as e:
            print(f"Failed to add AutoCES: {e}")
        
        try:
            # AutoMFLES - Multiple Frequency Linear Exponential Smoothing
            if STATSFORECAST_MFLES_AVAILABLE and StatsForecastAutoMFLES is not None:
                models.append(("AutoMFLES", StatsForecastAutoMFLES()))
                print("Added AutoMFLES model")
            else:
                print("StatsForecastAutoMFLES not available")
        except Exception as e:
            print(f"Failed to add AutoMFLES: {e}")
        
        try:
            # AutoTBATS - TBATS with automatic parameter selection
            if STATSFORECAST_TBATS_AVAILABLE and StatsForecastAutoTBATS is not None:
                models.append(("AutoTBATS", StatsForecastAutoTBATS()))
                print("Added AutoTBATS model")
            else:
                print("StatsForecastAutoTBATS not available")
        except Exception as e:
            print(f"Failed to add AutoTBATS: {e}")
        
        # Baseline Models
        try:
            # NaiveMean - Simple mean forecast
            models.append(("NaiveMean", NaiveMean()))
            print("Added NaiveMean model")
        except Exception as e:
            print(f"Failed to add NaiveMean: {e}")
        
        try:
            # NaiveMovingAverage - Moving average forecast - Enhanced
            models.append(("NaiveMovingAverage", NaiveMovingAverage(input_chunk_length=6)))  # Shorter window
            print("Added NaiveMovingAverage model")
        except Exception as e:
            print(f"Failed to add NaiveMovingAverage: {e}")
        
        try:
            # NaiveSeasonal - Seasonal naive forecast
            models.append(("NaiveSeasonal", NaiveSeasonal(K=12)))  # 12-month seasonal pattern
            print("Added NaiveSeasonal model")
        except Exception as e:
            print(f"Failed to add NaiveSeasonal: {e}")
        
        try:
            # NaiveDrift - Drift method forecast
            models.append(("NaiveDrift", NaiveDrift()))
            print("Added NaiveDrift model")
        except Exception as e:
            print(f"Failed to add NaiveDrift: {e}")
        
        # Machine Learning Models
        try:
            # XGBoost - Gradient boosting for time series
            # XGBoost - Extreme Gradient Boosting with time series features - Optimized
            models.append(("XGBoost", XGBModel(
                lags=24,                    # Increased from 12 for better patterns
                random_state=42,
                # XGBoost specific parameters for better performance
                **{
                    'n_estimators': 200,     # More trees for better learning
                    'max_depth': 6,          # Moderate depth to prevent overfitting
                    'learning_rate': 0.1,    # Learning rate
                    'subsample': 0.8,        # Row sampling for regularization
                    'colsample_bytree': 0.8, # Column sampling for regularization
                    'reg_alpha': 0.1,        # L1 regularization
                    'reg_lambda': 0.1,       # L2 regularization
                }
            )))
            print("Added XGBoost model")
        except Exception as e:
            print(f"Failed to add XGBoost: {e}")
        
        try:
            # LightGBM - Fast gradient boosting with optimization - Improved
            models.append(("LightGBM", LightGBMModel(
                lags=12,                    # Reduced lags for simpler model
                random_state=42,
                # LightGBM specific parameters for better performance
                **{
                    'n_estimators': 100,     # Fewer trees to prevent overfitting
                    'max_depth': 4,          # Shallower trees
                    'learning_rate': 0.05,   # Lower learning rate
                    'subsample': 0.9,        # Higher row sampling
                    'colsample_bytree': 0.9, # Higher column sampling
                    'reg_alpha': 0.5,        # Stronger L1 regularization
                    'reg_lambda': 0.5,       # Stronger L2 regularization
                    'min_child_samples': 10, # Fewer minimum samples in leaf
                    'verbose': -1,           # Suppress warnings
                    'force_row_wise': True,  # Force row-wise tree construction
                }
            )))
            print("Added LightGBM model")
        except Exception as e:
            print(f"Failed to add LightGBM: {e}")
        
        try:
            # RandomForest - Ensemble tree model with optimization
            models.append(("RandomForest", RandomForestModel(
                lags=24,                    # Increased from 12 for better patterns
                random_state=42,
                # RandomForest specific parameters for better performance
                **{
                    'n_estimators': 200,     # More trees for better learning
                    'max_depth': 8,          # Deeper trees for complex patterns
                    'min_samples_split': 5,  # Minimum samples to split
                    'min_samples_leaf': 2,   # Minimum samples in leaf
                    'max_features': 'sqrt',  # Feature sampling strategy
                    'bootstrap': True,       # Bootstrap sampling
                }
            )))
            print("Added RandomForest model")
        except Exception as e:
            print(f"Failed to add RandomForest: {e}")
        
        try:
            # CatBoost - Another powerful gradient boosting model with optimization
            models.append(("CatBoost", CatBoostModel(
                lags=24,                    # Increased from 12 for better patterns
                random_state=42,
                # CatBoost specific parameters for better performance
                **{
                    'iterations': 200,       # More iterations for better learning
                    'depth': 6,              # Tree depth
                    'learning_rate': 0.1,    # Learning rate
                    'l2_leaf_reg': 3,        # L2 regularization
                    'bootstrap_type': 'Bernoulli',  # Bootstrap type
                    'subsample': 0.8,        # Row sampling
                    'random_strength': 1,    # Random strength
                    'verbose': False,        # Silent training
                }
            )))
            print("Added CatBoost model")
        except Exception as e:
            print(f"Failed to add CatBoost: {e}")
        
        # PyTorch Lightning-based Deep Learning Models (with CPU-only training)
        try:
            # TCN (Temporal Convolutional Network) - Optimized for long sequences
            models.append(("TCN", TCNModel(
                input_chunk_length=24,      # Increased for better long-term patterns
                output_chunk_length=1,
                kernel_size=3,              # Kernel size for convolution
                num_filters=32,             # Number of filters
                dilation_base=2,            # Dilation base for TCN
                dropout=0.1,                # Dropout for regularization
                n_epochs=40,                # Increased for better convergence
                batch_size=16,              # Smaller batch for stability
                random_state=42,
                force_reset=True,
                save_checkpoints=False,
                pl_trainer_kwargs={
                    "enable_progress_bar": False, 
                    "enable_model_summary": False,
                    "accelerator": "cpu",  # Force CPU to avoid MPS issues
                    "max_epochs": 40
                }
            )))
            print("Added TCN model")
        except Exception as e:
            print(f"Failed to add TCN: {e}")
        
        try:
            # N-BEATS - Neural basis expansion analysis for time series - Improved
            models.append(("NBEATS", NBEATSModel(
                input_chunk_length=12,     # Reduced for simpler patterns
                output_chunk_length=1,
                num_stacks=2,              # Fewer stacks to reduce complexity
                num_blocks=2,              # Fewer blocks per stack
                num_layers=2,              # Fewer layers per block
                layer_widths=32,           # Smaller hidden layer width
                n_epochs=30,               # Reduced epochs
                batch_size=32,             # Larger batch for stability
                random_state=42,
                force_reset=True,
                save_checkpoints=False,
                pl_trainer_kwargs={
                    "enable_progress_bar": False, 
                    "enable_model_summary": False,
                    "accelerator": "cpu",
                    "max_epochs": 30
                }
            )))
            print("Added NBEATS model")
        except Exception as e:
            print(f"Failed to add NBEATS: {e}")
        
        try:
            # NHiTS - Neural Hierarchical Interpolation for Time Series - Optimized
            models.append(("NHiTS", NHiTSModel(
                input_chunk_length=24,     # Increased from 12 for better patterns
                output_chunk_length=1,
                num_stacks=3,              # Added hierarchical stacks
                num_blocks=1,              # Blocks per stack
                num_layers=2,              # Layers per block
                layer_widths=64,           # Hidden layer width
                n_epochs=40,               # Increased from 25
                batch_size=16,             # Added for stability
                random_state=42,
                force_reset=True,
                save_checkpoints=False,
                pl_trainer_kwargs={
                    "enable_progress_bar": False, 
                    "enable_model_summary": False,
                    "accelerator": "cpu",
                    "max_epochs": 40
                }
            )))
            print("Added NHiTS model")
        except Exception as e:
            print(f"Failed to add NHiTS: {e}")
        
        try:
            # DLinear - Simple but effective linear model for time series
            models.append(("DLinear", DLinearModel(
                input_chunk_length=12,
                output_chunk_length=1,
                n_epochs=25,
                random_state=42,
                force_reset=True,
                save_checkpoints=False,
                pl_trainer_kwargs={
                    "enable_progress_bar": False, 
                    "enable_model_summary": False,
                    "accelerator": "cpu"
                }
            )))
            print("Added DLinear model")
        except Exception as e:
            print(f"Failed to add DLinear: {e}")
        
        try:
            # NLinear - Normalized linear model - Improved configuration
            models.append(("NLinear", NLinearModel(
                input_chunk_length=24,     # Increased for better pattern capture
                output_chunk_length=1,
                n_epochs=40,               # More epochs for better convergence
                batch_size=32,             # Larger batch for stability
                random_state=42,
                force_reset=True,
                save_checkpoints=False,
                pl_trainer_kwargs={
                    "enable_progress_bar": False, 
                    "enable_model_summary": False,
                    "accelerator": "cpu",
                    "max_epochs": 40
                }
            )))
            print("Added NLinear model")
        except Exception as e:
            print(f"Failed to add NLinear: {e}")
        
        try:
            # TSMixer - Time Series Mixer model - Improved configuration
            models.append(("TSMixer", TSMixerModel(
                input_chunk_length=12,      # Reduced for simpler patterns
                output_chunk_length=1,
                n_epochs=30,                # Reduced epochs to prevent overfitting
                batch_size=32,              # Larger batch for more stable training
                random_state=42,
                force_reset=True,
                save_checkpoints=False,
                pl_trainer_kwargs={
                    "enable_progress_bar": False, 
                    "enable_model_summary": False,
                    "accelerator": "cpu",
                    "max_epochs": 30
                }
            )))
            print("Added TSMixer model")
        except Exception as e:
            print(f"Failed to add TSMixer: {e}")
        
        # Store individual models for ensemble creation (use best statistical models)
        individual_models = []
        for name, model in models:
            if name in ["AutoARIMA", "AutoETS", "ExponentialSmoothing"]:
                try:
                    # Create a copy for ensemble use
                    if name == "AutoARIMA":
                        if STATSFORECAST_ARIMA_AVAILABLE and StatsForecastAutoARIMA is not None:
                            individual_models.append(StatsForecastAutoARIMA())
                        else:
                            individual_models.append(DartsAutoARIMA())
                    elif name == "AutoETS":
                        if STATSFORECAST_ETS_AVAILABLE and StatsForecastAutoETS is not None:
                            individual_models.append(StatsForecastAutoETS())
                        # Skip if not available
                    elif name == "ExponentialSmoothing":
                        individual_models.append(ExponentialSmoothing())
                except:
                    continue
        
        # Ensemble Models - Combine multiple models for better performance
        if len(individual_models) >= 3:
            try:
                # NaiveEnsemble - Simple average of multiple models
                naive_ensemble = NaiveEnsembleModel(individual_models)
                models.append(("NaiveEnsemble", naive_ensemble))
                print("Added NaiveEnsemble model")
            except Exception as e:
                print(f"Failed to add NaiveEnsemble: {e}")
        
        # Add additional fallback models if we don't have enough yet
        while len(models) < 25:
            try:
                if len(models) == 23:
                    # ARIMA - Classic time series model
                    models.append(("ARIMA", ARIMA()))
                    print("Added ARIMA as additional model")
                elif len(models) == 24:
                    # LinearRegression - Simple baseline
                    models.append(("LinearRegression", LinearRegressionModel(lags=12)))
                    print("Added LinearRegression as additional model")
                else:
                    break
            except Exception as e:
                print(f"Failed to add additional model: {e}")
                break
        
        print(f"Prepared {len(models)} forecasting models")
        return models
    
    def evaluate_models(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Train and evaluate all models with comprehensive metrics"""
        print("Converting data to Darts TimeSeries...")
        
        # Convert to Darts TimeSeries with monthly frequency
        ts = TimeSeries.from_dataframe(
            data,
            time_col='date',
            value_cols='cve_count',
            freq='MS',  # Month Start frequency
            fill_missing_dates=True
        )
        
        # Split data for validation - use all complete months from current year for validation
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Find the index where current year starts
        ts_df = ts.to_dataframe()
        current_year_start = None
        current_month_index = None
        
        for i, date in enumerate(ts_df.index):
            if date.year == current_year:
                if current_year_start is None:
                    current_year_start = i
                if date.year == current_year and date.month == current_month:
                    current_month_index = i
                    break
        
        if current_year_start is not None and current_month_index is not None:
            # Split: train on all data before current year, validate on current year EXCLUDING current month
            train_ts = ts[:current_year_start]
            val_ts = ts[current_year_start:current_month_index]  # Exclude current incomplete month
        elif current_year_start is not None:
            # If no current month found, validate on all current year data
            train_ts = ts[:current_year_start]
            val_ts = ts[current_year_start:]
        else:
            # Fallback to original logic if no current year data found
            train_ts, val_ts = ts[:-6], ts[-6:]
        
        print(f"Training set size: {len(train_ts)} months")
        print(f"Validation set size: {len(val_ts)} months")
        
        models = self.prepare_models()
        results = []
        
        for model_name, model in models:
            print(f"\nTraining {model_name}...")
            try:
                # Train the model
                model.fit(train_ts)
                
                # Generate forecast
                forecast = model.predict(len(val_ts))
                
                # Calculate comprehensive metrics ONLY on completed months (val_ts excludes current month)
                # Using recommended time series accuracy metrics:
                # - MAPE: Mean Absolute Percentage Error (unit-free, interpretable)
                # - MASE: Mean Absolute Scaled Error (recommended by experts, unit-free)
                # - RMSSE: Root Mean Squared Scaled Error (scaled version of RMSE)
                # - MAE: Mean Absolute Error (simple, interpretable)
                mape_score = mape(val_ts, forecast)
                mae_score = mae(val_ts, forecast)
                
                # Calculate MASE and RMSSE with proper error handling
                try:
                    mase_score = mase(val_ts, forecast, train_ts)
                except Exception as e:
                    print(f"Warning: Could not calculate MASE for {model_name}: {e}")
                    mase_score = None
                    
                try:
                    rmsse_score = rmsse(val_ts, forecast, train_ts)
                except Exception as e:
                    print(f"Warning: Could not calculate RMSSE for {model_name}: {e}")
                    rmsse_score = None
                
                # Calculate validation data for the table (completed months only for metrics)
                validation_data = []
                val_df = val_ts.to_dataframe()
                forecast_df = forecast.to_dataframe()
                
                for i in range(len(val_df)):
                    val_date = val_df.index[i]
                    forecast_value = forecast_df.iloc[i]['cve_count']
                    actual_value = val_df.iloc[i]['cve_count']
                    predicted_value = max(0, round(forecast_value))
                    
                    # Calculate percent error, handling perfect predictions
                    if actual_value == predicted_value:
                        percent_error = 0.0
                    else:
                        percent_error = round(abs((actual_value - predicted_value) / max(actual_value, 1)) * 100, 2)
                    
                    validation_data.append({
                        'date': val_date.strftime('%Y-%m'),
                        'actual': int(actual_value),
                        'predicted': predicted_value,
                        'error': int(predicted_value - actual_value),  # Predicted - Actual (positive = overestimate, negative = underestimate)
                        'percent_error': percent_error,
                        'is_current_month': False  # Mark as complete month used in validation
                    })
                
                # Add current month for display purposes (if it exists and wasn't included in validation)
                # This is NOT used in MAPE calculations, only for display
                if current_month_index is not None:
                    current_month_ts = ts[current_month_index:current_month_index+1]
                    if len(current_month_ts) > 0:
                        # Generate forecast for current month using the trained model
                        current_forecast = model.predict(1)
                        current_df = current_month_ts.to_dataframe()
                        current_forecast_df = current_forecast.to_dataframe()
                        
                        current_date = current_df.index[0]
                        current_forecast_value = current_forecast_df.iloc[0]['cve_count']
                        current_actual_value = current_df.iloc[0]['cve_count']
                        current_predicted_value = max(0, round(current_forecast_value))
                        
                        # Calculate percent error, handling perfect predictions
                        if current_actual_value == current_predicted_value:
                            current_percent_error = 0.0
                        else:
                            current_percent_error = round(abs((current_actual_value - current_predicted_value) / max(current_actual_value, 1)) * 100, 2)
                        
                        validation_data.append({
                            'date': current_date.strftime('%Y-%m'),
                            'actual': int(current_actual_value),
                            'predicted': current_predicted_value,
                            'error': int(current_predicted_value - current_actual_value),
                            'percent_error': current_percent_error,
                            'is_current_month': True  # Mark as current partial month (NOT used in validation metrics)
                        })
                
                results.append({
                    'model_name': model_name,
                    'mape': mape_score,
                    'mase': mase_score,
                    'rmsse': rmsse_score,
                    'mae': mae_score,
                    'model': model,
                    'validation_data': validation_data
                })
                
                mase_str = f"{mase_score:.4f}" if mase_score is not None else "N/A"
                rmsse_str = f"{rmsse_score:.4f}" if rmsse_score is not None else "N/A"
                print(f"{model_name} - MAPE: {mape_score:.4f}, MASE: {mase_str}, RMSSE: {rmsse_str}, MAE: {mae_score:.4f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # Sort by MAPE (lower is better)
        results.sort(key=lambda x: x['mape'])
        
        print("\nModel Rankings:")
        for i, result in enumerate(results):
            mase_str = f"{result['mase']:.4f}" if result['mase'] is not None else "N/A"
            rmsse_str = f"{result['rmsse']:.4f}" if result['rmsse'] is not None else "N/A"
            print(f"{i+1}. {result['model_name']}: MAPE = {result['mape']:.4f}, MASE = {mase_str}, RMSSE = {rmsse_str}")
        
        return results
    
    def generate_forecasts(self, data: pd.DataFrame, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate monthly forecasts including the current month and remainder of current year and all of next year"""
        print("Generating monthly forecasts including current month through end of next year...")
        
        # Convert to Darts TimeSeries with monthly frequency
        ts = TimeSeries.from_dataframe(
            data,
            time_col='date',
            value_cols='cve_count',
            freq='MS',  # Month Start frequency
            fill_missing_dates=True
        )
        
        # Calculate months until end of current year + January 2026 (needed for chart display)
        current_date = datetime.now().date()
        current_month = datetime(current_date.year, current_date.month, 1).date()
        # Extend through January 2026 to support extended chart timeline
        end_forecast_date = datetime(current_date.year + 1, 1, 1).date()
        
        # For forecasting, we want to predict the current month (since it's incomplete) and all future months
        # Use data up to the previous month for training to predict current and future months
        previous_month_date = current_month - timedelta(days=1)  # Get last day of previous month
        previous_month = datetime(previous_month_date.year, previous_month_date.month, 1).date()
        
        # Find the data cutoff point - train on data up to previous month
        train_data = data[data['date'] < pd.to_datetime(current_month)]
        
        # Create training time series
        train_ts = TimeSeries.from_dataframe(
            train_data,
            time_col='date',
            value_cols='cve_count',
            freq='MS',
            fill_missing_dates=True
        )
        
        # Calculate number of months to forecast (current month through January 2026)
        months_to_forecast = ((end_forecast_date.year - current_month.year) * 12 + 
                             end_forecast_date.month - current_month.month + 1)
        
        print(f"Training on data through {previous_month.strftime('%Y-%m')}")
        print(f"Forecasting {months_to_forecast} months starting from {current_month.strftime('%Y-%m')} (through {end_forecast_date.strftime('%Y-%m')})")
        
        # Use top 5 models
        top_models = model_results[:5]
        forecasts = {}
        
        for result in top_models:
            model_name = result['model_name']
            model = result['model']
            
            # DEBUG: Check if models are unique objects
            print(f"🔍 DEBUG: {model_name} model object ID: {id(model)}")
            print(f"🔍 DEBUG: {model_name} model type: {type(model).__name__}")
            
            try:
                # Train on data up to previous month (excluding incomplete current month)
                model.fit(train_ts)
                
                # Generate forecast starting from current month
                forecast = model.predict(months_to_forecast)
                
                # Convert to list of dictionaries
                forecast_df = forecast.to_dataframe().reset_index()
                forecast_data = []
                
                for _, row in forecast_df.iterrows():
                    forecast_data.append({
                        'date': row['date'].strftime('%Y-%m'),
                        'cve_count': max(0, round(row['cve_count']))  # Ensure non-negative integers
                    })
                
                forecasts[model_name] = forecast_data
                
                # DEBUG: Show first few forecast values to check for differences
                first_3_values = [item['cve_count'] for item in forecast_data[:3]]
                print(f"🔍 DEBUG: {model_name} first 3 forecast values: {first_3_values}")
                
                # Debug output to see what dates were generated
                forecast_dates = [item['date'] for item in forecast_data]
                print(f"Generated forecast for {model_name}: {forecast_dates}")
                
                # Check specifically for January 2026
                jan_2026_found = any('2026-01' in date for date in forecast_dates)
                print(f"  January 2026 included: {jan_2026_found}")
                
            except Exception as e:
                print(f"Error generating forecast for {model_name}: {e}")
                continue
        
        return forecasts
    
    def save_data_file(self, data: pd.DataFrame, model_results: List[Dict[str, Any]], 
                      forecasts: Dict[str, Any], output_path: str):
        """Save data file for web interface"""
        print("Saving data file...")
        
        # Prepare historical data (exclude current month since it's being forecasted)
        current_date = datetime.now().date()
        current_month = datetime(current_date.year, current_date.month, 1).date()
        
        historical_data = []
        for _, row in data.iterrows():
            row_date = row['date'].date() if hasattr(row['date'], 'date') else pd.to_datetime(row['date']).date()
            if row_date < current_month:  # Only include complete months in historical data
                historical_data.append({
                    'date': row['date'].strftime('%Y-%m'),
                    'cve_count': int(row['cve_count'])
                })
        
        # Prepare model rankings with comprehensive metrics
        rankings = []
        for result in model_results:
            rankings.append({
                'model_name': result['model_name'],
                'mape': round(result['mape'], 4),
                'mase': round(result['mase'], 4) if result['mase'] is not None else None,
                'rmsse': round(result['rmsse'], 4) if result['rmsse'] is not None else None,
                'mae': round(result['mae'], 4)
            })
        
        # Prepare validation data for the best model
        # Prepare validation data for all models
        all_models_validation = {}
        for result in model_results:
            all_models_validation[result['model_name']] = result['validation_data']
        
        best_model_validation = model_results[0]['validation_data'] if model_results else []
        
        # Prepare current month progress data (will calculate cumulative after timelines are generated)
        current_date = datetime.now().date()
        current_month = datetime(current_date.year, current_date.month, 1).date()
        current_month_data = None
        current_month_raw_data = None
        
        # Find current month's raw data first
        for _, row in data.iterrows():
            row_date = row['date'].date() if hasattr(row['date'], 'date') else pd.to_datetime(row['date']).date()
            if row_date.year == current_month.year and row_date.month == current_month.month:
                current_month_raw_data = {
                    'date': row['date'].strftime('%Y-%m'),
                    'cve_count': int(row['cve_count']),
                    'days_elapsed': current_date.day,
                    'total_days': (datetime(current_date.year, current_date.month + 1, 1) - timedelta(days=1)).day if current_date.month < 12 else 31,
                    'progress_percentage': round((current_date.day / ((datetime(current_date.year, current_date.month + 1, 1) - timedelta(days=1)).day if current_date.month < 12 else 31)) * 100, 1)
                }
                break
        
        # Handle year boundary for total_days calculation
        if current_month_data and current_date.month == 12:
            next_month = datetime(current_date.year + 1, 1, 1)
        else:
            next_month = datetime(current_date.year, current_date.month + 1, 1) if current_date.month < 12 else datetime(current_date.year + 1, 1, 1)
        
        days_in_month = (next_month - datetime(current_date.year, current_date.month, 1)).days
        if current_month_data:
            current_month_data['total_days'] = days_in_month
            current_month_data['progress_percentage'] = round((current_date.day / days_in_month) * 100, 1)
        
        # Calculate yearly forecast totals for each model (backend single source of truth)
        yearly_forecast_totals = {}
        
        # Step 1: Sum only 2025 historical data for yearly forecast totals
        # (historical_data contains all data from 1999+, but yearly totals should only include 2025)
        historical_total = sum(item['cve_count'] for item in historical_data 
                             if item['date'].startswith('2025'))
        
        # Step 2: Calculate individual model totals
        for model_name, model_forecasts in forecasts.items():
            model_forecast_total = 0
            for forecast in model_forecasts:
                # Include current month and future months within 2025 only (2025-only scope)
                forecast_date = datetime.strptime(forecast['date'], '%Y-%m')
                if forecast_date.year == 2025 and forecast_date.month >= current_date.month:
                    model_forecast_total += forecast['cve_count']
                    print(f"DEBUG: yearly_forecast_totals - Including {forecast['date']}: {forecast['cve_count']} CVEs for {model_name}")
            
            # Store complete yearly total: historical + future forecasts
            yearly_forecast_totals[model_name] = round(historical_total + model_forecast_total)
        
        # Step 3: Calculate "All Models" average
        if yearly_forecast_totals:
            all_models_average = sum(yearly_forecast_totals.values()) / len(yearly_forecast_totals)
            yearly_forecast_totals['all_models_average'] = round(all_models_average)
        
        # Step 4: Identify best model and store its total
        if rankings:
            best_model_name = rankings[0]['model_name']
            yearly_forecast_totals['best_model_total'] = yearly_forecast_totals.get(best_model_name, 0)
            yearly_forecast_totals['best_model_name'] = best_model_name
        
        print(f"📊 Yearly forecast totals calculated: {yearly_forecast_totals}")
        
        # Calculate complete cumulative timelines for each model (eliminates all JavaScript calculations)
        cumulative_timelines = {}
        
        # Create comprehensive timeline for cumulative calculations (Jan 2025 - Dec 2025 + Jan 2026 endpoint)
        timeline_months = []
        for month in range(1, 13):  # January to December 2025
            timeline_months.append(f"2025-{month:02d}")
        timeline_months.append("2026-01")  # January 2026 endpoint to show full 2025 cumulative total
        
        # CRITICAL FIX: Calculate cumulative timeline correctly 
        # Each month shows cumulative total of CVEs published BEFORE that month starts
        for model_name in forecasts.keys():
            model_timeline = []
            
            print(f"🔍 DEBUG: Processing cumulative timeline for {model_name}")
            print(f"🔍 DEBUG: {model_name} forecast data: {forecasts[model_name][:3]}")
            
            for i, month_date in enumerate(timeline_months):
                cumulative_total = 0
                
                # Calculate cumulative total of all data published BEFORE this month starts
                for j in range(i):  # All months BEFORE current month (restore original logic)
                    prev_month_date = timeline_months[j]
                    prev_month_num = int(prev_month_date.split('-')[1])
                    prev_year = int(prev_month_date.split('-')[0])
                    
                    # Add historical data for past months
                    if prev_year == current_date.year and prev_month_num < current_date.month:
                        historical_month = next((item for item in historical_data if item['date'] == prev_month_date), None)
                        if historical_month:
                            cumulative_total += historical_month['cve_count']
                            if month_date == '2025-08':  # Debug August calculation (should include July forecast)
                                print(f"🔍 DEBUG: {model_name} Aug calc - added historical {prev_month_date}: {historical_month['cve_count']}")
                    
                    # Add forecast data for current and future months that are BEFORE this timeline month
                    elif (prev_year == current_date.year and prev_month_num >= current_date.month) or prev_year > current_date.year:
                        model_forecasts = forecasts[model_name]
                        forecast_month = next((item for item in model_forecasts if item['date'] == prev_month_date), None)
                        if forecast_month:
                            cumulative_total += forecast_month['cve_count']
                            if month_date == '2025-08':  # Debug August calculation (should include July forecast)
                                print(f"🔍 DEBUG: {model_name} Aug calc - added forecast {prev_month_date}: {forecast_month['cve_count']}")
                
                if month_date == '2025-07':  # Focus on July for debugging
                    print(f"🔍 DEBUG: {model_name} {month_date} FINAL cumulative_total: {cumulative_total:,}")
                
                # Store cumulative total for this month (total published BEFORE this month)
                model_timeline.append({
                    'date': month_date,
                    'cumulative_total': round(cumulative_total)
                })
            
            cumulative_timelines[f"{model_name}_cumulative"] = model_timeline
        
        # Calculate "All Models" average cumulative timeline
        all_models_timeline = []
        for i, month_date in enumerate(timeline_months):
            month_totals = []
            for model_name in forecasts.keys():
                model_timeline = cumulative_timelines[f"{model_name}_cumulative"]
                month_totals.append(model_timeline[i]['cumulative_total'])
            
            average_total = sum(month_totals) / len(month_totals) if month_totals else 0
            all_models_timeline.append({
                'date': month_date,
                'cumulative_total': round(average_total)
            })
        
        cumulative_timelines['all_models_cumulative'] = all_models_timeline
        
        print(f"📈 Cumulative timelines calculated for {len(cumulative_timelines)} datasets")
        
        # Calculate current month data using cumulative timeline (after timelines are generated)
        if current_month_raw_data:
            current_month_str = current_month_raw_data['date']
            current_month_progress = current_month_raw_data['cve_count']
            
            # Calculate cumulative total at start of current month using ONLY data published BEFORE current month
            # For July, this should be Jan+Feb+Mar+Apr+May (20,353), NOT including June
            cumulative_at_month_start = 0
            current_month_num = int(current_month_str.split('-')[1])  # Extract month number
            
            for item in historical_data:
                item_month_num = int(item['date'].split('-')[1])  # Extract month number
                # Only include months that are strictly before the current month
                if item['date'].startswith('2025') and item_month_num < current_month_num:
                    cumulative_at_month_start += item['cve_count']
                    print(f"📊 DEBUG: Including {item['date']}: {item['cve_count']:,} CVEs")
            
            # Final cumulative = cumulative at month start + current month progress
            cumulative_total = cumulative_at_month_start + current_month_progress
            
            print(f"🔍 DEBUG: Cumulative at month start: {cumulative_at_month_start:,}")
            print(f"🔍 DEBUG: Current month progress: {current_month_progress:,}")
            print(f"🔍 DEBUG: Final cumulative total: {cumulative_total:,}")
            
            # Create final current_month_data with cumulative total
            current_month_data = {
                'date': current_month_str,
                'cve_count': current_month_progress,  # Monthly progress count (for display)
                'cumulative_total': cumulative_total,  # Cumulative total (for chart)
                'days_elapsed': current_month_raw_data['days_elapsed'],
                'total_days': current_month_raw_data['total_days'],
                'progress_percentage': current_month_raw_data['progress_percentage']
            }
            print(f"📊 Current month cumulative total: {cumulative_total:,} CVEs (monthly: {current_month_progress:,})")
        
        # Prepare output data
        output_data = {
            'generated_at': datetime.now().isoformat(),
            'model_rankings': rankings,
            'historical_data': historical_data,
            'current_month_actual': current_month_data,
            'forecasts': forecasts,
            'yearly_forecast_totals': yearly_forecast_totals,
            'cumulative_timelines': cumulative_timelines,
            'validation_against_actuals': best_model_validation,
            'all_models_validation': all_models_validation,
            'summary': {
                'total_historical_cves': int(data['cve_count'].sum()),
                'data_period': {
                    'start': data['date'].min().strftime('%Y-%m-%d'),
                    'end': data['date'].max().strftime('%Y-%m-%d')
                },
                'forecast_period': {
                    'start': datetime(datetime.now().year, datetime.now().month, 1).strftime('%Y-%m-%d'),
                    'end': datetime(datetime.now().year + 1, 1, 31).strftime('%Y-%m-%d')
                }
            }
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save data file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Data file saved to {output_path}")
    
    def run(self, output_path: str = "web/data.json"):
        """Main execution method"""
        try:
            # Step 1: Parse CVE data
            data = self.parse_cve_data()
            
            # Step 2: Evaluate models
            model_results = self.evaluate_models(data)
            
            # Step 3: Generate forecasts
            forecasts = self.generate_forecasts(data, model_results)
            
            # Step 4: Save data file
            self.save_data_file(data, model_results, forecasts, output_path)
            
            print("\nCVE Forecast generation completed successfully!")
            
        except Exception as e:
            print(f"Error in CVE forecast generation: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate CVE forecasts from data')
    parser.add_argument('--data-path', 
                      help='Path to the CVE data directory (default: cvelistV5 in project root)')
    parser.add_argument('--output', default='web/data.json',
                      help='Output path for the generated data file (default: web/data.json)')
    
    args = parser.parse_args()
    
    engine = CVEForecastEngine(data_path=args.data_path)
    engine.run(output_path=args.output)


if __name__ == "__main__":
    main()
