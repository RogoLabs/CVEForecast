import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from darts import TimeSeries
from darts.metrics import mape, mase, rmsse, mae
from darts.utils.utils import SeasonalityMode

# Statistical Models
from darts.models import (
    # Exponential Smoothing
    ExponentialSmoothing,
    # Prophet
    Prophet,
    # ARIMA
    AutoARIMA,
    # Theta
    Theta,
    FourTheta,
    # TBATS (optional)
    TBATS,
    # Croston
    Croston,
    # Kalman Filter
    KalmanForecaster
)

# Tree-based Models
from darts.models import (
    XGBModel,
    LightGBMModel,
    CatBoostModel,
    RandomForestModel,
    LinearRegressionModel
)

# Deep Learning Models
from darts.models import (
    TCNModel,
    NBEATSModel,
    NHiTSModel,
    TiDEModel,
    DLinearModel
)

# Baseline Models
from darts.models.forecasting.baselines import (
    NaiveMean,
    NaiveDrift,
    NaiveSeasonal,
    NaiveEnsembleModel
)

# Handle optional imports
try:
    from darts.models import TBATS
except ImportError:
    try:
        from darts.models.forecasting.tbats import BATS as TBATS
    except ImportError:
        TBATS = None

try:
    from darts.models import LightGBMModel
except ImportError:
    try:
        from darts.models.forecasting.lightgbm import LightGBMModel
    except ImportError:
        LightGBMModel = None

logger = logging.getLogger(__name__)

# Create model mapping with all available models
MODEL_MAPPING = {
    # Statistical Models
    "Prophet": Prophet,
    "ExponentialSmoothing": ExponentialSmoothing,
    "AutoARIMA": AutoARIMA,
    "Theta": Theta,
    "FourTheta": FourTheta,
    "Croston": Croston,
    "KalmanFilter": KalmanForecaster,
    
    # Tree-based Models
    "XGBoost": XGBModel,
    "CatBoost": CatBoostModel,
    "RandomForest": RandomForestModel,
    "LinearRegression": LinearRegressionModel,
    
    # Deep Learning Models
    "TCN": TCNModel,
    "NBEATS": NBEATSModel,
    "NHiTS": NHiTSModel,
    "TiDE": TiDEModel,
    "DLinear": DLinearModel,
    
    # Baseline Models
    "NaiveMean": NaiveMean,
    "NaiveDrift": NaiveDrift,
    "NaiveSeasonal": NaiveSeasonal,
    "NaiveEnsemble": NaiveEnsembleModel,
}

# Add optional models if available
if TBATS is not None:
    MODEL_MAPPING["TBATS"] = TBATS

if LightGBMModel is not None:
    MODEL_MAPPING["LightGBM"] = LightGBMModel

# Log available models
available_models = [name for name in MODEL_MAPPING.keys() if MODEL_MAPPING[name] is not None]
logger.info(f"Available models: {', '.join(available_models)}")

# Log any missing models
missing_models = [name for name, model in MODEL_MAPPING.items() if model is None]
if missing_models:
    logger.warning(f"The following models are not available: {', '.join(missing_models)}")

def train_and_evaluate_model(model_name: str, model_config: dict, series: TimeSeries, eval_config: dict):
    """
    Trains and evaluates a single forecasting model.

    Args:
        model_name: The name of the model to train.
        model_config: The configuration for the model, including hyperparameters.
        series: The full time series data.
        eval_config: The evaluation configuration, including split ratio.

    Returns:
        A tuple containing the trained model and its MAPE score, or (None, None) if it fails.
    """
    logger.info(f"-- Training and evaluating {model_name} --")
    model_class = MODEL_MAPPING.get(model_name)
    if not model_class:
        logger.warning(f"Model {model_name} not found in mapping. Skipping.")
        return None, None, None

    try:
        # Split data
        split_point = int(eval_config['split_ratio'] * len(series))
        train, val = series[:split_point], series[split_point:]

        # Convert to float32 for DL models
        if model_name in ["TCN", "NBEATS", "NHiTS", "TiDE", "DLinear"]:
            train = train.astype(np.float32)
            val = val.astype(np.float32)

        # Handle special cases for hyperparameters
        params = model_config['hyperparameters'].copy()
        
        # Pre-process time series data for better numerical stability
        if model_name in ['KalmanFilter']:
            # Scale the time series to a more stable range
            train_mean = train.pd_dataframe().mean().values[0]
            train_std = train.pd_dataframe().std().values[0]
            train_std = max(train_std, 1e-6)  # Avoid division by zero
            train_scaled = (train - train_mean) / train_std
            val_scaled = (val - train_mean) / train_std
        else:
            train_scaled, val_scaled = train, val
            
        if model_name == 'ExponentialSmoothing':
            if 'trend' in params:
                # Convert string to proper trend component
                trend = params.pop('trend')
                if trend == 'add':
                    params['trend'] = True
                    params['damped_trend'] = params.get('damped_trend', False)
                elif trend == 'mul':
                    params['trend'] = True
                    params['damped_trend'] = params.get('damped_trend', False)
                    params['trend_mode'] = 'multiplicative'
                else:  # None or False
                    params['trend'] = False
            
            if 'seasonal' in params:
                seasonal = params.pop('seasonal')
                if seasonal.lower() == 'add':
                    params['seasonal'] = 12  # 12 months for monthly data
                    params['seasonal_mode'] = 'additive'
                elif seasonal.lower() == 'mul':
                    params['seasonal'] = 12  # 12 months for monthly data
                    params['seasonal_mode'] = 'multiplicative'
                else:  # None or False
                    params['seasonal'] = None
        
        elif model_name in ["Theta", "FourTheta"] and 'season_mode' in params:
            season_mode = params['season_mode'].lower()
            if season_mode == 'add':
                params['season_mode'] = SeasonalityMode.ADDITIVE
            elif season_mode == 'mul':
                params['season_mode'] = SeasonalityMode.MULTIPLICATIVE
            else:
                params['season_mode'] = SeasonalityMode.NONE
        
        # Instantiate model
        model = model_class(**params)

        try:
            # Train model on scaled data if needed
            train_to_use = train_scaled if model_name in ['KalmanFilter'] else train
            val_to_use = val_scaled if model_name in ['KalmanFilter'] else val
            
            # Add additional validation for KalmanFilter
            if model_name == 'KalmanFilter':
                # Ensure data doesn't contain NaNs or Infs
                train_df = train_to_use.pd_dataframe()
                if train_df.isnull().any().any() or np.isinf(train_df).any().any():
                    logger.warning(f"{model_name} received data with NaNs or Infs. Filling with zeros.")
                    train_df = train_df.fillna(0).replace([np.inf, -np.inf], 0)
                    train_to_use = TimeSeries.from_dataframe(train_df)
            
            # Train model
            model.fit(train_to_use)

            # Make predictions
            predictions = model.predict(len(val_to_use))
            
            # Rescale predictions back to original scale if needed
            if model_name in ['KalmanFilter']:
                predictions = (predictions * train_std) + train_mean
                predictions = TimeSeries.from_times_and_values(
                    predictions.time_index,
                    np.maximum(0, predictions.values())  # Ensure non-negative predictions
                )

            return model, train, val, predictions
            
        except Exception as e:
            logger.error(f"Error during {model_name} training/prediction: {str(e)}")
            return None, None, None, None

    except Exception as e:
        logger.error(f"Failed to train or evaluate {model_name}: {e}", exc_info=True)
        return None, None, None, None
