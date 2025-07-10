import logging
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import (
    ExponentialSmoothing,
    Prophet, ExponentialSmoothing, TBATS, XGBModel, LightGBMModel, CatBoostModel,
    TCNModel, NBEATSModel, NHiTSModel, TiDEModel, DLinearModel, TSMixerModel,
    RandomForest, LinearRegressionModel, AutoARIMA, Theta, FourTheta,
    KalmanForecaster, Croston, NaiveMean, NaiveDrift, NaiveSeasonal, NaiveEnsembleModel
)
from darts.utils.utils import SeasonalityMode
from darts.metrics import mape, mase, rmsse, mae

logger = logging.getLogger(__name__)

MODEL_MAPPING = {
    "Prophet": Prophet,
    "ExponentialSmoothing": ExponentialSmoothing,
    "TBATS": TBATS,
    "XGBoost": XGBModel,
    "LightGBM": LightGBMModel,
    "CatBoost": CatBoostModel,
    "TCN": TCNModel,
    "NBEATS": NBEATSModel,
    "NHiTS": NHiTSModel,
    "TiDE": TiDEModel,
    "DLinear": DLinearModel,
    "TSMixer": TSMixerModel,
    "RandomForest": RandomForest,
    "LinearRegression": LinearRegressionModel,
    "AutoARIMA": AutoARIMA,
    "Theta": Theta,
    "FourTheta": FourTheta,
    "KalmanFilter": KalmanForecaster,
    "Croston": Croston,
    "NaiveMean": NaiveMean,
    "NaiveDrift": NaiveDrift,
    "NaiveSeasonal": NaiveSeasonal,
    "NaiveEnsemble": NaiveEnsembleModel,
}

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
        if model_name in ["TCN", "NBEATS", "NHiTS", "TiDE", "DLinear", "TSMixer"]:
            train = train.astype(np.float32)
            val = val.astype(np.float32)

        # Handle special cases for hyperparameters
        params = model_config['hyperparameters'].copy()
        if model_name == 'ExponentialSmoothing':
            if 'trend' in params:
                params['trend'] = Trend[params['trend'].upper()]
            if 'seasonal' in params:
                params['seasonal'] = SeasonalityMode[params['seasonal'].upper()]
        elif model_name in ["Theta", "FourTheta"] and 'season_mode' in params:
            params['season_mode'] = SeasonalityMode[params['season_mode'].upper()]
        
        # Instantiate model
        model = model_class(**params)

        # Train model
        model.fit(train)

        # Make predictions
        predictions = model.predict(len(val))

        return model, train, val, predictions

    except Exception as e:
        logger.error(f"Failed to train or evaluate {model_name}: {e}", exc_info=True)
        return None, None, None, None
