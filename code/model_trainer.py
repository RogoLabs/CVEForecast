import logging
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import (
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
        return None, None

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
        if model_name in ["Theta", "FourTheta"] and 'season_mode' in params:
            params['season_mode'] = SeasonalityMode[params['season_mode']]
        
        # Instantiate model
        model = model_class(**params)

        # Train model
        model.fit(train)

        # Make predictions
        predictions = model.predict(len(val))

        # Evaluate model
        metrics = {}
        for metric_name, metric_func in [('mape', mape), ('mae', mae), ('mase', mase), ('rmsse', rmsse)]:
            try:
                # MASE and RMSSE require the training series
                if metric_name in ['mase', 'rmsse']:
                    metrics[metric_name] = metric_func(val, train, predictions)
                else:
                    metrics[metric_name] = metric_func(val, predictions)
            except Exception as e:
                logger.warning(f"Could not compute {metric_name.upper()} for {model_name}: {e}")
                metrics[metric_name] = None

        mape_str = f"{metrics.get('mape', 0):.2f}%" if metrics.get('mape') is not None else "N/A"
        mase_str = f"{metrics.get('mase', 0):.2f}" if metrics.get('mase') is not None else "N/A"
        rmsse_str = f"{metrics.get('rmsse', 0):.2f}" if metrics.get('rmsse') is not None else "N/A"
        mae_str = f"{metrics.get('mae', 0):.2f}" if metrics.get('mae') is not None else "N/A"
        logger.info(f"{model_name} - MAPE: {mape_str}, MASE: {mase_str}, RMSSE: {rmsse_str}, MAE: {mae_str}")

        return model, metrics

    except Exception as e:
        logger.error(f"Failed to train or evaluate {model_name}: {e}", exc_info=True)
        return None, None
