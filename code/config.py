#!/usr/bin/env python3
"""
Configuration module for CVE Forecast application.
Contains all constants, configuration variables, and model imports.
"""

import warnings
from typing import Dict, List, Any, Optional

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

# File paths and directories (relative to code directory)
DEFAULT_CVE_DATA_PATH = "../cvelistV5"
DEFAULT_OUTPUT_PATH = "../web/data.json"
DEFAULT_PERFORMANCE_HISTORY_PATH = "../web/performance_history.json"

# Model availability flags - will be set during import attempts
STATSFORECAST_ARIMA_AVAILABLE = False
STATSFORECAST_ETS_AVAILABLE = False  
STATSFORECAST_THETA_AVAILABLE = False
STATSFORECAST_CES_AVAILABLE = False
STATSFORECAST_MFLES_AVAILABLE = False
STATSFORECAST_TBATS_AVAILABLE = False

# Core Darts imports
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

from darts.metrics import mape, rmse, mae, mase, rmsse
from darts.utils.utils import ModelMode, SeasonalityMode

# Try to import StatsForecast models with fallback handling
def _try_import_statsforecast_models():
    """Attempt to import StatsForecast models and set availability flags."""
    global STATSFORECAST_ARIMA_AVAILABLE, STATSFORECAST_ETS_AVAILABLE
    global STATSFORECAST_THETA_AVAILABLE, STATSFORECAST_CES_AVAILABLE
    global STATSFORECAST_MFLES_AVAILABLE, STATSFORECAST_TBATS_AVAILABLE
    
    global StatsForecastAutoARIMA, StatsForecastAutoETS, StatsForecastAutoTheta
    global StatsForecastAutoCES, StatsForecastAutoMFLES, StatsForecastAutoTBATS
    
    # StatsForecast AutoARIMA
    try:
        from darts.models.forecasting.sf_auto_arima import StatsForecastAutoARIMA
        STATSFORECAST_ARIMA_AVAILABLE = True
    except ImportError:
        try:
            from darts.models import StatsForecastAutoARIMA
            STATSFORECAST_ARIMA_AVAILABLE = True
        except ImportError:
            StatsForecastAutoARIMA = None

    # StatsForecast AutoETS
    try:
        from darts.models.forecasting.sf_auto_ets import StatsForecastAutoETS
        STATSFORECAST_ETS_AVAILABLE = True
    except ImportError:
        try:
            from darts.models import StatsForecastAutoETS
            STATSFORECAST_ETS_AVAILABLE = True
        except ImportError:
            StatsForecastAutoETS = None

    # StatsForecast AutoTheta
    try:
        from darts.models.forecasting.sf_auto_theta import StatsForecastAutoTheta
        STATSFORECAST_THETA_AVAILABLE = True
    except ImportError:
        try:
            from darts.models import StatsForecastAutoTheta
            STATSFORECAST_THETA_AVAILABLE = True
        except ImportError:
            StatsForecastAutoTheta = None

    # StatsForecast AutoCES
    try:
        from darts.models.forecasting.sf_auto_ces import StatsForecastAutoCES
        STATSFORECAST_CES_AVAILABLE = True
    except ImportError:
        try:
            from darts.models import StatsForecastAutoCES
            STATSFORECAST_CES_AVAILABLE = True
        except ImportError:
            StatsForecastAutoCES = None

    # StatsForecast AutoMFLES
    try:
        from darts.models.forecasting.sf_auto_mfles import StatsForecastAutoMFLES
        STATSFORECAST_MFLES_AVAILABLE = True
    except ImportError:
        try:
            from darts.models import StatsForecastAutoMFLES
            STATSFORECAST_MFLES_AVAILABLE = True
        except ImportError:
            StatsForecastAutoMFLES = None

    # StatsForecast AutoTBATS
    try:
        from darts.models.forecasting.sf_auto_tbats import StatsForecastAutoTBATS
        STATSFORECAST_TBATS_AVAILABLE = True
    except ImportError:
        try:
            from darts.models import StatsForecastAutoTBATS
            STATSFORECAST_TBATS_AVAILABLE = True
        except ImportError:
            StatsForecastAutoTBATS = None

# Initialize StatsForecast imports
_try_import_statsforecast_models()

# Model configuration constants
MODEL_EVALUATION_SPLIT = 0.8  # 80% for training, 20% for validation
FORECAST_HORIZON_MONTHS = 12  # Number of months to forecast
ENSEMBLE_SIZE = 5  # Number of top models for ensemble

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
