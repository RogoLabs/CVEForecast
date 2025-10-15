"""
Core forecasting architecture - Base classes and interfaces.
"""

from .base_forecaster import BaseForecaster, ForecastResult
from .data_adapter import DataAdapter
from .validation_mixin import ValidationMixin

__all__ = [
    'BaseForecaster',
    'ForecastResult',
    'DataAdapter',
    'ValidationMixin'
]
