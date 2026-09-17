"""Shared test fixtures for CVEForecast test suite."""

import sys
from pathlib import Path

import pytest

# Add code directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))


@pytest.fixture
def sample_config():
    """Minimal config for testing."""
    return {
        'models': {'ExponentialSmoothing': {'enabled': True, 'hyperparameters': {'damped_trend': True}}},
        'forecast_constraints': {
            'max_annual_growth': 4.0,
            'min_annual_growth': 0.25,
            'max_monthly_spike': 4.0,
            'enable_sanity_guards': True,
        },
        'forecasting': {
            'log_space': True,
            'business_day_normalise': True,
            'damping_phi': 0.98,
            'training_window_months': None,
            'use_future_covariates': False,
        },
    }
