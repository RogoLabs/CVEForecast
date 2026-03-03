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
        'models': {
            'ExponentialSmoothing': {
                'enabled': True,
                'hyperparameters': {'damped_trend': True}
            }
        },
        'forecast_constraints': {
            'min_annual_growth_rate': 0.05,
            'max_annual_growth_rate': 0.40,
            'historical_avg_growth': 0.18,
            'enable_growth_floor': True,
            'enable_trend_adjustment': True,
            'enable_ytd_floor': True,
            'trend_adjustment_confidence': 0.7,
            'trend_adjustment_threshold': 0.75,
            'ytd_minimum_factor': 0.85,
        }
    }
