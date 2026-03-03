"""Tests for core.model_utils shared utilities."""
import importlib
import sys
from pathlib import Path

import pytest

# Import fix_hyperparameters directly from the module file to avoid
# pulling in heavy dependencies through core/__init__.py
_spec = importlib.util.spec_from_file_location(
    'model_utils',
    Path(__file__).parent.parent / 'code' / 'core' / 'model_utils.py',
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fix_hyperparameters = _mod.fix_hyperparameters


class TestFixHyperparameters:
    def test_exponential_smoothing_damped_trend_true(self):
        result = fix_hyperparameters('ExponentialSmoothing', {'damped_trend': True})
        assert result['damping_trend'] == 0.98
        assert 'damped_trend' not in result

    def test_exponential_smoothing_damped_trend_false(self):
        result = fix_hyperparameters('ExponentialSmoothing', {'damped_trend': False})
        assert result['damping_trend'] is None

    def test_exponential_smoothing_damped_trend_none(self):
        result = fix_hyperparameters('ExponentialSmoothing', {'damped_trend': None})
        assert result['damping_trend'] is None

    def test_exponential_smoothing_damped_trend_float(self):
        result = fix_hyperparameters('ExponentialSmoothing', {'damped_trend': 0.85})
        assert result['damping_trend'] == 0.85

    def test_exponential_smoothing_damped_trend_zero(self):
        result = fix_hyperparameters('ExponentialSmoothing', {'damped_trend': 0})
        assert result['damping_trend'] is None

    def test_exponential_smoothing_removes_unsupported_params(self):
        result = fix_hyperparameters('ExponentialSmoothing', {
            'initialization_method': 'estimated',
            'missing': 'drop',
            'trend': 'add'
        })
        assert 'initialization_method' not in result
        assert 'missing' not in result
        assert result['trend'] == 'add'

    def test_linear_regression_clamps_output_chunk_shift(self):
        result = fix_hyperparameters('LinearRegression', {'output_chunk_shift': 5})
        assert result['output_chunk_shift'] == 0

    def test_linear_regression_zero_shift_unchanged(self):
        result = fix_hyperparameters('LinearRegression', {'output_chunk_shift': 0})
        assert result['output_chunk_shift'] == 0

    def test_unknown_model_returns_copy(self):
        params = {'foo': 'bar', 'baz': 42}
        result = fix_hyperparameters('UnknownModel', params)
        assert result == params
        assert result is not params

    def test_does_not_modify_original(self):
        original = {'damped_trend': True, 'trend': 'add'}
        fix_hyperparameters('ExponentialSmoothing', original)
        assert 'damped_trend' in original
        assert 'damping_trend' not in original

    def test_empty_params(self):
        result = fix_hyperparameters('ExponentialSmoothing', {})
        assert isinstance(result, dict)
