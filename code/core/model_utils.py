"""
Shared model utilities for parameter fixing and safe model creation.

Extracted from cve_adapter.py and cna_adapter.py to eliminate duplication.
"""

import logging
from typing import Dict, Any, Optional


def fix_hyperparameters(model_name: str, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix known hyperparameter compatibility issues for Darts models.

    Returns a new dict with corrected parameters (never modifies the original).

    Handles:
    - ExponentialSmoothing: damped_trend -> damping_trend conversion,
      removal of unsupported initialization_method and missing params
    - Theta/FourTheta: season_mode string -> SeasonalityMode enum
    - LinearRegression: clamp output_chunk_shift to 0 if > 0

    Args:
        model_name: Name of the model (e.g. 'ExponentialSmoothing', 'Theta')
        hyperparameters: Original hyperparameters dict

    Returns:
        New dict with fixed hyperparameters
    """
    params = hyperparameters.copy()

    if model_name == 'ExponentialSmoothing':
        # Fix parameter name changes in newer Darts versions
        if 'damped_trend' in params:
            val = params.pop('damped_trend')
            # damping_trend must be float (0.0-1.0) or None
            # If False/0, set to None (no damping)
            # If True/1, use 0.98 as default damping
            # If float, use as-is
            if val is None or val is False or val == 0:
                params['damping_trend'] = None
            elif val is True or val == 1:
                params['damping_trend'] = 0.98
            elif isinstance(val, (int, float)):
                params['damping_trend'] = float(val)
            else:
                params['damping_trend'] = None

        # Ensure damping_trend is correct type if already exists
        if 'damping_trend' in params:
            val = params['damping_trend']
            if isinstance(val, bool):
                params['damping_trend'] = 0.98 if val else None
            elif val is not None:
                params['damping_trend'] = float(val) if val != 0 else None

        # Remove unsupported params
        params.pop('initialization_method', None)
        params.pop('missing', None)  # Not supported in current version

    if model_name in ['Theta', 'FourTheta']:
        # Fix season_mode: must be SeasonalityMode enum, not string
        if 'season_mode' in params:
            from darts.utils.utils import SeasonalityMode
            mode_str = str(params['season_mode']).lower()
            if mode_str in ['additive', 'add']:
                params['season_mode'] = SeasonalityMode.ADDITIVE
            elif mode_str in ['multiplicative', 'mult', 'mul']:
                params['season_mode'] = SeasonalityMode.MULTIPLICATIVE
            else:
                params.pop('season_mode')  # Remove invalid value

    if model_name == 'LinearRegression':
        # Fix incompatible parameter combination
        if params.get('output_chunk_shift', 0) > 0:
            # Can't use output_chunk_shift with auto-regression
            params['output_chunk_shift'] = 0

    return params


def create_model_safe(model_class, model_name: str, hyperparameters: Dict[str, Any],
                      logger: Optional[logging.Logger] = None):
    """
    Safely create a model instance with parameter fixing and fallback.

    1. Fixes hyperparameters using fix_hyperparameters()
    2. Attempts to create model with fixed params
    3. On failure, falls back to default params (no args)
    4. On total failure, returns None

    Args:
        model_class: The Darts model class to instantiate
        model_name: Name of the model (for fix_hyperparameters lookup)
        hyperparameters: Raw hyperparameters dict
        logger: Optional logger for warnings

    Returns:
        Model instance, or None if creation failed entirely
    """
    fixed_params = fix_hyperparameters(model_name, hyperparameters)

    try:
        return model_class(**fixed_params)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to create {model_name} with params, using defaults: {e}")
        try:
            return model_class()
        except Exception:
            return None
