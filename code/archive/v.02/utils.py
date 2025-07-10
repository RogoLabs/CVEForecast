#!/usr/bin/env python3
"""
Utility functions for the CVE Forecast application.
Contains logging setup and common helper functions.
"""

import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from config import LOG_FORMAT, LOG_LEVEL


def setup_logging(level: str = LOG_LEVEL) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (default from config)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger('cve_forecast')


def ensure_directory_exists(path: str) -> None:
    """
    Ensure that the directory for a given file path exists.
    
    Args:
        path: File path for which to ensure directory exists
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def get_model_category(model_name: str) -> str:
    """
    Categorize model type for better organization.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Category string for the model
    """
    model_name_lower = model_name.lower()
    
    # Tree-based models
    if any(keyword in model_name_lower for keyword in ['xgb', 'lightgbm', 'catboost', 'randomforest']):
        return 'tree_based'
    
    # Statistical models
    elif any(keyword in model_name_lower for keyword in 
             ['arima', 'ets', 'theta', 'prophet', 'exponential', 'tbats', 'croston', 'fft']):
        return 'statistical'
    
    # Deep learning models
    elif any(keyword in model_name_lower for keyword in 
             ['tcn', 'tft', 'nbeats', 'nhits', 'transformer', 'rnn', 'tide', 'linear', 'mixer']):
        return 'deep_learning'
    
    # Ensemble models
    elif 'ensemble' in model_name_lower:
        return 'ensemble'
    
    # Naive/baseline models
    elif any(keyword in model_name_lower for keyword in ['naive', 'mean', 'drift', 'kalman']):
        return 'baseline'
    
    # Regression models
    elif 'regression' in model_name_lower:
        return 'regression'
    
    else:
        return 'other'


def validate_date_format(date_str: str) -> bool:
    """
    Validate if a string is in YYYY-MM-DD format.
    
    Args:
        date_str: String to validate
        
    Returns:
        True if valid date format, False otherwise
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value to format
        decimal_places: Number of decimal places to include
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value  
        default: Default value to return if division by zero
        
    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length with optional suffix.
    
    Args:
        text: String to truncate
        max_length: Maximum length of the string
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_current_year_month() -> tuple[int, int]:
    """
    Get the current year and month.
    
    Returns:
        Tuple of (year, month)
    """
    now = datetime.now()
    return now.year, now.month


def is_file_readable(file_path: str) -> bool:
    """
    Check if a file exists and is readable.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if file is readable, False otherwise
    """
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)
