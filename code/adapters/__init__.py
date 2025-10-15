"""
Forecast Adapters - Specific implementations for different forecast types.
"""

from .cve_adapter import CVEForecaster
from .cna_adapter import CNAForecaster

__all__ = ['CVEForecaster', 'CNAForecaster']
