"""
Forecast Adapters - Specific implementations for different forecast types.
"""

from .cna_adapter import CNAForecaster
from .cve_adapter import CVEForecaster

__all__ = ['CVEForecaster', 'CNAForecaster']
