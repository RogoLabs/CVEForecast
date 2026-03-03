"""
Data Adapter - Abstract interface for data loading and preparation.

Defines the contract for how different data sources (CVE, CNA, etc.)
should be loaded and prepared for forecasting.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import pandas as pd
from darts import TimeSeries


class DataAdapter(ABC):
    """
    Abstract base class for data loading and preparation.

    Each forecast type (CVE, CNA) implements this interface to handle
    its specific data source and preparation requirements.
    """

    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from source.

        Returns:
            DataFrame with raw data
        """
        pass

    @abstractmethod
    def prepare_time_series(self, df: pd.DataFrame) -> TimeSeries:
        """
        Convert raw data to Darts TimeSeries.

        Args:
            df: Raw data DataFrame

        Returns:
            Prepared TimeSeries
        """
        pass

    @abstractmethod
    def validate_data(self, series: TimeSeries) -> Tuple[bool, str]:
        """
        Validate data quality and completeness.

        Args:
            series: Time series to validate

        Returns:
            Tuple of (is_valid, message)
        """
        pass

    @abstractmethod
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of loaded data.

        Returns:
            Dictionary with summary info
        """
        pass

    def load_and_prepare(self) -> Tuple[TimeSeries, Dict[str, Any]]:
        """
        Complete data loading pipeline.

        Returns:
            Tuple of (time_series, summary)
        """
        # Load raw data
        df = self.load_raw_data()

        # Prepare time series
        series = self.prepare_time_series(df)

        # Validate
        is_valid, message = self.validate_data(series)
        if not is_valid:
            raise ValueError(f'Data validation failed: {message}')

        # Get summary
        summary = self.get_data_summary()

        return series, summary
