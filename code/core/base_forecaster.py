"""
Base Forecaster - Abstract base class for all forecast systems.

This provides the unified interface that both CVE and CNA forecasting
must implement, ensuring consistency across the entire system.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from darts import TimeSeries


@dataclass
class ForecastResult:
    """
    Standardized forecast result structure.

    All forecast adapters return this format for consistency.
    """

    forecast_values: Dict[str, float]  # {date: value}
    model_name: str
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = None  # {date: {'lower': x, 'upper': y}}
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting systems.

    Defines the interface that CVE and CNA adapters must implement,
    ensuring consistent validation, diagnostics, and forecasting methodology.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize forecaster with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.series: Optional[TimeSeries] = None
        self.model_results: Dict[str, Any] = {}
        self.validation_results: Dict[str, Any] = {}
        self.diagnostic_results: Dict[str, Any] = {}

    @abstractmethod
    def load_data(self) -> TimeSeries:
        """
        Load and prepare data for forecasting.

        Returns:
            TimeSeries: Prepared time series data
        """
        pass

    @abstractmethod
    def get_forecast_horizon(self) -> Tuple[datetime, datetime]:
        """
        Determine forecast period.

        Returns:
            Tuple of (start_date, end_date) for forecast
        """
        pass

    @abstractmethod
    def get_model_list(self) -> List[str]:
        """
        Get list of models to use for this forecast type.

        Returns:
            List of model names
        """
        pass

    @abstractmethod
    def create_model(self, model_name: str, hyperparameters: Dict[str, Any]):
        """
        Create model instance with hyperparameters.

        Args:
            model_name: Name of model
            hyperparameters: Model hyperparameters

        Returns:
            Model instance
        """
        pass

    @abstractmethod
    def apply_constraints(self, forecasts: Dict[str, ForecastResult]) -> Dict[str, ForecastResult]:
        """
        Apply domain-specific constraints to forecasts.

        Args:
            forecasts: Raw forecasts

        Returns:
            Constrained forecasts
        """
        pass

    @abstractmethod
    def save_results(self, forecasts: Dict[str, ForecastResult]) -> str:
        """
        Save forecast results to appropriate format/location.

        Args:
            forecasts: Forecast results

        Returns:
            Path to saved results
        """
        pass

    # ========================================================================
    # Shared Methods (Implemented in Base)
    # ========================================================================

    def train_model(
        self,
        model_name: str,
        hyperparameters: Dict[str, Any],
        train_data: TimeSeries,
        val_data: Optional[TimeSeries] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train a single model and evaluate.

        Args:
            model_name: Model name
            hyperparameters: Hyperparameters
            train_data: Training data
            val_data: Optional validation data

        Returns:
            Tuple of (trained_model, metrics)
        """
        self.logger.info(f'Training {model_name}...')

        try:
            # Create model
            model = self.create_model(model_name, hyperparameters)

            # Handle case where model creation returned None
            if model is None:
                raise ValueError('Model creation returned None')

            # Train
            model.fit(train_data)

            # Evaluate if validation data provided and non-empty
            metrics = {}
            if val_data is not None and len(val_data) > 0:
                predictions = model.predict(len(val_data))

                # Calculate metrics
                from darts.metrics import mae, mape, rmse

                metrics = {
                    'mape': float(mape(val_data, predictions)),
                    'mae': float(mae(val_data, predictions)),
                    'rmse': float(rmse(val_data, predictions)),
                }

                self.logger.info(f'✅ {model_name}: MAPE = {metrics["mape"]:.2f}%')
            else:
                self.logger.info(f'✅ {model_name}: Trained (no validation)')

            return model, metrics

        except Exception as e:
            self.logger.error(f'❌ {model_name} training failed: {e}')
            raise

    def train_all_models(self, train_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Train all models for this forecast type.

        Args:
            train_ratio: Train/test split ratio

        Returns:
            Dictionary of trained models and metrics
        """
        if self.series is None:
            raise ValueError('Data not loaded. Call load_data() first.')

        self.logger.info('=' * 70)
        self.logger.info(f'TRAINING MODELS FOR {self.__class__.__name__}')
        self.logger.info('=' * 70)

        # Exclude current incomplete month from training
        # The series includes all data (for actuals display), but we only train on complete months

        import pandas as pd

        training_series = self.series
        current_month_start = pd.Timestamp.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Filter out any data points from the current month
        series_df = training_series.to_dataframe()
        complete_months_df = series_df[series_df.index < current_month_start]

        if len(complete_months_df) < len(series_df):
            from darts import TimeSeries

            training_series = TimeSeries.from_dataframe(
                complete_months_df, freq=training_series.freq_str, fill_missing_dates=False
            )
            self.logger.info(
                f'Excluded current incomplete month from training. Using {len(training_series)} complete months (was {len(self.series)})'
            )

        # Split data
        split_point = int(train_ratio * len(training_series))
        train_data = training_series[:split_point]
        val_data = training_series[split_point:]

        self.logger.info(f'Train: {len(train_data)} periods, Val: {len(val_data)} periods')

        # Get models to train
        models_to_train = self.get_model_list()
        self.logger.info(f'Training {len(models_to_train)} models: {models_to_train}')

        results = {}

        for model_name in models_to_train:
            # Get hyperparameters from config
            hyperparameters = self.config.get('models', {}).get(model_name, {}).get('hyperparameters', {})

            try:
                model, metrics = self.train_model(model_name, hyperparameters, train_data, val_data)

                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'hyperparameters': hyperparameters,
                    'trained': True,
                }

            except Exception as e:
                self.logger.warning(f'Skipping {model_name}: {e}')
                results[model_name] = {'model': None, 'metrics': {}, 'error': str(e), 'trained': False}

        self.model_results = results
        self.logger.info('=' * 70)

        return results

    def generate_forecasts(self, forecast_horizon: int) -> Dict[str, ForecastResult]:
        """
        Generate forecasts using all trained models.

        Args:
            forecast_horizon: Number of periods to forecast

        Returns:
            Dictionary of forecasts by model
        """
        if not self.model_results:
            raise ValueError('No models trained. Call train_all_models() first.')

        self.logger.info('=' * 70)
        self.logger.info('GENERATING FORECASTS')
        self.logger.info('=' * 70)

        forecasts = {}

        for model_name, model_data in self.model_results.items():
            if not model_data.get('trained', False):
                continue

            try:
                model = model_data['model']

                # Generate forecast
                forecast_ts = model.predict(forecast_horizon)

                # Convert to dictionary format
                forecast_values = {
                    str(date): float(value)
                    for date, value in zip(forecast_ts.time_index, forecast_ts.values().flatten())
                }

                forecasts[model_name] = ForecastResult(
                    forecast_values=forecast_values,
                    model_name=model_name,
                    metrics=model_data.get('metrics', {}),
                    metadata={'hyperparameters': model_data.get('hyperparameters', {})},
                )

                self.logger.info(f'✅ {model_name}: Generated {len(forecast_values)} forecasts')

            except Exception as e:
                self.logger.error(f'❌ {model_name} forecast failed: {e}')

        self.logger.info('=' * 70)

        return forecasts

    def run_validation(self, validation_type: str = 'cv', **kwargs) -> Dict[str, Any]:
        """
        Run validation on trained models.

        Args:
            validation_type: Type of validation ('cv', 'statistical', 'diagnostics')
            **kwargs: Additional arguments for validation

        Returns:
            Validation results
        """
        self.logger.info(f'Running {validation_type} validation...')

        # This would call appropriate validation methods
        # Implementation depends on validation type

        results = {}
        self.validation_results[validation_type] = results

        return results

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of forecaster state.

        Returns:
            Summary dictionary
        """
        return {
            'forecaster_type': self.__class__.__name__,
            'data_loaded': self.series is not None,
            'data_length': len(self.series) if self.series is not None else 0,
            'models_trained': len([m for m in self.model_results.values() if m.get('trained', False)]),
            'total_models': len(self.model_results),
            'validations_run': list(self.validation_results.keys()),
            'diagnostics_run': list(self.diagnostic_results.keys()),
        }

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f'{summary["forecaster_type"]}('
            f'data_length={summary["data_length"]}, '
            f'models_trained={summary["models_trained"]}/{summary["total_models"]})'
        )
