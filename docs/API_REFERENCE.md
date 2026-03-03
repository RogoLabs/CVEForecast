# CVE Forecast API Reference

**Version**: 0.11 "Phoenix" 🔥🐦
**Last Updated**: March 2026

## Table of Contents
- [Core Classes](#core-classes)
- [Data Loading](#data-loading)
- [Configuration](#configuration)
- [Utilities](#utilities)
- [Data Structures](#data-structures)

## Core Classes

### BaseForecaster

Abstract base class for all forecasters.

```python
from core.base_forecaster import BaseForecaster

class BaseForecaster(ABC):
    """
    Abstract base class defining the forecasting interface.
    
    Attributes:
        config (dict): Configuration dictionary
        logger (Logger): Python logger instance
        series (TimeSeries): Loaded time series data
        model_results (dict): Trained models and metrics
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize forecaster.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        pass
    
    @abstractmethod
    def load_data(self) -> TimeSeries:
        """
        Load and prepare data for forecasting.
        
        Returns:
            TimeSeries object with historical data
            
        Raises:
            DataLoadError: If data loading fails
        """
        pass
    
    @abstractmethod
    def create_model(self, model_name: str, hyperparameters: Dict[str, Any]):
        """
        Create a model instance with given hyperparameters.
        
        Args:
            model_name: Name of the model (e.g., "Prophet", "LightGBM")
            hyperparameters: Model-specific hyperparameters
            
        Returns:
            Configured model instance
            
        Raises:
            ModelCreationError: If model creation fails
        """
        pass
    
    def train_all_models(self, train_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Train all enabled models.
        
        Args:
            train_ratio: Fraction of data to use for training (default: 0.8)
                        In production mode (1.0), uses all complete months
            
        Returns:
            Dictionary mapping model names to results:
            {
                "model_name": {
                    "model": trained_model,
                    "metrics": {"mape": float, "mae": float},
                    "trained": bool
                }
            }
            
        Note:
            Automatically filters out current incomplete month from training
        """
        pass
    
    def generate_forecasts(self, horizon: int) -> Dict[str, ForecastResult]:
        """
        Generate forecasts for all trained models.
        
        Args:
            horizon: Number of periods to forecast
            
        Returns:
            Dictionary mapping model names to ForecastResult objects
        """
        pass
```

### CVEForecaster

CVE-specific forecaster implementation.

```python
from adapters.cve_adapter import CVEForecaster

class CVEForecaster(BaseForecaster, ValidationMixin):
    """
    Forecaster for CVE data.
    
    Attributes:
        current_datetime (datetime): Current timestamp for forecasting
        forecast_constraints (ForecastConstraints): Constraint manager
        cna_momentum (dict): CNA growth statistics
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize CVE forecaster.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.current_datetime = datetime.now(timezone.utc)
    
    def load_data(self) -> TimeSeries:
        """
        Load CVE data from cvelistV5 repository.
        
        Returns:
            TimeSeries with monthly CVE counts
            
        Raises:
            DataLoadError: If CVE data cannot be loaded
        """
        pass
    
    def create_model(self, model_name: str, hyperparameters: Dict[str, Any]):
        """
        Create CVE forecasting model.
        
        Supported models:
            - Prophet, AutoARIMA, TBATS, Theta, FourTheta
            - ExponentialSmoothing, KalmanFilter, Croston
            - XGBoost, LightGBM, CatBoost, RandomForest
            - LinearRegression
            
        Args:
            model_name: Name of model to create
            hyperparameters: Model-specific hyperparameters
            
        Returns:
            Configured model instance
        """
        pass
    
    def run_full_pipeline(self, train_ratio: float = 1.0,
                         run_validation: bool = False,
                         run_diagnostics: bool = False) -> Dict[str, Any]:
        """
        Execute complete CVE forecasting pipeline.
        
        Args:
            train_ratio: Training data ratio (1.0 for production)
            run_validation: Whether to run validation tests
            run_diagnostics: Whether to generate diagnostic plots
            
        Returns:
            Dictionary with pipeline results:
            {
                "models_trained": int,
                "forecasts_generated": int,
                "output_path": str,
                "execution_time": float
            }
        """
        pass
    
    def save_results(self, forecasts: Dict[str, ForecastResult]) -> str:
        """
        Save forecast results to web/data.json.
        
        Generates:
            - Model rankings (sorted by MAPE)
            - Yearly forecast totals
            - Cumulative timelines
            - Actuals cumulative (current year progress)
            - Forecast vs published validation data
            - Summary statistics
            
        Args:
            forecasts: Dictionary of forecast results
            
        Returns:
            Path to saved file (e.g., "web/data.json")
        """
        pass
```

### CNAForecaster

CNA-specific forecaster implementation.

```python
from adapters.cna_adapter import CNAForecaster

class CNAForecaster(BaseForecaster):
    """
    Forecaster for CNA data.
    
    Handles individual forecasting for 120+ CVE Numbering Authorities.
    """
    
    def forecast_all_cnas(self) -> Dict[str, Any]:
        """
        Generate forecasts for all CNAs.
        
        Process:
            1. Load CNA list from API
            2. For each CNA:
               - Load historical data
               - Test multiple models
               - Select best model (lowest MAPE)
               - Generate forecast
            3. Aggregate results
            
        Returns:
            Dictionary mapping CNA names to forecast data:
            {
                "cna_name": {
                    "forecasts": {"2025-10": 123, ...},
                    "model_selection": {
                        "selected_model": "LightGBM",
                        "validation_mape": 15.2
                    },
                    "cumulative_timeline": [...]
                }
            }
        """
        pass
    
    def select_best_model(self, cna_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Select optimal model for a CNA.
        
        Tests models:
            - Prophet, LightGBM, XGBoost
            - LinearRegression, ExponentialSmoothing
            
        Args:
            cna_data: Historical CVE data for the CNA
            
        Returns:
            Tuple of (model_name, validation_mape)
        """
        pass
```

### UnifiedPipeline

Orchestrates CVE and CNA forecasting.

```python
from unified_pipeline import UnifiedPipeline

class UnifiedPipeline:
    """
    Unified pipeline for CVE and CNA forecasting.
    
    Single entry point for production runs.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize unified pipeline.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        pass
    
    def run_all(self, run_cve: bool = True, run_cna: bool = True,
                cve_train_ratio: float = 1.0,
                cve_validation: bool = False,
                cve_diagnostics: bool = False) -> Dict[str, Any]:
        """
        Execute complete forecasting pipeline.
        
        Args:
            run_cve: Whether to run CVE forecasting
            run_cna: Whether to run CNA forecasting
            cve_train_ratio: Training ratio for CVE models
            cve_validation: Run CVE validation tests
            cve_diagnostics: Generate CVE diagnostic plots
            
        Returns:
            Dictionary with results from both pipelines:
            {
                "cve_results": {...},
                "cna_results": {...},
                "total_execution_time": float
            }
        """
        pass
```

### ForecastTracker

Tracks forecast accuracy over time.

```python
from forecast_tracker import ForecastTracker

class ForecastTracker:
    """
    Track forecast predictions over time for accuracy analysis.
    
    Attributes:
        history_path (Path): Path to forecast_history.json
        history (dict): Loaded history data
    """
    
    def __init__(self, history_path: str = "web/forecast_history.json"):
        """
        Initialize forecast tracker.
        
        Args:
            history_path: Path to history JSON file
        """
        pass
    
    def add_snapshot(self, forecasts: Dict[str, Dict[str, float]],
                    actuals: Dict[str, float],
                    model_performance: Dict[str, Dict],
                    snapshot_date: datetime,
                    metadata: Optional[Dict] = None):
        """
        Add new forecast snapshot to history.
        
        Args:
            forecasts: Forecasts by month and model
                      {"2025-10": {"Prophet": 4500, "LightGBM": 4450}}
            actuals: Actual CVE counts for completed months
                    {"2025-09": 4325, "2025-08": 4210}
            model_performance: Model metrics
                             {"Prophet": {"mape": 10.13, "mae": 412.78}}
            snapshot_date: Date of this snapshot
            metadata: Optional metadata (dataset_size, etc.)
        """
        pass
    
    def get_forecast_evolution(self, target_month: str) -> List[Dict]:
        """
        Get all forecasts made for a specific month over time.
        
        Args:
            target_month: Month in YYYY-MM format (e.g., "2025-10")
            
        Returns:
            List of snapshots containing forecasts for that month
        """
        pass
    
    def get_accuracy_summary(self) -> Dict:
        """
        Get summary of forecast accuracy across all tracked months.
        
        Returns:
            Dictionary with average errors, convergence stats:
            {
                "months_completed": int,
                "mean_absolute_error_pct": float,
                "median_absolute_error_pct": float,
                "convergence_breakdown": {
                    "improved": int,
                    "degraded": int,
                    "stable": int
                }
            }
        """
        pass
```

### Model Utilities (`core.model_utils`)

Shared utilities for parameter fixing and safe model creation, extracted from
adapters to eliminate duplication.

```python
from core.model_utils import fix_hyperparameters, create_model_safe

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
        model_name: Name of the model (e.g., "ExponentialSmoothing", "Theta")
        hyperparameters: Original hyperparameters dict

    Returns:
        New dict with fixed hyperparameters
    """
    pass


def create_model_safe(model_class, model_name: str, hyperparameters: Dict[str, Any],
                      logger: logging.Logger):
    """
    Safely create a Darts model instance with fallback.

    Attempts to create a model with the given hyperparameters.
    If creation fails, retries with an empty parameter set as a fallback.

    Args:
        model_class: The Darts model class to instantiate
        model_name: Name of the model (for logging)
        hyperparameters: Model-specific hyperparameters
        logger: Logger instance

    Returns:
        Configured model instance, or None if both attempts fail
    """
    pass
```

### ForecastConstraints

Enforces domain-specific constraints on forecast outputs.

```python
from forecast_constraints import ForecastConstraints

class ForecastConstraints:
    """
    Apply growth floor and trend constraints to yearly forecast totals.

    Attributes:
        min_growth_rate (float): Minimum allowed year-over-year growth
        max_growth_rate (float): Maximum allowed year-over-year growth
        enable_floor (bool): Whether growth floor is active
        enable_trend (bool): Whether trend constraint is active
    """

    def apply_constraints(self, yearly_totals: Dict[int, Dict[str, int]],
                         ytd_growth: Optional[float] = None,
                         previous_year_actuals: Optional[Dict[int, int]] = None
                         ) -> Dict[int, Dict[str, int]]:
        """
        Apply all constraints to yearly forecast totals.

        Args:
            yearly_totals: Dictionary of {year: {model_name: total}}
            ytd_growth: Year-to-date growth rate (optional, for trend adjustment)
            previous_year_actuals: Actual yearly CVE totals from historical data,
                                  used as baseline when prior year is not in
                                  yearly_totals (e.g., {2024: 39941, 2023: 34700})

        Returns:
            Constrained yearly totals with the same structure
        """
        pass

    def apply_growth_floor(self, forecast: int, previous_year: int) -> int:
        """
        Ensure forecast meets minimum growth rate relative to previous year.

        Args:
            forecast: Forecasted yearly total
            previous_year: Previous year's actual total

        Returns:
            Adjusted forecast (>= previous_year * (1 + min_growth_rate))
        """
        pass
```

## Data Loading

### load_cve_data

Load and aggregate CVE data.

```python
from data_loader import load_cve_data

def load_cve_data(config: Dict[str, Any],
                 filter_by_date: bool = True,
                 start_date_filter: str = "2017-01-01") -> pd.DataFrame:
    """
    Load CVE data from cvelistV5 repository.
    
    Process:
        1. Scan cvelistV5 directory for JSON files
        2. Parse each file and extract publication date
        3. Aggregate by month-end
        4. Fill missing months with 0
        5. Filter by date range
        
    Args:
        config: Configuration dictionary with:
            - data_source.cve_repo_path: Path to cvelistV5
            - data_source.start_date: Start date for data
        filter_by_date: Whether to apply date filtering
        start_date_filter: Start date (YYYY-MM-DD format)
        
    Returns:
        DataFrame with DatetimeIndex and 'cve_count' column:
        
        date         cve_count
        2017-01-31   2345
        2017-02-28   2156
        ...
        
    Raises:
        DataLoadError: If CVE data cannot be loaded
        
    Performance:
        - Processes 297K+ files in ~30 seconds
        - Memory usage: ~500MB peak
    """
    pass
```

## Configuration

### Configuration Structure

```python
{
    "data_source": {
        "cve_repo_path": "cvelistV5",
        "start_date": "2017-01-01",
        "cna_api_url": "https://cveproject.github.io/cvelistV5/cna.json"
    },
    "forecast": {
        "forecast_end_year": 2026,
        "horizon_months": 15,
        "include_current_month": true
    },
    "models": {
        "Prophet": {
            "enabled": true,
            "hyperparameters": {
                "yearly_seasonality": true,
                "weekly_seasonality": false,
                "daily_seasonality": false,
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 0.1
            },
            "tuning_results": {
                "mape": 10.13,
                "mae": 412.78,
                "tuned_at": "2025-10-01T00:00:00Z"
            }
        },
        "LightGBM": {
            "enabled": true,
            "hyperparameters": {
                "lags": 12,
                "output_chunk_length": 1,
                "n_estimators": 100
            },
            "tuning_results": {
                "mape": 6.22,
                "mae": 257.44,
                "tuned_at": "2025-10-01T00:00:00Z"
            }
        }
    },
    "file_paths": {
        "output": "web/data.json",
        "cna_output": "web/cna_data.json",
        "forecast_history": "web/forecast_history.json",
        "pipeline_results": "web/pipeline_results.json"
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(levelname)s - %(message)s"
    }
}
```

### Loading Configuration

```python
import json

def load_config(config_path: str = "code/config.json") -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)
```

## Utilities

### Logging Setup

```python
import logging

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)
```

## Data Structures

### ForecastResult

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class ForecastResult:
    """
    Container for forecast results.
    
    Attributes:
        model_name: Name of the forecasting model
        forecast_values: Dictionary mapping dates to predicted values
                        {"2025-10-31": 4049, "2025-11-30": 3968}
        metrics: Performance metrics
                {"mape": 10.13, "mae": 412.78, "rmse": 520.45}
        metadata: Additional information
                 {"training_time": 45.2, "data_points": 105}
    """
    model_name: str
    forecast_values: Dict[str, float]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
```

### Output Data Structure (data.json)

```python
{
    "generated_at": "2025-10-13T15:30:00Z",
    "model_rankings": [
        {
            "model_name": "LightGBM",
            "mape": 6.22,
            "mae": 257.44,
            "rmse": null,
            "hyperparameters": {...}
        }
    ],
    "yearly_forecast_totals": {
        "2025": {
            "Prophet": 47501,
            "LightGBM": 46892,
            "all_models": 47200
        },
        "2026": {
            "Prophet": 52345,
            "LightGBM": 51234,
            "all_models": 51800
        }
    },
    "current_month_actual": {
        "date": "2025-10",
        "cve_count": 1576,
        "cumulative_total": 35397
    },
    "actuals_cumulative": [
        {"date": "2025-01-01T00:00:00Z", "cumulative_total": 0},
        {"date": "2025-02-01T00:00:00Z", "cumulative_total": 4274},
        {"date": "2025-10-13T15:30:00Z", "cumulative_total": 36973}
    ],
    "cumulative_timelines": {
        "Prophet_cumulative": [
            {"date": "2025-01-01T00:00:00Z", "cumulative_total": 0},
            {"date": "2025-10-01T00:00:00Z", "cumulative_total": 35397},
            {"date": "2025-11-01T00:00:00Z", "cumulative_total": 39447}
        ]
    },
    "forecasts": {
        "Prophet": [
            {"date": "2025-10", "cve_count": 4049},
            {"date": "2025-11", "cve_count": 3968}
        ]
    },
    "summary": {
        "data_period": {
            "start": "2017-01-31",
            "end": "2025-09-30"
        },
        "forecast_period": {
            "start": "2025-10-01",
            "end": "2026-12-31"
        },
        "cumulative_cves_2025": 35397,
        "previous_year_total": 34700
    },
    "forecast_vs_published": {
        "Prophet": {
            "table_data": [
                {
                    "MONTH": "2025-01",
                    "PUBLISHED": 4274,
                    "FORECAST": 3284,
                    "ERROR": -990,
                    "PERCENT_ERROR": -23.16,
                    "PERFORMANCE": "Poor"
                }
            ],
            "summary_stats": {
                "mean_absolute_error": 412.78,
                "mean_absolute_percentage_error": 10.13
            }
        }
    }
}
```

---

**Next**: [Deployment Guide](DEPLOYMENT.md) | [Development Guide](DEVELOPMENT.md)
