# CVE Forecast Architecture Guide

**Version**: 0.10 "Phoenix" 🔥🐦  
**Last Updated**: October 2025

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Module Descriptions](#module-descriptions)
- [Design Patterns](#design-patterns)
- [Scalability & Performance](#scalability--performance)

## Overview

CVE Forecast is built on a modular, extensible architecture that separates concerns between data loading, model training, forecasting, and validation. The system follows object-oriented design principles with base classes, adapters, and mixins to promote code reuse and maintainability.

### Key Design Goals
1. **Modularity**: Independent components that can be developed and tested separately
2. **Extensibility**: Easy to add new models, data sources, or validation strategies
3. **Maintainability**: Clear separation of concerns with well-defined interfaces
4. **Production-Ready**: Robust error handling, logging, and monitoring
5. **Performance**: Efficient data processing and model training

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CVE Forecast System                          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│  Data Layer  │      │ Core Layer   │     │  Web Layer   │
└──────────────┘      └──────────────┘     └──────────────┘
        │                     │                     │
        ├─ data_loader.py     ├─ base_forecaster.py├─ index.html
        ├─ cve_adapter.py     ├─ validation_mixin  ├─ script.js
        └─ cna_adapter.py     ├─ unified_pipeline  ├─ cna_forecast.html
                              └─ forecast_tracker  └─ technical_details.html
```

### Layer Responsibilities

#### Data Layer
- **Purpose**: Load, parse, and prepare CVE data
- **Components**: `data_loader.py`, data adapters
- **Output**: Pandas DataFrames with monthly CVE counts

#### Core Layer
- **Purpose**: Model training, forecasting, and validation
- **Components**: Base classes, forecasters, pipeline orchestration
- **Output**: Forecast results, metrics, validation data

#### Web Layer
- **Purpose**: Visualization and user interface
- **Components**: HTML pages, JavaScript, CSS
- **Input**: JSON data files from core layer

## Core Components

### 1. Base Forecaster (`core/base_forecaster.py`)

Abstract base class that defines the forecasting interface.

```python
class BaseForecaster(ABC):
    """
    Abstract base class for all forecasters.
    Defines common interface and shared functionality.
    """
    
    @abstractmethod
    def load_data(self) -> TimeSeries:
        """Load and prepare data for forecasting."""
        pass
    
    @abstractmethod
    def create_model(self, model_name: str, hyperparameters: Dict):
        """Create a model instance with given hyperparameters."""
        pass
    
    def train_all_models(self, train_ratio: float = 0.8):
        """Train all enabled models."""
        # Filters out current incomplete month
        # Trains models on complete historical data
        pass
    
    def generate_forecasts(self, horizon: int):
        """Generate forecasts for all trained models."""
        pass
```

**Key Features**:
- Automatic filtering of incomplete current month from training
- Configurable train/validation split
- Model lifecycle management (create, train, predict)
- Comprehensive error handling

### 2. Validation Mixin (`core/validation_mixin.py`)

Provides validation and backtesting capabilities.

```python
class ValidationMixin:
    """
    Mixin class providing validation functionality.
    Can be combined with any forecaster.
    """
    
    def validate_model(self, model, train_data, val_data):
        """Validate model on held-out data."""
        pass
    
    def calculate_metrics(self, actual, predicted):
        """Calculate MAPE, MAE, RMSE."""
        pass
    
    def backtest(self, model_name, train_end_date):
        """Historical backtest on past data."""
        pass
```

**Key Features**:
- Walk-forward validation
- Multiple error metrics (MAPE, MAE, RMSE)
- Historical backtesting
- Performance tracking

### 3. CVE Forecaster (`adapters/cve_adapter.py`)

Concrete implementation for CVE forecasting.

```python
class CVEForecaster(BaseForecaster, ValidationMixin):
    """
    CVE-specific forecaster implementation.
    Handles CVE data loading, model training, and forecast generation.
    """
    
    def load_data(self):
        """Load CVE data from cvelistV5 repository."""
        # Calls data_loader.load_cve_data()
        # Returns TimeSeries object
        pass
    
    def create_model(self, model_name, hyperparameters):
        """Create CVE forecasting model."""
        # Supports 13+ models
        # Returns configured model instance
        pass
    
    def _calculate_forecast_vs_published(self, model_name):
        """Generate backtest validation data."""
        # Trains on data through 2024
        # Forecasts 2025 months
        # Compares against actuals
        pass
    
    def save_results(self, forecasts):
        """Save forecasts to web/data.json."""
        # Generates model rankings
        # Creates cumulative timelines
        # Saves forecast history
        pass
```

**Key Features**:
- CVE-specific data processing
- 13 optimized models
- Historical backtest validation
- Comprehensive output generation

### 4. CNA Forecaster (`adapters/cna_adapter.py`)

Handles CNA-specific forecasting.

```python
class CNAForecaster(BaseForecaster):
    """
    CNA-specific forecaster.
    Generates individual forecasts for 120+ CNAs.
    """
    
    def load_data(self, cna_name):
        """Load data for specific CNA."""
        pass
    
    def select_best_model(self, cna_data):
        """Choose optimal model for this CNA."""
        # Tests multiple models
        # Selects based on validation MAPE
        pass
    
    def forecast_all_cnas(self):
        """Generate forecasts for all CNAs."""
        # Parallel processing
        # Per-CNA model selection
        pass
```

**Key Features**:
- Per-CNA model selection
- Handles sparse data
- Parallel processing
- Organization-specific optimization

### 5. Unified Pipeline (`unified_pipeline.py`)

Orchestrates the entire forecasting workflow.

```python
class UnifiedPipeline:
    """
    Orchestrates CVE and CNA forecasting pipelines.
    Single entry point for production runs.
    """
    
    def run_all(self, run_cve=True, run_cna=True):
        """Execute complete pipeline."""
        # 1. Run CVE forecasting
        # 2. Run CNA forecasting
        # 3. Save results
        # 4. Generate summary
        pass
    
    def run_cve_pipeline(self):
        """Execute CVE forecasting workflow."""
        # Load data
        # Train models
        # Generate forecasts
        # Calculate backtest
        # Save results
        pass
    
    def run_cna_pipeline(self):
        """Execute CNA forecasting workflow."""
        # Load CNA list
        # Process each CNA
        # Select best models
        # Generate forecasts
        # Save results
        pass
```

**Key Features**:
- Single command execution
- Error handling and recovery
- Progress tracking
- Result aggregation

### 6. Forecast Tracker (`forecast_tracker.py`)

Tracks forecast accuracy over time.

```python
class ForecastTracker:
    """
    Tracks forecast predictions over time.
    Enables accuracy analysis and prediction evolution.
    """
    
    def add_snapshot(self, forecasts, actuals, model_performance):
        """Save current forecast snapshot."""
        # Stores forecasts for future comparison
        # Records model performance
        # Updates accuracy tracking
        pass
    
    def get_forecast_evolution(self, target_month):
        """Get all forecasts made for a specific month."""
        # Shows how predictions changed over time
        pass
    
    def get_model_stability_ranking(self):
        """Rank models by prediction stability."""
        # Measures forecast consistency
        pass
```

**Key Features**:
- Historical snapshot storage
- Accuracy tracking
- Prediction evolution analysis
- Stability metrics

## Data Flow

### 1. CVE Forecasting Pipeline

```
┌─────────────────┐
│  CVE Data Repo  │ (cvelistV5)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Loader    │ (data_loader.py)
│  - Parse JSON   │
│  - Aggregate    │
│  - Filter       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CVE Forecaster  │ (cve_adapter.py)
│  - Load data    │
│  - Train models │
│  - Generate     │
│    forecasts    │
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
┌─────────────────┐  ┌──────────────────┐
│   Backtest      │  │  Forecast        │
│   Validation    │  │  Generation      │
│  - Train on     │  │  - Oct-Dec 2025  │
│    2024 data    │  │  - 2026 forecasts│
│  - Test on      │  │                  │
│    2025 data    │  │                  │
└────────┬────────┘  └────────┬─────────┘
         │                    │
         └──────────┬─────────┘
                    │
                    ▼
         ┌─────────────────┐
         │  Save Results   │
         │  - data.json    │
         │  - forecast_    │
         │    history.json │
         └─────────────────┘
```

### 2. CNA Forecasting Pipeline

```
┌─────────────────┐
│  CNA List API   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CNA Adapter    │
│  - Load CNAs    │
│  - Filter data  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  For Each CNA:  │
│  - Load data    │
│  - Test models  │
│  - Select best  │
│  - Forecast     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Aggregate      │
│  - Combine      │
│    results      │
│  - Calculate    │
│    totals       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save Results   │
│  - cna_data.json│
└─────────────────┘
```

### 3. Monthly Tuning Workflow

```
┌─────────────────┐
│  GitHub Actions │
│  (1st of month) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Comprehensive   │
│     Tuner       │
│  - Load data    │
│  - Test configs │
│  - Validate     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Update Config  │
│  - config.json  │
│  - Backup old   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Commit & Push  │
│  - Auto-commit  │
│  - Trigger      │
│    daily run    │
└─────────────────┘
```

## Module Descriptions

### Data Loading (`data_loader.py`)

**Purpose**: Parse CVE JSON files and aggregate into monthly counts.

**Key Functions**:
```python
def load_cve_data(config):
    """
    Load CVE data from cvelistV5 repository.
    
    Returns:
        DataFrame with monthly CVE counts
    """
    # Parse 297K+ JSON files
    # Extract publication dates
    # Aggregate by month
    # Filter date range
    pass
```

**Performance**:
- Processes 297K+ files in ~30 seconds
- Memory-efficient streaming
- Robust error handling for malformed JSON

### Model Creation (`core/base_forecaster.py`)

**Supported Models**:

**Statistical Models**:
- Prophet (Facebook's forecasting tool)
- AutoARIMA (Automatic ARIMA selection)
- TBATS (Exponential smoothing with Box-Cox)
- Theta (Theta method)
- FourTheta (Theta with seasonality)
- ExponentialSmoothing (Holt-Winters)
- KalmanFilter (State-space model)
- Croston (Intermittent demand)

**Machine Learning Models**:
- XGBoost (Gradient boosting)
- LightGBM (Light gradient boosting)
- CatBoost (Categorical boosting)
- RandomForest (Ensemble trees)
- LinearRegression (OLS regression)

**Model Selection Criteria**:
- Historical performance (MAPE)
- Training time
- Memory requirements
- Stability (prediction consistency)

### Configuration Management (`config.json`)

**Structure**:
```json
{
  "data_source": {
    "cve_repo_path": "cvelistV5",
    "start_date": "2017-01-01"
  },
  "forecast": {
    "forecast_end_year": 2026,
    "horizon_months": 15
  },
  "models": {
    "Prophet": {
      "enabled": true,
      "hyperparameters": {
        "yearly_seasonality": true,
        "changepoint_prior_scale": 0.05
      },
      "tuning_results": {
        "mape": 10.13,
        "tuned_at": "2025-10-01"
      }
    }
  },
  "file_paths": {
    "output": "web/data.json",
    "cna_output": "web/cna_data.json"
  }
}
```

**Configuration Priority**:
1. Environment variables (if set)
2. `config.json` values
3. Default fallbacks

## Design Patterns

### 1. Template Method Pattern

`BaseForecaster` defines the forecasting workflow template:
```python
def run_full_pipeline(self):
    self.load_data()           # Implemented by subclass
    self.train_all_models()    # Common implementation
    self.generate_forecasts()  # Common implementation
    self.save_results()        # Implemented by subclass
```

### 2. Strategy Pattern

Model selection uses strategy pattern:
```python
class ModelStrategy:
    def create_model(self, model_name, hyperparameters):
        if model_name == "Prophet":
            return Prophet(**hyperparameters)
        elif model_name == "LightGBM":
            return LightGBMModel(**hyperparameters)
        # ...
```

### 3. Adapter Pattern

`CVEForecaster` and `CNAForecaster` adapt the base interface to specific domains:
```python
class CVEForecaster(BaseForecaster):
    def load_data(self):
        # CVE-specific data loading
        return load_cve_data(self.config)

class CNAForecaster(BaseForecaster):
    def load_data(self):
        # CNA-specific data loading
        return load_cna_data(self.config)
```

### 4. Mixin Pattern

`ValidationMixin` adds validation capabilities without inheritance:
```python
class CVEForecaster(BaseForecaster, ValidationMixin):
    # Inherits forecasting from BaseForecaster
    # Inherits validation from ValidationMixin
    pass
```

### 5. Factory Pattern

Model creation uses factory pattern:
```python
def create_model(self, model_name, hyperparameters):
    model_factory = {
        "Prophet": lambda: Prophet(**hyperparameters),
        "LightGBM": lambda: LightGBMModel(**hyperparameters),
        # ...
    }
    return model_factory[model_name]()
```

## Scalability & Performance

### Current Performance

**CVE Forecasting**:
- Data loading: ~30 seconds (297K files)
- Model training: ~5-10 minutes (13 models)
- Forecast generation: ~1 minute
- **Total**: ~10-15 minutes

**CNA Forecasting**:
- Per-CNA processing: ~5-10 seconds
- Total CNAs: 120+
- **Total**: ~10-15 minutes

**Combined Pipeline**: ~25-30 minutes

### Optimization Strategies

1. **Parallel Processing**
   - Train models in parallel (future enhancement)
   - Process CNAs concurrently

2. **Caching**
   - Cache parsed CVE data
   - Reuse trained models when possible

3. **Incremental Updates**
   - Only process new CVE files
   - Update existing forecasts

4. **Resource Management**
   - CPU-only models for GitHub Actions
   - Memory-efficient data structures

### Scalability Limits

**Current System**:
- Handles 300K+ CVE files
- Supports 13 CVE models
- Processes 120+ CNAs
- Runs on GitHub Actions (6 hour limit)

**Future Scaling**:
- Add GPU support for deep learning models
- Implement distributed training
- Use cloud infrastructure for larger workloads

## Error Handling

### Graceful Degradation

```python
try:
    model.fit(train_data)
except Exception as e:
    logger.warning(f"Model {model_name} failed: {e}")
    # Continue with other models
    # System remains functional
```

### Fallback Mechanisms

1. **Model Failures**: Skip failed models, continue with successful ones
2. **Data Issues**: Use cached data if fresh data unavailable
3. **Configuration Errors**: Fall back to default configurations
4. **Network Issues**: Retry with exponential backoff

### Logging Strategy

```python
# Structured logging with levels
logger.info("Starting CVE forecast pipeline")
logger.debug(f"Loaded {len(data)} data points")
logger.warning("Model X failed, using fallback")
logger.error("Critical failure in data loading")
```

## Testing Strategy

### Unit Tests
- Test individual functions and methods
- Mock external dependencies
- Fast execution (<1 second per test)

### Integration Tests
- Test component interactions
- Use sample data
- Moderate execution (~1 minute)

### End-to-End Tests
- Test complete pipeline
- Use production configuration
- Slow execution (~30 minutes)

### Continuous Integration
- Run tests on every commit
- Block merges if tests fail
- Generate coverage reports

## Future Enhancements

### Planned Features
1. **Real-time Forecasting**: Update forecasts as new CVEs published
2. **Ensemble Methods**: Combine multiple models for better accuracy
3. **Confidence Intervals**: Provide uncertainty estimates
4. **API Endpoints**: RESTful API for programmatic access
5. **Model Explainability**: SHAP values and feature importance

### Architecture Evolution
1. **Microservices**: Split into independent services
2. **Event-Driven**: Use message queues for async processing
3. **Cloud-Native**: Deploy on Kubernetes
4. **Streaming**: Real-time data processing with Apache Kafka

---

**Next**: [API Reference](API_REFERENCE.md) | [Deployment Guide](DEPLOYMENT.md)
