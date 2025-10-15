# CVEForecast Code Directory

Production-ready forecasting system with unified architecture.

---

## Quick Start

### Production Forecast (Fast):
```bash
python3 run_production_forecast.py
```
**Output:** `web/data.json`, `web/cna_data.json`

### With Validation:
```bash
python3 run_unified_pipeline.py --with-validation
```

### Complete Testing:
```bash
python3 run_unified_pipeline.py --with-validation --with-diagnostics
```

---

## Directory Structure

```
code/
├── core/                      # Base forecasting architecture
│   ├── base_forecaster.py     # Abstract base class
│   ├── data_adapter.py        # Data loading interface
│   └── validation_mixin.py    # Shared validation methods
│
├── adapters/                  # Forecast implementations
│   ├── cve_adapter.py         # CVE forecasting
│   └── cna_adapter.py         # CNA forecasting
│
├── validation/                # Validation suite
│   ├── time_series_cv.py      # Cross-validation
│   ├── statistical_tests.py   # DM & MCS tests
│   └── interval_validation.py # Interval validation
│
├── diagnostics/               # Diagnostic suite
│   ├── residual_analysis.py   # Residual tests
│   └── horizon_analysis.py    # Horizon evaluation
│
├── tuner/                     # Hyperparameter optimization
│   └── comprehensive_tuner.py # Model tuning
│
├── unified_pipeline.py        # Main coordinator
├── run_production_forecast.py # Production entry point
└── run_unified_pipeline.py    # Full-featured entry point
```

---

## Entry Points

### 1. `run_production_forecast.py`
**Purpose:** Quick production forecasts  
**Usage:** `python3 run_production_forecast.py`  
**Features:**
- Fast execution (no validation)
- Generates CVE + CNA forecasts
- Perfect for cron jobs

### 2. `run_unified_pipeline.py`
**Purpose:** Full-featured pipeline with options  
**Usage:**
```bash
# Full pipeline
python3 run_unified_pipeline.py

# With validation
python3 run_unified_pipeline.py --with-validation

# With diagnostics
python3 run_unified_pipeline.py --with-diagnostics

# CVE only
python3 run_unified_pipeline.py --cve-only

# CNA only
python3 run_unified_pipeline.py --cna-only
```

### 3. `test_unified_architecture.py`
**Purpose:** Verify architecture integrity  
**Usage:** `python3 test_unified_architecture.py`

---

## Architecture Overview

### BaseForecaster (Abstract Base Class)
All forecasters extend this base class:
```python
from core.base_forecaster import BaseForecaster

class MyForecaster(BaseForecaster):
    def load_data(self): ...
    def get_forecast_horizon(self): ...
    def get_model_list(self): ...
    def create_model(self, name, params): ...
    def apply_constraints(self, forecasts): ...
    def save_results(self, forecasts): ...
```

### ValidationMixin
Provides shared validation methods:
- `perform_cross_validation()` - 5-fold CV
- `perform_statistical_tests()` - DM & MCS
- `run_residual_diagnostics()` - Residual tests
- `run_horizon_analysis()` - Horizon evaluation

---

## Programmatic Usage

```python
from unified_pipeline import UnifiedForecastPipeline

# Initialize
pipeline = UnifiedForecastPipeline()

# Run both CVE and CNA
results = pipeline.run_all(
    run_cve=True,
    run_cna=True,
    cve_validation=True
)

# Or run individually
cve_results = pipeline.run_cve_pipeline(run_validation=True)
cna_results = pipeline.run_cna_pipeline()

# Access forecasters directly
cve_forecaster = pipeline.get_cve_forecaster()
cna_forecaster = pipeline.get_cna_forecaster()

# Run custom validation
cv_results = cve_forecaster.perform_cross_validation(n_splits=5)
diag_results = cve_forecaster.run_residual_diagnostics()
```

---

## Configuration

### `config.json`
Main CVE forecasting configuration:
- Model hyperparameters
- File paths
- Validation settings

### `cna_config.json`
CNA-specific configuration:
- CNA models
- Output paths
- Minimum CVE thresholds

---

## Utilities

### `data_loader.py`
Load CVE data from database

### `forecast_constraints.py`
Apply domain-specific forecast constraints

### `forecast_tracker.py`
Track forecast accuracy over time

### `cna_trend_data.py`
Calculate CNA momentum and trends

---

## Testing

```bash
# Verify architecture
python3 test_unified_architecture.py

# Run with validation
python3 run_unified_pipeline.py --with-validation

# Complete diagnostic suite
python3 run_unified_pipeline.py --with-diagnostics
```

---

## Common Tasks

### Add New Model:
1. Add hyperparameters to `config.json`
2. Model automatically available in adapters

### Extend Forecast Type:
1. Create new adapter extending `BaseForecaster`
2. Implement 6 abstract methods
3. Add to `unified_pipeline.py`

### Add Validation Method:
1. Add method to `core/validation_mixin.py`
2. Automatically available to all forecasters

---

## Dependencies

- Python 3.8+
- darts (time series)
- pandas, numpy
- scikit-learn
- statsmodels
- LightGBM, XGBoost, CatBoost (ML models)
- Prophet (forecasting)

---

## Output Files

### `web/data.json`
CVE forecast data:
```json
{
  "metadata": {...},
  "historical": {"2024-01": 1000, ...},
  "forecasts": {
    "LightGBM": {"2025-08": 5000, ...},
    "XGBoost": {"2025-08": 5100, ...}
  }
}
```

### `web/cna_data.json`
CNA-specific forecasts:
```json
{
  "cna_uuid": {
    "id": "cna_uuid",
    "name": "CNA Name",
    "historical": {...},
    "forecasts": {
      "BestModel": {"2025-08": 100, ...}
    },
    "model_selection": {
      "selected_model": "LightGBM",
      "validation_mape": 15.2
    }
  }
}
```

---

## Support

See documentation in `v.10 - Documentation/`:
- `WEEK5_UNIFIED_ARCHITECTURE.md` - Architecture details
- `CLEANUP_COMPLETE.md` - Cleanup summary
- `CODE_CLEANUP_AUDIT.md` - Audit details

---

**Last Updated:** 2025-10-07  
**Status:** Production Ready  
**Grade:** A- (Excellent Architecture)
