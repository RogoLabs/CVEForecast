# March Update - Comprehensive Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete overhaul of the CVEForecast repository — fix all Python code issues, modernize the web dashboard, harden CI/CD, remove all cruft, and update documentation.

**Architecture:** Layered commits on the MarchUpdate branch. Each task is a self-contained unit of work that can be committed independently. Tasks are grouped into 6 layers: cleanup, Python fixes, tests, web overhaul, CI/CD hardening, and documentation.

**Tech Stack:** Python 3.13, Darts, Tailwind CSS 3.x, Chart.js, GitHub Actions, ruff, pytest

---

## Task 1: Delete All Stale Files

**Files:**
- Delete: `code/run_forecast_only.sh` (empty)
- Delete: `code/run_full_pipeline.sh` (empty)
- Delete: `code/run_week3_validation.py` (empty)
- Delete: `code/run_week4_diagnostics.py` (empty)
- Delete: `code/PIPELINE_TESTING.md` (empty)
- Delete: `web/cna_forecast_clean.html` (empty)
- Delete: `web/cna-chart.js` (empty)
- Delete: `web/cna-table.js` (empty)
- Delete: `web/cna-utils.js` (empty)
- Delete: `web/test_config_display.html` (empty)
- Delete: `web/favicon.ico` (empty, will be regenerated)
- Delete: `web/cna_data_test.json` (old test data)
- Delete: `clean_run_20251007_152320.log` (stale log in repo root)
- Delete: `v.10 - Documentation/` (entire directory, 3 empty files)
- Delete: All files in `code/tuner/logs/` (35 old log files)
- Delete: All files in `code/tuner/config_backups/` (12 backup files)
- Delete: All files in `code/tuner/tuner/config_backups/` (9 backup files)
- Delete: All files in `code/tuner/results/` (old tuning results)

**Step 1: Remove all empty and stale files**

```bash
# Empty files
git rm code/run_forecast_only.sh code/run_full_pipeline.sh code/run_week3_validation.py code/run_week4_diagnostics.py code/PIPELINE_TESTING.md
git rm web/cna_forecast_clean.html web/cna-chart.js web/cna-table.js web/cna-utils.js web/test_config_display.html web/favicon.ico
git rm web/cna_data_test.json
git rm clean_run_20251007_152320.log

# v.10 documentation directory
git rm -r "v.10 - Documentation/"

# Old tuner artifacts
git rm -r code/tuner/logs/ code/tuner/config_backups/
git rm -r code/tuner/tuner/config_backups/
git rm -r code/tuner/results/
```

**Step 2: Move mislocated test file**

```bash
git mv code/test_unified_architecture.py tests/test_unified_architecture.py
```

**Step 3: Update .gitignore**

Add these entries to `.gitignore`:

```
# Tuner artifacts
code/tuner/logs/
code/tuner/config_backups/
code/tuner/tuner/config_backups/
code/tuner/results/

# macOS
.DS_Store

# Debug logs
debug.log
```

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove stale files and clean up repository

Delete 60+ empty, stale, and redundant files including old tuner
logs/backups, empty shell scripts, unused web files, and obsolete
v.10 documentation. Move test_unified_architecture.py to tests/."
```

---

## Task 2: Extract Shared Model Utilities

**Files:**
- Create: `code/core/model_utils.py`
- Modify: `code/adapters/cve_adapter.py:149-250`
- Modify: `code/adapters/cna_adapter.py:229-290`
- Modify: `code/core/__init__.py`

**Step 1: Create `code/core/model_utils.py`**

Extract the duplicated parameter-fixing logic from both adapters into a shared utility:

```python
"""
Shared model utilities - parameter fixing, model creation helpers.

Consolidates logic duplicated between CVE and CNA adapters.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def fix_hyperparameters(model_name: str, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix hyperparameter names and types for backwards compatibility with Darts.

    Handles parameter name changes across Darts versions and ensures
    correct types for enum parameters.

    Args:
        model_name: Name of the model
        hyperparameters: Raw hyperparameters from config

    Returns:
        Fixed hyperparameters dict (new copy, original not modified)
    """
    params = hyperparameters.copy()

    if model_name == 'ExponentialSmoothing':
        if 'damped_trend' in params:
            val = params.pop('damped_trend')
            if val is None or val is False or val == 0:
                params['damping_trend'] = None
            elif val is True or val == 1:
                params['damping_trend'] = 0.98
            elif isinstance(val, (int, float)):
                params['damping_trend'] = float(val)
            else:
                params['damping_trend'] = None

        if 'damping_trend' in params:
            val = params['damping_trend']
            if isinstance(val, bool):
                params['damping_trend'] = 0.98 if val else None
            elif val is not None:
                params['damping_trend'] = float(val) if val != 0 else None

        params.pop('initialization_method', None)
        params.pop('missing', None)

    if model_name in ('Theta', 'FourTheta'):
        if 'season_mode' in params:
            from darts.utils.utils import SeasonalityMode
            mode_str = str(params['season_mode']).lower()
            if mode_str in ('additive', 'add'):
                params['season_mode'] = SeasonalityMode.ADDITIVE
            elif mode_str in ('multiplicative', 'mult', 'mul'):
                params['season_mode'] = SeasonalityMode.MULTIPLICATIVE
            else:
                params.pop('season_mode')

    if model_name == 'LinearRegression':
        if params.get('output_chunk_shift', 0) > 0:
            params['output_chunk_shift'] = 0

    return params


def create_model_safe(model_class, model_name: str, hyperparameters: Dict[str, Any],
                      logger: Optional[logging.Logger] = None):
    """
    Safely create a model instance with fallback to defaults.

    Args:
        model_class: The Darts model class to instantiate
        model_name: Name for logging
        hyperparameters: Fixed hyperparameters
        logger: Optional logger

    Returns:
        Model instance or None if creation fails entirely
    """
    log = logger or logging.getLogger(__name__)
    fixed = fix_hyperparameters(model_name, hyperparameters)

    try:
        return model_class(**fixed)
    except Exception as e:
        log.warning(f"Failed to create {model_name} with params, using defaults: {e}")
        try:
            return model_class()
        except Exception:
            return None
```

**Step 2: Update `cve_adapter.py` to use shared utilities**

Replace lines 149-250 in `code/adapters/cve_adapter.py`. The `create_model` method should call `fix_hyperparameters` and `create_model_safe` from the shared module instead of inlining the logic:

```python
from core.model_utils import fix_hyperparameters, create_model_safe

# In create_model method, replace the parameter fixing block with:
def create_model(self, model_name: str, hyperparameters: Dict[str, Any]):
    model_classes = {
        'Prophet': Prophet,
        'ExponentialSmoothing': ExponentialSmoothing,
        'AutoARIMA': AutoARIMA,
        'Theta': Theta,
        'FourTheta': FourTheta,
        'TBATS': TBATS,
        'Croston': Croston,
        'KalmanForecaster': KalmanForecaster,
        'KalmanFilter': KalmanForecaster,
        'XGBoost': XGBModel,
        'LightGBM': LightGBMModel,
        'CatBoost': CatBoostModel,
        'RandomForest': RandomForestModel,
        'LinearRegression': LinearRegressionModel,
        'TCN': TCNModel,
        'NBEATS': NBEATSModel,
        'NHiTS': NHiTSModel,
        'TiDE': TiDEModel,
        'DLinear': DLinearModel,
        'NaiveMean': NaiveMean,
        'NaiveDrift': NaiveDrift,
        'NaiveSeasonal': NaiveSeasonal
    }

    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")

    return create_model_safe(model_classes[model_name], model_name, hyperparameters, self.logger)
```

**Step 3: Update `cna_adapter.py` similarly**

Replace the duplicated parameter fixing block in `code/adapters/cna_adapter.py:229-290` with the same pattern using `create_model_safe`.

**Step 4: Update `code/core/__init__.py`**

Add exports for the new module:
```python
from .model_utils import fix_hyperparameters, create_model_safe
```

**Step 5: Commit**

```bash
git add code/core/model_utils.py code/core/__init__.py code/adapters/cve_adapter.py code/adapters/cna_adapter.py
git commit -m "refactor: extract shared model utilities from adapters

Move duplicated parameter-fixing and model-creation logic from
cve_adapter.py and cna_adapter.py into core/model_utils.py."
```

---

## Task 3: Complete Constraint Integration

**Files:**
- Modify: `code/adapters/cve_adapter.py:252-269` (apply_constraints method)
- Modify: `code/forecast_constraints.py:138-140` (hardcoded 2024 value)
- Modify: `code/forecast_constraints.py:150` (hardcoded 2025 year)

**Step 1: Fix hardcoded values in `forecast_constraints.py`**

At line 138-140, replace the hardcoded 2024 baseline:
```python
# OLD:
elif previous_year == 2024:
    prev_baseline = 39941

# NEW: Accept previous_year_actuals parameter
```

Modify `apply_constraints` signature to accept `previous_year_actuals`:
```python
def apply_constraints(self, yearly_totals: Dict[int, Dict[str, int]],
                     ytd_growth: Optional[float] = None,
                     previous_year_actuals: Optional[Dict[int, int]] = None) -> Dict[int, Dict[str, int]]:
```

And in the baseline lookup:
```python
elif previous_year_actuals and previous_year in previous_year_actuals:
    prev_baseline = previous_year_actuals[previous_year]
else:
    constrained[year] = yearly_totals[year].copy()
    continue
```

At line 150, generalize the YTD floor to use `current_year` instead of hardcoded 2025:
```python
# OLD:
if year == 2025 and ytd_growth_2025 is not None ...
# NEW:
import datetime
current_year = datetime.datetime.now(datetime.timezone.utc).year
if year == current_year and ytd_growth is not None ...
```

**Step 2: Implement constraint integration in `cve_adapter.py`**

Replace the TODO at line 265 with actual constraint application. The `apply_constraints` method should:
1. Convert ForecastResult objects to yearly totals format
2. Call `self.constraints.apply_constraints()`
3. Map constrained values back to ForecastResult objects

```python
def apply_constraints(self, forecasts: Dict[str, ForecastResult]) -> Dict[str, ForecastResult]:
    """Apply CVE-specific forecast constraints."""
    if not self.constraints:
        self.logger.info("No constraints configured, passing through")
        return forecasts

    # Build yearly totals from forecasts
    yearly_totals = {}
    for model_name, result in forecasts.items():
        for date_str, value in result.forecast_values.items():
            year = int(date_str[:4])
            if year not in yearly_totals:
                yearly_totals[year] = {}
            # Accumulate monthly values into yearly total
            yearly_totals[year][model_name] = yearly_totals[year].get(model_name, 0) + int(value)

    # Get previous year actuals from loaded data
    prev_year_actuals = self._get_previous_year_actuals()

    # Apply constraints
    constrained_totals = self.constraints.apply_constraints(
        yearly_totals,
        previous_year_actuals=prev_year_actuals
    )

    # Calculate scaling factors and apply to monthly forecasts
    for model_name, result in forecasts.items():
        for year in constrained_totals:
            if model_name in constrained_totals[year] and model_name in yearly_totals.get(year, {}):
                original = yearly_totals[year][model_name]
                constrained = constrained_totals[year][model_name]
                if original > 0 and constrained != original:
                    scale = constrained / original
                    for date_str in list(result.forecast_values.keys()):
                        if int(date_str[:4]) == year:
                            result.forecast_values[date_str] = result.forecast_values[date_str] * scale

    self.logger.info(f"Applied constraints to {len(forecasts)} models")
    return forecasts
```

Add helper method `_get_previous_year_actuals` that extracts year totals from the loaded series data.

**Step 3: Commit**

```bash
git add code/forecast_constraints.py code/adapters/cve_adapter.py
git commit -m "feat: complete forecast constraint integration

Wire up ForecastConstraints.apply_constraints() in CVE adapter.
Remove hardcoded 2024 baseline value, make year references dynamic."
```

---

## Task 4: Standardize Date Handling and Code Quality

**Files:**
- Modify: `code/data_loader.py:61` (timezone handling)
- Modify: `code/utils.py` (log rotation)
- Modify: `code/cna_trend_data.py` (request timeout, timezone)
- Modify: `requirements.txt` (add python-dateutil)

**Step 1: Fix date parsing in `data_loader.py`**

At line 61, replace manual timezone replacement:
```python
# OLD:
dt_obj = datetime.fromisoformat(published_date_str.replace('Z', '+00:00'))

# NEW:
from dateutil import parser as dateutil_parser
dt_obj = dateutil_parser.isoparse(published_date_str)
```

**Step 2: Add log rotation to `utils.py`**

Replace the existing 14-line file:
```python
import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(config):
    """Configure logging with rotation and configurable output."""
    level = getattr(logging, config.get('level', 'INFO').upper(), logging.INFO)
    fmt = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config.get('file', 'debug.log')

    handlers = [logging.StreamHandler(sys.stdout)]

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3
    )
    handlers.append(file_handler)

    logging.basicConfig(level=level, format=fmt, handlers=handlers)
```

**Step 3: Add request timeout in `cna_trend_data.py`**

Find the `requests.get()` call and add a 30-second timeout:
```python
# OLD:
response = requests.get(url)

# NEW:
response = requests.get(url, timeout=30)
```

Also fix timezone handling to use UTC:
```python
# OLD: Uses local time
# NEW: Use datetime.now(timezone.utc)
```

**Step 4: Update requirements.txt**

Add `python-dateutil>=2.8.0` to the file.

**Step 5: Commit**

```bash
git add code/data_loader.py code/utils.py code/cna_trend_data.py requirements.txt
git commit -m "fix: standardize date handling, add log rotation, request timeouts

Replace manual timezone parsing with dateutil. Add RotatingFileHandler
for log management. Add 30s timeout to external HTTP requests."
```

---

## Task 5: Add pyproject.toml and Ruff Configuration

**Files:**
- Create: `pyproject.toml`

**Step 1: Create `pyproject.toml`**

```toml
[project]
name = "cveforecast"
version = "0.11.0"
description = "Predictive analytics for CVE publications using machine learning"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}

[tool.ruff]
target-version = "py310"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "single"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks integration tests requiring CVE data",
]
addopts = "-v --tb=short"
```

**Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyproject.toml with ruff and pytest configuration"
```

---

## Task 6: Add Unit Tests for Core Modules

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_model_utils.py`
- Create: `tests/test_forecast_constraints.py`
- Create: `tests/test_data_loader.py`

**Step 1: Create `tests/__init__.py`** (empty file)

**Step 2: Create `tests/conftest.py`**

```python
"""Shared test fixtures for CVEForecast test suite."""
import sys
from pathlib import Path
import pytest

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))


@pytest.fixture
def sample_config():
    """Minimal config for testing."""
    return {
        'models': {
            'ExponentialSmoothing': {
                'enabled': True,
                'hyperparameters': {'damped_trend': True}
            }
        },
        'forecast_constraints': {
            'min_annual_growth_rate': 0.05,
            'max_annual_growth_rate': 0.40,
            'historical_avg_growth': 0.18,
            'enable_growth_floor': True,
            'enable_trend_adjustment': True,
            'enable_ytd_floor': True,
        }
    }
```

**Step 3: Create `tests/test_model_utils.py`**

```python
"""Tests for core.model_utils shared utilities."""
import pytest
from core.model_utils import fix_hyperparameters


class TestFixHyperparameters:
    def test_exponential_smoothing_damped_trend_true(self):
        result = fix_hyperparameters('ExponentialSmoothing', {'damped_trend': True})
        assert result['damping_trend'] == 0.98
        assert 'damped_trend' not in result

    def test_exponential_smoothing_damped_trend_false(self):
        result = fix_hyperparameters('ExponentialSmoothing', {'damped_trend': False})
        assert result['damping_trend'] is None

    def test_exponential_smoothing_damped_trend_float(self):
        result = fix_hyperparameters('ExponentialSmoothing', {'damped_trend': 0.85})
        assert result['damping_trend'] == 0.85

    def test_exponential_smoothing_removes_unsupported(self):
        result = fix_hyperparameters('ExponentialSmoothing', {
            'initialization_method': 'estimated',
            'missing': 'drop'
        })
        assert 'initialization_method' not in result
        assert 'missing' not in result

    def test_linear_regression_fixes_output_chunk_shift(self):
        result = fix_hyperparameters('LinearRegression', {'output_chunk_shift': 5})
        assert result['output_chunk_shift'] == 0

    def test_unknown_model_returns_copy(self):
        params = {'foo': 'bar'}
        result = fix_hyperparameters('UnknownModel', params)
        assert result == params
        assert result is not params  # Must be a copy

    def test_does_not_modify_original(self):
        original = {'damped_trend': True}
        fix_hyperparameters('ExponentialSmoothing', original)
        assert 'damped_trend' in original  # Original unchanged
```

**Step 4: Create `tests/test_forecast_constraints.py`**

```python
"""Tests for forecast_constraints.py."""
import pytest
from forecast_constraints import ForecastConstraints


@pytest.fixture
def constraints():
    config = {
        'min_annual_growth_rate': 0.05,
        'max_annual_growth_rate': 0.40,
        'historical_avg_growth': 0.18,
        'enable_growth_floor': True,
        'enable_trend_adjustment': True,
        'enable_ytd_floor': True,
        'trend_adjustment_confidence': 0.7,
        'trend_adjustment_threshold': 0.75,
        'ytd_minimum_factor': 0.85,
    }
    return ForecastConstraints(config)


class TestGrowthFloor:
    def test_below_minimum_growth(self, constraints):
        result = constraints.apply_growth_floor(40000, 40000)
        assert result >= 42000  # At least 5% growth

    def test_above_maximum_growth(self, constraints):
        result = constraints.apply_growth_floor(100000, 40000)
        assert result <= 56000  # At most 40% growth

    def test_within_range_unchanged(self, constraints):
        result = constraints.apply_growth_floor(45000, 40000)
        assert result == 45000

    def test_disabled(self, constraints):
        constraints.enable_floor = False
        result = constraints.apply_growth_floor(1, 40000)
        assert result == 1


class TestApplyConstraints:
    def test_empty_input(self, constraints):
        assert constraints.apply_constraints({}) == {}

    def test_constraints_with_previous_year_actuals(self, constraints):
        yearly = {2026: {'ModelA': 30000}}
        actuals = {2025: 40000}
        result = constraints.apply_constraints(yearly, previous_year_actuals=actuals)
        assert result[2026]['ModelA'] >= 42000  # At least 5% growth from 40000
```

**Step 5: Run tests**

```bash
cd /Users/gamblin/Documents/Github/CVEForecast && python -m pytest tests/test_model_utils.py tests/test_forecast_constraints.py -v
```

Expected: All tests PASS.

**Step 6: Commit**

```bash
git add tests/
git commit -m "test: add unit tests for model_utils and forecast_constraints

Cover parameter fixing, growth floor bounds, and constraint
application with previous year actuals."
```

---

## Task 7: Extract Shared CSS

**Files:**
- Create: `web/styles.css`
- Modify: `web/index.html` (remove inline styles, link to stylesheet)
- Modify: `web/cna_forecast.html` (remove inline styles, link to stylesheet)
- Modify: `web/technical_details.html` (remove inline styles, link to stylesheet)

**Step 1: Create `web/styles.css`**

Extract all the duplicated CSS from the three HTML files into a single shared stylesheet. Include CSS custom properties for theming:

```css
:root {
    --color-bg: #f3f4f6;
    --color-surface: #ffffff;
    --color-text: #1f2937;
    --color-text-secondary: #6b7280;
    --color-text-muted: #9ca3af;
    --color-border: #e5e7eb;
    --color-primary: #2563eb;
    --color-primary-light: #dbeafe;
    --color-primary-dark: #1e40af;
    --gradient-start: #1f2937;
    --gradient-mid: #374151;
    --gradient-end: #111827;
    --font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

body {
    font-family: var(--font-family);
    background-color: var(--color-bg);
    color: var(--color-text);
}

/* ... all shared styles extracted from HTML files ... */
```

**Step 2: Replace inline `<style>` blocks in all 3 HTML files**

Add `<link rel="stylesheet" href="styles.css">` to each `<head>` and remove the `<style>...</style>` blocks. Keep only page-specific styles if any.

**Step 3: Commit**

```bash
git add web/styles.css web/index.html web/cna_forecast.html web/technical_details.html
git commit -m "refactor: extract shared CSS into styles.css

Remove duplicated style blocks from 3 HTML files. Add CSS custom
properties for theming support."
```

---

## Task 8: Add Accessibility (WCAG AA)

**Files:**
- Modify: `web/index.html`
- Modify: `web/cna_forecast.html`
- Modify: `web/technical_details.html`
- Modify: `web/styles.css`

**Step 1: Add ARIA labels to all interactive elements**

In each HTML file, update:

```html
<!-- Selects: add aria-label -->
<select id="validationModelSelector" aria-label="Select forecasting model for validation comparison" class="...">

<!-- SVG icons: add aria-hidden or title -->
<svg aria-hidden="true" class="w-8 h-8 text-blue-600" ...>

<!-- Tooltip elements: add role and aria -->
<div class="tooltip" role="tooltip" aria-label="Shows how well the selected model forecasted actual CVE counts" data-tooltip="...">

<!-- Table headers: add scope -->
<th scope="col" class="...">Month</th>

<!-- Buttons: add aria-label if icon-only -->
<button onclick="location.reload()" aria-label="Reload page" class="...">Try Again</button>
```

**Step 2: Add focus indicators to `web/styles.css`**

```css
*:focus-visible {
    outline: 2px solid var(--color-primary);
    outline-offset: 2px;
}
```

**Step 3: Add skip-to-content link**

At top of each HTML `<body>`:
```html
<a href="#main-content" class="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-blue-600 text-white px-4 py-2 rounded z-50">Skip to main content</a>
```

And add `id="main-content"` to the `<main>` element.

**Step 4: Commit**

```bash
git add web/
git commit -m "a11y: add ARIA labels, focus indicators, skip navigation

Add aria-label to all interactive elements, scope to table headers,
aria-hidden to decorative SVGs, and visible focus indicators."
```

---

## Task 9: Add SEO Metadata

**Files:**
- Modify: `web/index.html`
- Modify: `web/cna_forecast.html`
- Modify: `web/technical_details.html`

**Step 1: Add meta tags to each page's `<head>`**

For `index.html`:
```html
<title>CVEForecast - CVE Publication Predictions with Machine Learning</title>
<meta name="description" content="Predictive analytics for CVE vulnerability publications using 13+ machine learning and statistical models. Updated daily with real-time accuracy tracking.">
<meta property="og:title" content="CVEForecast">
<meta property="og:description" content="Predictive analytics for CVE vulnerability publications using machine learning">
<meta property="og:type" content="website">
<meta property="og:url" content="https://cveforecast.org">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="CVEForecast">
<meta name="twitter:description" content="Predictive analytics for CVE vulnerability publications using machine learning">
<link rel="canonical" href="https://cveforecast.org">
```

Similar tags for `cna_forecast.html` and `technical_details.html` with page-appropriate content.

**Step 2: Commit**

```bash
git add web/index.html web/cna_forecast.html web/technical_details.html
git commit -m "seo: add meta descriptions, Open Graph, and Twitter Card tags"
```

---

## Task 10: Add Dark Mode Support

**Files:**
- Modify: `web/styles.css`
- Modify: `web/index.html`
- Modify: `web/cna_forecast.html`
- Modify: `web/technical_details.html`
- Modify: `web/script.js`
- Modify: `web/cna.js`

**Step 1: Add dark theme CSS custom properties to `web/styles.css`**

```css
@media (prefers-color-scheme: dark) {
    :root {
        --color-bg: #111827;
        --color-surface: #1f2937;
        --color-text: #f9fafb;
        --color-text-secondary: #d1d5db;
        --color-text-muted: #9ca3af;
        --color-border: #374151;
        --color-primary: #60a5fa;
        --color-primary-light: #1e3a5f;
        --color-primary-dark: #93c5fd;
        --gradient-start: #0f172a;
        --gradient-mid: #1e293b;
        --gradient-end: #020617;
    }
}

[data-theme="dark"] {
    --color-bg: #111827;
    --color-surface: #1f2937;
    /* ... same overrides for manual toggle ... */
}
```

**Step 2: Add theme toggle button to each page's header nav**

```html
<button id="themeToggle" aria-label="Toggle dark mode" class="text-gray-300 hover:text-white transition-colors p-2">
    <svg id="themeIcon" class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path>
    </svg>
</button>
```

**Step 3: Add toggle logic to `web/script.js`**

```javascript
// Theme toggle with localStorage persistence
const themeToggle = document.getElementById('themeToggle');
if (themeToggle) {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
    themeToggle.addEventListener('click', () => {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        document.documentElement.setAttribute('data-theme', isDark ? 'light' : 'dark');
        localStorage.setItem('theme', isDark ? 'light' : 'dark');
    });
}
```

**Step 4: Update Chart.js colors to respect theme**

In chart configuration sections of `script.js` and `cna.js`, read CSS custom properties for chart colors.

**Step 5: Commit**

```bash
git add web/
git commit -m "feat: add dark mode with system preference detection and manual toggle

Uses CSS custom properties for theming, respects prefers-color-scheme,
persists manual choice in localStorage. Charts adapt to theme."
```

---

## Task 11: JavaScript Improvements

**Files:**
- Modify: `web/index.html` (add defer to script tags)
- Modify: `web/script.js`
- Modify: `web/cna.js`

**Step 1: Add `defer` to script tags in all HTML files**

```html
<!-- OLD -->
<script src="script.js"></script>
<!-- NEW -->
<script src="script.js" defer></script>
```

**Step 2: Wrap globals in module pattern in `web/script.js`**

```javascript
const CVEForecast = (() => {
    // Private state
    let forecastData = null;
    let modelInfoData = null;
    let chartInstance = null;
    let selectedYear = new Date().getFullYear();

    // ... existing functions become methods ...

    return { init: loadForecastData };
})();

document.addEventListener('DOMContentLoaded', CVEForecast.init);
```

**Step 3: Update chart to use `.update()` instead of destroy/recreate**

In the `createOrUpdateChart` function:
```javascript
if (chartInstance) {
    chartInstance.data = chartData;
    chartInstance.options = chartOptions;
    chartInstance.update();
} else {
    chartInstance = new Chart(ctx, { type: 'line', data: chartData, options: chartOptions });
}
```

**Step 4: Replace magic numbers with named constants**

```javascript
const CHART_HEIGHT = 400;
const YEAR_START_MONTH = 0;  // January
const YEAR_END_MONTH = 11;   // December
const TOP_MODELS_COUNT = 5;
```

**Step 5: Commit**

```bash
git add web/
git commit -m "refactor: improve JavaScript with module pattern, chart updates, defer

Wrap globals in IIFE, use chart.update() instead of destroy/recreate,
add defer to script tags, replace magic numbers with constants."
```

---

## Task 12: Upgrade Tailwind CSS and Clean Up Web

**Files:**
- Modify: `web/index.html`
- Modify: `web/cna_forecast.html`
- Modify: `web/technical_details.html`
- Create: `web/favicon.svg`

**Step 1: Upgrade Tailwind CDN reference**

In all HTML files:
```html
<!-- OLD -->
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<!-- NEW -->
<script src="https://cdn.tailwindcss.com"></script>
```

Note: Tailwind v3+ uses a CDN script instead of a CSS link for JIT compilation.

**Step 2: Create a proper favicon**

Create `web/favicon.svg`:
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" rx="6" fill="#1e40af"/>
  <text x="50%" y="55%" dominant-baseline="middle" text-anchor="middle" font-family="system-ui" font-weight="700" font-size="18" fill="white">CF</text>
</svg>
```

Add to all HTML `<head>`:
```html
<link rel="icon" type="image/svg+xml" href="favicon.svg">
```

**Step 3: Remove stale version string from footer**

In `index.html`, update the version badge in the footer:
```html
<!-- OLD -->
<span class="inline-block bg-blue-600 text-white text-xs px-2 py-1 rounded-full font-mono">v.05 Adolfo Suárez Madrid-Baraja 🇪🇸</span>
<!-- NEW -->
<span class="inline-block bg-blue-600 text-white text-xs px-2 py-1 rounded-full font-mono">v0.11</span>
```

**Step 4: Remove the broken `tests/verify_totals.js` script reference**

In `index.html`, remove:
```html
<script src="tests/verify_totals.js"></script>
```

**Step 5: Commit**

```bash
git add web/
git commit -m "chore: upgrade Tailwind to v3, add SVG favicon, clean up web files

Upgrade Tailwind CSS CDN to v3 JIT, replace empty favicon with SVG,
remove stale version string, remove broken script reference."
```

---

## Task 13: Add CI/CD Test Workflow

**Files:**
- Create: `.github/workflows/test.yml`

**Step 1: Create test workflow**

```yaml
name: Tests

on:
  pull_request:
    branches: [main, master]
  push:
    branches: [main, master]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v4
        with:
          python-version: '3.13'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest

      - name: Run tests
        run: python -m pytest tests/ -v -m "not slow and not integration"
```

**Step 2: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: add test workflow for pull requests and pushes"
```

---

## Task 14: Add Lint Workflow and Dependabot

**Files:**
- Create: `.github/workflows/lint.yml`
- Create: `.github/dependabot.yml`

**Step 1: Create lint workflow**

```yaml
name: Lint

on:
  pull_request:
    branches: [main, master]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: Install ruff
        run: pip install ruff

      - name: Check formatting
        run: ruff format --check code/ tests/

      - name: Check linting
        run: ruff check code/ tests/
```

**Step 2: Create dependabot config**

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

**Step 3: Commit**

```bash
git add .github/workflows/lint.yml .github/dependabot.yml
git commit -m "ci: add lint workflow and Dependabot configuration"
```

---

## Task 15: Improve Existing Workflows

**Files:**
- Create: `code/scripts/normalize_tuner_paths.py`
- Create: `code/scripts/generate_tuning_report.py`
- Create: `code/scripts/validate_forecast_data.py`
- Modify: `.github/workflows/main.yml`
- Modify: `.github/workflows/monthly_tuning.yml`

**Step 1: Extract inline Python from `monthly_tuning.yml` to scripts**

Create `code/scripts/normalize_tuner_paths.py`:
```python
"""Normalize tuner file paths for CI environment."""
import json
import os
import pathlib

cfg_path = pathlib.Path("code/tuner/tuner_config.json")
data = json.loads(cfg_path.read_text())
workspace = pathlib.Path(os.environ["GITHUB_WORKSPACE"])

updated = False
for key, rel in data.get("file_paths", {}).items():
    abs_path = (workspace / rel).resolve()
    if data["file_paths"][key] != str(abs_path):
        data["file_paths"][key] = str(abs_path)
        updated = True

if updated:
    cfg_path.write_text(json.dumps(data, indent=2))
    print(f"Updated {sum(1 for _ in data['file_paths'])} paths")
else:
    print("Paths already normalized")
```

Create `code/scripts/generate_tuning_report.py`:
```python
"""Generate summary report after hyperparameter tuning."""
import datetime as dt
import json
import os
import pathlib

workspace = pathlib.Path(os.environ["GITHUB_WORKSPACE"])
status = os.environ["JOB_STATUS"]
now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

summary_lines = [
    "# Hyperparameter Tuning Summary",
    "",
    f"**Date:** {now}",
    f"**Status:** {status}",
    ""
]

config_path = workspace / "code" / "config.json"
if config_path.exists():
    data = json.loads(config_path.read_text())
    models = sorted(data.get("models", {}).keys())
    summary_lines.append("## Models in config.json")
    summary_lines.extend([f"- {model}" for model in models])
    summary_lines.append("")

(workspace / "tuning_summary.md").write_text("\n".join(summary_lines))

status_payload = {
    "status": status,
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat() + "Z"
}
(workspace / "tuning_status.json").write_text(json.dumps(status_payload, indent=2))
```

Create `code/scripts/validate_forecast_data.py`:
```python
"""Validate forecast data JSON files before deployment."""
import json
import sys

REQUIRED_KEYS = ['generated_at', 'model_rankings', 'cumulative_timelines']

def validate():
    with open('web/data.json') as f:
        data = json.load(f)

    for key in REQUIRED_KEYS:
        if key not in data:
            print(f"FAIL: Missing required key '{key}' in data.json")
            return False

    if not data['model_rankings']:
        print("FAIL: model_rankings is empty")
        return False

    print(f"OK: data.json valid ({len(data['model_rankings'])} models)")
    return True

if __name__ == '__main__':
    sys.exit(0 if validate() else 1)
```

**Step 2: Update `main.yml`**

Add validation step before deployment, add retention-days to artifacts, and add all generated web files to the commit:

```yaml
    - name: Validate forecast data
      run: python code/scripts/validate_forecast_data.py

    - name: Commit and push changes
      if: steps.verify-changed-files.outputs.changed == 'true'
      run: |
        git add web/data.json web/cna_data.json web/forecast_history.json web/pipeline_results.json web/model_info.json
        git commit -m "Update CVE forecast data - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        git push

    - name: Upload artifacts
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: forecast-data
        path: web/data.json
        retention-days: 30
```

**Step 3: Update `monthly_tuning.yml`**

Replace inline Python heredocs with script calls:
```yaml
    - name: Normalize tuner file paths
      run: python code/scripts/normalize_tuner_paths.py
      env:
        GITHUB_WORKSPACE: ${{ github.workspace }}

    - name: Generate tuning report
      if: always()
      run: python code/scripts/generate_tuning_report.py
      env:
        GITHUB_WORKSPACE: ${{ github.workspace }}
        JOB_STATUS: ${{ job.status }}
```

**Step 4: Commit**

```bash
git add code/scripts/ .github/workflows/main.yml .github/workflows/monthly_tuning.yml
git commit -m "ci: extract inline scripts, add data validation, improve workflows

Move inline Python from workflows to code/scripts/. Add JSON
validation before deployment. Add retention-days to daily artifacts."
```

---

## Task 16: Update README

**Files:**
- Modify: `README.md`

**Step 1: Rewrite README.md**

Update to reflect the current state of the project post-overhaul:
- Update version to v0.11
- Remove stale v0.10 "Phoenix" release notes section
- Update model performance table with current data
- Update file structure section to match cleaned-up repo
- Add badges section (CI status)
- Update quick start instructions
- Remove emoji-heavy formatting, keep it clean and professional
- Add section on contributing with ruff/pytest instructions
- Update GitHub Actions documentation references
- Remove "What's New in v0.10" section
- Remove migration instructions from v0.9

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for v0.11 March Update

Refresh project description, update model performance, clean up
file structure references, add CI badges, remove stale v0.10 notes."
```

---

## Task 17: Update Documentation

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/DEPLOYMENT.md`
- Modify: `docs/DEVELOPMENT.md`
- Modify: `docs/API_REFERENCE.md`
- Delete: `docs/RELEASE_NOTES_v0.10.md`

**Step 1: Update docs to reflect new module structure**

- `ARCHITECTURE.md`: Add `core/model_utils.py`, document constraint integration, document extracted CSS and dark mode
- `DEPLOYMENT.md`: Document new test/lint/security workflows
- `DEVELOPMENT.md`: Add ruff and pytest instructions, update contributing workflow
- `API_REFERENCE.md`: Add `fix_hyperparameters()`, `create_model_safe()`, updated `apply_constraints()` signature

**Step 2: Remove stale release notes**

```bash
git rm docs/RELEASE_NOTES_v0.10.md
```

**Step 3: Commit**

```bash
git add docs/
git commit -m "docs: update architecture, deployment, and development guides

Reflect new module structure, CI/CD workflows, and development
tooling (ruff, pytest). Remove stale v0.10 release notes."
```

---

## Task 18: Final Verification

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

**Step 2: Run ruff**

```bash
ruff check code/ tests/
ruff format --check code/ tests/
```

Expected: No errors (or only pre-existing issues in untouched files).

**Step 3: Validate web files load**

```bash
python -m http.server 8000 --directory web &
# Open browser to http://localhost:8000 and verify dashboard loads
```

**Step 4: Verify git log is clean**

```bash
git log --oneline MarchUpdate --not main
```

Expected: ~17 clean, descriptive commits.

**Step 5: Summary commit if needed**

If any loose ends remain, create a final cleanup commit.
