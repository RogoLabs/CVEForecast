# March Update - Comprehensive Overhaul Design

**Date**: 2026-03-03
**Branch**: MarchUpdate
**Approach**: Layered commits on a single branch

## Goal

A "big bang" spring cleaning of the entire CVEForecast repository: fix all known issues in the Python codebase, overhaul the web dashboard, harden CI/CD, remove all cruft, and update documentation.

---

## Layer 1: Repository Cleanup (Spring Cleaning)

Remove all stale, empty, and redundant files to start fresh.

### Delete empty files (0 bytes, tracked in git)
- `code/run_forecast_only.sh`
- `code/run_full_pipeline.sh`
- `code/run_week3_validation.py`
- `code/run_week4_diagnostics.py`
- `code/PIPELINE_TESTING.md`
- `web/cna_forecast_clean.html`
- `web/cna-chart.js`
- `web/cna-table.js`
- `web/cna-utils.js`
- `web/test_config_display.html`
- `web/favicon.ico` (will be replaced with real favicon later)

### Delete old tuner artifacts
- All 35 log files in `code/tuner/logs/`
- All 12 config backups in `code/tuner/config_backups/`
- All 9 config backups in `code/tuner/tuner/config_backups/`
- Tuning result JSON files in `code/tuner/results/`

### Delete stale directories and files
- `v.10 - Documentation/` directory (3 empty markdown files)
- `clean_run_20251007_152320.log` (root-level stale log)
- `web/cna_data_test.json` (old test data, 212 KB)

### Move mislocated test
- Move `code/test_unified_architecture.py` to `tests/test_unified_architecture.py`

### Update .gitignore
- Add `*.log` pattern for log files
- Add `.DS_Store`
- Add `code/tuner/logs/`
- Add `code/tuner/config_backups/`
- Add `code/tuner/tuner/config_backups/`
- Add `code/tuner/results/`

---

## Layer 2: Python Code Cleanup

### 2a. Extract shared utilities - `code/core/model_utils.py`
- Move duplicated parameter fixing logic from `cve_adapter.py` and `cna_adapter.py` (ExponentialSmoothing, Theta, LinearRegressionModel fixes)
- Move duplicated current-month-exclusion training logic into base class

### 2b. Complete constraint integration
- Fix TODO at `cve_adapter.py:265` - wire `ForecastConstraints.apply_constraints()`
- Make `forecast_constraints.py` dynamic: replace hardcoded 2024 value (39941) with data-derived value
- Generalize YTD floor calculation (currently assumes 2025)

### 2c. Standardize date/time handling
- Replace manual timezone string replacement in `data_loader.py:61` with `dateutil.parser`
- Standardize on UTC-aware datetime objects across all modules
- Add `python-dateutil` to requirements.txt

### 2d. Code quality fixes
- Replace broad exception handlers with specific types in `data_loader.py`, `cna_adapter.py`
- Move inline import at `base_forecaster.py:161` to module level
- Use context managers for file I/O in `validation_mixin.py:296-305`
- Fix logging levels: DEBUG for verbose, INFO for important events
- Add log rotation in `utils.py`
- Add file size limits to JSON parsing in `data_loader.py`
- Add explicit request timeout to `cna_trend_data.py`
- Fix `cna_adapter.py:332` - restore logging level after temporary change

### 2e. Decompose the tuner
Break `comprehensive_tuner.py` (2,358 LOC) into:
- `tuner/search_spaces.py` - parameter search space definitions
- `tuner/evaluator.py` - model evaluation logic
- `tuner/optimizer.py` - optimization orchestration
- `tuner/reporter.py` - report generation
- `tuner/comprehensive_tuner.py` - slim orchestrator importing the above

### 2f. Add `pyproject.toml`
- Project metadata
- Tool configuration (ruff, pytest)
- Replace or supplement requirements.txt

---

## Layer 3: Unit Tests

### 3a. Core tests - `tests/test_base_forecaster.py`
- Test abstract interface contracts
- Test training with mock models
- Test forecast generation edge cases (empty data, single point)

### 3b. Constraint tests - `tests/test_forecast_constraints.py`
- Test growth floor/ceiling bounds
- Test YTD floor calculation
- Test constraint application on sample forecasts

### 3c. Tracker tests - `tests/test_forecast_tracker.py`
- Test snapshot creation and retrieval
- Test forecast evolution tracking
- Test stability ranking

### 3d. Utility tests - `tests/test_data_loader.py`
- Test date parsing edge cases
- Test rejected CVE filtering
- Test monthly aggregation

### 3e. Pytest configuration
- Add `pytest.ini` or `[tool.pytest]` section in `pyproject.toml`
- Configure test discovery paths
- Add markers for slow tests (integration vs unit)

---

## Layer 4: Web Overhaul

### 4a. Extract shared CSS - `web/styles.css`
- Pull all duplicated `<style>` blocks from 3 HTML files into shared stylesheet
- Define CSS custom properties for theming (colors, spacing, shadows)
- Keep only page-specific styles inline

### 4b. Accessibility (WCAG AA compliance)
- Add `aria-label` to all interactive elements (selects, buttons, tooltips)
- Add `role` attributes to custom components
- Add `scope` to all table headers
- Add `<label>` associations for all form inputs
- Add keyboard navigation for tooltips and expandable rows
- Add visible focus indicators (`:focus-visible`)
- Add `aria-label` or `<title>` to all inline SVGs

### 4c. SEO and metadata
- Add `<meta name="description">` to all pages
- Add Open Graph tags (og:title, og:description, og:type, og:url)
- Add Twitter Card metadata
- Add canonical URL tags
- Improve `<title>` tags to be descriptive
- Add JSON-LD structured data for the website
- Add real favicon (generate from project identity)

### 4d. Dark mode
- Add CSS custom properties for light/dark themes
- Add `@media (prefers-color-scheme: dark)` support
- Add manual toggle button in header with localStorage persistence
- Theme chart colors for dark mode

### 4e. JavaScript improvements
- Add `defer` attribute to all script tags
- Update chart instances via `.update()` instead of destroy/recreate
- Wrap globals in module pattern or IIFE
- Extract magic numbers to named constants
- Memoize filtered model results in `prepareChartData()`

### 4f. Upgrade dependencies
- Upgrade Tailwind CSS CDN reference to v3.x
- Verify Chart.js version is current

---

## Layer 5: CI/CD Hardening

### 5a. Test workflow - `.github/workflows/test.yml`
- Trigger on pull requests and pushes to main
- Run pytest suite with coverage reporting
- Fail on test failures

### 5b. Code quality workflow - `.github/workflows/lint.yml`
- Run `ruff check` for linting
- Run `ruff format --check` for formatting
- Trigger on pull requests

### 5c. Security scanning
- Add `.github/dependabot.yml` for dependency updates
- Add CodeQL analysis workflow

### 5d. Improve existing workflows
- Extract inline Python scripts from `main.yml` and `monthly_tuning.yml` to `code/scripts/`
- Add JSON schema validation step before deployment
- Add `retention-days: 30` to daily workflow artifacts
- Improve error handling and notifications

### 5e. Data validation script - `code/scripts/validate_forecast_data.py`
- Validate JSON structure of data.json and cna_data.json
- Check for required keys, reasonable value ranges
- Used by both CI and local development

---

## Layer 6: Documentation Update

### 6a. README.md overhaul
- Update project description to reflect current state
- Update architecture section
- Update model performance metrics
- Add badges (build status, last updated, license)
- Clean up structure to match current file layout
- Add contributing section
- Add quick-start that actually works

### 6b. Update docs/
- Update ARCHITECTURE.md to reflect new module structure (model_utils, decomposed tuner)
- Update DEPLOYMENT.md with new CI/CD workflows
- Update DEVELOPMENT.md with ruff/pytest instructions
- Update API_REFERENCE.md with new shared utilities
- Remove or archive stale release notes

### 6c. Code-level documentation
- Add/update docstrings for new shared utility functions
- Document constraint integration
- Document the new tuner module structure
