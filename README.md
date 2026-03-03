# CVE Forecast

Predictive analytics platform for CVE (Common Vulnerabilities and Exposures) publications using machine learning and statistical models. Updated daily with automated accuracy tracking.

**Live Dashboard:** [cveforecast.org](https://cveforecast.org)

## Features

- **13+ Forecasting Models** — Statistical (Prophet, ARIMA, TBATS), ML (XGBoost, LightGBM, CatBoost), deep learning (TCN, NBEATS), and baselines
- **Daily Automated Updates** — GitHub Actions pipeline generates fresh forecasts at midnight UTC
- **120+ CNA Forecasts** — Individual predictions for CVE Numbering Authorities with per-organization model selection
- **Real-World Validation** — Historical backtest on current year data with forecast vs. actual comparisons
- **Monthly Self-Tuning** — Automated hyperparameter optimization on the 1st of each month
- **Forecast Tracking** — Historical snapshots track prediction evolution and accuracy over time
- **Accessible Dashboard** — WCAG AA compliant with dark mode, responsive design, and keyboard navigation

## Quick Start

### Prerequisites
- Python 3.10+
- 8GB+ RAM recommended

### Installation

```bash
git clone https://github.com/RogoLabs/CVEForecast.git
cd CVEForecast
pip install -r requirements.txt
```

### Run Forecasts

```bash
# Clone CVE data (required)
git clone --depth 1 https://github.com/CVEProject/cvelistV5.git

# Run full pipeline
python code/run_production_forecast.py
```

### View Results

```bash
python -m http.server 8000 --directory web
# Open http://localhost:8000
```

## Architecture

```
code/
├── core/                  # Base classes and shared utilities
│   ├── base_forecaster.py # Abstract forecaster interface
│   ├── model_utils.py     # Shared parameter fixing and model creation
│   ├── validation_mixin.py# Cross-validation and diagnostics
│   └── data_adapter.py    # Data loading interface
├── adapters/              # Domain-specific implementations
│   ├── cve_adapter.py     # Total CVE forecasting
│   └── cna_adapter.py     # Per-CNA forecasting
├── validation/            # Statistical validation suite
├── diagnostics/           # Residual and horizon analysis
├── tuner/                 # Hyperparameter optimization
├── scripts/               # CI/CD helper scripts
└── run_production_forecast.py  # Main entry point

web/                       # Dashboard (GitHub Pages)
tests/                     # Test suite
docs/                      # Documentation
```

## Model Performance

Real-world backtest accuracy on current year data:

| Rank | Model | MAPE | Performance |
|------|-------|------|-------------|
| 1 | LightGBM | 6.22% | Excellent |
| 2 | KalmanFilter | 6.26% | Excellent |
| 3 | TBATS | 7.21% | Excellent |
| 4 | RandomForest | 9.16% | Good |
| 5 | AutoARIMA | 9.70% | Good |

Full rankings available on the [live dashboard](https://cveforecast.org).

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

### Linting

```bash
pip install ruff
ruff check code/ tests/
ruff format code/ tests/
```

### Configuration

Models and hyperparameters are configured in `code/config.json`. CNA-specific settings are in `code/cna_config.json`.

## CI/CD

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| Daily Forecast | Midnight UTC | Generate forecasts, deploy to GitHub Pages |
| Monthly Tuning | 1st of month, 2 AM UTC | Optimize hyperparameters |
| Tests | On PR/push | Run test suite |
| Lint | On PR | Check code style with ruff |

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Tuning Guide](docs/TUNING_GUIDE.md)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [CVE Project](https://github.com/CVEProject/cvelistV5) for the vulnerability data
- [Darts](https://unit8co.github.io/darts/) for the time series forecasting framework

---

**Version:** 0.11 | **Website:** [cveforecast.org](https://cveforecast.org)
