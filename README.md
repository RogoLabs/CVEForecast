# CVE Forecast

**CVE Forecast** is an enterprise-grade, self-improving forecasting platform that predicts Common Vulnerabilities and Exposures (CVEs) using advanced machine learning, statistical models, and automated hyperparameter optimization. The system provides actionable insights into future vulnerability disclosure trends through an intelligent, continuously-evolving pipeline with real-time accuracy tracking.

> **Version 0.10 "Phoenix" 🔥🐦 (October 2025)**: Complete architectural rebirth with unified pipeline, historical backtest validation, forecast accuracy tracking, modular codebase, and production-ready automation. The system now features real-world performance metrics, automated monthly tuning, and comprehensive documentation.

## 🚀 Key Features

### 📊 **Production-Ready Forecasting**
- **13 Optimized Models**: Statistical (Prophet, ARIMA, TBATS), ML (XGBoost, LightGBM, CatBoost), and baseline models
- **Real-World Validation**: Historical backtest on 2025 data (Jan-Sep) with actual vs. predicted comparisons
- **Accuracy Metrics**: MAPE ranging from 6.22% (LightGBM) to 21.65% (Croston) on real 2025 data
- **120+ CNA Forecasts**: Individual predictions for CVE Numbering Authorities with per-organization model selection
- **Dynamic Forecasting**: Automatically forecasts current incomplete month through end of next year

### 🔄 **Intelligent Automation**
- **Unified Pipeline**: Single command (`run_production_forecast.py`) handles CVE + CNA forecasting
- **Daily Updates**: Automated GitHub Actions workflow generates fresh forecasts at midnight UTC
- **Monthly Tuning**: Separate workflow optimizes hyperparameters on the 1st of each month
- **Forecast Tracking**: Historical snapshot system tracks prediction evolution and accuracy over time
- **Zero Downtime**: Continuous deployment to GitHub Pages with automatic rollback on failure

### 🎯 **Accuracy & Transparency**
- **Forecast vs Published Table**: Month-by-month comparison of predictions against actual CVE counts
- **Model Rankings**: Real-time performance leaderboard based on backtest MAPE
- **Historical Tracking**: `forecast_history.json` accumulates prediction snapshots for long-term analysis
- **Performance Badges**: Visual indicators (Excellent < 5%, Good < 10%, Fair < 20%, Poor > 20%)
- **Detailed Metrics**: MAE, MAPE, error percentages, and performance ratings for every model

### 🏗️ **Modern Architecture**
- **Modular Design**: Clean separation between data loading, training, forecasting, and validation
- **Base Classes**: `BaseForecaster` and `ValidationMixin` provide extensible framework
- **Adapters Pattern**: `CVEForecaster` and `CNAForecaster` implement domain-specific logic
- **Configuration-Driven**: Centralized `config.json` with optimized hyperparameters
- **Comprehensive Logging**: Detailed execution logs with progress tracking and error reporting

## 🌐 Live Dashboard

Experience the full power of CVE Forecast on the live dashboard:

**[cveforecast.org](https://cveforecast.org)**

## 🛠️ Technical Deep Dive

For a comprehensive understanding of the project's architecture, data processing pipeline, forecasting models, and deployment strategy, please refer to our detailed technical documentation:

**[Technical Details Page](web/technical_details.html)**

## 📦 Quick Start

### Prerequisites
- Python 3.10+
- 8GB+ RAM recommended
- CVE data repository (auto-cloned by pipeline)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[your-username]/CVEForecast.git
    cd CVEForecast
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Clone CVE data** (required for forecasting):
    ```bash
    git clone --depth 1 https://github.com/CVEProject/cvelistV5.git
    ```

### Running Forecasts

**Option 1: Full Production Pipeline** (Recommended)
```bash
python code/run_production_forecast.py
```
Generates:
- `web/data.json` - CVE forecasts and metrics
- `web/cna_data.json` - CNA forecasts
- `web/forecast_history.json` - Historical tracking
- `web/pipeline_results.json` - Execution summary

**Option 2: CVE Forecasts Only**
```bash
python code/adapters/cve_adapter.py
```

**Option 3: CNA Forecasts Only**
```bash
python code/adapters/cna_adapter.py
```

### View Results

**Local Dashboard:**
```bash
# Simple HTTP server
python -m http.server 8000 --directory web

# Open browser to http://localhost:8000
```

**Production:** Visit [cveforecast.org](https://cveforecast.org)

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design, components, and data flow
- **[API Reference](docs/API_REFERENCE.md)** - Classes, methods, and configuration options
- **[Deployment Guide](docs/DEPLOYMENT.md)** - GitHub Actions, hosting, and CI/CD
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing, testing, and best practices
- **[Tuning Guide](docs/TUNING_GUIDE.md)** - Hyperparameter optimization workflows

## 🎯 Model Performance (2025 Backtest)

Real-world accuracy on Jan-Sep 2025 data:

| Rank | Model | MAPE | MAE | Performance |
|------|-------|------|-----|-------------|
| 1 | LightGBM | 6.22% | 257.44 | 🥇 Excellent |
| 2 | KalmanFilter | 6.26% | 244.33 | 🥈 Excellent |
| 3 | TBATS | 7.21% | 293.67 | 🥉 Excellent |
| 4 | RandomForest | 9.16% | 374.11 | Good |
| 5 | AutoARIMA | 9.70% | 395.78 | Good |
| 6 | ExponentialSmoothing | 9.83% | 400.00 | Good |
| 7 | Prophet | 10.13% | 412.78 | Good |
| 8 | XGBoost | 10.39% | 420.67 | Good |

*Full rankings available in the dashboard's "Model Performance Rankings" section.*

## 🔧 Configuration

### Model Selection
Edit `code/config.json` to enable/disable models:
```json
{
  "models": {
    "Prophet": {
      "enabled": true,
      "hyperparameters": { ... }
    }
  }
}
```

### Forecast Horizon
Automatically forecasts from current month through December of next year. Override in config:
```json
{
  "forecast_end_year": 2026
}
```

### GitHub Actions
- **Daily Forecast**: `.github/workflows/main.yml` (midnight UTC)
- **Monthly Tuning**: `.github/workflows/monthly_tuning.yml` (1st of month, 2 AM UTC)

See [Deployment Guide](docs/DEPLOYMENT.md) for details.

## 🐛 Troubleshooting

### Common Issues

**"No CVE data found"**
```bash
# Clone CVE data repository
git clone --depth 1 https://github.com/CVEProject/cvelistV5.git
```

**"Model training failed"**
```bash
# Check logs for specific model errors
# Disable problematic models in config.json
```

**"Out of memory"**
```bash
# Reduce number of enabled models
# Use CPU-only models (disable deep learning models)
```

See [Development Guide](docs/DEVELOPMENT.md) for more troubleshooting tips.

## 🚀 What's New in v0.10 "Phoenix" 🔥🐦

### Major Changes
- ✨ **Unified Pipeline**: Single command for CVE + CNA forecasting
- 📊 **Historical Backtest**: Real-world validation on 2025 data
- 📈 **Forecast Tracking**: Accuracy monitoring over time
- 🏗️ **Modular Architecture**: Clean, extensible codebase
- 📝 **Comprehensive Docs**: Complete documentation suite
- 🔄 **Automated Tuning**: Monthly hyperparameter optimization
- 🎯 **Performance Metrics**: Transparent accuracy reporting

### Breaking Changes
- `code/main.py` replaced by `code/run_production_forecast.py`
- New data structure in `web/data.json` (includes `forecast_vs_published`)
- Configuration moved to `code/config.json` (from multiple files)

### Migration from v0.9
```bash
# Update to v0.10
git pull origin main

# Install new dependencies
pip install -r requirements.txt

# Run new pipeline
python code/run_production_forecast.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](docs/DEVELOPMENT.md) for:
- Code style guidelines
- Testing requirements
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CVE Project** for maintaining the cvelistV5 repository
- **Darts** library for time series forecasting framework
- **Contributors** who helped shape this release

---

**Version**: 0.10 "Phoenix" 🔥🐦  
**Release Date**: October 2025  
**Status**: Production Ready  
**Website**: [cveforecast.org](https://cveforecast.org)
