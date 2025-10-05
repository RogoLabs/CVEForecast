# CVE Forecast

**CVE Forecast** is a sophisticated, self-improving automated platform that leverages advanced hyperparameter optimization and multiple time series forecasting models to predict the number of Common Vulnerabilities and Exposures (CVEs). It provides a comprehensive, data-driven view of future trends in vulnerability disclosures, all accessible through a sleek, interactive web dashboard.

> **Version 0.9 "Edinburgh" 🏴󠁧󠁢󠁳󠁣󠁴󠁿 (October 2025)**: Year Rollover Automation & Enhanced Forecasting - Complete year rollover readiness for 2026 with fully dynamic YoY growth calculations, automatic chart axis updates, and intelligent forecast period management. Main forecast page now seamlessly transitions across year boundaries with zero manual intervention.

## 🚀 Key Features

### 🧠 **Intelligent Hyperparameter Optimization**
- **Comprehensive Tuner**: Advanced optimization engine that systematically explores hyperparameter spaces for 19+ models
- **Self-Improving Workflow**: Tuner learns from previous runs and builds on discoveries, continuously improving over time
- **Intelligent Search Strategies**: Adaptive grid/random search selection based on model complexity and search space size
- **Production-Ready Results**: Automatically saves optimal configurations and integrates with main forecasting pipeline

### 🔄 **Automated Self-Optimization**
- **Daily GitHub Actions Integration**: Automated hyperparameter tuning runs before each forecast generation
- **Continuous Learning**: System remembers and compares against previous optimization results
- **Smart Config Management**: Automatic backup and update of configuration files with improvement tracking
- **End-to-End Validation**: Complete pipeline from optimization to forecasting with comprehensive testing

### 📊 **Advanced Forecasting Engine**
- **25+ Models Supported**: Comprehensive suite including statistical (Prophet, ARIMA, Theta), tree-based (XGBoost, LightGBM, CatBoost), and deep learning models (TCN, NBEATS, DLinear)
- **CNA-Specific Forecasting**: Individual predictions for 166+ CVE Numbering Authorities with organization-specific model selection
- **Dynamic Forecasting**: Automatically adapts forecast periods based on current date and data availability
- **Optimized for Production**: CPU-optimized models with robust error handling and numerical stability
- **Performance Validation**: Rigorous model evaluation with historical backtesting and MAPE-based performance metrics

### 🛠️ **Enterprise-Grade Infrastructure**
- **Unified CI/CD Pipeline**: Single automated daily workflow with GitHub Actions for CVE and CNA forecast generation and deployment
- **Robust Error Handling**: Comprehensive error handling and fallback mechanisms throughout the entire pipeline
- **Interactive Web Dashboard**: Real-time visualization of historical data, model comparisons, and CNA-specific analytics
- **Complete Documentation**: In-depth technical documentation covering architecture, optimization strategies, and CNA forecasting workflows

## 🌐 Live Dashboard

Experience the full power of CVE Forecast on the live dashboard:

**[cveforecast.org](https://cveforecast.org)**

## 🛠️ Technical Deep Dive

For a comprehensive understanding of the project's architecture, data processing pipeline, forecasting models, and deployment strategy, please refer to our detailed technical documentation:

**[Technical Details Page](web/technical_details.html)**

## 📦 Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/gamblin/CVEForecast.git
    cd CVEForecast
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r code/requirements.txt
    ```

3.  **Run the forecast:**
    ```bash
    python code/main.py
    ```

4.  **View the dashboard locally:**
    Open `web/index.html` in your browser.

## 📅 Annual Maintenance

### Year Rollover (Dec 31, 2025)

The **main forecast system is fully dynamic** and will automatically roll over on January 1, 2026. The following components update automatically:
- ✅ Chart axis limits (Jan 2026 → Jan 2027)
- ✅ YoY growth calculations (compares 2026 vs 2025)
- ✅ Chart descriptions and labels
- ✅ Forecast generation periods
- ✅ Backend data processing

However, the **CNA forecast page** requires minor label updates (5 minutes):

```bash
# 1. Update CNA page labels
# Edit web/cna_forecast.html:
#   Lines 160-170 (table headers):
#     "2024 Published" → "2025 Published"
#     "2025 Forecasted" → "2026 Forecasted"
#     "2026 Forecasted" → "2027 Forecasted"
#     "2024→2025 Growth" → "2025→2026 Growth"
#
#   Lines 245-264 (summary cards):
#     "2024 Published" → "2025 Published"
#     "2025 Forecasted" → "2026 Forecasted"
#     "2024 → 2025 change" → "2025 → 2026 change"

# 2. Optional: Update forecast_end_year in code/config.json
#    (Not required - system auto-detects and uses next year if config is outdated)

# 3. Test the main forecast page
python code/main.py
# Verify web/data.json contains 2026 forecast data

# 4. Commit and deploy
git add web/cna_forecast.html
git commit -m "Update CNA labels for 2026"
git push
```

**Why CNA page needs manual updates**: The CNA page uses a 3-year rolling window design with year-specific logic. A full refactor to make it dynamic would require significant changes to data processing, chart rendering, and table sorting (4-6 hours work). The annual 5-minute label update is safer and more maintainable.

**Expected behavior on Jan 1, 2026**:
- Main page: Automatically shows 2026 forecasts vs 2025 actuals ✅
- CNA page: Data calculations work correctly, labels will be off by one year until updated ⚠️

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss your ideas.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
