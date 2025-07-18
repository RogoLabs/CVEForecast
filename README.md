# CVE Forecast

**CVE Forecast** is a sophisticated, self-improving automated platform that leverages advanced hyperparameter optimization and multiple time series forecasting models to predict the number of Common Vulnerabilities and Exposures (CVEs). It provides a comprehensive, data-driven view of future trends in vulnerability disclosures, all accessible through a sleek, interactive web dashboard.

> **Version 0.6 "Karlův mos" 🇨🇿 (July 2025)**: Revolutionary self-improving forecasting system with intelligent hyperparameter optimization, automated GitHub Actions integration, and comprehensive model stability enhancements. This major release transforms CVE Forecast into a truly intelligent, continuously-improving forecasting platform.

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
- **25+ Models Supported**: Comprehensive suite including statistical (Prophet, ARIMA, Theta), tree-based (XGBoost, LightGBM, CatBoost), and deep learning models (TCN, NBEATS, TiDE)
- **Dynamic Forecasting**: Automatically adapts forecast periods based on current date and data availability
- **Optimized for Production**: CPU-optimized models with robust error handling and numerical stability
- **Performance Validation**: Rigorous model evaluation with historical backtesting and performance metrics

### 🛠️ **Enterprise-Grade Infrastructure**
- **Automated CI/CD Pipeline**: Fully automated daily workflow with GitHub Actions for optimization, training, and deployment
- **Robust Error Handling**: Comprehensive error handling and fallback mechanisms throughout the entire pipeline
- **Interactive Web Dashboard**: Real-time visualization of historical data, model comparisons, and performance analytics
- **Complete Documentation**: In-depth technical documentation covering architecture, optimization strategies, and deployment

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss your ideas.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
