# CVE Forecast

**CVE Forecast** is a sophisticated, automated platform that leverages over 25 time series forecasting models to predict the number of Common Vulnerabilities and Exposures (CVEs). It provides a comprehensive, data-driven view of future trends in vulnerability disclosures, all accessible through a sleek, interactive web dashboard.

## 🚀 Key Features

- **Advanced Forecasting Engine**: Utilizes a diverse suite of models, from statistical classics like ARIMA and Prophet to cutting-edge deep learning architectures like N-BEATS, N-HiTS, and Transformers.
- **Automated CI/CD Pipeline**: Employs GitHub Actions for a fully automated daily workflow, including data ingestion, model training, evaluation, and deployment.
- **Interactive Web Dashboard**: A user-friendly interface for visualizing historical data, comparing model forecasts, and analyzing performance metrics.
- **Rigorous Model Evaluation**: Systematically ranks models based on a variety of performance metrics, ensuring the most accurate forecasts are always highlighted.
- **In-Depth Technical Documentation**: A comprehensive guide to the system's architecture, data pipeline, and forecasting models.

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
