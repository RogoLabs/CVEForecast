# CVE Forecast Dashboard

An automated web-based tool that analyzes historical CVE data and uses time series forecasting models to predict the number of new CVEs for the remainder of the current calendar year.

## 🔥 Features

- **Automated Data Collection**: Daily fetching of CVE data from the official CVE Project repository
- **Multiple Forecasting Models**: Implements 6+ time series models including ARIMA, Prophet, ExponentialSmoothing, Kalman Filter, and more
- **Comprehensive Model Evaluation**: Ranks models by multiple metrics (MAPE, RMSE, MAE, sMAPE)
- **Validation Against Actuals**: Real-time comparison of predictions vs actual CVE counts with detailed accuracy metrics
- **Interactive Dashboard**: Beautiful web interface with charts and real-time data visualization
- **Model Performance Transparency**: Detailed validation table showing monthly prediction accuracy
- **Enhanced Accuracy Metrics**: Multi-dimensional model performance assessment
- **GitHub Actions Automation**: Fully automated daily updates
- **GitHub Pages Ready**: Static site deployment compatible

## 🚀 Quick Start

### 1. Setup Repository

```bash
git clone https://github.com/yourusername/CVEForecast.git
cd CVEForecast
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Local Development and Testing

Use the test scripts in the `tests/` folder for local development with real CVE data:

**Quick validation:**
```bash
cd tests
python validate_forecast.py
```

**Development with data subset (faster):**
```bash
cd tests
python generate_sample_data.py
```

**Full integration test:**
```bash
cd tests
python test_forecast.py
```

All test scripts use real CVE data from the official CVElistV5 repository. See `tests/README.md` for detailed information about each test script and performance characteristics.

### 4. Manual Forecast Generation

For direct control over the forecast process:

```bash
# First, clone the CVE data repository
git clone --depth 1 https://github.com/CVEProject/cvelistV5.git cve_data

# Run the forecast analysis
python cve_forecast.py --data-path cve_data --output web/data.json

# Start the local server
python serve.py
```

### 5. View Dashboard

The dashboard will be available at `http://localhost:8000` when using the local server.

## 📊 Dashboard Components

### Summary Cards
- **Total Historical CVEs**: Complete count of analyzed CVE records
- **Best Model**: Top-performing forecasting model
- **Forecast Accuracy**: MAPE score of the best model

### Interactive Chart
- Historical CVE publication data
- Forecasts from multiple models
- Adjustable time ranges (6 months, 1 year, all data)
- Model selection filter

### Model Rankings Table
- Performance comparison of all models with comprehensive metrics (MAPE, RMSE, MAE, sMAPE)
- Color-coded performance indicators
- Detailed accuracy assessment

### Validation Against Actuals Table ⭐ **NEW**
- **Monthly comparison** of predicted vs actual CVE counts for the last 6 months
- **Absolute and percentage error** calculations
- **Color-coded accuracy ratings** (Excellent/Good/Fair/Poor)
- **Summary statistics** including average error and accuracy rate
- **Transparency** in model performance and reliability

## 🤖 Forecasting Models

The system evaluates the following time series models:

### Statistical Models
- **ARIMA**: AutoRegressive Integrated Moving Average
- **ExponentialSmoothing**: Exponential smoothing with trend and seasonality
- **Prophet**: Facebook's forecasting procedure
- **Theta**: Theta method for forecasting

### Machine Learning Models
- **LinearRegression**: Linear regression with time-based features
- **RandomForest**: Random forest with lagged features
- **LightGBM**: Gradient boosting with time series features
- **XGBoost**: Extreme gradient boosting
- **CatBoost**: Categorical boosting

### Deep Learning Models
- **NBEATS**: Neural Basis Expansion Analysis
- **TFT**: Temporal Fusion Transformer
- **RNN**: Recurrent Neural Networks
- **TCN**: Temporal Convolutional Networks
- **Transformer**: Transformer architecture for time series

## 🔄 Automation

### GitHub Actions Workflow

The system runs automatically via GitHub Actions:

- **Schedule**: Daily at midnight UTC
- **Process**: 
  1. Clone CVE data repository
  2. Parse and aggregate CVE publication dates
  3. Train and evaluate all forecasting models
  4. Generate forecasts for remainder of current year
  5. Update web dashboard data
  6. Commit and push changes

### Manual Trigger

You can manually trigger the workflow:
1. Go to the "Actions" tab in your GitHub repository
2. Select "CVE Forecast Daily Update"
3. Click "Run workflow"

## 🔄 Automated Deployment

The project includes GitHub Actions automation for daily updates:

### Workflow Process
1. **Scheduled Execution**: Runs daily at midnight UTC
2. **Data Download**: Downloads the latest CVE data from the official repository
3. **Forecast Generation**: Processes data and generates new forecasts
4. **Dashboard Update**: Updates the web dashboard with new data
5. **Automatic Commit**: Commits updated data back to the repository

### Setting Up Automation
1. Fork this repository to your GitHub account
2. Enable GitHub Actions in your repository settings
3. Optionally, configure GitHub Pages to serve the `web/` directory
4. The workflow will run automatically on schedule

### Manual Trigger
You can also trigger the workflow manually:
- Go to the "Actions" tab in your GitHub repository
- Select "CVE Forecast Daily Update"
- Click "Run workflow"

## 📁 Project Structure

```
CVEForecast/
├── .github/
│   └── workflows/
│       └── main.yml              # GitHub Actions workflow
├── web/
│   ├── index.html               # Dashboard HTML
│   ├── script.js                # Dashboard JavaScript
│   └── data.json                # Generated forecast data
├── cve_forecast.py              # Main forecasting script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore rules
```

## 📈 Data Flow

1. **Data Collection**: Clone CVE repository and parse JSON files
2. **Time Series Creation**: Aggregate CVE counts by publication date
3. **Model Training**: Train multiple forecasting models on historical data
4. **Model Evaluation**: Calculate MAPE scores and rank models
5. **Forecast Generation**: Generate predictions using top-performing models
6. **Data Export**: Save structured data for web dashboard
7. **Visualization**: Interactive charts and statistics in web interface

## 🌐 GitHub Pages Setup

To deploy the dashboard using GitHub Pages:

1. Go to your repository settings
2. Navigate to "Pages" section
3. Select "Deploy from a branch"
4. Choose "main" branch
5. Select "/ (root)" folder
6. Your dashboard will be available at: `https://yourusername.github.io/CVEForecast/web/`

## 🛠️ Configuration

### Environment Variables

No special environment variables are required. The system uses public APIs and repositories.

### Model Parameters

Model parameters can be adjusted in `cve_forecast.py`:

```python
# Example: Adjust NBEATS parameters
NBEATSModel(
    input_chunk_length=30,    # Input sequence length
    output_chunk_length=7,    # Forecast horizon
    n_epochs=50,              # Training epochs
    random_state=42
)
```

## 📊 Data Sources

- **CVE Data**: [CVE Project Official Repository](https://github.com/CVEProject/cvelistV5)
- **Update Frequency**: Daily at midnight UTC
- **Data Format**: JSON files containing CVE metadata and publication dates

## 🔍 Technical Details

### Performance Optimization

- **Shallow Git Clone**: Only fetches latest CVE data to reduce bandwidth
- **Efficient Parsing**: Streams JSON files without loading entire repository in memory
- **Model Caching**: Trains models efficiently with optimized parameters
- **Data Aggregation**: Pre-processes time series for faster model training

### Error Handling

- **Robust Data Parsing**: Skips malformed CVE entries
- **Model Fallbacks**: Continues if individual models fail to train
- **Network Resilience**: Handles repository cloning failures gracefully
- **Data Validation**: Ensures forecast data integrity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CVE Project](https://cve.mitre.org/) for providing public CVE data
- [Darts](https://github.com/unit8co/darts) for the excellent time series forecasting library
- [Chart.js](https://www.chartjs.org/) for interactive data visualization
- [Tailwind CSS](https://tailwindcss.com/) for beautiful styling

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/CVEForecast/issues) page
2. Create a new issue with detailed information
3. Include logs and error messages when applicable

---

**Made with ❤️ for the cybersecurity community**
