# Hyperparameter Tuning Guide

This guide explains how to run hyperparameter tuning for the CVE forecast models.

## Overview

Hyperparameter tuning optimizes model configurations to improve forecast accuracy. The tuning process:
- Tests multiple hyperparameter combinations for each model
- Uses walk-forward validation to measure performance
- Selects the best configuration based on MAPE (Mean Absolute Percentage Error)
- Saves optimized hyperparameters to `code/config.json`

## When to Run Tuning

Run hyperparameter tuning when:
- **Adding new models** to the pipeline
- **Monthly maintenance** (automated via GitHub Actions)
- **Significant data pattern changes** (e.g., major CVE trend shifts)
- **After major code changes** to forecasting logic
- **Performance degradation** detected in forecasts

## Option 1: Manual Tuning (Local)

### Prerequisites
```bash
# Ensure you have the latest CVE data
git clone https://github.com/CVEProject/cvelistV5.git cvelistV5

# Install dependencies
pip install -r requirements.txt
```

### Run Tuning
```bash
# Run comprehensive hyperparameter tuning
python code/tuner/comprehensive_tuner.py
```

### What Happens
1. **Data Loading**: Loads CVE data from `cvelistV5/` directory
2. **Model Testing**: Tests each model with multiple hyperparameter combinations
3. **Validation**: Uses walk-forward validation to measure accuracy
4. **Selection**: Selects best hyperparameters based on MAPE
5. **Saving**: Updates `code/config.json` with optimized configurations

### Expected Duration
- **Full tuning**: 2-4 hours (all 13 models)
- **Single model**: 10-30 minutes

### Output
```
code/config.json - Updated with optimized hyperparameters
```

### Commit Changes
```bash
# Review changes
git diff code/config.json

# Commit optimized configurations
git add code/config.json
git commit -m "🔧 Update optimized hyperparameters - Manual tuning $(date +%Y-%m-%d)"
git push
```

## Option 2: Automated Monthly Tuning (GitHub Actions)

### Automatic Execution
The workflow runs automatically on the **1st of each month at 2 AM UTC**.

### Manual Trigger
You can also trigger tuning manually:

1. Go to **Actions** tab in GitHub
2. Select **"Monthly Hyperparameter Tuning"** workflow
3. Click **"Run workflow"** button
4. Select branch (usually `main`)
5. Click **"Run workflow"**

### Monitoring
- **Progress**: Check the Actions tab for real-time logs
- **Duration**: Typically 2-4 hours
- **Artifacts**: Download tuning results from the workflow run
- **Notifications**: Creates an issue if tuning fails

### What Gets Updated
- `code/config.json` - Optimized hyperparameters
- Workflow artifacts - Tuning summary and results

## Tuning Configuration

### Models Tuned
The tuner optimizes hyperparameters for all enabled models:
- Prophet
- ExponentialSmoothing
- TBATS
- XGBoost
- LightGBM
- CatBoost
- RandomForest
- LinearRegression
- AutoARIMA
- Theta
- FourTheta
- KalmanFilter
- Croston

### Validation Strategy
- **Method**: Walk-forward validation
- **Metric**: MAPE (Mean Absolute Percentage Error)
- **Objective**: Minimize MAPE on validation set

### Search Space
Each model has a predefined hyperparameter search space in `code/tuner/comprehensive_tuner.py`.

## Troubleshooting

### Tuning Takes Too Long
```bash
# Tune specific models only
# Edit code/tuner/comprehensive_tuner.py to disable models
```

### Out of Memory
```bash
# Reduce batch sizes or disable memory-intensive models
# Check system resources before running
```

### Poor Results After Tuning
```bash
# Verify data quality
python code/data_loader.py

# Check for data anomalies
# Review validation metrics in tuner output
```

## Best Practices

1. **Backup Current Config**: Save `code/config.json` before tuning
2. **Review Changes**: Check `git diff` before committing
3. **Test Forecasts**: Run `python code/run_production_forecast.py` to verify
4. **Monitor Performance**: Compare MAPE before/after tuning
5. **Document Changes**: Note significant hyperparameter changes in commit message

## Advanced: Custom Tuning

### Tune Specific Models
Edit `code/tuner/comprehensive_tuner.py`:
```python
# Only tune Prophet and LightGBM
models_to_tune = ['Prophet', 'LightGBM']
```

### Adjust Search Space
Modify hyperparameter ranges in `comprehensive_tuner.py`:
```python
prophet_search_space = {
    'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],  # Add more values
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    # ...
}
```

### Change Validation Period
Adjust validation split in tuner:
```python
train_ratio = 0.8  # Use 80% for training, 20% for validation
```

## Results Interpretation

### Good Tuning Results
- MAPE decreases by 5-20%
- Consistent performance across validation periods
- Reasonable hyperparameter values

### Poor Tuning Results
- MAPE increases or stays the same
- Extreme hyperparameter values
- High variance in validation scores

### Example Output
```
✅ Prophet optimized: MAPE 10.13% → 8.45%
✅ LightGBM optimized: MAPE 6.22% → 5.87%
✅ XGBoost optimized: MAPE 10.39% → 9.12%
```

## Support

For issues or questions:
1. Check workflow logs in GitHub Actions
2. Review tuner output for error messages
3. Verify data quality and availability
4. Check system resources (memory, CPU)

---

**Last Updated**: 2025-10-13
