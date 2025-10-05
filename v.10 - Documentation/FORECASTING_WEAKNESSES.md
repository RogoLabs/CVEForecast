# CVE Forecast System - Identified Weaknesses

**Date:** 2025-10-05  
**Review Scope:** Forecasting logic, validation methodology, data handling, performance

---

## Critical Issues (High Priority)

### 1. ⚠️ No Temporal Prediction Tracking

**Current State:**
- Forecasts are generated fresh each run with no historical comparison
- No mechanism to track how predictions evolve as new data arrives
- Cannot measure forecast drift or convergence over time
- Website shows only current forecasts, not prediction evolution

**Impact:**
- Users cannot see if forecasts are getting more/less accurate
- No visibility into how predictions change month-over-month
- Impossible to debug systematic forecast bias
- Missing key trust-building metrics

**Example:**
```
When we forecast in July 2025 that August would have 5,000 CVEs:
- August 1:  Predicted 5,000 CVEs
- August 15: Updated to 5,200 CVEs (with 2,100 actual so far)
- September 1: Final actual was 5,150 CVEs

Current system: Only shows 5,150 vs last prediction
Needed system: Shows evolution: 5,000 → 5,200 → 5,150 (actual)
```

**Files Affected:**
- `code/main.py` - No snapshot saving logic
- `code/cna_main.py` - No historical tracking
- `web/data.json` - Only current forecasts

---

### 2. ⚠️ Validation Data Mismatch

**Problem:**
```json
"LightGBM": {
  "optimal_split_ratio": 0.99,  // Only ~1 month for validation!
  "tuning_results": {
    "mape": 0.03719807943546785  // 0.037% - unrealistically low
  }
}
```

**Analysis:**
- Split ratio of 0.99 on 100 months = only 1 month validation data
- XGBoost at 0.99 = 1 month validation, reporting 0.064% MAPE
- Prophet at 0.99 = 1 month validation, reporting 1.12% MAPE
- These results are overly optimistic and not representative

**Root Cause:**
- Lines 114, 208 in `code/main.py` use `optimal_split_ratio` directly
- Comprehensive tuner optimized for best MAPE, which favors high splits
- No minimum validation size constraint
- No cross-validation or rolling window validation

**Evidence from config.json:**
- LightGBM: split_ratio=0.99, MAPE=0.037%
- XGBoost: split_ratio=0.99, MAPE=0.064%
- LinearRegression: split_ratio=0.99, MAPE=0.054%
- Prophet: split_ratio=0.99, MAPE=1.12%
- RandomForest: split_ratio=0.70, MAPE=26.55% (more realistic)

**Impact:**
- Inflated confidence in model performance
- Risk of overfitting to recent patterns
- Poor generalization to future months
- Model rankings may not reflect true accuracy

---

### 3. ⚠️ Performance Bottleneck: Current Month Counting

**Problem:**
```python
# code/main.py lines 308-430
def _get_current_month_actual(self):
    # Scans ALL 284,000+ JSON files every single run!
    json_files = list(cve_data_path.rglob("cves/**/*.json"))
    for json_file in json_files:
        # Parse each file to count current month CVEs
```

**Measurements:**
- Processing 284,968 CVE JSON files
- Takes 5-10 minutes per run
- Duplicates work between main and CNA pipelines
- No caching or incremental updates

**Files Affected:**
- `code/main.py` lines 314-380
- `code/cna_main.py` lines 158-167 (similar scanning)
- `code/data_loader.py` lines 29-68 (full dataset scan)

**Impact:**
- Long run times (10+ minutes just for current month count)
- Wasted computation on unchanged data
- Inefficient CI/CD pipeline execution
- Poor developer experience

---

### 4. ⚠️ No Confidence Intervals

**Current Output:**
```json
"Prophet": {"2025-08": 4500}
```

**What's Missing:**
```json
"Prophet": {
  "2025-08": {
    "point_estimate": 4500,
    "lower_80": 4100,
    "upper_80": 4900,
    "lower_95": 3800,
    "upper_95": 5200
  }
}
```

**Impact:**
- Users cannot assess prediction reliability
- No uncertainty quantification
- Missing context for decision-making
- Cannot communicate model confidence
- High and low estimates would provide valuable planning context

**Technical Note:**
- Darts supports quantile regression for tree models
- Can implement conformal prediction for all models
- Would require data structure changes in `web/data.json`

---

## Moderate Issues

### 5. Weak Model Ensemble Strategy

**Current Approach:**
```python
# Simple average of top 5 models
ensemble_size = self.config['model_evaluation']['ensemble_size']
top_models = sorted(models, key=lambda x: x['metrics']['mape'])[:ensemble_size]
# All weighted equally
```

**Problems:**
- All models weighted equally regardless of recent performance
- No consideration of model strengths for different patterns
- Fixed ensemble size (5) may not be optimal
- Poor models dilute ensemble quality

**Better Approach:**
- Weighted ensemble based on inverse MAPE with recency bias
- Dynamic model selection based on data regime detection
- Adaptive weights that respond to performance changes
- Consider Bayesian model averaging or stacking

---

### 6. Limited Validation Metrics

**Current Metrics:**
- MAPE (primary) - Mean Absolute Percentage Error
- MAE, MASE, RMSSE (often set to 0 in config)

**Missing Critical Metrics:**

**A. Forecast Bias**
```python
# Do we systematically over or under-predict?
bias = np.mean(forecasts - actuals)
# Positive = overestimate, Negative = underestimate
```

**B. Directional Accuracy**
```python
# Do we correctly predict increases vs decreases?
direction_correct = np.mean(
    np.sign(forecasts[1:] - forecasts[:-1]) == 
    np.sign(actuals[1:] - actuals[:-1])
)
```

**C. Peak Detection Accuracy**
```python
# How well do we predict unusual spikes?
# (log4j, Spectre, Meltdown, etc.)
```

**D. Calibration**
```python
# Are our confidence intervals properly calibrated?
# (Should 80% intervals contain 80% of actuals)
```

---

### 7. Data Quality & Preprocessing Gaps

**Current Preprocessing:**
- Simple rejected CVE filtering (`data_loader.py` lines 46-55)
- Missing months filled with zeros (line 86)
- No outlier detection
- No seasonality decomposition
- No trend break detection

**Problems:**

**A. Outlier Handling**
```
Major disclosure events create spikes:
- Dec 2021: log4j (1,000+ CVEs in days)
- Jan 2018: Spectre/Meltdown
- These distort model training
```

**B. Zero-Filling**
```python
# Line 86 in data_loader.py
monthly_counts = monthly_counts.reindex(full_date_range, fill_value=0)
# Should distinguish: truly 0 vs missing data
```

**C. No Anomaly Detection**
- Models learn patterns from anomalous events
- No mechanism to flag unusual months
- No adaptive response to regime changes

**Impact:**
- Models may overfit to historical anomalies
- Forecasts unreliable during atypical periods
- No warning system for unusual predictions

---

### 8. Hyperparameter Staleness

**Current State:**
```json
"comprehensive_tuning": {
  "last_tuned_at": "2025-07-31T04:03:21.183828",
  "best_model": "LightGBM",
  "best_mape": 0.03719807943546785
}
```

**Tuning Dates (from config.json):**
- LightGBM: July 31, 2025
- XGBoost: July 30, 2025
- Prophet: August 1, 2025
- TiDE: September 28, 2025
- Various others: July-September 2025

**Questions:**
1. Are hyperparameters tuned 3+ months ago still optimal?
2. How often should re-tuning occur?
3. Are we overfitting hyperparameters to historical patterns?
4. Do current hyperparameters generalize to future data regimes?

**Recommendation:**
- Establish re-tuning schedule (monthly? quarterly?)
- Track hyperparameter stability over time
- Implement drift detection for parameter performance
- Consider online learning approaches

---

### 9. CNA Pipeline Model Selection Issues

**Current Approach:**
```python
# cna_main.py lines 291-298
model_candidates = [
    ('ExponentialSmoothing', ExponentialSmoothing, params),
    ('LightGBM', LightGBMModel, params),
    ('XGBoost', XGBModel, params),
    ('LinearRegression', LinearRegressionModel, params),
    ('Prophet', DartsProphet, params),
]
```

**Problems:**

**A. Fixed 6-Month Validation Window**
- Small CNAs may not have enough data for 6-month holdout
- Large CNAs might benefit from longer validation
- No adaptive validation sizing

**B. Hard-Coded Fallback**
```python
# Lines 344-350
if not valid_scores:
    logger.warning("All models failed validation, using LightGBM as fallback")
    return {'best_model': 'LightGBM', 'mape_score': 100.0}
```
- Fallback doesn't consider data characteristics
- No alternative strategies for poor-quality data

**C. Parameter Cleaning Issues**
```python
# Lines 308-330 - Complex parameter filtering per model
# Fragile logic that may break with config changes
```

---

## Lower Priority Issues

### 10. Forecast Horizon Inflexibility

**Current:**
```python
# main.py lines 260-269
forecast_end_year = self.config['model_evaluation'].get('forecast_end_year', 2026)
forecast_end_month = self.config['model_evaluation'].get('forecast_end_month', 1)
```

**Problems:**
- Hardcoded to forecast through January 2026
- Requires manual config updates as time progresses
- Different horizons for main vs CNA pipelines
- Website expects specific date ranges

---

### 11. No Parallel Processing

**Current:**
- Sequential model training (one model at a time)
- No multi-core utilization for model evaluation
- No batch processing for CNA forecasts

**Potential Speedup:**
```python
# Could parallelize:
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(train_model, m) for m in models]
    results = [f.result() for f in futures]
```

**Expected Improvement:**
- 5-8x speedup with 8-core parallelization
- Especially beneficial for CNA pipeline (250+ CNAs)

---

### 12. Website Integration Limitations

**Current:**
- Static JSON output (`web/data.json`, `web/cna_data.json`)
- No real-time updates
- No API for querying specific dates/models
- No interactive forecast exploration

**Missing Features:**
- Custom date range selection
- Model-specific views
- Downloadable data exports
- Forecast methodology explanations
- Historical accuracy dashboards

---

## Summary Statistics

| Category | Count | Priority |
|----------|-------|----------|
| Critical Issues | 4 | High |
| Moderate Issues | 5 | Medium |
| Lower Priority | 3 | Low |
| **Total Issues** | **12** | - |

**Most Critical:**
1. No temporal prediction tracking
2. Validation data mismatch (0.99 split ratio)
3. Performance bottleneck (full dataset scans)
4. Missing confidence intervals

**Next Steps:**
See `PREDICTION_TRACKING_PLAN.md` and `IMPROVEMENT_ROADMAP.md` for detailed implementation recommendations.
