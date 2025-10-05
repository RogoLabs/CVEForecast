# CVE Forecast System - Improvement Roadmap

**Date:** 2025-10-05  
**Timeline:** 6-8 weeks for complete implementation

---

## Priority Overview

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 🔴 P0 | Temporal Prediction Tracking | 3 days | Very High |
| 🔴 P0 | Fix Validation Split Ratios | 1 day | High |
| 🟡 P1 | Implement Data Caching | 2 days | High |
| 🟡 P1 | Add Confidence Intervals | 3 days | High |
| 🟡 P1 | Walk-Forward Validation | 2 days | Medium |
| 🟢 P2 | Enhanced Ensemble | 2 days | Medium |
| 🟢 P2 | Additional Metrics | 2 days | Medium |

---

## Phase 1: Critical Fixes (Week 1-2)

### 1. Temporal Prediction Tracking 🔴 [3 days]

**Implementation:** See `PREDICTION_TRACKING_PLAN.md`

**Quick Actions:**
- Create `code/forecast_tracker.py`
- Integrate into `main.py` and `cna_main.py`
- Create `web/forecast_history.json` structure
- Build visualization page

---

### 2. Fix Validation Split Ratios 🔴 [1 day]

**Problem:** 0.99 split = only 1 month validation

**Solution:**
```python
# code/main.py - enforce minimum validation size
def load_optimized_models(self):
    for model_name, model_config in enabled_models.items():
        split_ratio = model_config.get('optimal_split_ratio', 0.8)
        
        # NEW: Enforce minimum 12 months validation
        min_val_months = 12
        max_split = 1 - (min_val_months / len(self.series))
        split_ratio = min(split_ratio, max_split)
        
        self.logger.info(f"Using adjusted split {split_ratio:.2f} for {model_name}")
```

**Re-tune Models:**
```bash
cd code/tuner
python comprehensive_tuner.py --min-validation-months 12
```

---

## Phase 2: Performance Improvements (Week 3-4)

### 3. Implement Data Caching 🟡 [2 days]

**Create:** `code/data_cache.py`

**Key Features:**
```python
class CVEDataCache:
    def get_cache_key(self, cvelist_path):
        # Use git commit hash as cache key
        return git_head_hash
    
    def load_cached_data(self, cache_key):
        # Load from .cache/processed_data_{hash}.pkl
    
    def save_to_cache(self, data, cache_key):
        # Save processed monthly counts
```

**Expected Speedup:** 300x (10 min → 2 sec)

---

### 4. Add Confidence Intervals 🟡 [3 days]

**Approach:** Conformal Prediction

```python
def generate_forecast_with_intervals(model, train, horizon, alpha=0.1):
    # Train/calibration split
    cal_size = int(len(train) * 0.2)
    proper_train = train[:-cal_size]
    calibration = train[-cal_size:]
    
    # Fit and get calibration errors
    model.fit(proper_train)
    cal_pred = model.predict(len(calibration))
    errors = abs(calibration.values() - cal_pred.values())
    
    # Quantile for intervals
    q = np.quantile(errors, 1 - alpha)
    
    # Forecast with intervals
    forecast = model.predict(horizon)
    return {
        'point': forecast,
        'lower_90': forecast - q,
        'upper_90': forecast + q
    }
```

**Update Output:**
```json
{
  "forecasts": {
    "Prophet": {
      "2025-08": {
        "point": 4500,
        "lower_90": 4100,
        "upper_90": 4900
      }
    }
  }
}
```

---

### 5. Walk-Forward Cross-Validation 🟡 [2 days]

**Replace single split with rolling validation:**

```python
def walk_forward_validation(series, model_class, params, n_splits=6):
    min_train = 60  # months
    test_size = (len(series) - min_train) // n_splits
    
    results = []
    for i in range(n_splits):
        split = min_train + (i * test_size)
        train = series[:split]
        test = series[split:split + test_size]
        
        model = model_class(**params)
        model.fit(train)
        pred = model.predict(len(test))
        
        results.append({
            'fold': i + 1,
            'mape': mape(test, pred),
            'period': f"{test.start_time()} to {test.end_time()}"
        })
    
    return {
        'mean_mape': np.mean([r['mape'] for r in results]),
        'std_mape': np.std([r['mape'] for r in results]),
        'folds': results
    }
```

---

## Phase 3: Model Enhancements (Week 5-6)

### 6. Enhanced Ensemble 🟢 [2 days]

**Weighted ensemble with recency bias:**

```python
class AdaptiveEnsemble:
    def calculate_weights(self, validation_results):
        weights = {}
        for model, results in validation_results.items():
            recent_folds = results['fold_results'][-6:]  # Last 6 months
            recent_mapes = [f['mape'] for f in recent_folds]
            
            # Exponential recency weighting
            recency_weights = np.exp(np.linspace(-1, 0, len(recent_mapes)))
            weighted_mape = np.average(recent_mapes, weights=recency_weights)
            
            # Inverse MAPE for ensemble weight
            weights[model] = 1.0 / (weighted_mape + 0.01)
        
        # Normalize
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
```

---

### 7. Additional Metrics 🟢 [2 days]

**Add to validation:**

```python
def calculate_comprehensive_metrics(actual, forecast):
    return {
        # Accuracy
        'mape': calculate_mape(actual, forecast),
        'mae': calculate_mae(actual, forecast),
        
        # Bias
        'mean_error': np.mean(forecast - actual),
        'bias_pct': np.mean((forecast - actual) / actual) * 100,
        
        # Directional
        'directional_accuracy': calculate_direction_accuracy(actual, forecast),
        
        # Distribution
        'error_std': np.std(forecast - actual),
        'max_error': np.max(np.abs(forecast - actual))
    }
```

---

## Phase 4: Infrastructure (Week 7-8)

### 8. Parallel Processing 🟢 [1 day]

```python
from concurrent.futures import ProcessPoolExecutor

def train_all_models_parallel(models, data):
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(train_model, m, data): m 
                  for m in models}
        
        results = {}
        for future in futures:
            model_name = futures[future]
            results[model_name] = future.result()
    
    return results
```

**Expected Speedup:** 5-8x

---

### 9. Automated Re-tuning Schedule

**Create:** `.github/workflows/retuning.yml`

```yaml
name: Monthly Hyperparameter Retuning

on:
  schedule:
    - cron: '0 0 1 * *'  # 1st of each month
  workflow_dispatch:

jobs:
  retune:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run comprehensive tuner
        run: python code/tuner/comprehensive_tuner.py
      - name: Commit updated config
        run: |
          git config user.name "Tuning Bot"
          git add code/config.json
          git commit -m "Auto-tune: $(date +%Y-%m-%d)"
          git push
```

---

### 10. Data Quality Monitoring

**Create:** `code/data_quality.py`

```python
class DataQualityMonitor:
    def detect_outliers(self, series):
        """Flag unusual months using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
        return outliers.index.tolist()
    
    def detect_regime_change(self, series, window=12):
        """Detect structural breaks in trend."""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        
        # Check for significant trend shifts
        changes = rolling_mean.diff().abs() > 2 * rolling_std
        return changes[changes].index.tolist()
```

---

## Implementation Timeline

### Week 1-2: Critical Fixes
- [ ] Day 1-3: Prediction tracking system
- [ ] Day 4: Fix validation splits
- [ ] Day 5: Re-tune models with correct splits

### Week 3-4: Performance
- [ ] Day 1-2: Data caching implementation
- [ ] Day 3-5: Confidence intervals
- [ ] Day 6-7: Walk-forward validation

### Week 5-6: Model Improvements
- [ ] Day 1-2: Adaptive ensemble
- [ ] Day 3-4: Additional metrics
- [ ] Day 5: Testing and validation

### Week 7-8: Infrastructure
- [ ] Day 1: Parallel processing
- [ ] Day 2-3: Automated re-tuning
- [ ] Day 4-5: Data quality monitoring
- [ ] Day 6-7: Documentation and deployment

---

## Success Criteria

### Technical
- [ ] Forecast history tracks predictions over time
- [ ] Validation uses minimum 12 months of data
- [ ] Data processing < 30 seconds (vs 10+ minutes)
- [ ] All forecasts include confidence intervals
- [ ] Walk-forward validation shows realistic MAPE

### User Experience
- [ ] Users can view forecast evolution charts
- [ ] Model stability rankings visible
- [ ] Confidence intervals displayed on charts
- [ ] Performance improvements noticeable

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Breaking changes to data.json | Implement versioning, maintain backward compatibility |
| Performance regression | Benchmark before/after, implement caching strategically |
| Model accuracy degradation | A/B test new validation approach, rollback if worse |
| Storage growth from history | Implement rotation policy, compress old data |

---

## Post-Implementation

### Monitoring
- Track forecast accuracy weekly
- Monitor system performance metrics
- Review model stability scores
- Check cache hit rates

### Maintenance
- Re-tune models monthly
- Rotate old forecast snapshots quarterly
- Review and update validation thresholds
- Archive historical data annually

---

## Next Steps

1. **Review this roadmap** with team
2. **Create GitHub issues** for each priority
3. **Start with P0 items** (prediction tracking + validation fixes)
4. **Deploy incrementally** with feature flags
5. **Monitor and adjust** based on results

---

**For detailed implementation of prediction tracking, see:** `PREDICTION_TRACKING_PLAN.md`  
**For complete list of issues, see:** `FORECASTING_WEAKNESSES.md`
