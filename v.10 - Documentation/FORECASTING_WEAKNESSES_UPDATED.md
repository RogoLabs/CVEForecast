# CVE Forecast System - Remaining Weaknesses (v.10 Updated)

**Date:** 2025-10-06  
**Review Scope:** Post-v.10 implementation status  
**Updated After:** Issues #1, #2, #4 resolved

---

## ✅ Resolved in v.10

### ~~1. No Temporal Prediction Tracking~~ ✅ RESOLVED
**Status:** Implemented via `forecast_tracker.py`
- Forecast snapshots saved to `web/forecast_history.json`
- Tracks forecast evolution over time
- Calculates model stability metrics
- Measures prediction convergence
- **Impact:** Users can now see forecast changes and reliability

### ~~2. Validation Data Mismatch~~ ✅ RESOLVED  
**Status:** Fixed via minimum validation enforcement
- Enforces 12-month minimum validation period
- Adjusted 10 out of 11 models from 1-4 months to 11+ months
- Logs warnings when adjustments made
- **Impact:** More reliable model evaluation, better generalization

### ~~4. No Confidence Intervals~~ ✅ RESOLVED
**Status:** Implemented via conformal prediction
- 80% and 95% confidence intervals for all forecasts
- Works with 4 out of 5 models
- New `forecast_intervals` section in data.json
- **Impact:** Users can assess prediction reliability and plan for scenarios

---

## 🔴 Critical Issues Remaining

### 1. **Tuner Encourages Invalid Split Ratios**

**Problem:**
```python
# code/tuner/comprehensive_tuner.py lines 426, 441, 455, etc.
"split_ratios": [0.70, 0.72, 0.74, ..., 0.96, 0.98, 0.99]
#                                                      ^^^^ 
# PROBLEM: 0.99 is STILL in the search space!
```

**What Happens:**
1. Tuner tests 19 different split ratios including 0.99
2. Models with 0.99 split get artificially low MAPE (1 month validation)
3. Tuner selects 0.99 as "optimal" because it has lowest MAPE
4. Config.json gets updated with 0.99 split
5. **Our Issue #2 fix catches this at runtime**, but root cause persists

**Current State (After Issue #2 fix):**
```
✅ Runtime: main.py enforces 12-month minimum (fixed)
❌ Tuner: Still recommends 0.99 splits (not fixed)
```

**Analogy:**
> It's like we fixed the safety belt (main.py catches bad splits), 
> but the driver (tuner) is still steering toward the cliff (recommending 0.99).

**Impact:**
- **MEDIUM Risk:** Runtime enforcement prevents bad validation
- **HIGH Waste:** Tuner spends time evaluating invalid configurations
- **Confusing:** Logs show "adjusted from 0.99" after every tuning
- **Misleading:** Tuner reports "optimal split = 0.99" (not actually optimal)

**Evidence:**
```python
# All models have 0.99 in search space
"Prophet": {
    "split_ratios": [..., 0.96, 0.98, 0.99],  # Line 426
},
"XGBoost": {
    "split_ratios": [..., 0.96, 0.98, 0.99],  # Line 441
},
"LightGBM": {
    "split_ratios": [..., 0.96, 0.98, 0.99],  # Line 455
},
# ... all 16+ models have same pattern
```

**Fix Required:**
```python
# Remove invalid splits from search space
"split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 
                 0.82, 0.84, 0.85, 0.86, 0.88, 0.90],  # Stop at 0.90
# With 106 months data:
# 0.90 = 11 months validation ✓
# 0.92 = 8 months validation (borderline)
# 0.94 = 6 months validation (too small)
# 0.96 = 4 months validation ❌
# 0.98 = 2 months validation ❌
# 0.99 = 1 month validation ❌❌❌
```

**Recommended Action:**
- Update `comprehensive_tuner.py` lines 426-637
- Remove splits > 0.90 from all model search spaces
- Re-run tuner to get truly optimal parameters
- Priority: **HIGH** (should fix in v.10)

---

### 2. **Tuner Hyperparameter Staleness**

**Problem:**
```json
// From config.json
"comprehensive_tuning": {
  "last_tuned_at": "2025-07-31T04:03:21",  // 3+ months ago!
  "best_model": "LightGBM",
  "best_mape": 0.037  // Based on invalid 0.99 split
}
```

**Current Tuning Dates:**
- LightGBM: July 31, 2025 (tuned with 0.99 split)
- XGBoost: July 30, 2025 (tuned with 0.99 split)
- Prophet: August 1, 2025 (tuned with 0.99 split)
- Most models: July-September 2025

**Questions:**
1. **Are 3-month-old hyperparameters still optimal?**
   - Data has grown by ~9 months since tuning
   - Patterns may have shifted
   - Answer: Probably not

2. **Were they ever optimal?**
   - All tuned with invalid 0.99 splits
   - MAPE scores artificially low
   - Rankings potentially wrong
   - Answer: No, they were optimized for invalid validation

3. **What happens after Issue #2 fix?**
   - Runtime enforces 0.89 split (ignoring tuned 0.99)
   - Using hyperparameters tuned for different split ratio
   - May not be optimal for 0.89 split
   - Answer: Sub-optimal configuration

**Impact:**
- **Current Performance:** Unknown if optimal
- **Model Rankings:** Based on invalid validation
- **Hyperparameters:** Optimized for wrong validation setup

**Fix Required:**
1. Fix tuner search space (Issue #1 above)
2. Re-tune all models with correct validation
3. Establish re-tuning schedule (monthly? quarterly?)
4. Track hyperparameter stability over time

**Recommended Action:**
- Fix tuner split ratios first
- Re-run comprehensive tuner (4-8 hours)
- Compare new vs old MAPE scores
- Update model rankings
- Priority: **MEDIUM** (can wait until Issue #1 fixed)

---

## 🟡 Moderate Issues Remaining

### 3. **Performance Bottleneck: Current Month Counting**

**Status:** ⏸️ **Intentionally Not Fixed**

**Reason:** GitHub Actions overnight runs don't need speed optimization

**Analysis:**
- Scans 312,000+ CVE JSON files per run
- Takes ~2 minutes of the total runtime
- **Your use case:** Overnight automated runs (time doesn't matter)

**Decision:** Skip this optimization (correct choice for your workflow)

**When to revisit:**
- If moving to real-time/interactive forecasting
- If hitting GitHub Actions timeout limits
- If running multiple times per day manually

**Priority:** **LOW** (not needed for your use case)

---

### 4. **Weak Model Ensemble Strategy**

**Current Approach:**
```python
# Simple average of top 5 models
ensemble_size = 5
top_models = sorted(models, key=lambda x: x['mape'])[:5]
# All weighted equally
```

**Problems:**
- All models weighted equally (simple average)
- No consideration of:
  - Recent performance trends
  - Model strengths for different patterns
  - Forecast diversity
  - Prediction confidence

**Better Approach:**
```python
# Weighted ensemble based on inverse MAPE with recency bias
weights = []
for model in top_models:
    # Weight inversely proportional to MAPE
    # Recent performance weighted higher
    weight = (1 / model['mape']) * recency_factor
    weights.append(weight)

weighted_forecast = sum(w * f for w, f in zip(weights, forecasts))
```

**Expected Impact:**
- 5-10% improvement in ensemble accuracy
- Better handling of regime changes
- More responsive to performance shifts

**Priority:** **MEDIUM** (nice to have, not critical)

---

### 5. **Limited Validation Metrics**

**Current Metrics:**
- MAPE (primary)
- MAE, MASE, RMSSE (tracked but not emphasized)

**Missing Metrics:**

**A. Forecast Bias**
```python
bias = np.mean(forecasts - actuals)
# Do we systematically over/under-predict?
```

**B. Directional Accuracy**
```python
direction_correct = np.mean(
    np.sign(forecasts[1:] - forecasts[:-1]) == 
    np.sign(actuals[1:] - actuals[:-1])
)
# Do we correctly predict increases vs decreases?
```

**C. Confidence Interval Calibration**
```python
# After Issue #4, we can now measure this!
coverage_80 = np.mean(
    (actuals >= intervals['lower_80']) & 
    (actuals <= intervals['upper_80'])
)
# Should be ~0.80 if properly calibrated
```

**Priority:** **MEDIUM** (temporal tracking enables this analysis)

---

### 6. **Data Quality & Preprocessing Gaps**

**Current Preprocessing:**
- Simple rejected CVE filtering
- Missing months filled with zeros
- No outlier detection
- No anomaly flagging

**Problems:**

**A. Outlier Events**
```
Major disclosure events create spikes:
- Dec 2021: log4j (1,000+ CVEs in days)
- Jan 2018: Spectre/Meltdown
- These distort model training
```

**B. Zero-Filling Logic**
```python
monthly_counts = monthly_counts.reindex(full_date_range, fill_value=0)
# Can't distinguish: truly 0 CVEs vs missing data
```

**Impact:**
- Models may overfit to historical anomalies
- No warning system for unusual predictions
- Forecasts unreliable during atypical periods

**Priority:** **MEDIUM** (affects forecast quality)

---

### 7. **CNA Pipeline Model Selection Issues**

**Fixed 6-Month Validation Window:**
- Small CNAs may not have enough data
- Large CNAs might benefit from longer validation
- No adaptive validation sizing

**Hard-Coded Fallback:**
```python
if not valid_scores:
    logger.warning("All models failed, using LightGBM as fallback")
    return {'best_model': 'LightGBM', 'mape_score': 100.0}
```
- Fallback doesn't consider data characteristics
- No alternative strategies

**Priority:** **LOW** (CNA pipeline working adequately)

---

## 🟢 Lower Priority Issues

### 8. **Forecast Horizon Inflexibility**

**Current:**
- Hardcoded to January 2026
- Requires manual config updates
- **Note:** Dynamic forecasting was implemented (Memory: d779fe7f)
- **Status:** Partially resolved, could be improved

**Priority:** **LOW** (working adequately)

---

### 9. **No Parallel Processing**

**Current:**
- Sequential model training
- No multi-core utilization
- No batch processing for CNAs

**Potential:** 5-8x speedup with 8-core parallelization

**Your Use Case:** Not needed for overnight runs

**Priority:** **LOW** (not needed)

---

### 10. **Website Integration Limitations**

**Current:**
- Static JSON output
- No real-time updates
- No API for querying
- No interactive exploration

**Missing:**
- Custom date range selection
- Model-specific views
- Downloadable exports
- Methodology explanations
- Historical accuracy dashboards

**Note:** v.11 frontend will address visualization

**Priority:** **LOW** (frontend work, not backend)

---

## 📊 Updated Summary Statistics

| Category | Resolved | Remaining | Priority |
|----------|----------|-----------|----------|
| **Critical** | 3 | 2 | HIGH |
| **Moderate** | 0 | 5 | MEDIUM |
| **Lower** | 0 | 3 | LOW |
| **Total** | **3/12** | **10/12** | - |

### Issues by Status

**✅ Resolved (v.10):**
1. ~~No temporal prediction tracking~~ → `forecast_tracker.py`
2. ~~Validation data mismatch~~ → Minimum validation enforcement
3. ~~No confidence intervals~~ → Conformal prediction

**🔴 Critical Remaining:**
1. **Tuner encourages invalid split ratios** (HIGH priority for v.10)
2. **Tuner hyperparameter staleness** (MEDIUM priority)

**🟡 Moderate Remaining:**
3. Performance bottleneck (⏸️ intentionally skipped)
4. Weak ensemble strategy
5. Limited validation metrics
6. Data quality gaps
7. CNA pipeline issues

**🟢 Lower Priority:**
8. Forecast horizon inflexibility
9. No parallel processing
10. Website limitations

---

## 🎯 Recommended Actions for v.10

### **Priority 1: Fix Tuner Split Ratios (1 hour)**

**File:** `code/tuner/comprehensive_tuner.py`

**Change:** Lines 426-637 (all model definitions)

**Before:**
```python
"split_ratios": [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 
                 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 
                 0.94, 0.95, 0.96, 0.98, 0.99],
```

**After:**
```python
"split_ratios": [0.70, 0.75, 0.80, 0.85, 0.88, 0.90],
# Removed invalid splits (>0.90)
# Reduced search space (was 19 → now 6)
# Faster tuning, valid results
```

**Impact:**
- Tuner finds truly optimal splits
- No more "adjusted from 0.99" warnings
- More efficient tuning (less time wasted)
- Aligned with runtime validation enforcement

**Effort:** 1 hour (find/replace + testing)

### **Priority 2: Re-run Tuner (4-8 hours automated)**

**After fixing split ratios:**
```bash
cd code/tuner
python comprehensive_tuner.py
# Let it run overnight via GitHub Actions
```

**Benefits:**
- Get accurate MAPE scores
- Proper model rankings
- Hyperparameters optimized for correct validation
- Confidence in model selection

**Effort:** Passive (automated run)

### **Priority 3: Evaluate Results (30 min)**

**Compare:**
- Old MAPE scores (with 0.99 split)
- New MAPE scores (with valid splits)
- Model ranking changes
- Performance improvements

**Document findings in v.10 documentation**

---

## 🏁 v.10 Completion Checklist

**Resolved:**
- [x] Issue #1: Temporal Tracking
- [x] Issue #2: Validation Splits (runtime)
- [x] Issue #4: Confidence Intervals

**Remaining for v.10:**
- [ ] Fix tuner split ratios
- [ ] Re-run comprehensive tuner
- [ ] Validate new hyperparameters
- [ ] Document tuner improvements

**Deferred to v.11:**
- Enhanced ensemble strategy
- Additional validation metrics
- Website frontend for new features

**Intentionally Skipped:**
- Performance optimization (not needed)
- Parallel processing (not needed)

---

## 📈 Impact Assessment

### **What v.10 Delivers:**

✅ **Transparency:** Temporal tracking shows forecast evolution  
✅ **Reliability:** Better validation (11+ months vs 1 month)  
✅ **Trust:** Confidence intervals quantify uncertainty  
✅ **Accuracy:** After tuner fix, truly optimal parameters  

### **Remaining Work:**

🔴 **Fix tuner** (1 hour) → Prevents future 0.99 split recommendations  
🟡 **Enhancements** (v.11) → Ensemble improvements, metrics, frontend  

---

## 💡 Conclusion

**v.10 Progress:** Excellent! 3 out of 4 attempted issues resolved.

**Critical Discovery:** Tuner still recommends invalid splits (0.99)
- Runtime fix (Issue #2) catches this ✅
- But root cause (tuner search space) remains ❌
- Easy fix: Remove splits > 0.90 from tuner

**Recommendation:** Fix tuner in v.10 before calling it complete
- 1 hour to fix search space
- 4-8 hours to re-tune (automated)
- Results in truly optimal configuration

**Then v.10 is production-ready!**

---

**Updated:** 2025-10-06  
**Status:** 3/4 resolved, 1 fix recommended  
**Next:** Fix tuner split ratios, re-run tuner, document results
