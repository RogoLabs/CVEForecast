# Temporal Prediction Tracking - Quick Start Guide

**Issue:** No way to track how forecasts change over time  
**Solution:** Implement snapshot-based forecast history tracking  
**Effort:** 2-3 days development  
**Priority:** 🔴 Critical (Issue #1)

---

## Problem Summary

**Current State:**
- Every run generates fresh forecasts
- No history of previous predictions
- Can't answer: "How did our August forecast change from June → July → August?"
- Users can't see forecast convergence or stability

**What We Need:**
```
August 2025 CVE Forecast Evolution:
June 1:  5,000 CVEs predicted (8 weeks out) ─┐
July 1:  5,200 CVEs predicted (4 weeks out) ─┤ Track these!
Aug 1:   5,150 CVEs predicted (actual MTD)  ─┤
Sept 1:  5,080 CVEs ACTUAL ──────────────────┘
```

---

## Implementation Plan (2-3 Days)

### Day 1: Create Tracking Module (4-6 hours)

#### Step 1: Create `code/forecast_tracker.py`

**File:** `code/forecast_tracker.py` (new file)

```python
"""
Forecast tracking module - capture prediction snapshots over time.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ForecastTracker:
    """Track forecast predictions over time for accuracy analysis."""
    
    def __init__(self, history_path: str = "web/forecast_history.json"):
        self.history_path = Path(history_path)
        self.history = self._load_history()
    
    def _load_history(self) -> Dict:
        """Load existing forecast history or create new."""
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data.get('forecast_snapshots', []))} snapshots")
                return data
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
        
        return {
            "version": "1.0",
            "last_updated": datetime.utcnow().isoformat(),
            "forecast_snapshots": [],
            "accuracy_tracking": {},
            "stability_metrics": {}
        }
    
    def add_snapshot(
        self,
        forecasts: Dict[str, Dict[str, float]],
        actuals: Dict[str, float],
        model_performance: Dict[str, Dict],
        snapshot_date: datetime,
        metadata: Optional[Dict] = None
    ):
        """
        Add new forecast snapshot.
        
        Args:
            forecasts: {month: {model: value}} - e.g., {"2025-08": {"Prophet": 4500}}
            actuals: {month: value} - e.g., {"2025-07": 4325}
            model_performance: {model: {metrics}} - e.g., {"Prophet": {"mape": 1.12}}
            snapshot_date: Date of this snapshot
            metadata: Optional info (dataset size, etc.)
        """
        snapshot = {
            "snapshot_id": snapshot_date.strftime("%Y-%m-%d_%H%M%S"),
            "snapshot_date": snapshot_date.isoformat(),
            "generation_time": datetime.utcnow().isoformat(),
            "data_through": max(actuals.keys()) if actuals else None,
            "forecasts": forecasts,
            "model_performance": model_performance,
            "metadata": metadata or {}
        }
        
        # Add actuals if we have new data
        if actuals and len(self.history["forecast_snapshots"]) > 0:
            snapshot["actuals"] = actuals
        
        self.history["forecast_snapshots"].append(snapshot)
        self.history["last_updated"] = datetime.utcnow().isoformat()
        
        # Update derived metrics
        self._update_accuracy_tracking()
        self._update_stability_metrics()
        self._save_history()
        
        logger.info(f"Saved snapshot {snapshot['snapshot_id']}")
    
    def _update_accuracy_tracking(self):
        """Calculate accuracy for months that now have actuals."""
        snapshots = self.history["forecast_snapshots"]
        
        # Find all months with actuals
        months_with_actuals = {}
        for snapshot in snapshots:
            if "actuals" in snapshot:
                months_with_actuals.update(snapshot["actuals"])
        
        # For each completed month, track all forecasts made for it
        for month, actual in months_with_actuals.items():
            forecast_evolution = []
            
            for snapshot in snapshots:
                if month in snapshot["forecasts"]:
                    snapshot_date = datetime.fromisoformat(snapshot["snapshot_date"])
                    month_date = datetime.strptime(month, "%Y-%m")
                    weeks_ahead = (month_date - snapshot_date).days / 7
                    
                    for model, forecast_value in snapshot["forecasts"][month].items():
                        if model == "all_models_avg":
                            continue
                        
                        error = forecast_value - actual
                        error_pct = (error / actual * 100) if actual != 0 else 0
                        
                        forecast_evolution.append({
                            "snapshot_date": snapshot["snapshot_date"],
                            "model": model,
                            "forecast": int(forecast_value),
                            "error": int(error),
                            "error_pct": round(error_pct, 2),
                            "weeks_ahead": round(weeks_ahead, 1)
                        })
            
            if forecast_evolution:
                # Calculate volatility (how much did forecasts change?)
                all_forecasts = [item["forecast"] for item in forecast_evolution]
                volatility = (np.std(all_forecasts) / np.mean(all_forecasts) * 100 
                             if all_forecasts else 0)
                
                self.history["accuracy_tracking"][month] = {
                    "actual": int(actual),
                    "forecasts_over_time": forecast_evolution,
                    "prediction_volatility": round(volatility, 2)
                }
    
    def _update_stability_metrics(self):
        """Calculate how much forecasts changed between runs."""
        snapshots = self.history["forecast_snapshots"]
        
        if len(snapshots) < 2:
            return
        
        model_revisions = {}
        
        for i in range(1, len(snapshots)):
            prev = snapshots[i-1]["forecasts"]
            curr = snapshots[i]["forecasts"]
            
            common_months = set(prev.keys()) & set(curr.keys())
            
            for month in common_months:
                common_models = set(prev[month].keys()) & set(curr[month].keys())
                
                for model in common_models:
                    if model == "all_models_avg":
                        continue
                    
                    prev_val = prev[month][model]
                    curr_val = curr[month][model]
                    
                    if prev_val != 0:
                        revision_pct = abs((curr_val - prev_val) / prev_val * 100)
                        
                        if model not in model_revisions:
                            model_revisions[model] = []
                        model_revisions[model].append(revision_pct)
        
        # Calculate stability scores
        for model, revisions in model_revisions.items():
            if revisions:
                mean_rev = np.mean(revisions)
                stability = 1 / (1 + mean_rev / 10)  # 0=unstable, 1=stable
                
                self.history["stability_metrics"][model] = {
                    "mean_revision_pct": round(mean_rev, 2),
                    "max_revision_pct": round(np.max(revisions), 2),
                    "stability_score": round(stability, 3)
                }
    
    def _save_history(self):
        """Save to disk."""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        return {
            "total_snapshots": len(self.history["forecast_snapshots"]),
            "months_tracked": len(self.history.get("accuracy_tracking", {})),
            "models_tracked": len(self.history.get("stability_metrics", {}))
        }
```

**Test it:**
```bash
python -c "from code.forecast_tracker import ForecastTracker; t = ForecastTracker(); print('✓ Module created')"
```

---

### Day 2: Integrate into Main Pipeline (4-6 hours)

#### Step 2: Modify `code/main.py`

**Add import at top:**
```python
from forecast_tracker import ForecastTracker
```

**Add to `__init__` method:**
```python
def __init__(self, config_path='config.json'):
    # ... existing code ...
    
    # NEW: Initialize forecast tracker
    self.forecast_tracker = ForecastTracker(
        history_path=self.config['file_paths'].get(
            'forecast_history', 
            'web/forecast_history.json'
        )
    )
```

**Add new method after `save_results()` (around line 720):**
```python
def _save_forecast_snapshot(self):
    """Save current forecasts to temporal tracking system."""
    try:
        # Prepare forecasts: {month: {model: value}}
        forecasts_by_month = {}
        
        for model_name, forecast_series in self.final_forecasts.items():
            for ts, val in zip(forecast_series.time_index, forecast_series.values()):
                month = ts.strftime('%Y-%m')
                if month not in forecasts_by_month:
                    forecasts_by_month[month] = {}
                forecasts_by_month[month][model_name] = float(val[0])
        
        # Add ensemble average
        for month in forecasts_by_month:
            values = list(forecasts_by_month[month].values())
            forecasts_by_month[month]['all_models_avg'] = np.mean(values)
        
        # Prepare actuals: {month: value}
        actuals_dict = {}
        historical_data = self._get_historical_data()
        for record in historical_data:
            actuals_dict[record['date']] = record['cve_count']
        
        # Add current month
        current_month_actual = self._get_current_month_actual()
        if current_month_actual['cve_count'] > 0:
            current_month_key = current_month_actual['date'][:7]
            actuals_dict[current_month_key] = current_month_actual['cve_count']
        
        # Prepare model performance
        model_performance = {}
        for name, result in self.model_results.items():
            model_performance[name] = {
                'mape': result['metrics'].get('mape', 0),
                'mae': result['metrics'].get('mae', 0)
            }
        
        # Prepare metadata
        metadata = {
            'models_used': len(self.final_forecasts),
            'dataset_size': len(self.series),
            'forecast_horizon': len(forecasts_by_month)
        }
        
        # Save snapshot
        self.forecast_tracker.add_snapshot(
            forecasts=forecasts_by_month,
            actuals=actuals_dict,
            model_performance=model_performance,
            snapshot_date=self.current_datetime,
            metadata=metadata
        )
        
        # Log summary
        stats = self.forecast_tracker.get_summary_stats()
        self.logger.info(f"Forecast snapshot saved. Total snapshots: {stats['total_snapshots']}")
        
    except Exception as e:
        self.logger.error(f"Failed to save forecast snapshot: {e}")
        # Don't fail the whole run if tracking fails
```

**Modify `save_results()` method (add at end, before final save):**
```python
def save_results(self):
    # ... all existing code ...
    
    # NEW: Save forecast snapshot for temporal tracking
    self.logger.info("Saving forecast snapshot...")
    self._save_forecast_snapshot()
    
    # ... continue with existing save logic ...
```

---

### Day 3: Testing & Validation (4-6 hours)

#### Step 3: Test the Integration

**Test Script:** `tests/test_temporal_tracking.py`

```python
import json
from pathlib import Path

def test_tracking_works():
    """Verify forecast_history.json is created and populated."""
    history_path = Path("web/forecast_history.json")
    
    # Check file exists
    assert history_path.exists(), "forecast_history.json not created"
    
    # Load and validate structure
    with open(history_path) as f:
        history = json.load(f)
    
    # Validate structure
    assert "version" in history
    assert "forecast_snapshots" in history
    assert len(history["forecast_snapshots"]) > 0, "No snapshots saved"
    
    # Check first snapshot structure
    snapshot = history["forecast_snapshots"][0]
    assert "snapshot_id" in snapshot
    assert "snapshot_date" in snapshot
    assert "forecasts" in snapshot
    assert "model_performance" in snapshot
    
    print(f"✓ Found {len(history['forecast_snapshots'])} snapshots")
    print(f"✓ Tracking {len(history.get('accuracy_tracking', {}))} completed months")
    print(f"✓ Monitoring {len(history.get('stability_metrics', {}))} models")
    
    return True

if __name__ == "__main__":
    test_tracking_works()
    print("\n🎉 Temporal tracking is working!")
```

**Run Tests:**

```bash
# 1. Run main pipeline
cd /Users/gamblin/Documents/Github/CVEForecast
python code/main.py

# 2. Verify forecast_history.json created
ls -lh web/forecast_history.json

# 3. Check structure
cat web/forecast_history.json | jq '.forecast_snapshots | length'

# 4. Run validation test
python tests/test_temporal_tracking.py

# 5. Run again to get second snapshot
python code/main.py

# 6. Verify snapshot count increased
cat web/forecast_history.json | jq '.forecast_snapshots | length'
```

---

## Expected Results After Implementation

### 1. New File Created: `web/forecast_history.json`

```json
{
  "version": "1.0",
  "last_updated": "2025-10-05T20:30:00Z",
  "forecast_snapshots": [
    {
      "snapshot_id": "2025-10-05_153000",
      "snapshot_date": "2025-10-05T15:30:00Z",
      "data_through": "2025-09",
      "forecasts": {
        "2025-10": {
          "Prophet": 4500,
          "LightGBM": 4450,
          "all_models_avg": 4475
        }
      },
      "model_performance": {
        "Prophet": {"mape": 1.12, "mae": 41.58}
      }
    }
  ],
  "accuracy_tracking": {},
  "stability_metrics": {}
}
```

### 2. Log Output Shows Tracking

```
INFO - Saving forecast snapshot...
INFO - Saved snapshot 2025-10-05_153000
INFO - Forecast snapshot saved. Total snapshots: 1
```

### 3. After Multiple Runs

```json
{
  "forecast_snapshots": [
    {"snapshot_id": "2025-10-05_153000", ...},
    {"snapshot_id": "2025-10-06_153000", ...},
    {"snapshot_id": "2025-10-07_153000", ...}
  ],
  "accuracy_tracking": {
    "2025-09": {
      "actual": 4325,
      "forecasts_over_time": [
        {"snapshot_date": "2025-09-01", "forecast": 4400, "error": 75},
        {"snapshot_date": "2025-09-15", "forecast": 4350, "error": 25}
      ]
    }
  },
  "stability_metrics": {
    "Prophet": {
      "mean_revision_pct": 2.5,
      "stability_score": 0.87
    }
  }
}
```

---

## Configuration Update

**Add to `code/config.json`:**

```json
{
  "file_paths": {
    "cve_data": "cvelistV5",
    "output_data": "web/data.json",
    "performance_history": "web/performance_history.json",
    "forecast_history": "web/forecast_history.json"
  }
}
```

---

## Success Criteria

After implementation, you should be able to answer:

✅ **"How has our August forecast changed over time?"**
- Query `forecast_history.json` → `accuracy_tracking["2025-08"]`

✅ **"Which models are most stable?"**
- Query `stability_metrics` → sorted by `stability_score`

✅ **"Did our predictions converge as the month approached?"**
- Check `forecasts_over_time` → errors decrease over time

✅ **"How many forecasts have we tracked?"**
- Check `len(forecast_snapshots)`

---

## Troubleshooting

### Issue: forecast_history.json not created

**Solution:**
```bash
# Check file paths in config
cat code/config.json | grep forecast_history

# Verify write permissions
touch web/forecast_history.json
ls -l web/forecast_history.json
```

### Issue: Snapshots not accumulating

**Solution:**
```python
# Check if file is being overwritten vs appended
# In forecast_tracker.py, _load_history() should load existing data
# In add_snapshot(), should APPEND not replace
```

### Issue: accuracy_tracking empty

**Solution:**
- Need at least 2 runs where a month completes between them
- First run: forecasts for October
- Second run: October completes, actuals available
- Second run will populate accuracy_tracking["2025-10"]

---

## Next Steps After Implementation

### Phase 2: Visualization (Week 2)

Once tracking works, create visualization page:

**File:** `web/forecast_evolution.html`

**Features:**
1. **Evolution Chart** - Show how forecast for selected month changed
2. **Stability Dashboard** - Rank models by stability score  
3. **Convergence Analysis** - Plot error reduction over time

### Phase 3: Alerting (Week 3)

Add alerts for unusual changes:
- Forecast revision > 10%
- Sudden stability drop
- Systematic bias detected

---

## Time Breakdown

| Task | Time | Deliverable |
|------|------|-------------|
| Create tracker module | 4-6h | `forecast_tracker.py` |
| Integrate into main.py | 4-6h | Snapshot saving on each run |
| Testing & validation | 4-6h | Verified multi-run accumulation |
| **Total** | **12-18h** | **Working temporal tracking** |

---

## Getting Started NOW

```bash
# 1. Create the tracker module
touch code/forecast_tracker.py
# Copy the ForecastTracker class code above

# 2. Test it independently
python -c "from code.forecast_tracker import ForecastTracker; print('Works!')"

# 3. Integrate into main.py
# Add the import, __init__ change, and _save_forecast_snapshot method

# 4. Run and verify
python code/main.py
cat web/forecast_history.json | jq .

# 5. Run again tomorrow and see snapshots accumulate
```

---

**Full detailed implementation guide:** See `PREDICTION_TRACKING_PLAN.md`

**Status:** Ready to implement on v.10 branch 🚀
