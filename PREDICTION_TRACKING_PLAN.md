# Prediction Tracking Implementation Plan

**Purpose:** Track how CVE forecasts change over time to measure accuracy evolution and build user trust.

**Status:** Ready for Implementation  
**Estimated Effort:** 2-3 days development + 1 day testing

---

## Overview

Enable the CVE Forecast system to capture snapshots of predictions over time, allowing users to:
1. See how forecasts for a specific month evolved as it approached
2. Measure forecast stability and convergence
3. Track model accuracy improvements over time
4. Build trust through transparency

**Example Use Case:**
```
August 2025 Forecast Evolution:
- June 1:    Predicted 5,000 CVEs (8 weeks out)
- July 1:    Updated to 5,200 CVEs (4 weeks out)
- July 15:   Updated to 5,180 CVEs (2 weeks out)
- August 1:  Updated to 5,150 CVEs (with 2,100 actual MTD)
- September 1: Actual was 5,080 CVEs

Accuracy: June prediction was 1.6% low, improved to 1.4% high by July
```

---

## Phase 1: Data Structure Design

### New File: `web/forecast_history.json`

**Structure:**
```json
{
  "version": "1.0",
  "last_updated": "2025-10-05T15:30:00Z",
  
  "forecast_snapshots": [
    {
      "snapshot_id": "2025-07-01_000000",
      "snapshot_date": "2025-07-01T00:00:00Z",
      "generation_time": "2025-07-01T03:15:42Z",
      "data_through": "2025-06-30",
      
      "forecasts": {
        "2025-08": {
          "Prophet": 4500,
          "LightGBM": 4450,
          "XGBoost": 4480,
          "LinearRegression": 4520,
          "all_models_avg": 4487
        },
        "2025-09": {
          "Prophet": 4600,
          "LightGBM": 4550,
          "XGBoost": 4580,
          "LinearRegression": 4620,
          "all_models_avg": 4587
        }
      },
      
      "model_performance": {
        "Prophet": {"mape": 1.12, "mae": 41.58, "validation_months": 1},
        "LightGBM": {"mape": 0.037, "mae": 1.37, "validation_months": 1},
        "XGBoost": {"mape": 0.064, "mae": 2.29, "validation_months": 1},
        "LinearRegression": {"mape": 0.054, "mae": 2.17, "validation_months": 1}
      },
      
      "metadata": {
        "models_used": 4,
        "dataset_size": 100,
        "forecast_horizon": 7
      }
    },
    
    {
      "snapshot_id": "2025-08-01_000000",
      "snapshot_date": "2025-08-01T00:00:00Z",
      "generation_time": "2025-08-01T03:22:18Z",
      "data_through": "2025-07-31",
      
      "actuals": {
        "2025-07": 4325
      },
      
      "forecasts": {
        "2025-08": {
          "Prophet": 4520,
          "LightGBM": 4470,
          "XGBoost": 4500,
          "LinearRegression": 4540,
          "all_models_avg": 4507
        },
        "2025-09": {
          "Prophet": 4650,
          "LightGBM": 4580,
          "XGBoost": 4610,
          "LinearRegression": 4660,
          "all_models_avg": 4625
        }
      },
      
      "model_performance": {
        "Prophet": {"mape": 1.08, "mae": 40.12, "validation_months": 1},
        "LightGBM": {"mape": 0.035, "mae": 1.29, "validation_months": 1}
      }
    }
  ],
  
  "accuracy_tracking": {
    "2025-07": {
      "actual": 4325,
      "forecasts_over_time": [
        {
          "snapshot_date": "2025-06-01",
          "model": "Prophet",
          "forecast": 4400,
          "error": 75,
          "error_pct": 1.73,
          "weeks_ahead": 8
        },
        {
          "snapshot_date": "2025-07-01",
          "model": "Prophet",
          "forecast": 4500,
          "error": 175,
          "error_pct": 4.05,
          "weeks_ahead": 4
        }
      ],
      "final_error": {
        "Prophet": 175,
        "LightGBM": 125,
        "all_models_avg": 150
      },
      "convergence_quality": "improved",
      "prediction_volatility": 2.3
    }
  },
  
  "stability_metrics": {
    "Prophet": {
      "mean_revision_pct": 2.5,
      "max_revision_pct": 8.2,
      "stability_score": 0.87
    },
    "LightGBM": {
      "mean_revision_pct": 1.8,
      "max_revision_pct": 5.1,
      "stability_score": 0.92
    }
  }
}
```

---

## Phase 2: Backend Implementation

### New Module: `code/forecast_tracker.py`

```python
"""
Forecast tracking module for temporal prediction analysis.
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
                logger.info(f"Loaded forecast history with {len(data.get('forecast_snapshots', []))} snapshots")
                return data
            except Exception as e:
                logger.error(f"Failed to load forecast history: {e}")
                return self._empty_history()
        
        logger.info("Creating new forecast history")
        return self._empty_history()
    
    def _empty_history(self) -> Dict:
        """Create empty history structure."""
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
        Add new forecast snapshot to history.
        
        Args:
            forecasts: Dict of {month: {model: value}}
            actuals: Dict of {month: actual_value} for completed months
            model_performance: Dict of {model: {metrics}}
            snapshot_date: Date of this snapshot
            metadata: Optional additional metadata
        """
        snapshot_id = snapshot_date.strftime("%Y-%m-%d_%H%M%S")
        
        # Find latest completed month from actuals
        data_through = max(actuals.keys()) if actuals else None
        
        snapshot = {
            "snapshot_id": snapshot_id,
            "snapshot_date": snapshot_date.isoformat(),
            "generation_time": datetime.utcnow().isoformat(),
            "data_through": data_through,
            "forecasts": forecasts,
            "model_performance": model_performance,
            "metadata": metadata or {}
        }
        
        # Only include actuals if this is not the first snapshot
        if actuals and len(self.history["forecast_snapshots"]) > 0:
            snapshot["actuals"] = actuals
        
        self.history["forecast_snapshots"].append(snapshot)
        self.history["last_updated"] = datetime.utcnow().isoformat()
        
        # Update accuracy tracking
        self._update_accuracy_tracking()
        
        # Update stability metrics
        self._update_stability_metrics()
        
        # Save to disk
        self._save_history()
        
        logger.info(f"Added forecast snapshot {snapshot_id}")
    
    def _update_accuracy_tracking(self):
        """Calculate accuracy metrics for months that now have actuals."""
        snapshots = self.history["forecast_snapshots"]
        
        # Find all months that now have actuals
        months_with_actuals = {}
        for snapshot in snapshots:
            if "actuals" in snapshot:
                months_with_actuals.update(snapshot["actuals"])
        
        # For each month with actuals, gather all forecasts made for it
        for month, actual in months_with_actuals.items():
            forecast_evolution = []
            
            for snapshot in snapshots:
                snapshot_date = datetime.fromisoformat(snapshot["snapshot_date"])
                
                if month in snapshot["forecasts"]:
                    # Calculate weeks ahead
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
                # Calculate final errors by model
                final_errors = {}
                for item in forecast_evolution:
                    if item["weeks_ahead"] <= 1:  # Most recent forecast
                        model = item["model"]
                        if model not in final_errors or item["snapshot_date"] > final_errors[model]["date"]:
                            final_errors[model] = {
                                "date": item["snapshot_date"],
                                "error": item["error"],
                                "error_pct": item["error_pct"]
                            }
                
                # Calculate convergence quality
                errors_by_distance = {}
                for item in forecast_evolution:
                    week_bucket = int(item["weeks_ahead"] // 4) * 4  # 4-week buckets
                    if week_bucket not in errors_by_distance:
                        errors_by_distance[week_bucket] = []
                    errors_by_distance[week_bucket].append(abs(item["error_pct"]))
                
                # Check if errors decreased as we got closer
                sorted_buckets = sorted(errors_by_distance.keys(), reverse=True)
                if len(sorted_buckets) >= 2:
                    far_error = np.mean(errors_by_distance[sorted_buckets[0]])
                    near_error = np.mean(errors_by_distance[sorted_buckets[-1]])
                    convergence = "improved" if near_error < far_error else "degraded"
                else:
                    convergence = "unknown"
                
                # Calculate prediction volatility
                all_forecasts = [item["forecast"] for item in forecast_evolution]
                volatility = np.std(all_forecasts) / np.mean(all_forecasts) * 100 if all_forecasts else 0
                
                self.history["accuracy_tracking"][month] = {
                    "actual": int(actual),
                    "forecasts_over_time": forecast_evolution,
                    "final_errors": {model: data["error"] for model, data in final_errors.items()},
                    "convergence_quality": convergence,
                    "prediction_volatility": round(volatility, 2)
                }
    
    def _update_stability_metrics(self):
        """Calculate forecast stability metrics per model."""
        snapshots = self.history["forecast_snapshots"]
        
        if len(snapshots) < 2:
            return
        
        # Track revisions per model
        model_revisions = {}
        
        for i in range(1, len(snapshots)):
            prev_snapshot = snapshots[i-1]
            curr_snapshot = snapshots[i]
            
            # Find common months between snapshots
            prev_months = set(prev_snapshot["forecasts"].keys())
            curr_months = set(curr_snapshot["forecasts"].keys())
            common_months = prev_months & curr_months
            
            for month in common_months:
                prev_forecasts = prev_snapshot["forecasts"][month]
                curr_forecasts = curr_snapshot["forecasts"][month]
                
                # Find common models
                common_models = set(prev_forecasts.keys()) & set(curr_forecasts.keys())
                
                for model in common_models:
                    if model == "all_models_avg":
                        continue
                    
                    prev_val = prev_forecasts[model]
                    curr_val = curr_forecasts[model]
                    
                    if prev_val != 0:
                        revision_pct = abs((curr_val - prev_val) / prev_val * 100)
                        
                        if model not in model_revisions:
                            model_revisions[model] = []
                        model_revisions[model].append(revision_pct)
        
        # Calculate stability metrics
        for model, revisions in model_revisions.items():
            if revisions:
                mean_rev = np.mean(revisions)
                max_rev = np.max(revisions)
                # Stability score: 1 = perfect, 0 = highly unstable
                stability = 1 / (1 + mean_rev / 10)  # Normalize
                
                self.history["stability_metrics"][model] = {
                    "mean_revision_pct": round(mean_rev, 2),
                    "max_revision_pct": round(max_rev, 2),
                    "stability_score": round(stability, 3),
                    "num_revisions": len(revisions)
                }
    
    def _save_history(self):
        """Save forecast history to disk."""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with pretty printing
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            logger.info(f"Saved forecast history to {self.history_path}")
        except Exception as e:
            logger.error(f"Failed to save forecast history: {e}")
    
    def get_forecast_evolution(self, target_month: str) -> List[Dict]:
        """
        Get all forecasts made for a specific month over time.
        
        Args:
            target_month: Month in YYYY-MM format
            
        Returns:
            List of forecast snapshots for that month
        """
        evolution = []
        for snapshot in self.history["forecast_snapshots"]:
            if target_month in snapshot["forecasts"]:
                evolution.append({
                    "date": snapshot["snapshot_date"],
                    "forecasts": snapshot["forecasts"][target_month],
                    "data_through": snapshot.get("data_through")
                })
        return evolution
    
    def get_model_stability_ranking(self) -> List[tuple]:
        """
        Get models ranked by stability score.
        
        Returns:
            List of (model_name, stability_score) tuples, sorted by stability
        """
        metrics = self.history.get("stability_metrics", {})
        ranking = [(model, data["stability_score"]) 
                  for model, data in metrics.items()]
        return sorted(ranking, key=lambda x: x[1], reverse=True)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics about forecast history."""
        return {
            "total_snapshots": len(self.history["forecast_snapshots"]),
            "months_tracked": len(self.history.get("accuracy_tracking", {})),
            "models_tracked": len(self.history.get("stability_metrics", {})),
            "date_range": {
                "first": self.history["forecast_snapshots"][0]["snapshot_date"] 
                        if self.history["forecast_snapshots"] else None,
                "last": self.history["forecast_snapshots"][-1]["snapshot_date"]
                       if self.history["forecast_snapshots"] else None
            }
        }
```

---

## Phase 3: Integration into Main Pipeline

### Modify: `code/main.py`

Add to imports:
```python
from forecast_tracker import ForecastTracker
```

Add to `CVEForecastEngine.__init__`:
```python
def __init__(self, config_path='config.json'):
    # ... existing code ...
    self.forecast_tracker = ForecastTracker(
        history_path=self.config['file_paths'].get(
            'forecast_history', 
            'web/forecast_history.json'
        )
    )
```

Add to `save_results()` method (after line 667):
```python
def save_results(self):
    # ... existing save logic ...
    
    # NEW: Track forecasts for temporal analysis
    try:
        self._save_forecast_snapshot()
    except Exception as e:
        self.logger.error(f"Failed to save forecast snapshot: {e}")
    
    # ... rest of save_results ...

def _save_forecast_snapshot(self):
    """Save current forecasts to history tracker."""
    # Prepare forecasts dict: {month: {model: value}}
    forecasts_by_month = {}
    
    for model_name, forecast_series in self.final_forecasts.items():
        for ts, val in zip(forecast_series.time_index, forecast_series.values()):
            month = ts.strftime('%Y-%m')
            if month not in forecasts_by_month:
                forecasts_by_month[month] = {}
            forecasts_by_month[month][model_name] = float(val[0])
    
    # Add all_models_avg
    for month in forecasts_by_month:
        values = list(forecasts_by_month[month].values())
        forecasts_by_month[month]['all_models_avg'] = np.mean(values)
    
    # Prepare actuals dict: {month: value}
    actuals_dict = {}
    historical_data = self._get_historical_data()
    for record in historical_data:
        actuals_dict[record['date']] = record['cve_count']
    
    # Add current month if available
    current_month_actual = self._get_current_month_actual()
    if current_month_actual['cve_count'] > 0:
        current_month_key = current_month_actual['date'][:7]  # YYYY-MM
        actuals_dict[current_month_key] = current_month_actual['cve_count']
    
    # Prepare model performance
    model_performance = {}
    for name, result in self.model_results.items():
        model_performance[name] = {
            'mape': result['metrics'].get('mape', 0),
            'mae': result['metrics'].get('mae', 0),
            'validation_months': len(self.series) - int(result['split_ratio'] * len(self.series))
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
    
    self.logger.info("Forecast snapshot saved to history")
```

---

## Phase 4: Configuration Updates

### Add to `code/config.json`:

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

## Phase 5: Visualization (Web Frontend)

### New File: `web/forecast_evolution.html`

Create page with:

**1. Month Evolution Chart**
```
Shows how forecasts for a selected month changed over time

[Dropdown: Select Month] [August 2025 ▼]

Forecast Evolution for August 2025
5,200 ┤                              ╭── Actual: 5,080
       │                         ╭───╯
5,100 ┤                    ╭────╯
       │               ╭───╯
5,000 ┤          ╭────╯
       │     ╭───╯
4,900 ┤────╯
       └─────┬─────┬─────┬─────┬─────┬─────┬
           Jun 1  Jun 15 Jul 1 Jul 15 Aug 1 Sep 1
           (8 wks)(6 wks)(4 wks)(2 wks)(0 wk)(actual)
```

**2. Model Stability Dashboard**
```
Model Stability Rankings

1. LightGBM    ████████████░░ 92% stable (avg revision: 1.8%)
2. Prophet     ████████████░░ 87% stable (avg revision: 2.5%)
3. XGBoost     ███████████░░░ 85% stable (avg revision: 2.8%)
```

**3. Accuracy Convergence Analysis**
```
How Accuracy Improves as Month Approaches

Error %
 8% ┤ ●
    │  ╲
 6% ┤   ●
    │    ╲
 4% ┤     ●  ●
    │        ╲
 2% ┤         ● ●
    │            ╲
 0% ┤             ●──── Actual
    └────┬────┬────┬────┬────┬
       8 wks 6 wks 4 wks 2 wks 0 wk
```

### Update `web/index.html`:

Add navigation link:
```html
<nav>
  <a href="index.html">Main Forecast</a>
  <a href="cna_forecast.html">CNA Forecasts</a>
  <a href="forecast_evolution.html">Forecast History</a>
  <a href="technical_details.html">Technical Details</a>
</nav>
```

---

## Phase 6: Testing Strategy

### Unit Tests: `tests/test_forecast_tracker.py`

```python
import pytest
from datetime import datetime
from code.forecast_tracker import ForecastTracker

def test_empty_history_creation():
    tracker = ForecastTracker("test_history.json")
    assert len(tracker.history["forecast_snapshots"]) == 0

def test_add_snapshot():
    tracker = ForecastTracker("test_history.json")
    
    forecasts = {
        "2025-08": {"Prophet": 4500, "LightGBM": 4450}
    }
    actuals = {"2025-07": 4325}
    performance = {"Prophet": {"mape": 1.12}}
    
    tracker.add_snapshot(
        forecasts=forecasts,
        actuals=actuals,
        model_performance=performance,
        snapshot_date=datetime(2025, 7, 1)
    )
    
    assert len(tracker.history["forecast_snapshots"]) == 1
    assert "2025-08" in tracker.history["forecast_snapshots"][0]["forecasts"]

def test_accuracy_tracking():
    tracker = ForecastTracker("test_history.json")
    
    # Add first snapshot with forecast
    tracker.add_snapshot(
        forecasts={"2025-08": {"Prophet": 4500}},
        actuals={},
        model_performance={"Prophet": {"mape": 1.12}},
        snapshot_date=datetime(2025, 7, 1)
    )
    
    # Add second snapshot with actual
    tracker.add_snapshot(
        forecasts={"2025-09": {"Prophet": 4600}},
        actuals={"2025-08": 4550},  # Actual for August
        model_performance={"Prophet": {"mape": 1.10}},
        snapshot_date=datetime(2025, 8, 1)
    )
    
    assert "2025-08" in tracker.history["accuracy_tracking"]
    assert tracker.history["accuracy_tracking"]["2025-08"]["actual"] == 4550

def test_stability_metrics():
    tracker = ForecastTracker("test_history.json")
    
    # Add multiple snapshots
    for i in range(3):
        tracker.add_snapshot(
            forecasts={"2025-10": {"Prophet": 4500 + i*10}},
            actuals={},
            model_performance={"Prophet": {"mape": 1.12}},
            snapshot_date=datetime(2025, 7 + i, 1)
        )
    
    assert "Prophet" in tracker.history["stability_metrics"]
    assert "stability_score" in tracker.history["stability_metrics"]["Prophet"]
```

### Integration Test

```bash
# Run main.py multiple times and verify history grows
python code/main.py
# Check forecast_history.json has 1 snapshot

python code/main.py  
# Check forecast_history.json has 2 snapshots

# Verify structure
cat web/forecast_history.json | jq '.forecast_snapshots | length'
```

---

## Success Metrics

### Technical Metrics
- [ ] `forecast_history.json` created and updated on each run
- [ ] Snapshot count increases with each run
- [ ] Accuracy tracking populated for completed months
- [ ] Stability metrics calculated correctly
- [ ] No performance regression (< 5 sec overhead)

### User Metrics
- [ ] Users can view forecast evolution charts
- [ ] Model stability rankings displayed
- [ ] Convergence analysis shows error reduction
- [ ] Historical accuracy dashboards functional

---

## Rollout Plan

### Week 1: Backend Implementation
- Day 1-2: Implement `forecast_tracker.py`
- Day 3: Integrate into `main.py`
- Day 4: Unit tests
- Day 5: Integration testing

### Week 2: Frontend Implementation
- Day 1-2: Create `forecast_evolution.html`
- Day 3: Add charts and visualizations
- Day 4: Navigation and UX
- Day 5: Cross-browser testing

### Week 3: Validation & Documentation
- Day 1-2: Collect real snapshot data
- Day 3: Validate accuracy metrics
- Day 4: Write user documentation
- Day 5: Deploy to production

---

## Future Enhancements

### Phase 2 Features (Later)
1. **Email Alerts:** Notify when forecasts change significantly
2. **API Endpoint:** Query historical forecasts programmatically
3. **Comparison Tools:** Compare current vs historical model performance
4. **Export Functionality:** Download forecast evolution data
5. **Anomaly Detection:** Flag unusual forecast revisions

### Advanced Analytics
- Forecast quality scores (convergence + stability + accuracy)
- Model recommendation based on stability patterns
- Automatic re-training triggers based on drift detection

---

## Maintenance

### Storage Management
- Rotate old snapshots after 2 years
- Compress historical data
- Archive to separate file if > 10MB

### Monitoring
- Track file size growth
- Alert if snapshots missing
- Verify data integrity weekly

---

## Conclusion

This implementation provides comprehensive temporal tracking of CVE forecasts, enabling:
- **Transparency:** Users see how predictions evolve
- **Trust:** Demonstrable accuracy improvements
- **Debugging:** Identify systematic biases
- **Quality:** Measure forecast stability

**Next Step:** Begin Phase 1 implementation with `forecast_tracker.py` module.
