# Frontend Development Notes - Temporal Tracking Visualization

**Date:** 2025-10-05  
**For:** `web/forecast_evolution.html` implementation  
**Priority:** Phase 2 (after temporal tracking system stabilizes)

---

## ⚠️ Critical: Handle Single-Day Runs

### Issue
On the **first day** the system runs, `forecast_history.json` will contain only **1 snapshot**. This means:
- No stability metrics yet (need 2+ snapshots to compare)
- No accuracy tracking yet (need completed months)
- Minimal data for charts and visualizations

### Requirements for Frontend

**The web page MUST gracefully handle this scenario:**

#### 1. Single Snapshot Detection
```javascript
// Check if we have enough data
const history = await fetch('/web/forecast_history.json').then(r => r.json());
const snapshotCount = history.forecast_snapshots.length;

if (snapshotCount < 2) {
  // Show "collecting data" message
  displayInsufficientDataMessage(snapshotCount);
  return;
}
```

#### 2. Graceful Degradation
**Show appropriate messages when data is insufficient:**

```javascript
function displayInsufficientDataMessage(snapshotCount) {
  const message = `
    <div class="info-message">
      <h3>📊 Temporal Tracking Active</h3>
      <p>Currently collecting forecast data...</p>
      <ul>
        <li>Snapshots captured: ${snapshotCount}</li>
        <li>Minimum required: 2 (for stability metrics)</li>
        <li>Optimal: 7+ (for trend analysis)</li>
      </ul>
      <p><strong>Next step:</strong> Run the forecast again tomorrow to populate charts.</p>
    </div>
  `;
  document.getElementById('charts-container').innerHTML = message;
}
```

#### 3. Progressive Enhancement
**Show available data even with limited snapshots:**

**Day 1 (1 snapshot):**
- ✅ Show: Current forecasts table
- ✅ Show: Model performance metrics
- ❌ Hide: Stability charts
- ❌ Hide: Evolution graphs
- ❌ Hide: Convergence analysis

**Day 2 (2 snapshots):**
- ✅ Show: All above +
- ✅ Show: Stability metrics
- ✅ Show: Basic evolution comparison
- ❌ Hide: Trend lines (need 3+ points)

**Day 7+ (optimal):**
- ✅ Show: Everything
- ✅ Show: Trend lines
- ✅ Show: Statistical analysis
- ✅ Show: Convergence quality

---

## Data Structure Reference

### Day 1 Structure
```json
{
  "version": "1.0",
  "forecast_snapshots": [
    {
      "snapshot_id": "2025-10-05_204157",
      "forecasts": { "2025-11": { "Prophet": 4500 } },
      "model_performance": { "Prophet": { "mape": 1.12 } }
    }
  ],
  "accuracy_tracking": {},      // EMPTY on day 1
  "stability_metrics": {}        // EMPTY on day 1
}
```

### Day 2+ Structure
```json
{
  "forecast_snapshots": [
    { /* snapshot 1 */ },
    { /* snapshot 2 */ }
  ],
  "accuracy_tracking": {},       // Still empty until months complete
  "stability_metrics": {         // NOW POPULATED
    "Prophet": {
      "mean_revision_pct": 2.5,
      "stability_score": 0.87
    }
  }
}
```

---

## Recommended UI Components

### 1. Data Availability Indicator
```html
<div class="data-status">
  <div class="status-item">
    <span class="label">Snapshots:</span>
    <span class="value">5</span>
    <span class="badge">✓ Good</span>
  </div>
  <div class="status-item">
    <span class="label">Accuracy Data:</span>
    <span class="value">2 months</span>
    <span class="badge">✓ Available</span>
  </div>
  <div class="status-item">
    <span class="label">Stability Metrics:</span>
    <span class="value">12 models</span>
    <span class="badge">✓ Tracking</span>
  </div>
</div>
```

### 2. Chart Placeholder States
```html
<!-- When insufficient data -->
<div class="chart-placeholder">
  <div class="icon">📊</div>
  <h3>Stability Chart</h3>
  <p>Requires 2+ snapshots</p>
  <p class="muted">Run forecast again to populate</p>
</div>

<!-- When data is available -->
<div class="chart-container">
  <canvas id="stability-chart"></canvas>
</div>
```

### 3. Progressive Unlock Badges
Show what features unlock at different data thresholds:

```
🔒 Evolution Charts     (2+ snapshots needed)
🔒 Trend Analysis       (7+ snapshots needed)
🔒 Accuracy Tracking    (1+ completed month needed)
✅ Current Forecasts    (Available now)
✅ Model Performance    (Available now)
```

---

## Testing Checklist

### Frontend Developer Testing

**Day 1 Testing (Single Snapshot):**
- [ ] Load page with 1 snapshot - shows graceful message
- [ ] No JavaScript errors in console
- [ ] Model performance table displays correctly
- [ ] Evolution charts show "not enough data" placeholder
- [ ] Stability section shows "collecting data" message

**Day 2 Testing (Multiple Snapshots):**
- [ ] Stability metrics appear and display correctly
- [ ] Evolution comparison works with 2 data points
- [ ] Charts render without errors
- [ ] All transitions smooth from placeholders to live data

**Day 7+ Testing (Full Data):**
- [ ] Trend lines render correctly
- [ ] Statistical analysis calculates properly
- [ ] All features unlocked and functional
- [ ] Performance acceptable with large dataset

---

## Example: Feature Gating Logic

```javascript
class ForecastVisualization {
  constructor(historyData) {
    this.history = historyData;
    this.snapshotCount = historyData.forecast_snapshots.length;
    this.completedMonths = Object.keys(historyData.accuracy_tracking).length;
  }
  
  canShowStabilityMetrics() {
    return this.snapshotCount >= 2;
  }
  
  canShowTrendLines() {
    return this.snapshotCount >= 7;
  }
  
  canShowAccuracyTracking() {
    return this.completedMonths >= 1;
  }
  
  canShowConvergenceAnalysis() {
    return this.completedMonths >= 3;
  }
  
  render() {
    // Always show
    this.renderCurrentForecasts();
    this.renderModelPerformance();
    
    // Conditional rendering
    if (this.canShowStabilityMetrics()) {
      this.renderStabilityCharts();
    } else {
      this.renderStabilityPlaceholder();
    }
    
    if (this.canShowAccuracyTracking()) {
      this.renderAccuracyDashboard();
    } else {
      this.renderAccuracyPlaceholder();
    }
    
    if (this.canShowTrendLines()) {
      this.renderTrendAnalysis();
    }
  }
}
```

---

## Error Handling

### Missing File
```javascript
async function loadForecastHistory() {
  try {
    const response = await fetch('/web/forecast_history.json');
    if (!response.ok) {
      if (response.status === 404) {
        showError('Forecast history not generated yet. Run python code/main.py to initialize.');
        return null;
      }
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Failed to load forecast history:', error);
    showError('Unable to load forecast data. Please check console for details.');
    return null;
  }
}
```

### Invalid Data
```javascript
function validateHistoryData(data) {
  const required = ['version', 'forecast_snapshots', 'accuracy_tracking', 'stability_metrics'];
  const missing = required.filter(key => !(key in data));
  
  if (missing.length > 0) {
    console.error('Invalid forecast_history.json structure. Missing keys:', missing);
    return false;
  }
  
  if (!Array.isArray(data.forecast_snapshots)) {
    console.error('forecast_snapshots is not an array');
    return false;
  }
  
  return true;
}
```

---

## User Messaging Examples

### Day 1 Message
```
📊 Temporal Tracking Initialized!

Your forecast tracking system is now active and collecting data.

Current Status:
✅ 1 snapshot captured
⏳ Stability metrics: Available after next run
⏳ Accuracy tracking: Available after months complete

What to expect:
• Tomorrow: Stability metrics will appear
• After month completion: Accuracy tracking begins
• After 1 week: Full trend analysis available

Next Step: Run python code/main.py again tomorrow to unlock more features.
```

### Day 2 Message
```
🎉 Stability Metrics Now Available!

Your system has captured 2 snapshots. New features unlocked:

✅ Model Stability Rankings
✅ Forecast Revision Tracking
✅ Basic Evolution Comparison

Still collecting:
⏳ Accuracy tracking (waiting for months to complete)
⏳ Trend analysis (optimal after 7+ snapshots)

Keep running daily for richer insights!
```

---

## Implementation Priority

### Phase 1: Core Infrastructure (Week 1)
- [ ] Single snapshot detection and messaging
- [ ] Basic data loading and validation
- [ ] Current forecast display (always available)
- [ ] Model performance table

### Phase 2: Progressive Features (Week 2)
- [ ] Stability metrics dashboard (2+ snapshots)
- [ ] Evolution comparison charts
- [ ] Placeholder states for all features

### Phase 3: Advanced Analytics (Week 3)
- [ ] Accuracy tracking dashboard (completed months)
- [ ] Trend line analysis (7+ snapshots)
- [ ] Convergence quality visualization
- [ ] Statistical summaries

---

## Design Inspiration

**Consider these patterns:**
- **GitHub Insights:** Progressive data unlock as commits accumulate
- **Google Analytics:** "Collecting data" states for new properties
- **Stripe Dashboard:** Placeholder cards for upcoming features

---

## Related Files

**Backend:**
- `code/forecast_tracker.py` - Data generation logic
- `web/forecast_history.json` - Data source

**Frontend (to be created):**
- `web/forecast_evolution.html` - Main visualization page
- `web/css/forecast_evolution.css` - Styling
- `web/js/forecast_evolution.js` - Chart logic

**Testing:**
- `tests/test_temporal_tracking.py` - Backend validation
- Manual testing protocol (above)

---

## Questions to Resolve Before Implementation

1. **Chart Library:** Chart.js, D3.js, or Plotly?
2. **Update Frequency:** Real-time websocket or manual refresh?
3. **Data Retention:** How many snapshots to keep in browser cache?
4. **Mobile Support:** Responsive design requirements?
5. **Export Features:** Allow CSV/PNG download of charts?

---

**Status:** 📋 Planning Document  
**Next:** Build `web/forecast_evolution.html` after 1 week of snapshot accumulation

**Note:** This frontend should only be built after the system has run for at least 7 days to have meaningful data to design against.
