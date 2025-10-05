# Year Rollover Audit Report
## Date: October 5, 2025
## Target: January 1, 2026 Rollover

---

## 🔴 CRITICAL ISSUES FOUND

### 1. Chart Axis Limits (FRONTEND)
**File**: `web/script.js` Lines 502-503
**Status**: ❌ HARDCODED TO 2025/2026
```javascript
min: '2025-01-01',
max: '2026-01-05',
```
**Impact**: Chart will show wrong date range on Jan 1, 2026
**Fix Required**: Make dynamic based on current year

---

### 2. Forecast End Year (CONFIG FILES)
**Files**: 
- `code/config.json` Line 19
- `code/cna_config.json` Line 17
- `tuner/tuner_config.json` Line 21

**Status**: ❌ HARDCODED TO 2026
```json
"forecast_end_year": 2026,
"forecast_end_month": 1,
```
**Impact**: On Jan 1, 2026, system will only forecast through Jan 2026 (current month) instead of through Jan 2027
**Used By**: `main.py` line 257 for calculating `months_to_forecast`
**Fix Required**: Either make dynamic in code OR document annual config update requirement

---

### 3. CNA Forecast Table Headers (FRONTEND)
**File**: `web/cna_forecast.html` Lines 160-170
**Status**: ❌ HARDCODED YEAR LABELS
```html
<th>2024 Published</th>
<th>2025 Forecasted</th>
<th>2026 Forecasted</th>
<th>2024→2025 Growth</th>
```
**Impact**: Table headers will show wrong years on Jan 1, 2026
**Fix Required**: Make labels dynamic based on current year

---

### 4. CNA Summary Cards (FRONTEND)
**File**: `web/cna_forecast.html` Lines 245-264
**Status**: ❌ HARDCODED YEAR LABELS
```html
<h3>2024 Published</h3>
<h3>2025 Forecasted</h3>
<div>2024 → 2025 change</div>
```
**Impact**: Summary panel will show wrong years on Jan 1, 2026
**Fix Required**: Make labels dynamic based on current year

---

## ✅ ALREADY DYNAMIC (NO ACTION NEEDED)

### 1. Previous Year Calculation (BACKEND)
**File**: `code/main.py` Line 485
**Status**: ✅ DYNAMIC
```python
previous_year = self.current_datetime.year - 1
```

### 2. Actuals Cumulative Timeline (BACKEND)
**File**: `code/main.py` Line 508
**Status**: ✅ DYNAMIC
```python
current_year = self.current_datetime.year
current_year_historical = [item for item in historical_data if item['date'].startswith(str(current_year))]
```

### 3. CNA Forecast Horizon (BACKEND)
**File**: `code/cna_main.py` Lines 372-383
**Status**: ✅ DYNAMIC
```python
current_year = now.year
next_year = current_year + 1
remaining_current_year = 12 - current_month
horizon = remaining_current_year + next_year_months
```

### 4. YoY Growth Display (FRONTEND)
**File**: `web/script.js` Lines 107-114
**Status**: ✅ DYNAMIC
```javascript
const previousYear = forecastData.summary?.previous_year || 2024;
const currentYear = new Date().getFullYear();
document.getElementById('yoyGrowthDetail').textContent = `${bestModelTotal.toLocaleString()} vs ${lastYearTotal.toLocaleString()} (${currentYear} vs ${previousYear})`;
```

---

## RECOMMENDED FIXES

### Priority 1: Chart Axis (Critical - User Facing)
### Priority 2: Config Files (Critical - Forecast Generation)
### Priority 3: CNA UI Labels (High - User Facing)
