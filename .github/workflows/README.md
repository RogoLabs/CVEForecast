# GitHub Actions Workflows

This directory contains automated workflows for the CVE Forecast system.

## Workflows

### 1. Daily Forecast Update (`main.yml`)

**Purpose**: Generate daily CVE and CNA forecasts and deploy to GitHub Pages.

**Schedule**: Daily at midnight UTC

**Triggers**:
- Automatic: Daily at 00:00 UTC
- Manual: Via "Run workflow" button
- Push: On commits to `main` or `master` branch

**What It Does**:
1. Clones latest CVE data from CVEProject/cvelistV5
2. Runs production forecast pipeline (`run_production_forecast.py`)
   - Trains 13 optimized models
   - Generates CVE forecasts (Oct-Dec 2025, 2026)
   - Calculates backtest metrics (Jan-Sep 2025)
   - Generates CNA forecasts (120+ CNAs)
   - Saves forecast history for tracking
3. Commits updated data files
4. Deploys to GitHub Pages

**Output Files**:
- `web/data.json` - CVE forecasts and metrics
- `web/cna_data.json` - CNA forecasts
- `web/forecast_history.json` - Historical tracking
- `web/pipeline_results.json` - Execution summary

**Duration**: ~10-15 minutes

**Manual Trigger**:
```
Actions → "CVE & CNA Forecast Daily Update" → Run workflow
```

---

### 2. Monthly Hyperparameter Tuning (`monthly_tuning.yml`)

**Purpose**: Optimize model hyperparameters for improved forecast accuracy.

**Schedule**: 1st of each month at 2 AM UTC

**Triggers**:
- Automatic: Monthly on the 1st at 02:00 UTC
- Manual: Via "Run workflow" button

**What It Does**:
1. Clones latest CVE data
2. Runs comprehensive hyperparameter tuning (`comprehensive_tuner.py`)
   - Tests multiple hyperparameter combinations
   - Uses walk-forward validation
   - Selects best configurations based on MAPE
3. Updates `code/config.json` with optimized hyperparameters
4. Commits changes if hyperparameters improved
5. Creates artifacts with tuning summary
6. Opens issue if tuning fails

**Output Files**:
- `code/config.json` - Optimized hyperparameters
- `tuning_summary.md` - Tuning results summary

**Duration**: 2-4 hours

**Manual Trigger**:
```
Actions → "Monthly Hyperparameter Tuning" → Run workflow
```

**Artifacts**: Available for 90 days after each run

---

## Workflow Dependencies

```
Monthly Tuning (1st of month)
    ↓
Updates code/config.json
    ↓
Daily Forecast (uses optimized configs)
    ↓
Deploys to GitHub Pages
```

## Manual Execution

### Run Daily Forecast
```bash
# Local execution
python code/run_production_forecast.py
```

### Run Monthly Tuning
```bash
# Local execution
python code/tuner/comprehensive_tuner.py
```

See [TUNING_GUIDE.md](../../docs/TUNING_GUIDE.md) for detailed tuning instructions.

## Monitoring

### Check Workflow Status
1. Go to **Actions** tab
2. Select workflow
3. View recent runs and logs

### Download Artifacts
1. Open workflow run
2. Scroll to "Artifacts" section
3. Download results

### View Deployment
- **Production**: https://[your-username].github.io/CVEForecast/
- **Status**: Check "Deploy to GitHub Pages" step

## Troubleshooting

### Daily Forecast Fails
1. Check CVE data clone step
2. Verify model training logs
3. Check for data format issues
4. Review error messages in logs

### Monthly Tuning Fails
1. Check timeout (5 hour limit)
2. Verify sufficient memory
3. Review model-specific errors
4. Check tuning configuration

### Deployment Fails
1. Verify GitHub Pages is enabled
2. Check permissions (write access)
3. Review artifact upload logs
4. Verify web/ directory contents

## Configuration

### Modify Schedule

Edit cron expressions in workflow files:

**Daily Forecast** (`main.yml`):
```yaml
schedule:
  - cron: '0 0 * * *'  # Midnight UTC
```

**Monthly Tuning** (`monthly_tuning.yml`):
```yaml
schedule:
  - cron: '0 2 1 * *'  # 1st of month, 2 AM UTC
```

### Adjust Timeouts

**Daily Forecast**: No timeout (typically 10-15 min)

**Monthly Tuning**: 5 hour timeout
```yaml
timeout-minutes: 300
```

### Change Python Version

Both workflows use Python 3.13:
```yaml
python-version: '3.13'
```

## Best Practices

1. **Monitor First Run**: Watch logs for any issues
2. **Review Artifacts**: Check tuning results after monthly runs
3. **Verify Deployments**: Confirm website updates after daily runs
4. **Check History**: Review forecast_history.json for accuracy trends
5. **Manual Testing**: Test locally before relying on automation

## Support

- **Workflow Issues**: Check Actions logs and error messages
- **Tuning Questions**: See [TUNING_GUIDE.md](../../docs/TUNING_GUIDE.md)
- **Pipeline Issues**: Review `run_production_forecast.py` logs
- **Data Issues**: Verify CVE data availability and format

---

**Last Updated**: 2025-10-13
