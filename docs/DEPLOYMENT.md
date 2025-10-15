# CVE Forecast Deployment Guide

**Version**: 0.10 "Phoenix" 🔥🐦  
**Last Updated**: October 2025

## Table of Contents
- [GitHub Actions Deployment](#github-actions-deployment)
- [Manual Deployment](#manual-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## GitHub Actions Deployment

### Overview

CVE Forecast uses two automated workflows:
1. **Daily Forecast** - Generates forecasts every midnight UTC
2. **Monthly Tuning** - Optimizes hyperparameters on the 1st of each month

### Daily Forecast Workflow

**File**: `.github/workflows/main.yml`

**Schedule**: Daily at 00:00 UTC

**Triggers**:
- Automatic: Daily at midnight
- Manual: Via "Run workflow" button
- Push: On commits to `main` branch

**Steps**:
1. Clone CVE data repository
2. Install dependencies
3. Run production forecast pipeline
4. Commit results
5. Deploy to GitHub Pages

**Duration**: ~10-15 minutes

**Manual Trigger**:
```bash
# Via GitHub UI
Actions → "CVE & CNA Forecast Daily Update" → Run workflow

# Via GitHub CLI
gh workflow run main.yml
```

### Monthly Tuning Workflow

**File**: `.github/workflows/monthly_tuning.yml`

**Schedule**: 1st of each month at 02:00 UTC

**Triggers**:
- Automatic: Monthly on the 1st
- Manual: Via "Run workflow" button

**Steps**:
1. Clone CVE data repository
2. Install dependencies
3. Run comprehensive hyperparameter tuning
4. Update config.json with optimized hyperparameters
5. Commit changes
6. Create artifacts

**Duration**: ~2-4 hours

**Manual Trigger**:
```bash
# Via GitHub UI
Actions → "Monthly Hyperparameter Tuning" → Run workflow

# Via GitHub CLI
gh workflow run monthly_tuning.yml
```

### GitHub Pages Setup

1. **Enable GitHub Pages**:
   - Go to repository Settings
   - Navigate to Pages section
   - Source: GitHub Actions
   - Branch: main
   - Folder: web/

2. **Configure Custom Domain** (optional):
   ```bash
   # Add CNAME file
   echo "cveforecast.org" > web/CNAME
   git add web/CNAME
   git commit -m "Add custom domain"
   git push
   ```

3. **Verify Deployment**:
   - Check Actions tab for deployment status
   - Visit https://[username].github.io/CVEForecast/
   - Or custom domain if configured

### Secrets Configuration

No secrets required for public deployment. For private repositories:

```bash
# Set GitHub token (automatic)
GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Manual Deployment

### Local Production Run

```bash
# 1. Clone CVE data
git clone --depth 1 https://github.com/CVEProject/cvelistV5.git

# 2. Run production pipeline
python code/run_production_forecast.py

# 3. Verify outputs
ls -lh web/data.json web/cna_data.json web/forecast_history.json

# 4. Test locally
python -m http.server 8000 --directory web
# Open http://localhost:8000
```

### Deploy to Custom Server

```bash
# 1. Generate forecasts
python code/run_production_forecast.py

# 2. Copy web directory to server
rsync -avz web/ user@server:/var/www/cveforecast/

# 3. Configure web server (nginx example)
server {
    listen 80;
    server_name cveforecast.org;
    root /var/www/cveforecast;
    index index.html;
    
    location / {
        try_files $uri $uri/ =404;
    }
}

# 4. Restart web server
sudo systemctl restart nginx
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY code/ code/
COPY web/ web/

# Clone CVE data
RUN git clone --depth 1 https://github.com/CVEProject/cvelistV5.git

# Run forecast
CMD ["python", "code/run_production_forecast.py"]
```

```bash
# Build and run
docker build -t cve-forecast .
docker run -v $(pwd)/web:/app/web cve-forecast
```

## Configuration

### Environment Variables

```bash
# Optional overrides
export CVE_REPO_PATH="/path/to/cvelistV5"
export OUTPUT_PATH="/path/to/output"
export LOG_LEVEL="DEBUG"
```

### config.json

```json
{
  "data_source": {
    "cve_repo_path": "cvelistV5",
    "start_date": "2017-01-01"
  },
  "forecast": {
    "forecast_end_year": 2026,
    "horizon_months": 15
  },
  "file_paths": {
    "output": "web/data.json",
    "cna_output": "web/cna_data.json"
  }
}
```

### Workflow Customization

**Change Schedule**:
```yaml
# .github/workflows/main.yml
on:
  schedule:
    - cron: '0 0 * * *'  # Midnight UTC
    # Change to: '0 6 * * *' for 6 AM UTC
```

**Adjust Timeout**:
```yaml
# .github/workflows/monthly_tuning.yml
- name: Run Comprehensive Hyperparameter Tuning
  run: python code/tuner/comprehensive_tuner.py
  timeout-minutes: 300  # 5 hours (default)
  # Increase if needed: 360 (6 hours)
```

## Monitoring

### GitHub Actions Monitoring

**Check Workflow Status**:
```bash
# Via GitHub CLI
gh run list --workflow=main.yml
gh run view [run-id]

# Via GitHub UI
Actions tab → Select workflow → View runs
```

**Download Artifacts**:
```bash
# Via GitHub CLI
gh run download [run-id]

# Via GitHub UI
Actions → Run → Artifacts section → Download
```

### Log Analysis

**View Logs**:
```bash
# Via GitHub CLI
gh run view [run-id] --log

# Via GitHub UI
Actions → Run → Job → Step logs
```

**Common Log Patterns**:
```
# Success indicators
✓ Loaded 106 months of CVE data
✓ Saved forecasts for 120 CNAs
✅ CVE forecasts: web/data.json

# Warning indicators
⚠️ Model X failed, continuing with others
⚠️ CNA Y has insufficient data

# Error indicators
❌ FATAL ERROR: Data loading failed
❌ Pipeline execution failed
```

### Performance Metrics

**Track Execution Time**:
```python
# In pipeline_results.json
{
  "execution_time": {
    "data_loading": 32.5,
    "model_training": 487.2,
    "forecast_generation": 45.8,
    "total": 565.5
  }
}
```

**Monitor Resource Usage**:
```bash
# GitHub Actions provides:
# - CPU usage
# - Memory usage
# - Disk usage
# - Network usage

# View in Actions logs
```

## Troubleshooting

### Common Issues

**1. Workflow Fails to Start**
```bash
# Check workflow syntax
gh workflow view main.yml

# Validate YAML
yamllint .github/workflows/main.yml

# Check permissions
# Settings → Actions → General → Workflow permissions
```

**2. Data Loading Fails**
```bash
# Verify CVE repo clone
ls -la cvelistV5/

# Check disk space
df -h

# Verify file permissions
ls -l cvelistV5/cves/
```

**3. Model Training Fails**
```bash
# Check logs for specific model
grep "ERROR" workflow.log | grep "Model"

# Disable problematic model
# Edit config.json: "enabled": false

# Retry workflow
gh workflow run main.yml
```

**4. Deployment Fails**
```bash
# Check GitHub Pages status
# Settings → Pages

# Verify artifact upload
# Actions → Run → Artifacts

# Check deployment logs
# Actions → Deploy to GitHub Pages
```

**5. Out of Memory**
```bash
# Reduce enabled models
# Edit config.json: disable deep learning models

# Increase GitHub Actions memory (not possible)
# Consider self-hosted runner with more RAM
```

### Debug Mode

**Enable Debug Logging**:
```yaml
# .github/workflows/main.yml
env:
  PYTHONUNBUFFERED: 1
  LOG_LEVEL: DEBUG  # Add this
```

**Local Debug**:
```bash
# Run with debug logging
LOG_LEVEL=DEBUG python code/run_production_forecast.py

# Check specific component
python -c "from data_loader import load_cve_data; load_cve_data({})"
```

### Recovery Procedures

**Rollback Deployment**:
```bash
# Revert to previous commit
git revert HEAD
git push

# Or restore from backup
git checkout [previous-commit] -- web/
git commit -m "Restore previous data"
git push
```

**Manual Intervention**:
```bash
# If automated workflow fails repeatedly:

# 1. Run locally
python code/run_production_forecast.py

# 2. Verify outputs
ls -lh web/*.json

# 3. Commit manually
git add web/
git commit -m "Manual forecast update"
git push
```

## Best Practices

### 1. Monitor First Run
- Watch logs for any issues
- Verify all outputs generated
- Check website displays correctly

### 2. Regular Checks
- Weekly: Review workflow execution times
- Monthly: Verify tuning improvements
- Quarterly: Audit forecast accuracy

### 3. Backup Strategy
- GitHub automatically backs up repository
- Download artifacts for long-term storage
- Keep local copies of critical configs

### 4. Update Schedule
- Review and update dependencies quarterly
- Test major changes in separate branch
- Use semantic versioning for releases

### 5. Documentation
- Document any custom configurations
- Keep deployment notes up to date
- Record troubleshooting solutions

---

**Next**: [Development Guide](DEVELOPMENT.md) | [Tuning Guide](TUNING_GUIDE.md)
