#!/usr/bin/env python3
"""
CNA-specific CVE forecasting script.

- Parses CVEs from a local clone of CVEProject/cvelistV5
- Groups monthly publication counts by CNA (provider orgId)
- Filters to CNAs with at least --min_cves total published CVEs
- Trains 3 fast models (LightGBM, XGBoost, Prophet) with hyperparameters loaded from code/config.json
- Produces forecasts for rest of current year + all of next year for each eligible CNA
- Writes output to web/cna_data.json in the following structure:

{
  "<CNA_ID>": {
    "id": "<orgId>",
    "name": "<shortName or None>",
    "scope": null,  # if available in data in the future
    "historical": {"YYYY-MM": count, ...},
    "forecasts": {
      "LightGBM": {"YYYY-MM": forecast_count, ...},
      "XGBoost": {"YYYY-MM": forecast_count, ...},
      "Prophet": {"YYYY-MM": forecast_count, ...}
    }
  },
  ...
}

Usage:
  python code/cna_main.py --cvelist_dir cvelistV5 --output web/cna_data.json --min_cves 100

Notes:
- This script intentionally does not depend on the main forecasting pipeline to keep runtime fast and isolated.
- It uses the same optimized hyperparameters for consistency by reading code/config.json.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress sklearn feature names warning for LightGBM
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Darts time series and models
from darts import TimeSeries
from darts.models import LightGBMModel, XGBModel
from darts.models import Prophet as DartsProphet


# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("cna_forecast")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------
# Data Models
# ---------------------------
@dataclass
class CNARecord:
    org_id: str
    short_name: Optional[str]


# ---------------------------
# Helpers
# ---------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_hyperparameters(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Extracts hyperparameters for a model from code/config.json.

    Returns an empty dict if not found.
    """
    try:
        return dict(config["models"][model_name].get("hyperparameters", {}))
    except Exception:
        logger.warning("No hyperparameters found for %s in config.json; using defaults", model_name)
        return {}


def parse_cve_file(path: str) -> Optional[Tuple[datetime, CNARecord]]:
    """Parse a single cvelistV5 JSON to extract (published_date, CNARecord).

    Returns None if required fields are missing or the record is not published.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # datePublished is the authoritative publication date in v5
        # Some records may use 'datePublic' historically; prefer datePublished
        meta = data.get("cveMetadata", {})
        date_str = meta.get("datePublished") or meta.get("datePublic")
        if not date_str:
            return None

        # Parse date
        try:
            # Parse as UTC then drop timezone to ensure tz-naive timestamps
            ts = pd.to_datetime(date_str, utc=True)
            ts = ts.tz_convert(None)  # make tz-naive
            published = ts.to_pydatetime()
        except Exception:
            return None

        # Extract CNA org and short name
        containers = data.get("containers", {})
        cna = containers.get("cna", {})
        provider = cna.get("providerMetadata", {}) if isinstance(cna, dict) else {}

        org_id = provider.get("orgId") or meta.get("assignerOrgId")
        short_name = provider.get("shortName") or meta.get("assignerShortName")

        if not org_id:
            # Cannot attribute to a CNA; skip
            return None

        return published, CNARecord(org_id=org_id, short_name=short_name)
    except Exception:
        # Corrupt file, ignore
        return None


def scan_cvelist_for_cna_counts(cvelist_dir: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Scan cvelistV5 and build a DataFrame with rows [org_id, date].

    Returns:
        - DataFrame with columns: ['org_id', 'date'] (date is pandas.Timestamp)
        - Mapping org_id -> short_name (best effort)
    """
    pattern = os.path.join(cvelist_dir, "cves", "*", "*", "CVE-*.json")
    paths = glob(pattern)
    logger.info("Scanning %d CVE files from %s", len(paths), cvelist_dir)

    rows: List[Tuple[str, pd.Timestamp]] = []
    names: Dict[str, str] = {}

    for p in paths:
        parsed = parse_cve_file(p)
        if not parsed:
            continue
        published, cna = parsed
        # ensure tz-naive normalized month start
        rows.append((cna.org_id, pd.to_datetime(published).tz_localize(None).normalize()))
        if cna.short_name and cna.org_id not in names:
            names[cna.org_id] = cna.short_name

    if not rows:
        logger.warning("No publishable CVE records found. Is the repository clone correct?")
        return pd.DataFrame(columns=["org_id", "date"]), names

    df = pd.DataFrame(rows, columns=["org_id", "date"])  # type: ignore
    return df, names


def build_monthly_series(df: pd.DataFrame, org_id: str) -> tuple[pd.Series, int]:
    """Build a complete monthly count series (MS frequency) for a given CNA org_id.

    Missing months are filled with zeros to create a contiguous time series.
    Returns both the full series and current month partial count.
    Limits historical data to start from 2017-01-01 for consistency.
    """
    sub = df[df["org_id"] == org_id].copy()
    if sub.empty:
        return pd.Series(dtype=float), 0
    sub = sub.sort_values("date")

    # Limit historical data to start from 2017-01-01 for consistency with main page
    start_cutoff = pd.Timestamp("2017-01-01")
    sub = sub[sub["date"] >= start_cutoff]
    
    if sub.empty:
        return pd.Series(dtype=float), 0

    # Monthly start frequency - start from 2017-01-01 or first data point, whichever is later
    data_start = sub["date"].min().to_period("M").to_timestamp(how="start")
    start = max(start_cutoff, data_start)
    end = sub["date"].max().to_period("M").to_timestamp(how="start")

    # Count per month
    counts = sub.set_index("date").resample("MS").size()

    # Reindex to full monthly range
    full_index = pd.date_range(start=start, end=end, freq="MS")
    counts = counts.reindex(full_index, fill_value=0).astype(float)
    counts.index.name = "date"
    
    # Calculate current month partial count (up to today)
    current_month_start = pd.Timestamp.now().to_period("M").to_timestamp(how="start")
    current_month_data = sub[sub["date"] >= current_month_start]
    current_month_partial = len(current_month_data)
    
    return counts, current_month_partial


def series_to_darts(counts: pd.Series) -> TimeSeries:
    df = counts.reset_index()
    df.columns = ["date", "value"]
    return TimeSeries.from_dataframe(df, time_col="date", value_cols="value", fill_missing_dates=True, freq="MS")


def validate_model_performance(
    ts: TimeSeries,
    model_class,
    model_params: Dict[str, Any],
    validation_months: int = 6
) -> float:
    """Validate model performance using walk-forward validation.
    
    Returns MAPE (Mean Absolute Percentage Error) - lower is better.
    """
    if len(ts) < validation_months + 12:  # Need enough data for validation
        return float('inf')
    
    try:
        # Use last validation_months for testing
        train_ts = ts[:-validation_months]
        test_ts = ts[-validation_months:]
        
        # Train model
        if model_class.__name__ == 'Prophet':
            model = model_class(**model_params)
        elif model_class.__name__ in ['LightGBMModel', 'XGBModel']:
            lags = model_params.pop('lags', 12)
            if len(train_ts) <= lags:
                return float('inf')
            model = model_class(lags=lags, **model_params)
        else:
            model = model_class(**model_params)
        
        model.fit(train_ts)
        predictions = model.predict(validation_months)
        
        # Calculate MAPE
        actual = test_ts.values().flatten()
        predicted = predictions.values().flatten()
        
        # Avoid division by zero
        mask = actual != 0
        if not mask.any():
            return float('inf')
        
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return mape
        
    except Exception:
        return float('inf')


def select_best_models_for_cna(
    ts: TimeSeries,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Test multiple models and return the best performing one with its validation score.
    
    Returns dict with 'best_model', 'model_name', 'mape_score', and 'all_scores'.
    """
    from darts.models import ExponentialSmoothing, AutoARIMA, LinearRegressionModel
    from darts.models import LightGBMModel, XGBModel
    
    try:
        from darts.models import Prophet as DartsProphet
    except ImportError:
        try:
            from darts.models.forecasting.prophet import Prophet as DartsProphet
        except ImportError:
            DartsProphet = None
            logger.warning("Prophet model not available, skipping")
    
    # Top 5 CPU-only models based on validation MAPE performance
    model_candidates = [
        # Best performing models (ordered by MAPE) - ExponentialSmoothing replaces TiDE for reliability
        ('ExponentialSmoothing', ExponentialSmoothing, get_model_hyperparameters(config, "ExponentialSmoothing")),
        ('LightGBM', LightGBMModel, get_model_hyperparameters(config, "LightGBM")),
        ('XGBoost', XGBModel, get_model_hyperparameters(config, "XGBoost")),
        ('LinearRegression', LinearRegressionModel, get_model_hyperparameters(config, "LinearRegression") if LinearRegressionModel else {}),
        ('Prophet', DartsProphet, get_model_hyperparameters(config, "Prophet")),
    ]
    
    # Filter out unavailable models
    model_candidates = [(name, cls, params) for name, cls, params in model_candidates if cls is not None]
    
    validation_scores = {}
    
    for model_name, model_class, params in model_candidates:
        try:
            # Clean params for each model type
            clean_params = dict(params)
            
            if model_name == 'Prophet':
                # Prophet-specific parameter handling
                clean_params = {k: v for k, v in clean_params.items() 
                              if k in ['yearly_seasonality', 'weekly_seasonality', 'daily_seasonality',
                                     'seasonality_mode', 'growth', 'changepoint_prior_scale',
                                     'seasonality_prior_scale', 'n_changepoints', 'mcmc_samples',
                                     'interval_width']}
            elif model_name in ['LightGBM', 'XGBoost']:
                # Remove non-model parameters
                clean_params = {k: v for k, v in clean_params.items() 
                              if k not in ['random_state', 'feature_pre_filter', 'early_stopping_rounds']}
                clean_params['random_state'] = params.get('random_state', 42)
            elif model_name == 'ExponentialSmoothing':
                # ExponentialSmoothing-specific parameter handling
                clean_params = {k: v for k, v in clean_params.items() 
                              if k not in ['random_state']}
                # ExponentialSmoothing handles trend and seasonality
            elif model_name == 'LinearRegression':
                # LinearRegression-specific parameter handling
                clean_params = {k: v for k, v in clean_params.items() 
                              if k not in ['likelihood', 'quantiles']}
            
            mape = validate_model_performance(ts, model_class, clean_params)
            validation_scores[model_name] = mape
            logger.info(f"  {model_name}: MAPE = {mape:.2f}%")
            
        except Exception as e:
            logger.warning(f"  {model_name} validation failed: {e}")
            validation_scores[model_name] = float('inf')
    
    # Find best model with fallback logic
    valid_scores = {k: v for k, v in validation_scores.items() if v != float('inf')}
    
    if not valid_scores:
        # Fallback: if all models failed, use LightGBM as default
        logger.warning("All models failed validation, using LightGBM as fallback")
        return {
            'best_model': 'LightGBM',
            'mape_score': 100.0,  # High MAPE to indicate fallback
            'all_scores': validation_scores
        }
    
    best_model_name = min(valid_scores.keys(), key=lambda k: valid_scores[k])
    best_mape = valid_scores[best_model_name]
    
    return {
        'best_model': best_model_name,
        'mape_score': best_mape,
        'all_scores': validation_scores
    }


def forecast_with_models(
    ts: TimeSeries,
    config: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Select best model for this CNA and produce forecasts.

    Returns a dict with the best model's forecasts and validation info.
    """
    out: Dict[str, Dict[str, float]] = {}
    
    # Calculate dynamic horizon: remaining months in current year + all of next year
    now = pd.Timestamp.now()
    current_year = now.year
    current_month = now.month
    next_year = current_year + 1
    
    # Count remaining months in current year (after current month)
    remaining_current_year = 12 - current_month
    # All months of next year
    next_year_months = 12
    # Total horizon
    horizon = remaining_current_year + next_year_months
    
    # Select best model for this CNA
    logger.info("Validating models for CNA...")
    model_selection = select_best_models_for_cna(ts, config)
    best_model_name = model_selection['best_model']
    
    logger.info(f"Best model: {best_model_name} (MAPE: {model_selection['mape_score']:.2f}%)")
    
    # Train and forecast with best model
    try:
        if best_model_name == 'Prophet':
            prophet_params = get_model_hyperparameters(config, "Prophet")
            prophet = DartsProphet(
                yearly_seasonality=prophet_params.get("yearly_seasonality", True),
                weekly_seasonality=prophet_params.get("weekly_seasonality", False),
                daily_seasonality=prophet_params.get("daily_seasonality", False),
                seasonality_mode=prophet_params.get("seasonality_mode", "additive"),
                growth=prophet_params.get("growth", "linear"),
                changepoint_prior_scale=prophet_params.get("changepoint_prior_scale", 0.05),
                seasonality_prior_scale=prophet_params.get("seasonality_prior_scale", 0.1),
                n_changepoints=prophet_params.get("n_changepoints", 25),
                mcmc_samples=prophet_params.get("mcmc_samples", 0),
                interval_width=prophet_params.get("interval_width", 0.8),
            )
            prophet.fit(ts)
            forecast = prophet.predict(horizon)
            out[best_model_name] = {idx.strftime("%Y-%m"): max(0, round(float(val))) 
                                   for idx, val in zip(forecast.time_index, forecast.values().flatten())}
        
        elif best_model_name == 'LightGBM':
            lgb_params = get_model_hyperparameters(config, "LightGBM")
            lags = int(lgb_params.pop("lags", 12) or 12)
            if len(ts) <= lags:
                lags = max(1, min(lags, len(ts) - 1))
            if len(ts) - lags < 2:
                raise ValueError("Insufficient samples for LightGBM after applying lags")
            
            lgbm_specific = {k: v for k, v in lgb_params.items() 
                           if k not in {"lags", "random_state", "feature_pre_filter"} and v is not None}
            
            model_lgb = LightGBMModel(
                lags=lags,
                random_state=lgb_params.get("random_state", 42),
                **lgbm_specific,
            )
            model_lgb.fit(ts)
            forecast = model_lgb.predict(horizon)
            out[best_model_name] = {idx.strftime("%Y-%m"): max(0, round(float(val))) 
                                   for idx, val in zip(forecast.time_index, forecast.values().flatten())}
        
        elif best_model_name == 'XGBoost':
            xgb_params = get_model_hyperparameters(config, "XGBoost")
            lags = int(xgb_params.pop("lags", 12) or 12)
            if len(ts) <= lags:
                lags = max(1, min(lags, len(ts) - 1))
            if len(ts) - lags < 2:
                raise ValueError("Insufficient samples for XGBoost after applying lags")
            
            xgb_specific = {k: v for k, v in xgb_params.items() 
                          if k not in {"lags", "random_state", "early_stopping_rounds"} and v is not None}
            
            model_xgb = XGBModel(
                lags=lags,
                random_state=xgb_params.get("random_state", 42),
                **xgb_specific,
            )
            model_xgb.fit(ts)
            forecast = model_xgb.predict(horizon)
            out[best_model_name] = {idx.strftime("%Y-%m"): max(0, round(float(val))) 
                                   for idx, val in zip(forecast.time_index, forecast.values().flatten())}
        
        elif best_model_name == 'ExponentialSmoothing':
            from darts.models import ExponentialSmoothing
            model = ExponentialSmoothing()
            model.fit(ts)
            forecast = model.predict(horizon)
            out[best_model_name] = {idx.strftime("%Y-%m"): max(0, round(float(val))) 
                                   for idx, val in zip(forecast.time_index, forecast.values().flatten())}
        
        elif best_model_name == 'LinearRegression':
            lr_params = get_model_hyperparameters(config, "LinearRegression")
            lags = int(lr_params.pop("lags", 30) or 30)
            if len(ts) <= lags:
                lags = max(1, min(lags, len(ts) - 1))
            if len(ts) - lags < 2:
                raise ValueError("Insufficient samples for LinearRegression after applying lags")
            
            lr_specific = {k: v for k, v in lr_params.items() 
                          if k not in {"lags", "likelihood", "quantiles"} and v is not None}
            
            model_lr = LinearRegressionModel(
                lags=lags,
                **lr_specific,
            )
            model_lr.fit(ts)
            forecast = model_lr.predict(horizon)
            out[best_model_name] = {idx.strftime("%Y-%m"): max(0, round(float(val))) 
                                   for idx, val in zip(forecast.time_index, forecast.values().flatten())}
        
        elif best_model_name == 'ExponentialSmoothing':
            es_params = get_model_hyperparameters(config, "ExponentialSmoothing")
            
            # Clean ExponentialSmoothing parameters
            clean_params = {k: v for k, v in es_params.items() 
                          if k not in ['random_state'] and v is not None}
            
            model_es = ExponentialSmoothing(**clean_params)
            model_es.fit(ts)
            forecast = model_es.predict(horizon)
            out[best_model_name] = {idx.strftime("%Y-%m"): max(0, round(float(val))) 
                                   for idx, val in zip(forecast.time_index, forecast.values().flatten())}
        
    
    except Exception as e:
        logger.warning(f"{best_model_name} forecasting failed: {e}")
        # Fallback to simple exponential smoothing
        try:
            from darts.models import ExponentialSmoothing
            model = ExponentialSmoothing()
            model.fit(ts)
            forecast = model.predict(horizon)
            out['ExponentialSmoothing'] = {idx.strftime("%Y-%m"): max(0, round(float(val))) 
                                         for idx, val in zip(forecast.time_index, forecast.values().flatten())}
            best_model_name = 'ExponentialSmoothing'
        except Exception:
            logger.error("All models failed for this CNA")
            return {}
    
    # Add model selection metadata
    out['_metadata'] = {
        'selected_model': best_model_name,
        'validation_mape': model_selection['mape_score'],
        'all_model_scores': model_selection['all_scores']
    }
    
    return out


def run(cvelist_dir: str, output_path: str, min_cves: int) -> None:
    logger.info("Starting CNA forecast generation | min_cves=%s", min_cves)

    config = load_config(os.path.join("code", "cna_config.json"))

    df, cna_names = scan_cvelist_for_cna_counts(cvelist_dir)
    if df.empty:
        # still write an empty JSON to avoid CI churn
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        logger.info("No data found; wrote empty %s", output_path)
        return

    # Aggregate by CNA
    results: Dict[str, Any] = {}

    # Precompute total counts per CNA to filter
    totals = df.groupby("org_id").size().sort_values(ascending=False)
    eligible_orgs = totals[totals >= int(min_cves)].index.tolist()
    logger.info("Eligible CNAs (>= %d CVEs): %d", min_cves, len(eligible_orgs))

    for i, org_id in enumerate(eligible_orgs, start=1):
        try:
            counts, current_month_partial = build_monthly_series(df, org_id)
            if counts.empty or counts.sum() < min_cves:
                continue

            ts = series_to_darts(counts)
            forecasts = forecast_with_models(ts, config)

            # Format historical dict as YYYY-MM -> int
            hist = {idx.strftime("%Y-%m"): int(v) for idx, v in counts.items()}
            
            # Generate cumulative data for chart (matching main.py structure)
            historical_cumulative = []
            cumulative_timelines = {}
            
            # Add current month partial data
            now = pd.Timestamp.now()
            current_month = now.month
            current_year = now.year
            next_year = current_year + 1
            current_month_key = now.strftime("%Y-%m")

            # Extract metadata and clean forecasts
            metadata = forecasts.pop('_metadata', {})
            selected_model = metadata.get('selected_model', 'Unknown')
            
            # Generate historical cumulative data for current year (starting at 0 on Jan 1)
            cumulative_current_year = 0
            historical_cumulative.append({
                "date": f"{current_year}-01-01T12:00:00Z",
                "cumulative_total": 0
            })
            
            for month in range(1, current_month + 1):  # Jan through current month
                month_key = f"{current_year}-{month:02d}"
                if month_key in hist:
                    cumulative_current_year += hist[month_key]
                
                # Add data point at beginning of next month showing cumulative total up to current month
                if month < 12:  # Don't go beyond December
                    next_month = month + 1
                    if next_month <= current_month:  # Only add if next month is not in future
                        historical_cumulative.append({
                            "date": f"{current_year}-{next_month:02d}-01T12:00:00Z",
                            "cumulative_total": cumulative_current_year
                        })
            
            # Generate forecast cumulative data (matching main.py cumulative_timelines structure)
            if selected_model in forecasts:
                model_key = f"{selected_model}_cumulative"
                cumulative_timelines[model_key] = []
                
                # Start cumulative from current year total
                cumulative_forecast = cumulative_current_year
                
                # Add remaining months of current year (2025)
                for month in range(current_month + 1, 13):  # From next month through Dec
                    month_key = f"{current_year}-{month:02d}"
                    if month_key in forecasts[selected_model]:
                        # Add data point at beginning of month BEFORE adding this month's CVEs
                        # This shows cumulative total through previous month
                        cumulative_timelines[model_key].append({
                            "date": f"{current_year}-{month:02d}-01T12:00:00Z",
                            "cumulative_total": cumulative_forecast
                        })
                        
                        # Now add this month's CVEs to cumulative total
                        cumulative_forecast += forecasts[selected_model][month_key]
                
                # Add dedicated end-of-year total point (December 31st)
                # This shows the complete year-end total including all December CVEs
                if cumulative_forecast > 0:
                    cumulative_timelines[model_key].append({
                        "date": f"{current_year}-12-31T23:59:59Z",
                        "cumulative_total": cumulative_forecast
                    })
                
                # Add January 1st of next year starting at 0
                cumulative_timelines[model_key].append({
                    "date": f"{next_year}-01-01T12:00:00Z", 
                    "cumulative_total": 0
                })
                
                # Add all months of next year
                cumulative_next_year = 0
                for month in range(1, 13):  # Jan through Dec of next year
                    month_key = f"{next_year}-{month:02d}"
                    if month_key in forecasts[selected_model]:
                        # Add data point at beginning of month BEFORE adding this month's CVEs
                        # This shows cumulative total through previous month
                        cumulative_timelines[model_key].append({
                            "date": f"{next_year}-{month:02d}-01T12:00:00Z",
                            "cumulative_total": cumulative_next_year
                        })
                        
                        # Now add this month's CVEs to cumulative total
                        cumulative_next_year += forecasts[selected_model][month_key]
                
                # Add dedicated end-of-year total point for next year (December 31st)
                # This shows the complete year-end total including all December CVEs
                if cumulative_next_year > 0:
                    cumulative_timelines[model_key].append({
                        "date": f"{next_year}-12-31T23:59:59Z",
                        "cumulative_total": cumulative_next_year
                    })
            
            # Clean up Infinity values for JSON serialization
            validation_mape = metadata.get('validation_mape', float('inf'))
            if not np.isfinite(validation_mape):
                validation_mape = 999.99  # Use high but finite value
            
            all_model_scores = metadata.get('all_model_scores', {})
            cleaned_scores = {}
            for model, score in all_model_scores.items():
                if np.isfinite(score):
                    cleaned_scores[model] = score
                else:
                    cleaned_scores[model] = 999.99  # Use high but finite value for failed models
            
            results[org_id] = {
                "id": org_id,
                "name": cna_names.get(org_id),
                "scope": None,  # placeholder; can be enriched from a CNA registry later
                "historical": hist,
                "historical_cumulative": historical_cumulative,
                "cumulative_timelines": cumulative_timelines,  # Match main.py structure
                "current_month": {
                    "month": current_month_key,
                    "partial_count": current_month_partial,
                    "days_elapsed": pd.Timestamp.now().day,
                    "days_in_month": pd.Timestamp.now().days_in_month
                },
                "forecasts": forecasts,
                "model_selection": {
                    "selected_model": metadata.get('selected_model', 'LightGBM'),  # Fallback to LightGBM instead of Unknown
                    "validation_mape": validation_mape,
                    "all_model_scores": cleaned_scores
                }
            }
            if i % 25 == 0:
                logger.info("Processed %d / %d CNAs", i, len(eligible_orgs))
        except Exception as e:
            logger.warning("Failed processing org_id %s: %s", org_id, e)
            continue

    # Persist output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    logger.info("Wrote CNA forecasts: %s (CNAs: %d)", output_path, len(results))


def main() -> None:
    # Fixed parameters for GitHub Actions - no command line arguments needed
    cvelist_dir = "cvelistV5"
    output_path = os.path.join("web", "cna_data.json")
    min_cves = 50  # Lower threshold to include more CNAs
    
    run(cvelist_dir, output_path, min_cves)


if __name__ == "__main__":
    main()
