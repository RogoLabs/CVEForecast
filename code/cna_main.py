#!/usr/bin/env python3
"""
CNA-specific CVE forecasting script.

- Parses CVEs from a local clone of CVEProject/cvelistV5
- Groups monthly publication counts by CNA (provider orgId)
- Filters to CNAs with at least --min_cves total published CVEs
- Trains 3 fast models (LightGBM, XGBoost, Prophet) with hyperparameters loaded from code/config.json
- Produces 12-month forecasts for each eligible CNA
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
  python code/cna_main.py --cvelist_dir cvelistV5 --output web/cna_data.json --min_cves 100 --horizon 12

Notes:
- This script intentionally does not depend on the main forecasting pipeline to keep runtime fast and isolated.
- It uses the same optimized hyperparameters for consistency by reading code/config.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    pattern = os.path.join(cvelist_dir, "cves", "**", "**", "*.json")
    paths = glob(pattern, recursive=True)
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
    """
    sub = df[df["org_id"] == org_id].copy()
    if sub.empty:
        return pd.Series(dtype=float), 0
    sub = sub.sort_values("date")

    # Monthly start frequency
    start = sub["date"].min().to_period("M").to_timestamp(how="start")
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


def forecast_with_models(
    ts: TimeSeries,
    horizon: int,
    config: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Train LightGBM, XGBoost, and Prophet on ts and produce horizon-step forecasts.

    Returns a dict of model_name -> {"YYYY-MM": forecast_value}.
    """
    out: Dict[str, Dict[str, float]] = {}

    # Prophet
    try:
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
        f_prophet = prophet.predict(horizon)
        out["Prophet"] = {idx.strftime("%Y-%m"): float(val) for idx, val in zip(f_prophet.time_index, f_prophet.values().flatten())}
    except Exception as e:
        logger.warning("Prophet failed: %s", e)

    # LightGBM
    try:
        lgb_params = get_model_hyperparameters(config, "LightGBM")
        lags = int(lgb_params.pop("lags", 12) or 12)
        # Ensure enough history for lags
        if len(ts) <= lags:
            lags = max(1, min(lags, len(ts) - 1))
        # Require at least two effective samples after lagging
        if len(ts) - lags < 2:
            raise ValueError("Insufficient samples for LightGBM after applying lags")
        # Map config to LightGBM params
        lgbm_specific = {}
        for k, v in lgb_params.items():
            if k in {"lags", "random_state", "feature_pre_filter"}:
                continue
            if v is None:
                continue
            lgbm_specific[k] = v
        model_lgb = LightGBMModel(
            lags=lags,
            random_state=lgb_params.get("random_state", 42),
            **lgbm_specific,
        )
        model_lgb.fit(ts)
        f_lgb = model_lgb.predict(horizon)
        out["LightGBM"] = {idx.strftime("%Y-%m"): float(val) for idx, val in zip(f_lgb.time_index, f_lgb.values().flatten())}
    except Exception as e:
        logger.warning("LightGBM failed (org may have too little data): %s", e)

    # XGBoost
    try:
        xgb_params = get_model_hyperparameters(config, "XGBoost")
        lags = int(xgb_params.pop("lags", 12) or 12)
        if len(ts) <= lags:
            lags = max(1, min(lags, len(ts) - 1))
        if len(ts) - lags < 2:
            raise ValueError("Insufficient samples for XGBoost after applying lags")
        xgb_specific = {}
        for k, v in xgb_params.items():
            if k in {"lags", "random_state", "early_stopping_rounds"}:
                continue
            if v is None:
                continue
            xgb_specific[k] = v
        model_xgb = XGBModel(
            lags=lags,
            random_state=xgb_params.get("random_state", 42),
            **xgb_specific,
        )
        model_xgb.fit(ts)
        f_xgb = model_xgb.predict(horizon)
        out["XGBoost"] = {idx.strftime("%Y-%m"): float(val) for idx, val in zip(f_xgb.time_index, f_xgb.values().flatten())}
    except Exception as e:
        logger.warning("XGBoost failed (org may have too little data): %s", e)

    return out


def run(cvelist_dir: str, output_path: str, min_cves: int, horizon: int) -> None:
    logger.info("Starting CNA forecast generation | min_cves=%s horizon=%s", min_cves, horizon)

    config = load_config(os.path.join("code", "config.json"))

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
            forecasts = forecast_with_models(ts, horizon, config)

            # Format historical dict as YYYY-MM -> int
            hist = {idx.strftime("%Y-%m"): int(v) for idx, v in counts.items()}
            
            # Add current month partial data
            current_month_key = pd.Timestamp.now().strftime("%Y-%m")

            results[org_id] = {
                "id": org_id,
                "name": cna_names.get(org_id),
                "scope": None,  # placeholder; can be enriched from a CNA registry later
                "historical": hist,
                "current_month": {
                    "month": current_month_key,
                    "partial_count": current_month_partial,
                    "days_elapsed": pd.Timestamp.now().day,
                    "days_in_month": pd.Timestamp.now().days_in_month
                },
                "forecasts": forecasts,
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
    parser = argparse.ArgumentParser(description="Generate CNA-specific CVE forecasts")
    parser.add_argument("--cvelist_dir", type=str, default="cvelistV5", help="Path to cvelistV5 clone")
    parser.add_argument("--output", type=str, default=os.path.join("web", "cna_data.json"), help="Output JSON path")
    parser.add_argument("--min_cves", type=int, default=100, help="Minimum total CVEs per CNA to include")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon in months")

    args = parser.parse_args()
    run(args.cvelist_dir, args.output, args.min_cves, args.horizon)


if __name__ == "__main__":
    main()
