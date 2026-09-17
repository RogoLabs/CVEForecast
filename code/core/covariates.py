"""
Calendar covariates and business-day normalisation.

CVEs are published on working days, but the pipeline forecasts calendar months.
Months carry 20-23 business days - a 15% swing that no model can infer from the
count series alone, and which the calendar tells us exactly, for every month,
forever. Prior to v0.12 every model was left to learn this from 117 noisy
observations; the "February is a quiet month" seasonal index (0.889, the lowest
of any month) was largely just February being short.

Two mechanisms live here:

1. ``business_days_for_index`` powers a *normalisation* - divide counts by
   business days before modelling, multiply back afterwards. This removes the
   effect outright and works for every model, including the ones that accept no
   covariates at all (Theta, TBATS, Croston, Kalman).
2. ``build_future_covariates`` emits month-of-year dummies for the regression
   models that do accept ``lags_future_covariates``, to capture whatever
   seasonality survives the business-day adjustment.

Measured on the reconstructed monthly series (2017-2026), correlation between
detrended counts and business-day count was +0.287 (2017-2024) and +0.395
(2022-2025).
"""

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from darts import TimeSeries

logger = logging.getLogger(__name__)

# Models that accept lags_future_covariates in darts. Everything else silently
# ignores covariates, so we do not pay the cost of building them.
COVARIATE_CAPABLE_MODELS = frozenset(
    {
        'LinearRegression',
        'LightGBM',
        'XGBoost',
        'CatBoost',
        'RandomForest',
    }
)


def business_days_in_month(period_end: pd.Timestamp) -> int:
    """
    Count Mon-Fri days in the calendar month containing ``period_end``.

    Public holidays are deliberately not excluded: they vary by CNA jurisdiction
    and the dominant publishers span several countries, so a single holiday
    calendar would add noise rather than remove it.

    Args:
        period_end: Any timestamp inside the target month

    Returns:
        Number of business days in that month (20-23 in practice)
    """
    start = pd.Timestamp(period_end).replace(day=1)
    end = start + pd.offsets.MonthEnd(0)
    return len(pd.bdate_range(start, end))


def business_days_for_index(index: Sequence[pd.Timestamp]) -> np.ndarray:
    """
    Business-day count for every period in a monthly index.

    Args:
        index: Monthly DatetimeIndex (month-start or month-end anchored)

    Returns:
        Float array of business-day counts, aligned to ``index``
    """
    return np.array([business_days_in_month(ts) for ts in index], dtype=float)


def normalise_by_business_days(series: TimeSeries) -> TimeSeries:
    """
    Convert a monthly count series into counts per business day.

    Args:
        series: Monthly TimeSeries of raw counts

    Returns:
        TimeSeries of counts per business day, same index
    """
    bdays = business_days_for_index(series.time_index)
    values = series.values().flatten() / bdays
    return TimeSeries.from_times_and_values(series.time_index, values, columns=series.components)


def denormalise_by_business_days(series: TimeSeries) -> TimeSeries:
    """
    Convert a per-business-day series back into monthly counts.

    Args:
        series: Monthly TimeSeries of counts per business day

    Returns:
        TimeSeries of raw monthly counts, same index
    """
    bdays = business_days_for_index(series.time_index)
    values = series.values().flatten() * bdays
    return TimeSeries.from_times_and_values(series.time_index, values, columns=series.components)


def build_future_covariates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = 'ME',
    include_business_days: bool = True,
) -> TimeSeries:
    """
    Build the future-covariate block: month dummies and (optionally) business days.

    Future covariates must span training *and* forecast periods, so callers should
    pass a range that comfortably covers both.

    Args:
        start: First period to cover
        end: Last period to cover (inclusive)
        freq: Pandas frequency alias for the index
        include_business_days: Emit a scaled business-day column. Leave this on
            when the target series has *not* been business-day normalised, off
            when it has (the information is already gone from the target).

    Returns:
        TimeSeries with 12 month-dummy components, plus ``business_days`` if requested
    """
    index = pd.date_range(start=start, end=end, freq=freq)
    if len(index) == 0:
        raise ValueError(f'Empty covariate range: {start} .. {end} at freq={freq}')

    frame = pd.DataFrame(index=index)
    for month in range(1, 13):
        frame[f'month_{month:02d}'] = (index.month == month).astype(float)

    if include_business_days:
        bdays = business_days_for_index(index)
        # Centre and scale so the column sits on the same order as the dummies;
        # unscaled counts of ~21 dominate regularised regressions.
        frame['business_days'] = (bdays - bdays.mean()) / max(bdays.std(), 1e-9)

    logger.debug(f'Built future covariates: {len(frame)} periods, {len(frame.columns)} components')
    return TimeSeries.from_dataframe(frame, freq=freq, fill_missing_dates=False)


def slice_covariates(covariates: Optional[TimeSeries], series: TimeSeries) -> Optional[TimeSeries]:
    """
    Trim a covariate block to the span a model needs for a given target series.

    darts tolerates covariates that extend beyond the target, so this is a
    convenience for logging and for keeping fit/predict calls symmetric.

    Args:
        covariates: Covariate TimeSeries, or None
        series: Target series being modelled

    Returns:
        Covariates covering at least ``series``, or None if none were supplied
    """
    if covariates is None:
        return None
    if covariates.start_time() > series.start_time() or covariates.end_time() < series.end_time():
        logger.warning(
            f'Covariates ({covariates.start_time()}..{covariates.end_time()}) do not fully cover '
            f'target ({series.start_time()}..{series.end_time()})'
        )
    return covariates


def build_cna_covariate(
    monthly_counts: 'pd.DataFrame',
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = 'ME',
) -> Optional[TimeSeries]:
    """
    Cumulative active-CNA count as a covariate, extended over the forecast horizon.

    CNA onboarding looks like the clearest exogenous driver of publication volume:
    more numbering authorities issuing identifiers means more CVEs, and the raw
    correlation between the two series is +0.796.

    **It does not improve forecasts, and is off by default.** Measured over 24
    rolling origins at h=1..12, adding it cost the best model 2.12 -> 2.31 MASE
    and moved no other model by more than 0.06. That correlation is almost
    entirely shared trend - both series climb together, which a lag-based model
    already captures from the target alone - so the covariate contributes noise
    rather than signal. The extrapolation below adds more of it.

    Two further caveats if the picture ever changes. Join dates are inferred from
    the ``CNA-YYYY-NNNN`` identifier, so the real resolution is annual with a
    heuristic quarter; the monthly series is smoother than the truth. And future
    values must be extrapolated, since we cannot know who will onboard next: this
    uses the trailing 24-month linear trend, which suits a steadily growing series
    but will understate a sudden recruitment drive.

    Args:
        monthly_counts: DataFrame with 'date' and 'cna_count' columns
        start: First period to cover
        end: Last period to cover, typically past the forecast horizon
        freq: Pandas frequency alias

    Returns:
        Single-component TimeSeries of CNA counts, or None if unusable
    """
    if monthly_counts is None or monthly_counts.empty:
        logger.warning('No CNA count data available; covariate unavailable')
        return None

    counts = monthly_counts.set_index(pd.DatetimeIndex(monthly_counts['date']))['cna_count'].astype(float)
    counts = counts.resample(freq).last().dropna()
    if len(counts) < 24:
        logger.warning(f'Only {len(counts)} months of CNA history; covariate unavailable')
        return None

    index = pd.date_range(start=start, end=end, freq=freq)
    aligned = counts.reindex(index)

    # Back-fill before the record starts, then extrapolate the trailing trend forward.
    aligned = aligned.ffill()
    if aligned.isna().any():
        aligned = aligned.bfill()

    last_known = counts.index.max()
    future = index[index > last_known]
    if len(future) > 0:
        recent = counts.iloc[-24:]
        slope = float(np.polyfit(np.arange(len(recent)), recent.to_numpy(), 1)[0])
        base = float(counts.iloc[-1])
        aligned.loc[future] = [base + slope * (i + 1) for i in range(len(future))]
        logger.info(f'Extrapolated CNA count {len(future)} months at {slope:.2f}/month from {base:.0f}')

    values = aligned.to_numpy(dtype=float)
    # Scale so the covariate sits on the same order as the month dummies.
    scaled = (values - values.mean()) / max(values.std(), 1e-9)
    return TimeSeries.from_times_and_values(index, scaled, columns=['cna_count'])
