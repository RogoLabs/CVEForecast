"""
Log-space modelling and damped-trend post-processing.

The CVE series is multiplicative: 2017-2026 runs 17,950 -> ~63,500 with
compounding growth, step changes when a large CNA onboards, and variance that
scales with level. Modelling it in levels fights that on every axis - trend is
under-forecast, residuals are heteroscedastic, and symmetric prediction
intervals misrepresent a right-skewed quantity.

Prior to v0.12 the pipeline modelled in levels and then applied a hard-coded
"growth floor" to compensate for the resulting downward bias. Measured over 24
rolling origins, moving to log space plus business-day normalisation took the
best model from 2.39 to 2.12 MASE. Bias stays negative (-9.4%) because the 2026
surge outruns every method - but that is a forecasting limit to be shown through
prediction intervals, not papered over by inflating the point estimate.

Damping is applied to the forecast *path* rather than inside any one model, so
it works identically for all 15 model families. Without it, extrapolating the
2026 regime (+14.7%/month) 16 months forward produces an absurd 2027. The default
phi=0.98 is deliberately mild: harder damping scored no better and deepened an
already negative bias.
"""

import logging
from typing import Optional

import numpy as np
from darts import TimeSeries

logger = logging.getLogger(__name__)


def to_log_space(series: TimeSeries) -> TimeSeries:
    """
    Map a count series into log space.

    Uses log1p so that zero-count months (possible for small CNAs) stay finite.

    Args:
        series: TimeSeries of non-negative counts

    Returns:
        TimeSeries of log1p(counts)
    """
    values = series.values().flatten()
    if np.any(values < 0):
        raise ValueError('Cannot log-transform a series containing negative values')
    return TimeSeries.from_times_and_values(series.time_index, np.log1p(values), columns=series.components)


def from_log_space(series: TimeSeries, sigma: Optional[float] = None) -> TimeSeries:
    """
    Map a log-space series back to counts.

    Args:
        series: TimeSeries of log1p values
        sigma: Residual standard deviation in log space. When supplied, applies
            the ``exp(sigma^2 / 2)`` smearing correction so the result estimates
            the *mean* rather than the median. Leave as None to publish the
            median, which is the more natural central estimate for a skewed
            quantity and the one our prediction intervals are built around.

    Returns:
        TimeSeries of counts, floored at zero
    """
    values = np.expm1(series.values().flatten())
    if sigma is not None and sigma > 0:
        values = values * np.exp(sigma**2 / 2)
    return TimeSeries.from_times_and_values(series.time_index, np.maximum(values, 0.0), columns=series.components)


def damp_forecast_path(
    forecast: TimeSeries,
    last_observed: float,
    phi: float = 0.9,
    in_log_space: bool = True,
) -> TimeSeries:
    """
    Damp the trend implied by a forecast path.

    Takes the step-to-step increments the model projected and shrinks the i-th
    increment by ``phi**i``, then re-accumulates. For a model projecting a roughly
    constant per-step increment ``b`` this reproduces the textbook damped trend
    ``level + (phi + phi^2 + ... + phi^h) * b``, but it generalises to any path -
    including the non-linear ones gradient-boosted models produce.

    Args:
        forecast: Model forecast, in whichever space it was produced
        last_observed: Final observed value of the training series, same space
        phi: Damping factor in (0, 1]. 1.0 disables damping. Values around
            0.85-0.95 are typical; lower damps harder.
        in_log_space: True when ``forecast`` holds log values. Damping is only
            meaningful on additive increments, so in levels space the increments
            are converted to ratios first.

    Returns:
        Damped forecast, same index and space as the input
    """
    if not 0 < phi <= 1:
        raise ValueError(f'phi must be in (0, 1], got {phi}')
    if phi == 1.0:
        return forecast

    values = forecast.values().flatten().astype(float)
    if len(values) == 0:
        return forecast

    if in_log_space:
        anchored = np.concatenate([[last_observed], values])
        increments = np.diff(anchored)
        weights = phi ** np.arange(1, len(increments) + 1)
        damped = last_observed + np.cumsum(increments * weights)
    else:
        # Work on log-ratios so damping is scale-free, then map back.
        safe_last = max(last_observed, 1e-9)
        anchored = np.concatenate([[safe_last], np.maximum(values, 1e-9)])
        log_increments = np.diff(np.log(anchored))
        weights = phi ** np.arange(1, len(log_increments) + 1)
        damped = safe_last * np.exp(np.cumsum(log_increments * weights))

    logger.debug(f'Damped {len(values)} step path with phi={phi} (log_space={in_log_space})')
    return TimeSeries.from_times_and_values(forecast.time_index, damped, columns=forecast.components)


def trim_to_window(series: TimeSeries, window_months: Optional[int]) -> TimeSeries:
    """
    Restrict a series to its most recent ``window_months`` observations.

    Off by default. A short window helps a simple local-trend model at short
    horizons, but measured against the actual darts model set at h=1..12 it hurt:
    mean MASE 2.83 on full history, 3.02 at a 60-month window, 3.07 at 36 months
    (where LightGBM stopped fitting altogether). Kept because the picture may
    invert if the 2026 regime persists - re-run the ablation before switching it on.

    Args:
        series: Full history
        window_months: Number of trailing months to keep, or None for everything

    Returns:
        Trimmed series (returns the input unchanged when the window is None,
        non-positive, or longer than the series)
    """
    if window_months is None or window_months <= 0 or len(series) <= window_months:
        return series
    return series[-window_months:]
