"""
Rolling-origin backtesting - the basis for every accuracy number we publish.

Replaces the single-origin backtest used through v0.11, which trained through
31 December and scored one forecast path for the current year. With eight to
twelve scored months that ranking was mostly sampling noise: LightGBM sat first
at 6.22% in the v0.10 notes and tenth at 35.75% by September 2026, without any
code changing.

This module scores each model from many origins and reports:

* **MASE** as the primary metric, scaled by the in-sample seasonal-naive MAE of
  each fold's training data. Note that MASE > 1 does *not* by itself mean the
  model lost to a naive forecast: the denominator is in-sample seasonal
  volatility, and during the 2026 surge every model - naive ones included -
  scores above 1 because out-of-sample errors dwarf historical volatility. The
  honest comparison is against the naive models scored in the same run, which is
  what ``mark_naive_baselines`` sets. On 24 origins, LinearRegression (2.12) and
  TBATS (2.36) beat NaiveDrift (2.70); LightGBM (3.62) and Theta (3.66) lose to it.
* **MAPE** for continuity with earlier releases.
* **Bias**, because the pipeline's historic downward bias in levels space is what
  the old growth-floor hack existed to paper over.
* **Per-horizon breakdown**, because h=1 and h=12 are different problems
  (roughly 1.5 vs 3.5 MASE on current origins) and one number hides that.
* **Log-space residuals per horizon**, consumed by ``core.intervals`` to build
  empirically calibrated prediction intervals.

MAPE is deliberately not the ranking metric: it penalises over-forecasting more
than under-forecasting, which biases selection low on a growing series.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from darts import TimeSeries

logger = logging.getLogger(__name__)

# Seasonal period for the MASE denominator: monthly data, annual seasonality.
SEASONAL_PERIOD = 12


@dataclass
class BacktestResult:
    """Accuracy of one model across all scored origins."""

    model_name: str
    n_origins: int = 0
    mase: Optional[float] = None
    mase_std: Optional[float] = None
    mape: Optional[float] = None
    mae: Optional[float] = None
    bias_pct: Optional[float] = None
    mase_by_horizon: Dict[int, float] = field(default_factory=dict)
    log_residuals_by_horizon: Dict[int, List[float]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    # Set by mark_naive_baseline() once every model in the run has been scored.
    beats_naive: Optional[bool] = None

    @property
    def is_valid(self) -> bool:
        """True when enough origins scored for the numbers to mean anything."""
        return self.n_origins >= 3 and self.mase is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for web/validation.json (residuals excluded - they are bulky)."""
        return {
            'model_name': self.model_name,
            'n_origins': self.n_origins,
            'mase': _round(self.mase, 3),
            'mase_std': _round(self.mase_std, 3),
            'mape': _round(self.mape, 2),
            'mae': _round(self.mae, 2),
            'bias_pct': _round(self.bias_pct, 2),
            'mase_by_horizon': {str(h): _round(v, 3) for h, v in sorted(self.mase_by_horizon.items())},
            'beats_naive': self.beats_naive,
            'n_errors': len(self.errors),
        }


def _round(value: Optional[float], digits: int) -> Optional[float]:
    """Round, tolerating None and non-finite values."""
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def seasonal_naive_mae(values: np.ndarray, period: int = SEASONAL_PERIOD) -> float:
    """
    In-sample MAE of the seasonal-naive forecast - the MASE denominator.

    Args:
        values: Training values for a single fold
        period: Seasonal period (12 for monthly data)

    Returns:
        Mean absolute seasonal difference. Falls back to the first-difference MAE
        when the series is shorter than one seasonal cycle.
    """
    if len(values) > period:
        scale = float(np.mean(np.abs(values[period:] - values[:-period])))
    elif len(values) > 1:
        scale = float(np.mean(np.abs(np.diff(values))))
    else:
        scale = 0.0
    # Guard against a degenerate constant series producing a zero denominator.
    return scale if scale > 1e-9 else 1.0


class RollingOriginBacktest:
    """
    Expanding-window backtest over many forecast origins.

    Args:
        horizon: Months to forecast from each origin
        min_train: Minimum training observations before an origin is scored
        step: Months between consecutive origins (1 = every month)
        max_origins: Keep only this many of the most recent origins. Recency
            matters: the series changed regime in 2026 and origins from 2018 say
            little about how a model handles 2026.
    """

    def __init__(
        self,
        horizon: int = 12,
        min_train: int = 48,
        step: int = 1,
        max_origins: Optional[int] = 36,
    ):
        self.horizon = horizon
        self.min_train = min_train
        self.step = step
        self.max_origins = max_origins

    def origins_for(self, series: TimeSeries) -> List[int]:
        """
        Index positions to use as forecast origins.

        Args:
            series: Full history

        Returns:
            Ascending list of split points. Each origin trains on ``series[:o]``
            and scores against ``series[o:o + horizon]``, so origins stop far
            enough from the end to leave at least one month to score.
        """
        latest = len(series) - 1  # need >= 1 observation to score
        candidates = list(range(self.min_train, latest + 1, self.step))
        if self.max_origins is not None:
            candidates = candidates[-self.max_origins :]
        return candidates

    def evaluate(
        self,
        series: TimeSeries,
        forecast_fn: Callable[[TimeSeries, int], Optional[TimeSeries]],
        model_name: str,
    ) -> BacktestResult:
        """
        Score one model across every origin.

        Args:
            series: Full history of complete months, raw counts
            forecast_fn: ``(train_series, horizon) -> forecast TimeSeries or None``.
                Production passes ``ForecastEngine.forecast`` so the backtest
                exercises the same transforms that ship.
            model_name: Label for reporting

        Returns:
            BacktestResult; check ``is_valid`` before using the metrics
        """
        result = BacktestResult(model_name=model_name)
        origins = self.origins_for(series)
        if not origins:
            result.errors.append(f'No valid origins (series length {len(series)}, min_train {self.min_train})')
            return result

        values = series.values().flatten()
        scaled_errors: List[float] = []
        fold_mase: List[float] = []
        pct_errors: List[float] = []
        abs_errors: List[float] = []
        signed: List[float] = []
        per_horizon: Dict[int, List[float]] = {}

        for origin in origins:
            train = series[:origin]
            actual = values[origin : origin + self.horizon]
            if len(actual) == 0:
                continue

            forecast = forecast_fn(train, self.horizon)
            if forecast is None:
                result.errors.append(f'origin {origin}: no forecast')
                continue

            predicted = forecast.values().flatten()[: len(actual)]
            if len(predicted) != len(actual) or not np.all(np.isfinite(predicted)):
                result.errors.append(f'origin {origin}: malformed forecast')
                continue

            scale = seasonal_naive_mae(values[:origin])
            fold_scaled = np.abs(predicted - actual) / scale
            scaled_errors.extend(fold_scaled.tolist())
            fold_mase.append(float(np.mean(fold_scaled)))

            nonzero = actual != 0
            if nonzero.any():
                pct_errors.extend((np.abs(predicted[nonzero] - actual[nonzero]) / actual[nonzero] * 100).tolist())
            abs_errors.extend(np.abs(predicted - actual).tolist())
            signed.extend((predicted - actual).tolist())

            for i, err in enumerate(fold_scaled, start=1):
                per_horizon.setdefault(i, []).append(float(err))

            # Log-ratio residuals drive the empirical prediction intervals.
            safe = (predicted > 0) & (actual > 0)
            for i in np.where(safe)[0]:
                result.log_residuals_by_horizon.setdefault(int(i) + 1, []).append(
                    float(np.log(actual[i] / predicted[i]))
                )

        result.n_origins = len(fold_mase)
        if not fold_mase:
            result.errors.append('All origins failed')
            return result

        result.mase = float(np.mean(scaled_errors))
        result.mase_std = float(np.std(fold_mase))
        result.mape = float(np.mean(pct_errors)) if pct_errors else None
        result.mae = float(np.mean(abs_errors))
        mean_actual = float(np.mean(np.abs(values[origins[0] :]))) or 1.0
        result.bias_pct = float(np.mean(signed) / mean_actual * 100)
        result.mase_by_horizon = {h: float(np.mean(v)) for h, v in per_horizon.items()}

        logger.info(
            f'{model_name}: MASE {result.mase:.3f} (+/-{result.mase_std:.3f}) '
            f'MAPE {result.mape:.1f}% bias {result.bias_pct:+.1f}% over {result.n_origins} origins'
        )
        return result


# Models whose whole purpose is to be the benchmark everything else must clear.
NAIVE_MODELS = ('NaiveDrift', 'NaiveSeasonal', 'NaiveMean')


def mark_naive_baselines(results: Dict[str, BacktestResult]) -> Optional[float]:
    """
    Flag which models actually beat the naive benchmark.

    Must run after every model in a batch has been scored, since the threshold is
    the best naive MASE from that same run rather than a fixed constant.

    Args:
        results: Backtest results keyed by model name, mutated in place

    Returns:
        The naive MASE threshold used, or None when no naive model scored
    """
    naive_scores = [r.mase for name, r in results.items() if name in NAIVE_MODELS and r.is_valid]
    if not naive_scores:
        logger.warning('No naive baseline scored - "beats naive" cannot be determined')
        return None

    threshold = min(naive_scores)
    for name, result in results.items():
        if result.is_valid and name not in NAIVE_MODELS:
            result.beats_naive = bool(result.mase < threshold)

    beaten = sum(1 for n, r in results.items() if n not in NAIVE_MODELS and r.beats_naive)
    total = sum(1 for n, r in results.items() if n not in NAIVE_MODELS and r.is_valid)
    logger.info(f'Naive baseline MASE {threshold:.3f}: {beaten}/{total} models beat it')
    return threshold


def rank_models(results: Dict[str, BacktestResult]) -> List[BacktestResult]:
    """
    Order models best-first by MASE.

    Models that failed to score are appended at the end rather than dropped, so
    a silently broken model stays visible instead of disappearing from the table.

    Args:
        results: Backtest results keyed by model name

    Returns:
        Ranked list
    """
    valid = [r for r in results.values() if r.is_valid]
    invalid = [r for r in results.values() if not r.is_valid]
    return sorted(valid, key=lambda r: r.mase) + invalid
