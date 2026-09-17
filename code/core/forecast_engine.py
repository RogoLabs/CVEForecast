"""
The single forecast path, shared by production and evaluation.

Before v0.12 the published forecast and the published accuracy metric were
produced by different code: models were fitted on all history for the forecast,
and separately re-fitted from a single January origin for the ranking table. The
ranking therefore described a model that was never shipped.

Everything now goes through ``ForecastEngine.forecast``. The rolling-origin
backtest calls it with historical cut-offs, production calls it with the full
series, and the numbers on the dashboard describe the forecast on the dashboard.

Pipeline, in order:

    trim to training window -> business-day normalise -> log1p
      -> fit -> predict -> damp trend -> expm1 -> business-day denormalise

Each stage is individually switchable through ``ForecastSettings`` so the
backtest can quantify what each one is worth.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from darts import TimeSeries

from core.covariates import (
    COVARIATE_CAPABLE_MODELS,
    build_cna_covariate,
    build_future_covariates,
    denormalise_by_business_days,
    normalise_by_business_days,
)
from core.transforms import damp_forecast_path, from_log_space, to_log_space, trim_to_window

logger = logging.getLogger(__name__)


@dataclass
class ForecastSettings:
    """
    Modelling choices applied uniformly to every model.

    Defaults were chosen by ablation over 24 rolling origins at h=1..12, not by
    prior. Best-model MASE improved 2.39 (v0.11 behaviour) -> 2.12. Each field
    below records what the alternative cost. See docs/FORECAST_METHODOLOGY_REVIEW.md.
    """

    log_space: bool = True
    business_day_normalise: bool = True
    # Mild damping: phi=0.95 scored marginally better (2.11 vs 2.12 MASE) but pushed
    # bias from -9.4% to -11.7% on an already under-forecasting series. 0.98 buys
    # insurance against explosive 16-month extrapolation at almost no accuracy cost.
    damping_phi: float = 0.98
    # Trimming the training window helps a local-trend model but hurts these darts
    # models at h=12 (mean MASE 2.83 on all history vs 3.02 at 60 months, and
    # LightGBM fails outright at 36). Left off; switch it on per-model if that changes.
    training_window_months: Optional[int] = None
    # Month dummies do not pay for themselves once business days are normalised out
    # of the target - they cost the best model 2.15 -> 2.26 MASE. Capability retained.
    use_future_covariates: bool = False
    # Active-CNA count as an exogenous driver. Correlates +0.796 with monthly CVEs
    # but costs the best model 2.12 -> 2.31 MASE: the correlation is shared trend,
    # which the target's own lags already carry. Off by default; needs
    # use_future_covariates when enabled.
    use_cna_covariate: bool = False
    # Calendar of the month being predicted; [0] means "this period only".
    future_covariate_lags: Any = field(default_factory=lambda: [0])
    freq: str = 'ME'

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ForecastSettings':
        """
        Build settings from the ``forecasting`` block of config.json.

        Args:
            config: Full application config

        Returns:
            Populated settings, falling back to the defaults above
        """
        block = config.get('forecasting', {}) or {}
        defaults = cls()
        return cls(
            log_space=block.get('log_space', defaults.log_space),
            business_day_normalise=block.get('business_day_normalise', defaults.business_day_normalise),
            damping_phi=block.get('damping_phi', defaults.damping_phi),
            training_window_months=block.get('training_window_months', defaults.training_window_months),
            use_future_covariates=block.get('use_future_covariates', defaults.use_future_covariates),
            use_cna_covariate=block.get('use_cna_covariate', defaults.use_cna_covariate),
            future_covariate_lags=block.get('future_covariate_lags', defaults.future_covariate_lags),
            freq=block.get('freq', defaults.freq),
        )


@dataclass
class ForecastAttempt:
    """Outcome of a single forecast, successful or not."""

    model_name: str
    forecast: Optional[TimeSeries] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.forecast is not None


class ForecastEngine:
    """
    Applies the shared transform pipeline around any darts model.

    Args:
        settings: Modelling choices
        create_model: Callable ``(model_name, hyperparameters) -> darts model``.
            Must return a *fresh* instance on every call; refitting a model that
            has already been fitted leaks information across backtest folds.
    """

    def __init__(
        self,
        settings: ForecastSettings,
        create_model: Callable[[str, Dict[str, Any]], Any],
        cna_counts: Optional[Any] = None,
    ):
        self.settings = settings
        self.create_model = create_model
        # DataFrame of monthly CNA counts, supplied by the adapter when the
        # exogenous driver is enabled.
        self.cna_counts = cna_counts

    def _covariates_for(self, series: TimeSeries, horizon: int) -> Optional[TimeSeries]:
        """Build a covariate block spanning the training series plus the horizon."""
        if not self.settings.use_future_covariates:
            return None
        # Pad generously: lag-based models look back beyond the target start, and
        # darts raises rather than extrapolating a short covariate block.
        start = series.start_time() - pd.DateOffset(years=1)
        end = series.end_time() + pd.DateOffset(months=horizon + 24)
        covariates = build_future_covariates(
            start=start,
            end=end,
            freq=self.settings.freq,
            # When the target is per-business-day the effect is already removed,
            # so passing it again would double-count.
            include_business_days=not self.settings.business_day_normalise,
        )

        if self.settings.use_cna_covariate and self.cna_counts is not None:
            cna = build_cna_covariate(self.cna_counts, start=start, end=end, freq=self.settings.freq)
            if cna is not None:
                covariates = covariates.stack(cna)

        return covariates

    def forecast(
        self,
        series: TimeSeries,
        model_name: str,
        hyperparameters: Dict[str, Any],
        horizon: int,
    ) -> ForecastAttempt:
        """
        Produce a forecast through the full transform pipeline.

        Args:
            series: Training series of raw monthly counts (complete months only)
            model_name: Model identifier, used for covariate capability lookup
            hyperparameters: Model hyperparameters
            horizon: Number of months to forecast

        Returns:
            ForecastAttempt carrying either a counts-space forecast or an error
        """
        try:
            work = trim_to_window(series, self.settings.training_window_months)
            if len(work) < 24:
                return ForecastAttempt(model_name, error=f'Too few training periods ({len(work)})')

            if self.settings.business_day_normalise:
                work = normalise_by_business_days(work)
            if self.settings.log_space:
                work = to_log_space(work)

            last_observed = float(work.values().flatten()[-1])

            use_covariates = self.settings.use_future_covariates and model_name in COVARIATE_CAPABLE_MODELS
            if use_covariates:
                # darts refuses future_covariates at fit() unless the constructor
                # declared lags for them, so the hyperparameters have to carry it.
                hyperparameters = {**hyperparameters}
                hyperparameters.setdefault('lags_future_covariates', self.settings.future_covariate_lags)

            model = self.create_model(model_name, hyperparameters)
            if model is None:
                return ForecastAttempt(model_name, error='Model creation returned None')

            covariates = self._covariates_for(work, horizon) if use_covariates else None

            if covariates is not None:
                model.fit(work, future_covariates=covariates)
                predicted = model.predict(horizon, future_covariates=covariates)
            else:
                model.fit(work)
                predicted = model.predict(horizon)

            predicted = damp_forecast_path(
                predicted,
                last_observed=last_observed,
                phi=self.settings.damping_phi,
                in_log_space=self.settings.log_space,
            )

            if self.settings.log_space:
                predicted = from_log_space(predicted)
            if self.settings.business_day_normalise:
                predicted = denormalise_by_business_days(predicted)

            values = np.maximum(predicted.values().flatten(), 0.0)
            predicted = TimeSeries.from_times_and_values(predicted.time_index, values)

            return ForecastAttempt(
                model_name,
                forecast=predicted,
                metadata={
                    'train_periods': len(work),
                    'train_end': str(series.end_time().date()),
                    'used_covariates': covariates is not None,
                },
            )

        except Exception as e:  # noqa: BLE001 - one bad model must not sink the run
            logger.warning(f'{model_name}: forecast failed - {type(e).__name__}: {e}')
            return ForecastAttempt(model_name, error=f'{type(e).__name__}: {e}')
