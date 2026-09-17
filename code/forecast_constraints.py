"""
Sanity guards on forecast output - and year totals built the honest way.

**What changed in v0.12.** Through v0.11 this module enforced a "growth floor":
every model-year total was clamped to at least 5% growth over the prior year and
then blended 70% of the way toward an assumed 18% historical average. Two things
went wrong with that.

First, it compared unlike quantities. By September the forecast covers only the
*remaining* months of the year, but the floor compared that four-month sum against
the *full* previous year and inflated it to clear an annual growth bar. On
2026-09-17 the published Sep-Dec total was ``48,153 x 1.141 = 54,940`` - a number
produced entirely by this formula.

Second, because the floor bound before any model's own signal did, every model
converged on it. All twelve landed inside a five-CVE band (112,811-112,816). The
dashboard showed twelve models; it was showing one formula twelve times.

The floor existed to counter the downward bias of modelling an exponential series
in levels. ``core.transforms`` fixes that at the source. What remains here is a
guard against numerical blowup - deliberately wide enough that it should never
bind on a sane forecast, and loud when it does.

Year totals now come from ``build_year_projections``: published months are added
as *actuals*, and only the unpublished remainder is forecast. By September that
makes 9/12 of the annual figure a known quantity rather than a model output.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Deliberately loose. These catch a model that has diverged, not a model that
# disagrees with our expectations. 2026 ran +137% year over year and must pass.
DEFAULT_MAX_ANNUAL_GROWTH = 4.0  # 400% of prior year
DEFAULT_MIN_ANNUAL_GROWTH = 0.25  # 25% of prior year
DEFAULT_MAX_MONTHLY_SPIKE = 4.0  # vs. trailing 12-month maximum


@dataclass
class YearProjection:
    """A year's total, split into what is known and what is forecast."""

    year: int
    actual_ytd: int = 0
    forecast_remainder: int = 0
    months_actual: int = 0
    months_forecast: int = 0
    lower_80: Optional[int] = None
    upper_80: Optional[int] = None

    @property
    def total(self) -> int:
        return self.actual_ytd + self.forecast_remainder

    @property
    def is_complete(self) -> bool:
        """True once no month of the year is still being forecast."""
        return self.months_forecast == 0

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            'year': self.year,
            'total': self.total,
            'actual_ytd': self.actual_ytd,
            'forecast_remainder': self.forecast_remainder,
            'months_actual': self.months_actual,
            'months_forecast': self.months_forecast,
        }
        if self.lower_80 is not None:
            out['lower_80'] = self.lower_80
            out['upper_80'] = self.upper_80
        return out


class ForecastConstraints:
    """
    Blowup guards for forecast output.

    Args:
        config: The ``forecast_constraints`` block of config.json. Pass the block,
            not the whole config - v0.11 passed the whole document while this class
            read flat keys, so every configured value was silently ignored in
            favour of the defaults.
        logger: Optional logger
    """

    def __init__(self, config: Optional[dict] = None, logger_override: Optional[logging.Logger] = None):
        config = config or {}
        if 'forecast_constraints' in config:
            # Tolerate a caller that hands over the whole config document rather
            # than failing silently the way v0.11 did.
            logger.warning('ForecastConstraints received the full config; using its forecast_constraints block')
            config = config['forecast_constraints']

        self.max_annual_growth = config.get('max_annual_growth', DEFAULT_MAX_ANNUAL_GROWTH)
        self.min_annual_growth = config.get('min_annual_growth', DEFAULT_MIN_ANNUAL_GROWTH)
        self.max_monthly_spike = config.get('max_monthly_spike', DEFAULT_MAX_MONTHLY_SPIKE)
        self.enabled = config.get('enable_sanity_guards', True)
        self.logger = logger_override or logger

    def check_monthly(self, values: List[float], recent_history: List[float]) -> List[str]:
        """
        Flag individual months that look like numerical divergence.

        Args:
            values: Forecast monthly values
            recent_history: Trailing observed monthly values (12 is plenty)

        Returns:
            Human-readable warnings; empty when everything looks sane
        """
        if not self.enabled or not values or not recent_history:
            return []

        ceiling = max(recent_history) * self.max_monthly_spike
        warnings = [
            f'month {i + 1} forecast {v:,.0f} exceeds {self.max_monthly_spike}x the recent monthly max '
            f'({max(recent_history):,.0f})'
            for i, v in enumerate(values)
            if v > ceiling
        ]
        for message in warnings:
            self.logger.warning(f'Sanity guard: {message}')
        return warnings

    def check_annual(self, projection: YearProjection, previous_year_total: Optional[int]) -> List[str]:
        """
        Flag a year total that has diverged from the prior year.

        Compares whole year against whole year. A partial-year remainder is never
        measured against a full prior year - that was the v0.11 defect.

        Args:
            projection: The year being checked
            previous_year_total: Prior year's actual total, if known

        Returns:
            Human-readable warnings; empty when everything looks sane
        """
        if not self.enabled or not previous_year_total or previous_year_total <= 0:
            return []

        ratio = projection.total / previous_year_total
        warnings: List[str] = []
        if ratio > self.max_annual_growth:
            warnings.append(
                f'{projection.year} total {projection.total:,} is {ratio:.1f}x {projection.year - 1} '
                f'({previous_year_total:,}) - above the {self.max_annual_growth}x guard'
            )
        elif ratio < self.min_annual_growth:
            warnings.append(
                f'{projection.year} total {projection.total:,} is {ratio:.2f}x {projection.year - 1} '
                f'({previous_year_total:,}) - below the {self.min_annual_growth}x guard'
            )
        for message in warnings:
            self.logger.warning(f'Sanity guard: {message}')
        return warnings


def build_year_projections(
    monthly_actuals: Dict[str, float],
    monthly_forecasts: Dict[str, float],
    intervals: Optional[Dict[str, Dict[str, float]]] = None,
    partial_month: Optional[str] = None,
) -> Dict[int, YearProjection]:
    """
    Combine published months with forecast months into per-year totals.

    This is the number the dashboard leads with. Splitting it makes the headline
    honest and makes it degrade gracefully: in January almost all of it is model
    output, by September most of it is fact.

    Where a month appears in both inputs the actual wins - a published month is
    never overwritten by a forecast of itself. The one exception is
    ``partial_month``: the month currently in progress, where the actual covers
    only the days so far and the forecast value is the *remainder*. Both are
    counted, so the running month is a nowcast rather than either a stale partial
    count or a forecast that ignores what has already been published.

    Args:
        monthly_actuals: ``{'YYYY-MM': count}`` for published months
        monthly_forecasts: ``{'YYYY-MM': count}`` for forecast months
        intervals: Optional ``{'YYYY-MM': {'lower_80': x, 'upper_80': y}}``; the
            year's band is the actual YTD plus the summed monthly bounds, which
            assumes errors are perfectly correlated across months and so is the
            conservative (wider) of the reasonable choices
        partial_month: ``'YYYY-MM'`` of the in-progress month, whose forecast
            entry is a remainder to add on top of its partial actual

    Returns:
        ``{year: YearProjection}``
    """
    projections: Dict[int, YearProjection] = {}

    for month, value in sorted(monthly_actuals.items()):
        year = int(month[:4])
        proj = projections.setdefault(year, YearProjection(year=year))
        proj.actual_ytd += int(round(value))
        proj.months_actual += 1

    lower_acc: Dict[int, float] = {}
    upper_acc: Dict[int, float] = {}

    for month, value in sorted(monthly_forecasts.items()):
        if month in monthly_actuals and month != partial_month:
            continue
        year = int(month[:4])
        proj = projections.setdefault(year, YearProjection(year=year))
        proj.forecast_remainder += int(round(value))
        # The partial month is already counted in months_actual; counting it as a
        # forecast month too would overstate how much of the year is modelled.
        if month != partial_month:
            proj.months_forecast += 1

        if intervals and month in intervals:
            band = intervals[month]
            if 'lower_80' in band:
                lower_acc[year] = lower_acc.get(year, 0.0) + band['lower_80']
                upper_acc[year] = upper_acc.get(year, 0.0) + band['upper_80']

    for year, proj in projections.items():
        if year in lower_acc:
            proj.lower_80 = int(round(proj.actual_ytd + lower_acc[year]))
            proj.upper_80 = int(round(proj.actual_ytd + upper_acc[year]))

    for proj in projections.values():
        logger.info(
            f'{proj.year}: {proj.total:,} total = {proj.actual_ytd:,} published '
            f'({proj.months_actual} months) + {proj.forecast_remainder:,} forecast '
            f'({proj.months_forecast} months)'
        )
    return projections


def combine_model_forecasts(
    per_model: Dict[str, Dict[str, float]],
    members: Optional[List[str]] = None,
    method: str = 'trimmed_mean',
) -> Dict[str, float]:
    """
    Combine several models' monthly forecasts into one ensemble path.

    v0.11 took the median of all twelve models while describing itself as a
    "weighted ensemble", and never scored the result. Combination is a good
    variance-reduction play but only over a curated pool: in backtest the mean of
    all models (1.014 MASE) beat the average member but lost to the best single
    model (0.978), and the worst member sat at 1.689. Pass ``members`` to restrict
    the pool to models that cleared the naive baseline.

    Args:
        per_model: ``{model_name: {'YYYY-MM': value}}``
        members: Models to include; None uses all
        method: ``trimmed_mean`` (drops the extremes), ``mean`` or ``median``

    Returns:
        ``{'YYYY-MM': combined_value}``
    """
    pool = {k: v for k, v in per_model.items() if members is None or k in members}
    if not pool:
        logger.warning('No ensemble members available')
        return {}

    months = sorted({month for values in pool.values() for month in values})
    combined: Dict[str, float] = {}

    for month in months:
        values = [v[month] for v in pool.values() if month in v]
        if not values:
            continue
        if method == 'median' or len(values) < 4:
            combined[month] = float(np.median(values))
        elif method == 'mean':
            combined[month] = float(np.mean(values))
        else:
            # Trim the single best and worst so one diverged model cannot drag
            # the ensemble, while keeping more information than a bare median.
            combined[month] = float(np.mean(sorted(values)[1:-1]))

    logger.info(f'Ensemble ({method}) over {len(pool)} members: {sorted(pool)}')
    return combined
