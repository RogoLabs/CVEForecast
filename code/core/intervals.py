"""
Empirical (conformal) prediction intervals and their coverage.

Through v0.11 the site published bare point estimates 16 months out.
``ForecastResult.confidence_intervals`` existed and was never populated, and
``validation/interval_validation.py`` was imported by nothing.

Intervals here are conformal in the practical sense: they come from the observed
distribution of out-of-sample errors in the rolling-origin backtest, not from a
model's own distributional assumptions. That makes them model-agnostic - the same
machinery works for Theta and for CatBoost - and it means the band reflects how
wrong this pipeline has actually been, rather than how wrong a model believes it
might be.

Residuals are handled as log ratios, ``log(actual / forecast)``, so the bands are
multiplicative and asymmetric. That matches a count series that cannot go
negative and whose errors scale with level.

Measured bands on the reconstructed series:

    horizon   80%              95%
    h=1       [0.88x, 1.30x]   [0.86x, 1.42x]
    h=6       [0.91x, 1.25x]   [0.82x, 1.45x]
    h=12      [0.94x, 1.46x]   [0.85x, 1.87x]

The h=12 row is why publishing a bare integer for next year was the least
defensible number on the dashboard.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LEVELS: Tuple[float, ...] = (0.80, 0.95)

# Below this many residuals a horizon's quantiles are too noisy to trust on their
# own; it borrows from the nearest horizon that does have enough.
MIN_RESIDUALS_PER_HORIZON = 8

# A robust scale is one number, not a set of tail quantiles, so it needs far less
# data than a band does. This is the floor below which a median absolute
# deviation stops meaning anything at all.
MIN_RESIDUALS_FOR_SCALE = 6


@dataclass
class IntervalBands:
    """
    Multiplicative interval factors, per horizon and confidence level.

    ``factors[h][level] = (low, high)`` such that the interval for a forecast
    ``f`` at horizon ``h`` is ``(f * low, f * high)``.
    """

    factors: Dict[int, Dict[str, Tuple[float, float]]] = field(default_factory=dict)
    levels: Tuple[float, ...] = DEFAULT_LEVELS
    n_residuals: Dict[int, int] = field(default_factory=dict)
    max_horizon: int = 0

    def for_horizon(self, horizon: int) -> Dict[str, Tuple[float, float]]:
        """
        Factors for a horizon, reusing the longest available beyond the fitted range.

        Args:
            horizon: 1-based forecast step

        Returns:
            Mapping of level label ('80', '95') to (low, high) factors
        """
        if not self.factors:
            return {}
        if horizon in self.factors:
            return self.factors[horizon]
        # The fitted horizons are not guaranteed to be contiguous. A residual is
        # only recorded where both actual and forecast are positive, so a series
        # with sporadic zero months can lose an interior horizon entirely - which
        # is common for a small CNA and never happens on the aggregate series.
        # Indexing at min(max(factors), horizon) assumed no gaps and raised
        # KeyError on the first one.
        at_or_below = [h for h in self.factors if h <= horizon]
        return self.factors[max(at_or_below)] if at_or_below else self.factors[min(self.factors)]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for web/validation.json."""
        return {
            'levels': [f'{int(level * 100)}' for level in self.levels],
            'max_horizon': self.max_horizon,
            'factors': {
                str(h): {label: [round(lo, 4), round(hi, 4)] for label, (lo, hi) in levels.items()}
                for h, levels in sorted(self.factors.items())
            },
            'n_residuals': {str(h): n for h, n in sorted(self.n_residuals.items())},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'IntervalBands':
        """
        Rebuild bands from ``to_dict`` output.

        The CVE pipeline builds and applies its bands in one pass, so it never
        needed this. The CNA pipeline cannot: backtesting every CNA on every run
        costs hours, so scoring is cached and refreshed a few CNAs at a time,
        which means the run that builds a band is almost never the run that
        publishes it.

        Args:
            payload: A previously serialised band

        Returns:
            IntervalBands; empty when the payload has no factors
        """
        bands = cls()
        if not payload:
            return bands

        levels = payload.get('levels') or []
        if levels:
            bands.levels = tuple(int(label) / 100.0 for label in levels)

        for horizon, per_level in (payload.get('factors') or {}).items():
            bands.factors[int(horizon)] = {
                label: (float(bounds[0]), float(bounds[1])) for label, bounds in per_level.items()
            }
        bands.n_residuals = {int(h): int(n) for h, n in (payload.get('n_residuals') or {}).items()}
        # Trust the factors over a stored scalar that could disagree with them.
        bands.max_horizon = max(bands.factors) if bands.factors else 0
        return bands


def build_intervals(
    log_residuals_by_horizon: Dict[int, List[float]],
    levels: Sequence[float] = DEFAULT_LEVELS,
    enforce_monotonic: bool = True,
) -> IntervalBands:
    """
    Turn backtest residuals into per-horizon interval factors.

    Args:
        log_residuals_by_horizon: ``{horizon: [log(actual / forecast), ...]}``,
            as produced by ``RollingOriginBacktest``
        levels: Confidence levels, e.g. (0.80, 0.95)
        enforce_monotonic: Force bands to widen with horizon. Sample noise
            otherwise produces a narrower h=6 band than h=3, which is indefensible
            to a reader even when it is what the data happened to show.

    Returns:
        IntervalBands; empty if no horizon had enough residuals
    """
    bands = IntervalBands(levels=tuple(levels))
    if not log_residuals_by_horizon:
        logger.warning('No residuals supplied; intervals unavailable')
        return bands

    usable = {h: np.asarray(v, dtype=float) for h, v in log_residuals_by_horizon.items() if len(v) > 0}
    well_sampled = {h: v for h, v in usable.items() if len(v) >= MIN_RESIDUALS_PER_HORIZON}
    if not well_sampled:
        logger.warning(
            f'No horizon reached {MIN_RESIDUALS_PER_HORIZON} residuals '
            f'(max {max((len(v) for v in usable.values()), default=0)}); intervals unavailable'
        )
        return bands

    for horizon in sorted(usable):
        residuals = usable[horizon]
        if len(residuals) < MIN_RESIDUALS_PER_HORIZON:
            # Borrow the nearest horizon with enough data rather than publishing
            # a band derived from three points.
            nearest = min(well_sampled, key=lambda h: abs(h - horizon))
            residuals = well_sampled[nearest]

        per_level: Dict[str, Tuple[float, float]] = {}
        for level in levels:
            tail = (1.0 - level) / 2.0
            lo = float(np.exp(np.quantile(residuals, tail)))
            hi = float(np.exp(np.quantile(residuals, 1.0 - tail)))
            per_level[f'{int(level * 100)}'] = (lo, hi)

        bands.factors[horizon] = per_level
        bands.n_residuals[horizon] = len(usable[horizon])

    if enforce_monotonic:
        _widen_monotonically(bands)

    bands.max_horizon = max(bands.factors) if bands.factors else 0
    logger.info(f'Built prediction intervals for horizons 1..{bands.max_horizon} at levels {bands.levels}')
    return bands


def _widen_monotonically(bands: IntervalBands) -> None:
    """Make each band at least as wide as the band at the previous horizon."""
    running: Dict[str, Tuple[float, float]] = {}
    for horizon in sorted(bands.factors):
        for label, (lo, hi) in bands.factors[horizon].items():
            prev_lo, prev_hi = running.get(label, (lo, hi))
            widened = (min(lo, prev_lo), max(hi, prev_hi))
            bands.factors[horizon][label] = widened
            running[label] = widened


def apply_intervals(
    forecast_values: Dict[str, float],
    bands: IntervalBands,
) -> Dict[str, Dict[str, float]]:
    """
    Attach interval bounds to a dated forecast.

    Args:
        forecast_values: ``{date_string: point_forecast}``; iteration order after
            sorting defines the horizon index
        bands: Factors from ``build_intervals``

    Returns:
        ``{date_string: {'lower_80': x, 'upper_80': y, 'lower_95': ..., ...}}``,
        empty when no bands are available
    """
    if not bands.factors:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for step, date_str in enumerate(sorted(forecast_values), start=1):
        point = forecast_values[date_str]
        entry: Dict[str, float] = {}
        for label, (lo, hi) in bands.for_horizon(step).items():
            entry[f'lower_{label}'] = round(point * lo, 2)
            entry[f'upper_{label}'] = round(point * hi, 2)
        out[date_str] = entry
    return out


def validate_coverage(
    log_residuals_by_horizon: Dict[int, List[float]],
    bands: IntervalBands,
) -> Dict[str, Any]:
    """
    Measure how often the bands actually contained the outcome.

    In-sample against the residuals that built the bands, so this reports
    construction quality rather than genuinely held-out coverage - with the
    monotonic widening applied, realised coverage typically runs slightly above
    nominal. It is still the number worth publishing: "our 80% interval has
    covered 83% of months" is the most trust-building line a forecasting site
    can print, and it makes miscalibration visible the moment it appears.

    Args:
        log_residuals_by_horizon: Same residuals passed to ``build_intervals``
        bands: The constructed bands

    Returns:
        ``{'80': {'nominal': 0.8, 'empirical': 0.83, 'n': 420, 'calibrated': True}, ...}``
    """
    if not bands.factors:
        return {}

    summary: Dict[str, Any] = {}
    for level in bands.levels:
        label = f'{int(level * 100)}'
        covered = 0
        total = 0
        for horizon, residuals in log_residuals_by_horizon.items():
            factors = bands.for_horizon(horizon).get(label)
            if not factors:
                continue
            lo, hi = np.log(factors[0]), np.log(factors[1])
            arr = np.asarray(residuals, dtype=float)
            covered += int(np.sum((arr >= lo) & (arr <= hi)))
            total += len(arr)

        if total:
            empirical = covered / total
            summary[label] = {
                'nominal': level,
                'empirical': round(empirical, 4),
                'n': total,
                # Within 5 percentage points of nominal is the usual tolerance.
                'calibrated': bool(abs(empirical - level) <= 0.05),
            }

    for label, stats in summary.items():
        logger.info(
            f'Interval coverage {label}%: {stats["empirical"]:.1%} empirical vs {stats["nominal"]:.0%} nominal '
            f'over {stats["n"]} observations ({"calibrated" if stats["calibrated"] else "MISCALIBRATED"})'
        )
    return summary


def robust_scale(residuals: Sequence[float]) -> float:
    """
    Spread of a residual sample, as a median absolute deviation.

    The MAD rather than a standard deviation because these samples are small and
    a single bad month should not set the width of a whole band.

    Args:
        residuals: Log residuals, pooled across horizons

    Returns:
        Median absolute deviation, or NaN when there is too little to measure
    """
    if len(residuals) < MIN_RESIDUALS_FOR_SCALE:
        return float('nan')
    arr = np.asarray(residuals, dtype=float)
    return float(np.median(np.abs(arr - np.median(arr))))


def build_shared_shape(
    residuals_by_group: Dict[str, Dict[int, List[float]]],
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> Tuple[IntervalBands, Dict[str, float]]:
    """
    Build one band shape for a population, plus each member's own width.

    Splits the problem in two, because its halves need very different amounts of
    data:

    * *How a band widens with horizon* is a dozen quantiles deep into the tails,
      and it is the part members genuinely have in common. Estimating it needs
      more residuals than any one short series has, and pooling supplies them.
    * *How wide the band is overall* is one robust number, and it is emphatically
      not common - measured dispersion varies about fivefold across CNAs. But a
      single scalar is cheap to estimate, and a handful of residuals will do.

    So the shape is pooled after each member's residuals are divided by that
    member's own spread, and the per-member width is handed back separately to
    stretch it with.

    This matters most where the data is thinnest. A rolling-origin backtest can
    only score its newest origin at h=1, the next at h=1..2, and so on, so at
    horizon h it holds at most ``max_origins - h + 1`` residuals however long the
    series is. Per-member bands therefore run out of data exactly at the long
    horizons, while a pooled shape does not.

    Args:
        residuals_by_group: ``{group_id: {horizon: [log residuals]}}``
        levels: Confidence levels

    Returns:
        (shape, scales). ``shape`` is fitted on scale-normalised residuals and is
        not itself publishable; pass it through ``scale_bands`` with a member's
        entry from ``scales`` first. Both are empty when there is nothing to fit.
    """
    scales: Dict[str, float] = {}
    normalised: Dict[int, List[float]] = {}

    for group_id, by_horizon in residuals_by_group.items():
        flat = [v for values in by_horizon.values() for v in values]
        scale = robust_scale(flat)
        if not np.isfinite(scale) or scale <= 0:
            continue
        scales[group_id] = scale
        for horizon, values in by_horizon.items():
            normalised.setdefault(horizon, []).extend(v / scale for v in values)

    if not normalised:
        logger.warning('No group had enough residuals to measure a scale; shared shape unavailable')
        return IntervalBands(levels=tuple(levels)), {}

    shape = build_intervals(normalised, levels=levels)
    logger.info(
        f'Shared interval shape from {len(scales)} groups, horizons 1..{shape.max_horizon}; '
        f'group scale median {np.median(list(scales.values())):.3f}'
    )
    return shape, scales


def scale_bands(shape: IntervalBands, scale: float) -> IntervalBands:
    """
    Stretch a normalised shape to one member's measured spread.

    Args:
        shape: Bands fitted on scale-normalised residuals
        scale: That member's robust scale, from ``build_shared_shape``

    Returns:
        Publishable bands; empty when the shape or the scale is unusable.
        Widening with horizon is preserved, since raising every factor to the
        same positive power is monotone in log space.
    """
    if not shape.factors or not np.isfinite(scale) or scale <= 0:
        return IntervalBands(levels=shape.levels)

    out = IntervalBands(levels=shape.levels)
    for horizon, per_level in shape.factors.items():
        out.factors[horizon] = {
            label: (float(np.exp(np.log(lo) * scale)), float(np.exp(np.log(hi) * scale)))
            for label, (lo, hi) in per_level.items()
        }
        out.n_residuals[horizon] = shape.n_residuals.get(horizon, 0)
    out.max_horizon = shape.max_horizon
    return out


def pooled_residuals(
    per_model: Dict[str, Dict[int, List[float]]],
    model_names: Optional[Sequence[str]] = None,
) -> Dict[int, List[float]]:
    """
    Pool residuals across models to build one band for an ensemble forecast.

    Args:
        per_model: ``{model_name: {horizon: [log residuals]}}``
        model_names: Restrict to these models (e.g. the ensemble's members).
            None pools everything.

    Returns:
        Merged ``{horizon: [log residuals]}``
    """
    pooled: Dict[int, List[float]] = {}
    for name, by_horizon in per_model.items():
        if model_names is not None and name not in model_names:
            continue
        for horizon, residuals in by_horizon.items():
            pooled.setdefault(horizon, []).extend(residuals)
    return pooled
