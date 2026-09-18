"""
Cached per-CNA model selection.

Choosing a model for each CNA by rolling-origin backtest is the right way to do
it, and far too slow to do daily: 140 CNAs x 6 models x 24 origins is about
20,000 model fits, measured at 2h17m. The previous single-holdout approach was
fast because it was cheap in the wrong way - one 6-month split, scored on 6
points, picking winners that were mostly noise.

v0.14 made it 2.7x slower again, taking the backtest from six months to sixteen
so it covers what the site actually publishes, and from eight origins to
twenty-four so the long horizons have residuals to fit a band from. That is
11.7 minutes a run against 4.3, which the cap below is what makes affordable:
the cost of a run is set by the refresh cap, not by the population.

The resolution is that model *choice* does not need to be daily. A CNA's series
gains one observation a month; the model that suited it yesterday almost
certainly suits it today. So selection is cached, and each run refreshes only the
few entries that have gone stale.

Refreshes are also capped per run and taken oldest-first, which bounds the daily
cost to a constant instead of letting every entry expire on the same day. With
the defaults below, a handful of CNAs are re-selected per run and the whole
population cycles through roughly every two weeks.

New CNAs, and any whose history has grown substantially since they were last
scored, jump the queue - those are the cases where the cached choice is most
likely to be wrong.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# How long a selection stays valid before it is eligible for refresh.
DEFAULT_REFRESH_DAYS = 30

# Ceiling on re-selections per run. This is the knob that bounds daily runtime.
DEFAULT_MAX_REFRESH_PER_RUN = 12

# Growth in months of history that forces a refresh regardless of age: a CNA that
# has gained this much data is a different forecasting problem than when it was
# last scored.
SERIES_GROWTH_TRIGGER = 6


class ModelSelectionCache:
    """
    Persisted record of which model was chosen for each CNA, and when.

    Args:
        path: Location of the cache file
        refresh_days: Age at which an entry becomes eligible for refresh
        max_refresh_per_run: Hard cap on re-selections in a single run
    """

    def __init__(
        self,
        path: str = 'web/cna_model_selection.json',
        refresh_days: int = DEFAULT_REFRESH_DAYS,
        max_refresh_per_run: int = DEFAULT_MAX_REFRESH_PER_RUN,
    ):
        self.path = Path(path)
        self.refresh_days = refresh_days
        self.max_refresh_per_run = max_refresh_per_run
        self.entries: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                entries = data.get('selections', {})
                logger.info(f'Loaded {len(entries)} cached CNA model selections')
                return entries
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f'Could not read selection cache, starting fresh: {e}')
        return {}

    def get(self, cna_id: str) -> Optional[Dict[str, Any]]:
        """
        Cached selection for a CNA.

        Args:
            cna_id: CNA identifier

        Returns:
            The cached entry, or None if this CNA has never been scored
        """
        return self.entries.get(cna_id)

    def plan_refresh(self, candidates: Dict[str, int], now: Optional[datetime] = None) -> List[str]:
        """
        Decide which CNAs to re-score this run.

        Args:
            candidates: ``{cna_id: months_of_history}`` for every eligible CNA
            now: Current time, for testing

        Returns:
            CNA ids to re-select, at most ``max_refresh_per_run``
        """
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.refresh_days)

        never_scored: List[str] = []
        grown: List[str] = []
        aged: List[tuple] = []

        for cna_id, months in candidates.items():
            entry = self.entries.get(cna_id)
            if entry is None:
                never_scored.append(cna_id)
                continue

            if months - entry.get('n_months', months) >= SERIES_GROWTH_TRIGGER:
                grown.append(cna_id)
                continue

            try:
                scored_at = datetime.fromisoformat(entry['selected_at'])
            except (KeyError, ValueError):
                never_scored.append(cna_id)
                continue

            if scored_at < cutoff:
                aged.append((scored_at, cna_id))

        # Priority order: never scored, then materially more data, then oldest.
        # Aged entries are still usable, so they yield to the other two.
        aged.sort()
        queue = never_scored + grown + [cna_id for _, cna_id in aged]
        selected = queue[: self.max_refresh_per_run]

        logger.info(
            f'Selection refresh: {len(never_scored)} new, {len(grown)} grown, {len(aged)} aged; '
            f'refreshing {len(selected)} this run (cap {self.max_refresh_per_run})'
        )
        return selected

    def put(
        self,
        cna_id: str,
        model: str,
        mase: Optional[float],
        n_months: int,
        scores: Dict[str, Any],
        residuals: Optional[Dict[int, List[float]]] = None,
        window_residuals: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """
        Record a fresh selection.

        Args:
            cna_id: CNA identifier
            model: Chosen model name
            mase: Its backtest MASE
            n_months: Length of history it was scored on
            scores: All candidate scores, for transparency on the site
            residuals: ``{horizon: [log(actual / forecast), ...]}`` from the same
                backtest that chose the model - the raw material for this CNA's
                prediction interval.

                Cached for the same reason the model choice is. Scoring runs for
                at most a dozen CNAs per run, so residuals computed and dropped
                would leave the other ~128 with no band, and which CNAs had one
                would rotate daily. Stored raw rather than as a finished band
                because the band's shape is estimated across the whole
                population, so a CNA's own residuals are only half of what
                building it needs.
            window_residuals: ``{'YYYY': [log(actual total / forecast total)]}``
                for the year totals the site publishes. A separate measurement
                rather than something derivable from the monthly ones: summing
                monthly bounds would assume the model errs in the same direction
                all year, and the months largely cancel instead.

                Keyed by calendar year, and the span each year covers shifts as
                the forecast window rolls forward, so an entry more than a month
                old describes a slightly different span than today's. Over the
                30-day refresh cycle that is at most one month of drift on a
                twelve-month total.
        """
        entry = {
            'model': model,
            'mase': mase,
            'n_months': n_months,
            'selected_at': datetime.now(timezone.utc).isoformat(),
            'all_scores': scores,
        }
        # Absent rather than null when there are none: the publishing side treats
        # a missing key as "this CNA has no interval", and a null would have to
        # be special-cased into meaning the same thing.
        if residuals:
            # 4dp is far finer than the quantiles these feed, and keeps a file
            # that is rewritten on every run from carrying 17 digits of noise.
            entry['log_residuals'] = {
                str(h): [round(float(v), 4) for v in vals] for h, vals in sorted(residuals.items()) if vals
            }
        if window_residuals:
            entry['log_residuals_by_window'] = {
                str(name): [round(float(v), 4) for v in vals] for name, vals in sorted(window_residuals.items()) if vals
            }
        self.entries[cna_id] = entry

    def save(self) -> None:
        """Write the cache to disk, tolerating an unwritable path."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                # 1.1 adds the per-CNA backtest residuals that build the
                # prediction intervals. Entries written by 1.0 simply have no
                # 'log_residuals' key, which already reads as "no band yet" -
                # they gain one when their turn to re-score comes round, so no
                # migration is needed.
                'version': '1.1',
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'refresh_days': self.refresh_days,
                'selections': self.entries,
            }
            with open(self.path, 'w') as f:
                json.dump(payload, f, indent=2)
            logger.info(f'Saved {len(self.entries)} CNA model selections to {self.path}')
        except OSError as e:
            logger.error(f'Could not write selection cache: {e}')
