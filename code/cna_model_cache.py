"""
Cached per-CNA model selection.

Choosing a model for each CNA by rolling-origin backtest is the right way to do
it, and far too slow to do daily: roughly 140 CNAs x 6 models x 8 origins is
about 6,700 model fits, which measured at over an hour. The previous
single-holdout approach was fast because it was cheap in the wrong way - one
6-month split, scored on 6 points, picking winners that were mostly noise.

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

    def put(self, cna_id: str, model: str, mase: Optional[float], n_months: int, scores: Dict[str, Any]) -> None:
        """
        Record a fresh selection.

        Args:
            cna_id: CNA identifier
            model: Chosen model name
            mase: Its backtest MASE
            n_months: Length of history it was scored on
            scores: All candidate scores, for transparency on the site
        """
        self.entries[cna_id] = {
            'model': model,
            'mase': mase,
            'n_months': n_months,
            'selected_at': datetime.now(timezone.utc).isoformat(),
            'all_scores': scores,
        }

    def save(self) -> None:
        """Write the cache to disk, tolerating an unwritable path."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'version': '1.0',
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'refresh_days': self.refresh_days,
                'selections': self.entries,
            }
            with open(self.path, 'w') as f:
                json.dump(payload, f, indent=2)
            logger.info(f'Saved {len(self.entries)} CNA model selections to {self.path}')
        except OSError as e:
            logger.error(f'Could not write selection cache: {e}')
