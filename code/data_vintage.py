"""
Data vintage logging - measuring how much CVE counts get revised after the fact.

``cvelistV5`` is backfilled. A CVE published in month M keeps arriving in the
repository for days or weeks afterwards, so the count we observe for a month
depends on *when we looked*. The pipeline correctly drops the current, incomplete
month from training, but the most recent complete month is also still filling in,
and lag-based models weight it heavily.

Nobody can measure this retrospectively - it needs a record of what each month
looked like on each day. This module builds that record, one row per daily run.
After a few weeks it answers: how much does a month grow after it closes, and by
how much should the freshest observations be inflated before a model sees them?

Until enough vintages accumulate the revision factors are reported as ``None``
rather than guessed at.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Two years of daily observations, bounded so the file stays reviewable in git.
MAX_VINTAGES = 800

# Revision factors need several independent months before they mean anything.
MIN_MONTHS_FOR_FACTORS = 3


class VintageLog:
    """
    Append-only log of monthly counts as observed on each run.

    Args:
        path: Location of the vintage JSON file
    """

    def __init__(self, path: str = 'web/data_vintages.json'):
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                data.setdefault('vintages', [])
                data.setdefault('revision_factors', {})
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f'Could not read vintage log, starting fresh: {e}')
        return {'version': '1.0', 'vintages': [], 'revision_factors': {}}

    def record(self, monthly_counts: Dict[str, float], observed_at: Optional[datetime] = None) -> None:
        """
        Record what the monthly counts look like right now.

        Only the trailing months are stored - older months no longer move, and
        keeping all of them would bloat the file for no information gain.

        Args:
            monthly_counts: ``{'YYYY-MM': count}`` as currently observed
            observed_at: Observation timestamp (defaults to now, UTC)
        """
        observed_at = observed_at or datetime.now(timezone.utc)
        recent = dict(sorted(monthly_counts.items())[-6:])

        self.data['vintages'].append(
            {
                'observed_at': observed_at.isoformat(),
                'counts': {month: int(round(count)) for month, count in recent.items()},
            }
        )
        if len(self.data['vintages']) > MAX_VINTAGES:
            self.data['vintages'] = self.data['vintages'][-MAX_VINTAGES:]

        self.data['revision_factors'] = self._compute_revision_factors()
        self._save()
        logger.info(f'Recorded data vintage ({len(self.data["vintages"])} total)')

    def _compute_revision_factors(self) -> Dict[str, Any]:
        """
        Estimate how much a month grows in the days after it is first observed.

        For each month, compares the first observation to the latest one, bucketed
        by how many days after month end the first look happened.

        Returns:
            ``{'days_1': {'factor': 1.04, 'n': 5}, ...}`` plus a ``ready`` flag
        """
        first_seen: Dict[str, Dict[str, Any]] = {}
        latest: Dict[str, int] = {}

        for vintage in self.data['vintages']:
            observed = datetime.fromisoformat(vintage['observed_at'])
            for month, count in vintage['counts'].items():
                if month not in first_seen:
                    first_seen[month] = {'count': count, 'observed_at': observed}
                latest[month] = count

        buckets: Dict[str, list] = {}
        for month, first in first_seen.items():
            final = latest.get(month)
            if not final or first['count'] <= 0:
                continue
            # Only months that have since settled tell us anything about revision.
            month_end = datetime.fromisoformat(f'{month}-01T00:00:00+00:00')
            days_after = (first['observed_at'] - month_end).days
            if days_after < 0:
                continue
            bucket = 'days_0_7' if days_after <= 38 else 'days_8_30' if days_after <= 61 else 'days_30_plus'
            buckets.setdefault(bucket, []).append(final / first['count'])

        # With a single vintage, first_seen == latest for every month and every
        # ratio is exactly 1.0 - true but meaningless. Readiness needs repeat looks.
        factors: Dict[str, Any] = {
            'ready': len(first_seen) >= MIN_MONTHS_FOR_FACTORS and len(self.data['vintages']) >= 2,
            'months_observed': len(first_seen),
            'n_vintages': len(self.data['vintages']),
        }
        if not factors['ready']:
            return factors
        for bucket, ratios in buckets.items():
            if len(ratios) >= MIN_MONTHS_FOR_FACTORS:
                factors[bucket] = {'factor': round(float(np.median(ratios)), 4), 'n': len(ratios)}
        return factors

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, 'w') as f:
                json.dump(self.data, f, indent=2)
        except OSError as e:
            logger.error(f'Could not write vintage log: {e}')

    def summary(self) -> Dict[str, Any]:
        """
        Current state of the vintage record.

        Returns:
            Counts and revision factors, for display and for later correction work
        """
        return {
            'n_vintages': len(self.data['vintages']),
            'first_observed': self.data['vintages'][0]['observed_at'] if self.data['vintages'] else None,
            'revision_factors': self.data.get('revision_factors', {}),
        }
