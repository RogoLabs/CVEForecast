"""
Validate forecast output before deployment.

Runs in CI between the forecast step and the Pages deploy. The v0.11 version
checked that three keys existed and that the rankings list was non-empty - which
every one of the defects fixed in v0.12 would have passed. These checks target
the specific failures we have actually shipped:

* year totals that do not equal published + forecast
* a projection leaking into the actuals series (drawn as a real observation)
* every model collapsing onto one number, which is what a constraint layer
  overriding the models looks like from the outside
* prediction intervals that do not bracket their own point forecast
* an accuracy table with no baseline to compare against
* a CNA interval that inverts, excludes its own point forecast, or claims a
  horizon the backtest never reached
* a CNA forecast that has run away from anything its own history could reach

Exits non-zero on failure so the deploy does not proceed.
"""

import json
import sys
from typing import Any, Dict, List

DATA_PATH = 'web/data.json'
VALIDATION_PATH = 'web/validation.json'
CNA_PATH = 'web/cna_data.json'

# A forecast month above this multiple of the CNA's trailing peak is a runaway,
# not a prediction. Mirrors RUNAWAY_CEILING in the CNA adapter; this is the
# backstop that keeps one from reaching the site if the guard there regresses.
CNA_RUNAWAY_CEILING = 8.0

REQUIRED_KEYS = [
    'generated_at',
    'model_rankings',
    'cumulative_timelines',
    'yearly_forecast_totals',
    'actuals_cumulative',
    'forecasts',
    'methodology',
]

# Distinct models should not agree to within this fraction on a year total.
# v0.11 shipped twelve models inside a five-CVE band on ~55,000 CVEs.
MIN_MODEL_SPREAD = 0.005


def _fail(message: str, failures: List[str]) -> None:
    failures.append(message)


def check_structure(data: Dict[str, Any], failures: List[str]) -> None:
    for key in REQUIRED_KEYS:
        if key not in data:
            _fail(f"missing required key '{key}'", failures)
    if not data.get('model_rankings'):
        _fail('model_rankings is empty', failures)


def check_rankings(data: Dict[str, Any], failures: List[str]) -> None:
    rankings = data.get('model_rankings') or []
    if not any(r.get('is_baseline') for r in rankings):
        _fail('no naive baseline in model_rankings - nothing to judge models against', failures)

    scored = [r for r in rankings if r.get('mase') is not None]
    if not scored:
        _fail('no model has a MASE score', failures)
        return

    mase_values = [r['mase'] for r in scored]
    if mase_values != sorted(mase_values):
        _fail('model_rankings is not sorted by MASE', failures)

    for entry in scored:
        if entry['mase'] < 0:
            _fail(f'{entry["model_name"]}: negative MASE {entry["mase"]}', failures)
        if not entry.get('n_origins'):
            _fail(f'{entry["model_name"]}: scored with no origins', failures)


def check_year_totals(data: Dict[str, Any], failures: List[str]) -> None:
    for year, models in (data.get('yearly_forecast_totals') or {}).items():
        totals = []
        for name, proj in models.items():
            expected = proj['actual_ytd'] + proj['forecast_remainder']
            if proj['total'] != expected:
                _fail(
                    f'{year}/{name}: total {proj["total"]} != actual_ytd + forecast_remainder ({expected})',
                    failures,
                )
            if proj.get('lower_80') is not None and not (proj['lower_80'] <= proj['total'] <= proj['upper_80']):
                _fail(f'{year}/{name}: total {proj["total"]} outside its own 80% band', failures)
            if proj['months_forecast'] > 0:
                totals.append(proj['total'])

        # Guard against the v0.11 failure mode: a constraint layer overriding the
        # models makes every "model" report the same number.
        if len(totals) >= 4:
            spread = (max(totals) - min(totals)) / max(max(totals), 1)
            if spread < MIN_MODEL_SPREAD:
                _fail(
                    f'{year}: {len(totals)} models agree to within {spread:.4%} '
                    f'({min(totals):,}-{max(totals):,}) - forecasts are being overridden downstream',
                    failures,
                )


def check_actuals_are_actual(data: Dict[str, Any], failures: List[str]) -> None:
    """The actuals series must not carry a forecast. It is drawn as observed data."""
    generated_at = data.get('generated_at', '')[:10]
    for entry in data.get('actuals_cumulative') or []:
        if entry['date'][:10] > generated_at:
            _fail(
                f'actuals_cumulative contains a future-dated point ({entry["date"]}) - '
                'a projection is being presented as observed data',
                failures,
            )


def check_intervals(data: Dict[str, Any], failures: List[str]) -> None:
    intervals = data.get('monthly_intervals') or {}
    if not intervals:
        return  # allowed: too few backtest residuals to calibrate

    points = {row['date']: row['cve_count'] for row in (data.get('forecasts') or {}).get('Ensemble', [])}
    for month, band in intervals.items():
        if band['lower_80'] > band['upper_80'] or band['lower_95'] > band['upper_95']:
            _fail(f'{month}: interval bounds inverted', failures)
        if band['lower_95'] > band['lower_80'] or band['upper_95'] < band['upper_80']:
            _fail(f'{month}: 95% interval is narrower than 80%', failures)
        point = points.get(month)
        if point is not None and not (band['lower_80'] <= point <= band['upper_80']):
            _fail(f'{month}: point forecast {point} outside its own 80% band', failures)


def check_cumulative_band(data: Dict[str, Any], failures: List[str]) -> None:
    """The shaded chart band must actually contain the line it is drawn around."""
    band = data.get('cumulative_band') or {}
    if not band:
        return  # allowed: no calibrated intervals

    timeline = {
        e['date']: e['cumulative_total'] for e in data.get('cumulative_timelines', {}).get('Ensemble_cumulative', [])
    }
    lower = {e['date']: e['cumulative_total'] for e in band.get('lower', [])}
    upper = {e['date']: e['cumulative_total'] for e in band.get('upper', [])}

    if set(lower) != set(timeline) or set(upper) != set(timeline):
        _fail('cumulative_band does not align with the ensemble timeline', failures)
        return

    for date, point in timeline.items():
        if not (lower[date] <= point <= upper[date]):
            _fail(
                f'{date}: ensemble cumulative {point:,} outside its band [{lower[date]:,}, {upper[date]:,}]', failures
            )

    # The year band and the chart band are built by different code paths from the
    # same per-month bounds, so they must agree at year end. A mismatch means one
    # of them is accumulating the wrong quantity - which is exactly how the year
    # band once ended up above the total it was meant to bracket.
    for year, models in (data.get('yearly_forecast_totals') or {}).items():
        proj = models.get('Ensemble') or {}
        year_end = f'{year}-12-31T23:59:59Z'
        if proj.get('lower_80') is None or year_end not in lower:
            continue
        for label, from_band, from_year in (
            ('lower', lower[year_end], proj['lower_80']),
            ('upper', upper[year_end], proj['upper_80']),
        ):
            if abs(from_band - from_year) > 1:
                _fail(
                    f'{year}: {label} bound disagrees between chart band ({from_band:,}) '
                    f'and year projection ({from_year:,})',
                    failures,
                )


def check_methodology(data: Dict[str, Any], failures: List[str]) -> None:
    meta = data.get('methodology') or {}
    if meta.get('ranking_metric') != 'MASE':
        _fail(f'ranking_metric is {meta.get("ranking_metric")!r}, expected MASE', failures)

    coverage = meta.get('interval_coverage') or {}
    for level, stats in coverage.items():
        if not stats.get('calibrated'):
            _fail(
                f'{level}% interval is miscalibrated: {stats.get("empirical"):.1%} empirical '
                f'vs {stats.get("nominal"):.0%} nominal',
                failures,
            )


def validate(path: str = DATA_PATH) -> bool:
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'FAIL: {path} not found')
        return False
    except json.JSONDecodeError as e:
        print(f'FAIL: {path} is not valid JSON: {e}')
        return False

    failures: List[str] = []
    check_structure(data, failures)
    if failures:  # later checks assume the structure is there
        for message in failures:
            print(f'FAIL: {message}')
        return False

    check_rankings(data, failures)
    check_year_totals(data, failures)
    check_actuals_are_actual(data, failures)
    check_intervals(data, failures)
    check_cumulative_band(data, failures)
    check_methodology(data, failures)

    if failures:
        for message in failures:
            print(f'FAIL: {message}')
        return False

    rankings = data['model_rankings']
    beat = sum(1 for r in rankings if r.get('beats_naive'))
    print(
        f'OK: {path} valid - {len(rankings)} models ranked by MASE, '
        f'{beat} beat the naive baseline, years {sorted(data["yearly_forecast_totals"])}'
    )
    return True


def check_cna_intervals(data: Dict[str, Any], failures: List[str]) -> None:
    """
    Per-CNA bands, where a CNA has one.

    Absent intervals are fine and expected: a band arrives only as that CNA's
    backtest is re-scored, which happens a dozen CNAs at a time.
    """
    with_intervals = 0
    for cna_id, rec in data.items():
        intervals = rec.get('intervals')
        if not intervals:
            continue
        with_intervals += 1
        name = rec.get('name') or cna_id

        model = (rec.get('model_selection') or {}).get('selected_model')
        points = {m[:7]: v for m, v in (rec.get('forecasts') or {}).get(model, {}).items()}

        for month, band in (intervals.get('monthly') or {}).items():
            if band['lower_80'] > band['upper_80'] or band['lower_95'] > band['upper_95']:
                _fail(f'{name} {month}: interval bounds inverted', failures)
            if band['lower_95'] > band['lower_80'] or band['upper_95'] < band['upper_80']:
                _fail(f'{name} {month}: 95% interval is narrower than 80%', failures)
            point = points.get(month)
            if point is not None and not (band['lower_80'] <= point <= band['upper_80']):
                _fail(f'{name} {month}: point forecast {point} outside its own 80% band', failures)

        # A month past the fitted horizon must carry no band rather than one
        # extrapolated from the longest horizon that was measured.
        max_horizon = intervals.get('max_horizon')
        monthly = intervals.get('monthly') or {}
        if max_horizon is not None and len(monthly) > max_horizon:
            _fail(
                f'{name}: {len(monthly)} months carry a band but only {max_horizon} horizons were fitted',
                failures,
            )

        forecast_months = {m[:7] for m in (rec.get('forecasts') or {}).get(model, {})}
        for year, band in (intervals.get('annual') or {}).items():
            if band['lower_80'] > band['upper_80']:
                _fail(f'{name} {year}: annual interval bounds inverted', failures)
            # The month in progress belongs to the forecast, which predicts all
            # of it, not to the history, which holds only the part published so
            # far. Adding both counts it twice.
            published = sum(
                v
                for m, v in (rec.get('historical') or {}).items()
                if str(m)[:4] == year and str(m)[:7] not in forecast_months
            )
            forecast = sum(v for m, v in (rec.get('forecasts') or {}).get(model, {}).items() if m[:4] == year)
            total = published + forecast
            if not (band['lower_80'] <= total <= band['upper_80']):
                _fail(
                    f'{name} {year}: projected total {total:,.0f} outside its own 80% band '
                    f'({band["lower_80"]:,} to {band["upper_80"]:,})',
                    failures,
                )

    print(f'  {with_intervals}/{len(data)} CNAs publish a prediction interval')


def check_cna_runaways(data: Dict[str, Any], failures: List[str]) -> None:
    """No CNA may publish a month far beyond anything its own history reached."""
    for cna_id, rec in data.items():
        history = [v for v in (rec.get('historical') or {}).values() if isinstance(v, (int, float))]
        if not history:
            continue
        peak = max(history[-24:])
        if peak <= 0:
            continue
        model = (rec.get('model_selection') or {}).get('selected_model')
        forecast = (rec.get('forecasts') or {}).get(model) or {}
        worst = max((v for v in forecast.values() if isinstance(v, (int, float))), default=0)
        if worst > peak * CNA_RUNAWAY_CEILING:
            _fail(
                f'{rec.get("name") or cna_id}: forecast peaks at {worst:,.0f} against a 24-month high of '
                f'{peak:,.0f} ({worst / peak:,.0f}x) - a runaway reached the site',
                failures,
            )


def validate_cna(path: str = CNA_PATH) -> bool:
    """Validate web/cna_data.json. Absent file is not a failure; it is optional output."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'SKIP: {path} not present')
        return True
    except json.JSONDecodeError as e:
        print(f'FAIL: {path} is not valid JSON: {e}')
        return False

    failures: List[str] = []
    check_cna_intervals(data, failures)
    check_cna_runaways(data, failures)

    if failures:
        for message in failures:
            print(f'FAIL: {message}')
        return False

    print(f'OK: {path} valid - {len(data)} CNAs')
    return True


if __name__ == '__main__':
    ok = validate(sys.argv[1] if len(sys.argv) > 1 else DATA_PATH)
    ok = validate_cna() and ok
    sys.exit(0 if ok else 1)
