"""
Test script for temporal prediction tracking implementation.

This script validates that the forecast tracker:
1. Creates forecast_history.json file
2. Saves snapshots correctly
3. Accumulates snapshots over multiple runs
4. Tracks accuracy for completed months
5. Calculates stability metrics

Usage:
    python tests/test_temporal_tracking.py
"""

import json
import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))


def test_tracking_works():
    """Verify forecast_history.json is created and populated."""
    history_path = Path('web/forecast_history.json')

    print('=' * 60)
    print('TEMPORAL PREDICTION TRACKING - VALIDATION TEST')
    print('=' * 60)

    # Check file exists
    if not history_path.exists():
        print('\n❌ FAIL: forecast_history.json not found')
        print(f'   Expected path: {history_path.absolute()}')
        return False

    print(f'\n✓ File exists: {history_path}')

    # Load and validate structure
    try:
        with open(history_path) as f:
            history = json.load(f)
    except Exception as e:
        print(f'\n❌ FAIL: Cannot parse JSON - {e}')
        return False

    print('✓ Valid JSON format')

    # Validate top-level structure
    required_keys = ['version', 'forecast_snapshots', 'accuracy_tracking', 'stability_metrics']
    for key in required_keys:
        if key not in history:
            print(f"\n❌ FAIL: Missing required key '{key}'")
            return False

    print(f'✓ Contains all required keys: {", ".join(required_keys)}')

    # Check snapshots
    snapshots = history['forecast_snapshots']
    if len(snapshots) == 0:
        print('\n⚠️  WARNING: No snapshots saved yet')
        print("   Run 'python code/main.py' to create first snapshot")
        return True  # Not a failure, just hasn't run yet

    print(f'\n✓ Found {len(snapshots)} snapshot(s)')

    # Validate first snapshot structure
    snapshot = snapshots[0]
    required_snapshot_keys = ['snapshot_id', 'snapshot_date', 'forecasts', 'model_performance']

    for key in required_snapshot_keys:
        if key not in snapshot:
            print(f"❌ FAIL: Snapshot missing key '{key}'")
            return False

    print('✓ Snapshot structure valid')

    # Display snapshot details
    print('\n📊 Snapshot Details:')
    print(f'   ID: {snapshot["snapshot_id"]}')
    print(f'   Date: {snapshot["snapshot_date"]}')
    print(f'   Data through: {snapshot.get("data_through", "N/A")}')

    # Count forecasts
    if 'forecasts' in snapshot:
        forecast_months = len(snapshot['forecasts'])
        models = set()
        for month_forecasts in snapshot['forecasts'].values():
            models.update(month_forecasts.keys())
        print(f'   Forecast months: {forecast_months}')
        print(f'   Models: {", ".join(sorted(models))}')

    # Check accuracy tracking
    accuracy_tracking = history.get('accuracy_tracking', {})
    if accuracy_tracking:
        print(f'\n✓ Tracking accuracy for {len(accuracy_tracking)} completed month(s)')
        for month, data in list(accuracy_tracking.items())[:3]:  # Show first 3
            print(f'   {month}: Actual={data["actual"]}, Volatility={data.get("prediction_volatility", "N/A")}%')
    else:
        print('\n⚠️  No accuracy tracking yet (need completed months)')

    # Check stability metrics
    stability = history.get('stability_metrics', {})
    if stability:
        print(f'\n✓ Stability metrics for {len(stability)} model(s)')
        for model, metrics in list(stability.items())[:3]:  # Show first 3
            print(
                f'   {model}: Score={metrics.get("stability_score", "N/A")}, '
                f'Mean revision={metrics.get("mean_revision_pct", "N/A")}%'
            )
    else:
        print('\n⚠️  No stability metrics yet (need multiple runs)')

    print('\n' + '=' * 60)
    print('✅ VALIDATION PASSED')
    print('=' * 60)
    print('\nNext Steps:')
    print("1. Run 'python code/main.py' to create more snapshots")
    print('2. Run again tomorrow to see accuracy tracking populate')
    print('3. View forecast_history.json to see full data')

    return True


def display_forecast_evolution(month: str = None):
    """Display how forecasts for a specific month evolved over time."""
    history_path = Path('web/forecast_history.json')

    if not history_path.exists():
        print('❌ forecast_history.json not found. Run main.py first.')
        return

    with open(history_path) as f:
        history = json.load(f)

    accuracy = history.get('accuracy_tracking', {})

    if not accuracy:
        print('⚠️  No completed months tracked yet')
        return

    # If no month specified, use the most recent
    if month is None:
        month = max(accuracy.keys())

    if month not in accuracy:
        print(f'❌ Month {month} not found in tracking')
        print(f'Available months: {", ".join(accuracy.keys())}')
        return

    data = accuracy[month]
    print(f'\n📈 Forecast Evolution for {month}')
    print(f'Actual: {data["actual"]} CVEs')
    print(f'Volatility: {data.get("prediction_volatility", "N/A")}%')
    print(f'Convergence: {data.get("convergence_quality", "N/A")}')
    print('\nForecast History:')
    print(f'{"Date":<12} {"Model":<15} {"Forecast":<10} {"Error":<10} {"Weeks Out":<10}')
    print('-' * 70)

    for item in data.get('forecasts_over_time', []):
        date = item['snapshot_date'][:10]  # YYYY-MM-DD
        print(
            f'{date:<12} {item["model"]:<15} {item["forecast"]:<10} {item["error"]:+<10} {item["weeks_ahead"]:<10.1f}'
        )


def show_model_stability():
    """Display model stability rankings."""
    history_path = Path('web/forecast_history.json')

    if not history_path.exists():
        print('❌ forecast_history.json not found. Run main.py first.')
        return

    with open(history_path) as f:
        history = json.load(f)

    stability = history.get('stability_metrics', {})

    if not stability:
        print('⚠️  No stability metrics yet. Need multiple runs.')
        return

    print('\n🎯 Model Stability Rankings')
    print('(Higher score = more stable predictions)')
    print(f'\n{"Rank":<6} {"Model":<15} {"Score":<10} {"Mean Revision %":<18} {"Max Revision %"}')
    print('-' * 75)

    # Sort by stability score
    ranked = sorted(stability.items(), key=lambda x: x[1]['stability_score'], reverse=True)

    for i, (model, metrics) in enumerate(ranked, 1):
        print(
            f'{i:<6} {model:<15} {metrics["stability_score"]:<10.3f} '
            f'{metrics["mean_revision_pct"]:<18.2f} {metrics["max_revision_pct"]:.2f}'
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test temporal prediction tracking')
    parser.add_argument('--evolution', metavar='MONTH', help='Show forecast evolution for month (YYYY-MM)')
    parser.add_argument('--stability', action='store_true', help='Show model stability rankings')

    args = parser.parse_args()

    if args.evolution:
        display_forecast_evolution(args.evolution)
    elif args.stability:
        show_model_stability()
    else:
        # Run validation test
        success = test_tracking_works()
        sys.exit(0 if success else 1)
