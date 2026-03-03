"""
Forecast Horizon Analysis

Evaluates model performance at different forecast horizons (h=1, 3, 6, 12)
to understand how accuracy degrades with longer forecasts.

Scientific Justification:
- Forecast uncertainty increases with horizon
- Models may perform differently at different horizons
- Near-term vs long-term forecast quality assessment
- Critical for understanding forecast reliability over time
"""

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, rmse


class HorizonAnalysis:
    """
    Analyze model performance across different forecast horizons.

    Evaluates how forecast accuracy changes as we predict further into
    the future, which is critical for understanding model reliability.
    """

    def __init__(self):
        """Initialize horizon analysis."""
        self.logger = logging.getLogger(__name__)

    def evaluate_by_horizon(
        self, model, data: TimeSeries, horizons: List[int] = [1, 3, 6, 12], n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate model performance at different forecast horizons.

        For each horizon h, computes error metrics (MAPE, MAE, RMSE) to show
        how accuracy changes with forecast distance.

        Args:
            model: Darts model instance
            data: Historical time series
            horizons: List of horizons to evaluate (in periods)
            n_splits: Number of train/test splits

        Returns:
            dict: Performance metrics by horizon
        """
        self.logger.info(f'Evaluating performance at horizons: {horizons}')

        results = {}

        for h in horizons:
            self.logger.debug(f'Evaluating horizon h={h}')

            # Need enough data for this horizon
            min_data = 24 + h  # Minimum training + horizon
            if len(data) < min_data:
                self.logger.warning(f'Insufficient data for horizon {h}')
                continue

            metrics = self._evaluate_single_horizon(model, data, h, n_splits)
            results[f'h{h}'] = metrics

        # Add summary
        results['summary'] = self._summarize_horizon_results(results)

        return results

    def _evaluate_single_horizon(self, model, data: TimeSeries, horizon: int, n_splits: int) -> Dict[str, float]:
        """
        Evaluate model at a single horizon using rolling origin.

        Args:
            model: Model to evaluate
            data: Data
            horizon: Forecast horizon
            n_splits: Number of evaluation points

        Returns:
            dict: Average metrics at this horizon
        """
        mape_scores = []
        mae_scores = []
        rmse_scores = []

        # Calculate split points
        total_len = len(data)
        test_starts = np.linspace(
            total_len // 2,  # Start from halfway
            total_len - horizon,
            n_splits,
            dtype=int,
        )

        for test_start in test_starts:
            if test_start < 24:  # Need minimum training
                continue

            try:
                # Split data
                train = data[:test_start]

                # Get actual value at horizon
                if test_start + horizon > len(data):
                    continue

                actual = data[test_start + horizon - 1 : test_start + horizon]

                # Train and forecast
                model.fit(train)
                forecast = model.predict(horizon)

                # Get forecast at this horizon
                forecast_h = forecast[-1:]

                # Calculate metrics
                mape_scores.append(mape(actual, forecast_h))
                mae_scores.append(mae(actual, forecast_h))
                rmse_scores.append(rmse(actual, forecast_h))

            except Exception as e:
                self.logger.warning(f'Failed at test_start={test_start}: {e}')
                continue

        if len(mape_scores) == 0:
            return {'mape_mean': np.nan, 'mae_mean': np.nan, 'rmse_mean': np.nan, 'n_evaluations': 0}

        return {
            'mape_mean': float(np.mean(mape_scores)),
            'mape_std': float(np.std(mape_scores)),
            'mae_mean': float(np.mean(mae_scores)),
            'mae_std': float(np.std(mae_scores)),
            'rmse_mean': float(np.mean(rmse_scores)),
            'rmse_std': float(np.std(rmse_scores)),
            'n_evaluations': len(mape_scores),
        }

    def _summarize_horizon_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize horizon analysis results.

        Args:
            results: Results by horizon

        Returns:
            dict: Summary statistics
        """
        # Extract MAPE at each horizon
        horizons = []
        mapes = []

        for key, metrics in results.items():
            if key == 'summary':
                continue

            h = int(key[1:])  # Extract number from 'h1', 'h3', etc.
            mape_mean = metrics.get('mape_mean', np.nan)

            if not np.isnan(mape_mean):
                horizons.append(h)
                mapes.append(mape_mean)

        if len(horizons) == 0:
            return {'error': 'No valid horizon results'}

        # Calculate degradation rate
        if len(horizons) >= 2:
            # Fit linear trend: MAPE = a + b * horizon
            slope, intercept = np.polyfit(horizons, mapes, 1)
            degradation_rate = slope
        else:
            degradation_rate = np.nan

        # Best and worst horizons
        best_horizon = horizons[np.argmin(mapes)]
        worst_horizon = horizons[np.argmax(mapes)]

        return {
            'horizons_evaluated': horizons,
            'best_horizon': int(best_horizon),
            'best_mape': float(min(mapes)),
            'worst_horizon': int(worst_horizon),
            'worst_mape': float(max(mapes)),
            'mape_range': float(max(mapes) - min(mapes)),
            'degradation_rate': float(degradation_rate) if not np.isnan(degradation_rate) else None,
            'degradation_interpretation': self._interpret_degradation(degradation_rate),
        }

    def _interpret_degradation(self, rate: float) -> str:
        """
        Interpret degradation rate.

        Args:
            rate: MAPE increase per horizon unit

        Returns:
            str: Interpretation
        """
        if np.isnan(rate):
            return 'Insufficient data for trend analysis'

        if rate < 0:
            return '⚠️ Unexpected: accuracy improves at longer horizons (check data)'
        elif rate < 0.1:
            return '✅ Excellent: minimal degradation with horizon'
        elif rate < 0.3:
            return '✅ Good: gradual degradation with horizon'
        elif rate < 0.5:
            return '⚠️ Moderate: noticeable degradation with horizon'
        else:
            return '❌ Concerning: rapid degradation with horizon'

    def compare_models_by_horizon(
        self, models_dict: Dict[str, Any], data: TimeSeries, horizons: List[int] = [1, 3, 6, 12]
    ) -> pd.DataFrame:
        """
        Compare multiple models across horizons.

        Args:
            models_dict: Dictionary of {model_name: model_instance}
            data: Historical data
            horizons: Horizons to evaluate

        Returns:
            DataFrame: Comparison results
        """
        comparison = []

        for model_name, model in models_dict.items():
            self.logger.info(f'Evaluating {model_name} across horizons...')

            try:
                results = self.evaluate_by_horizon(model, data, horizons)

                for horizon_key, metrics in results.items():
                    if horizon_key == 'summary':
                        continue

                    h = int(horizon_key[1:])

                    comparison.append(
                        {
                            'model': model_name,
                            'horizon': h,
                            'mape': metrics.get('mape_mean', np.nan),
                            'mae': metrics.get('mae_mean', np.nan),
                            'rmse': metrics.get('rmse_mean', np.nan),
                        }
                    )

            except Exception as e:
                self.logger.error(f'Failed to evaluate {model_name}: {e}')

        return pd.DataFrame(comparison)

    def plot_horizon_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'mape',
        save_path: Optional[str] = None,
        title: str = 'Model Performance by Forecast Horizon',
    ):
        """
        Plot model performance across horizons.

        Args:
            comparison_df: DataFrame from compare_models_by_horizon()
            metric: Metric to plot ('mape', 'mae', or 'rmse')
            save_path: Optional save path
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot each model
        for model in comparison_df['model'].unique():
            model_data = comparison_df[comparison_df['model'] == model]
            model_data = model_data.sort_values('horizon')

            ax.plot(model_data['horizon'], model_data[metric], marker='o', label=model, linewidth=2, markersize=8)

        ax.set_xlabel('Forecast Horizon', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add annotation about degradation
        ax.text(
            0.02,
            0.98,
            'Note: Higher values at longer horizons indicate\nforecast accuracy degrades with time',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f'Horizon plot saved to {save_path}')
        else:
            plt.show()

        plt.close()

    def plot_single_model_horizon(
        self, horizon_results: Dict[str, Any], model_name: str = 'Model', save_path: Optional[str] = None
    ):
        """
        Plot horizon analysis for a single model.

        Args:
            horizon_results: Results from evaluate_by_horizon()
            model_name: Model name
            save_path: Optional save path
        """
        # Extract data
        horizons = []
        mapes = []
        mape_stds = []

        for key, metrics in horizon_results.items():
            if key == 'summary':
                continue

            h = int(key[1:])
            horizons.append(h)
            mapes.append(metrics.get('mape_mean', 0))
            mape_stds.append(metrics.get('mape_std', 0))

        if len(horizons) == 0:
            self.logger.warning('No horizon data to plot')
            return

        # Sort by horizon
        sort_idx = np.argsort(horizons)
        horizons = np.array(horizons)[sort_idx]
        mapes = np.array(mapes)[sort_idx]
        mape_stds = np.array(mape_stds)[sort_idx]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot with error bars
        ax.errorbar(
            horizons,
            mapes,
            yerr=mape_stds,
            marker='o',
            linewidth=2,
            markersize=10,
            capsize=5,
            capthick=2,
            label='MAPE ± std',
        )

        # Add trend line
        if len(horizons) >= 2:
            z = np.polyfit(horizons, mapes, 1)
            p = np.poly1d(z)
            ax.plot(horizons, p(horizons), '--', alpha=0.5, label=f'Trend (slope={z[0]:.3f})')

        ax.set_xlabel('Forecast Horizon (periods)', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Forecast Accuracy by Horizon: {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add summary text
        summary = horizon_results.get('summary', {})
        if summary:
            text = f'Best: h={summary.get("best_horizon")} (MAPE={summary.get("best_mape", 0):.2f}%)\n'
            text += f'Worst: h={summary.get("worst_horizon")} (MAPE={summary.get("worst_mape", 0):.2f}%)\n'
            text += f'{summary.get("degradation_interpretation", "")}'

            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f'Single model horizon plot saved to {save_path}')
        else:
            plt.show()

        plt.close()


def format_horizon_report(horizon_results: Dict[str, Any], model_name: str = 'Model') -> str:
    """
    Format horizon analysis results as readable report.

    Args:
        horizon_results: Results from evaluate_by_horizon()
        model_name: Model name

    Returns:
        Formatted string report
    """
    lines = ['', '=' * 70]
    lines.append(f'HORIZON ANALYSIS: {model_name}')
    lines.append('=' * 70)
    lines.append('')

    # Results by horizon
    lines.append('Performance by Forecast Horizon:')
    lines.append('-' * 70)

    for key in sorted([k for k in horizon_results.keys() if k != 'summary']):
        metrics = horizon_results[key]
        h = int(key[1:])

        mape = metrics.get('mape_mean', 0)
        mape_std = metrics.get('mape_std', 0)
        n = metrics.get('n_evaluations', 0)

        lines.append(f'  h={h:2d}: MAPE = {mape:6.2f}% ± {mape_std:5.2f}%  (n={n} evaluations)')

    lines.append('')

    # Summary
    summary = horizon_results.get('summary', {})
    if summary and 'error' not in summary:
        lines.append('=' * 70)
        lines.append('SUMMARY')
        lines.append('=' * 70)
        lines.append(f'Best Horizon:  h={summary.get("best_horizon")} (MAPE={summary.get("best_mape", 0):.2f}%)')
        lines.append(f'Worst Horizon: h={summary.get("worst_horizon")} (MAPE={summary.get("worst_mape", 0):.2f}%)')
        lines.append(f'MAPE Range:    {summary.get("mape_range", 0):.2f}%')

        rate = summary.get('degradation_rate')
        if rate is not None:
            lines.append(f'Degradation:   {rate:.3f}% MAPE per horizon unit')

        lines.append(f'\nInterpretation: {summary.get("degradation_interpretation", "")}')
        lines.append('=' * 70)

    lines.append('')

    return '\n'.join(lines)
