"""
Prediction Interval Validation

Validates that prediction intervals have correct empirical coverage,
addressing the critical gap where intervals are generated but never
verified to actually cover the claimed percentage of observations.

Key Concepts:
- A 95% prediction interval should cover ~95% of actual values
- Coverage can be checked empirically using historical data
- Intervals that are too narrow or wide need recalibration

Scientific Justification:
- Prediction intervals are only meaningful if properly calibrated
- Miscalibrated intervals mislead stakeholders about uncertainty
- Empirical validation is standard practice in forecast evaluation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class IntervalValidator:
    """
    Validates prediction interval calibration.

    Addresses the gap where prediction intervals are generated using
    conformal prediction or other methods but never validated to ensure
    they have correct empirical coverage.
    """

    def __init__(self, tolerance: float = 0.05):
        """
        Initialize validator.

        Args:
            tolerance: Acceptable deviation from target coverage
                      (e.g., 0.05 means 95% interval can be 90-100%)
        """
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)

    def validate_coverage(
        self, actuals: np.ndarray, intervals_dict: Dict[str, Dict[str, np.ndarray]], tolerance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Check if prediction intervals have correct empirical coverage.

        Args:
            actuals: Actual observed values
            intervals_dict: Dictionary like {
                '80': {'lower': array, 'upper': array},
                '95': {'lower': array, 'upper': array}
            }
            tolerance: Override default tolerance

        Returns:
            dict: Validation results for each confidence level

        Example:
            >>> actuals = np.array([100, 105, 98, 102, 110])
            >>> intervals = {
            ...     '95': {
            ...         'lower': np.array([95, 100, 93, 97, 105]),
            ...         'upper': np.array([105, 110, 103, 107, 115])
            ...     }
            ... }
            >>> results = validator.validate_coverage(actuals, intervals)
            >>> print(results['95']['is_valid'])  # True if coverage ~95%
        """
        if tolerance is None:
            tolerance = self.tolerance

        results = {}

        for conf_level, intervals in intervals_dict.items():
            target_coverage = int(conf_level) / 100
            lower = np.array(intervals['lower'])
            upper = np.array(intervals['upper'])
            actuals_arr = np.array(actuals)

            # Validate array lengths
            if len(actuals_arr) != len(lower) or len(actuals_arr) != len(upper):
                raise ValueError(f'Length mismatch: actuals={len(actuals_arr)}, lower={len(lower)}, upper={len(upper)}')

            # Check coverage
            covered = (actuals_arr >= lower) & (actuals_arr <= upper)
            empirical_coverage = np.mean(covered)

            # Calculate coverage error
            coverage_error = empirical_coverage - target_coverage

            # Is coverage acceptable?
            is_valid = abs(coverage_error) <= tolerance

            # Average interval width
            widths = upper - lower
            avg_width = np.mean(widths)
            min_width = np.min(widths)
            max_width = np.max(widths)

            # Interval efficiency (narrower is better, given correct coverage)
            # Normalized by actual values to get percentage width
            avg_actual = np.mean(np.abs(actuals_arr))
            if avg_actual > 0:
                relative_width = (avg_width / avg_actual) * 100
            else:
                relative_width = np.inf

            # Statistical test: Is coverage significantly different from target?
            # Using binomial test
            n_covered = np.sum(covered)
            n_total = len(covered)
            binom_pvalue = stats.binom_test(n_covered, n_total, target_coverage, alternative='two-sided')

            # Interpretation
            if is_valid:
                if binom_pvalue >= 0.05:
                    status = '✅ Valid'
                    interpretation = (
                        f'Coverage is {empirical_coverage:.1%}, within acceptable range '
                        f'[{target_coverage - tolerance:.1%}, {target_coverage + tolerance:.1%}]. '
                        f'Not significantly different from target (p = {binom_pvalue:.3f}).'
                    )
                else:
                    status = '⚠️ Borderline'
                    interpretation = (
                        f'Coverage is {empirical_coverage:.1%}, technically within tolerance '
                        f'but statistically different from target (p = {binom_pvalue:.3f}). '
                        f'Consider recalibration.'
                    )
            else:
                status = '❌ Invalid'
                if empirical_coverage < target_coverage - tolerance:
                    interpretation = (
                        f'Coverage is {empirical_coverage:.1%}, too low! Intervals are too narrow. Need recalibration.'
                    )
                else:
                    interpretation = (
                        f'Coverage is {empirical_coverage:.1%}, too high! Intervals are too wide. Can be tightened.'
                    )

            results[conf_level] = {
                'target_coverage': target_coverage,
                'empirical_coverage': empirical_coverage,
                'coverage_error': coverage_error,
                'is_valid': is_valid,
                'status': status,
                'interpretation': interpretation,
                'n_covered': int(n_covered),
                'n_total': n_total,
                'avg_width': float(avg_width),
                'relative_width_pct': float(relative_width),
                'min_width': float(min_width),
                'max_width': float(max_width),
                'binomial_pvalue': float(binom_pvalue),
            }

            self.logger.info(
                f'{conf_level}% Interval: {status} - '
                f'Coverage = {empirical_coverage:.1%} '
                f'(target = {target_coverage:.1%})'
            )

        return results

    def plot_interval_diagnostic(
        self,
        actuals: np.ndarray,
        forecasts: np.ndarray,
        intervals_dict: Dict[str, Dict[str, np.ndarray]],
        time_index: Optional[List] = None,
        save_path: Optional[str] = None,
        title: str = 'Prediction Interval Diagnostic',
    ):
        """
        Create diagnostic plot for prediction intervals.

        Args:
            actuals: Actual values
            forecasts: Point forecasts
            intervals_dict: Dictionary of intervals
            time_index: Optional time labels
            save_path: Optional path to save plot
            title: Plot title
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Use time index or simple range
        if time_index is None:
            x = np.arange(len(actuals))
            xlabel = 'Observation'
        else:
            x = time_index
            xlabel = 'Time'

        # Plot 1: Forecasts with intervals
        axes[0].plot(x, actuals, 'ko-', label='Actual', linewidth=2, markersize=6)
        axes[0].plot(x, forecasts, 'b--', label='Forecast', linewidth=2)

        # Add intervals (sorted by width, largest first)
        sorted_intervals = sorted(intervals_dict.items(), key=lambda x: int(x[0]), reverse=True)

        colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(sorted_intervals)))

        for idx, (conf_level, intervals) in enumerate(sorted_intervals):
            alpha_val = 0.2 + (idx * 0.1)
            axes[0].fill_between(
                x,
                intervals['lower'],
                intervals['upper'],
                alpha=alpha_val,
                color=colors[idx],
                label=f'{conf_level}% interval',
            )

        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel('Value')
        axes[0].set_title(f'{title}: Forecast with Prediction Intervals')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Coverage by observation
        for conf_level, intervals in sorted(intervals_dict.items()):
            covered = ((actuals >= intervals['lower']) & (actuals <= intervals['upper'])).astype(float)

            int(conf_level) / 100

            axes[1].plot(x, covered, marker='o', label=f'{conf_level}%', markersize=4)

        axes[1].axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Covered', linewidth=2)
        axes[1].axhline(0.0, color='red', linestyle='--', alpha=0.5, label='Not covered', linewidth=2)

        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel('Coverage (1=covered, 0=not)')
        axes[1].set_title('Prediction Interval Coverage Over Time')
        axes[1].set_ylim([-0.1, 1.1])
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f'Diagnostic plot saved to {save_path}')
        else:
            plt.show()

        plt.close()

    def recalibrate_intervals(
        self,
        actuals: np.ndarray,
        forecasts: np.ndarray,
        current_intervals: Dict[str, Dict[str, np.ndarray]],
        target_coverage: float = 0.95,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Recalibrate prediction intervals to achieve target coverage.

        This method adjusts interval width based on empirical coverage errors.

        Args:
            actuals: Actual values
            forecasts: Point forecasts
            current_intervals: Current interval estimates
            target_coverage: Desired coverage level

        Returns:
            Recalibrated intervals
        """
        self.logger.info(f'Recalibrating intervals for {target_coverage:.0%} coverage')

        # Calculate current errors
        errors = np.abs(actuals - forecasts)

        # Find quantile that gives target coverage
        quantile = np.quantile(errors, target_coverage)

        self.logger.info(f'Optimal quantile for {target_coverage:.0%} coverage: {quantile:.2f}')

        # Create recalibrated intervals
        recalibrated = {f'{int(target_coverage * 100)}': {'lower': forecasts - quantile, 'upper': forecasts + quantile}}

        # Verify coverage
        covered = (actuals >= recalibrated[f'{int(target_coverage * 100)}']['lower']) & (
            actuals <= recalibrated[f'{int(target_coverage * 100)}']['upper']
        )
        empirical_coverage = np.mean(covered)

        self.logger.info(f'Recalibrated coverage: {empirical_coverage:.1%}')

        return recalibrated

    def summarize_validation(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate summary report of validation results.

        Args:
            validation_results: Results from validate_coverage()

        Returns:
            Formatted summary string
        """
        lines = ['', '=' * 70]
        lines.append('PREDICTION INTERVAL VALIDATION SUMMARY')
        lines.append('=' * 70)
        lines.append('')

        for conf_level in sorted(validation_results.keys(), key=lambda x: int(x)):
            result = validation_results[conf_level]

            lines.append(f'{conf_level}% Prediction Interval:')
            lines.append(f'  Status:             {result["status"]}')
            lines.append(f'  Target Coverage:    {result["target_coverage"]:.1%}')
            lines.append(f'  Empirical Coverage: {result["empirical_coverage"]:.1%}')
            lines.append(f'  Coverage Error:     {result["coverage_error"]:+.1%}')
            lines.append(f'  Observations:       {result["n_covered"]}/{result["n_total"]} covered')
            lines.append(f'  Average Width:      ±{result["avg_width"]:.2f}')
            lines.append(f'  Relative Width:     {result["relative_width_pct"]:.1f}% of forecast')
            lines.append(f'  Binomial p-value:   {result["binomial_pvalue"]:.4f}')
            lines.append(f'  Interpretation:     {result["interpretation"]}')
            lines.append('')

        lines.append('=' * 70)

        # Overall assessment
        all_valid = all(r['is_valid'] for r in validation_results.values())

        if all_valid:
            lines.append('✅ OVERALL: All prediction intervals are properly calibrated')
        else:
            invalid_levels = [k for k, v in validation_results.items() if not v['is_valid']]
            lines.append(f'⚠️ OVERALL: Some intervals need recalibration: {", ".join(invalid_levels)}%')

        lines.append('=' * 70)
        lines.append('')

        return '\n'.join(lines)


def quick_validate(
    actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray, target_coverage: float = 0.95
) -> Tuple[float, bool]:
    """
    Quick validation check for a single interval.

    Args:
        actuals: Actual values
        lower: Lower bounds
        upper: Upper bounds
        target_coverage: Target coverage (default: 0.95)

    Returns:
        Tuple of (empirical_coverage, is_valid)

    Example:
        >>> coverage, valid = quick_validate(actuals, lower_95, upper_95)
        >>> if not valid:
        ...     print(f"Coverage is {coverage:.1%}, expected ~95%")
    """
    covered = (actuals >= lower) & (actuals <= upper)
    empirical_coverage = np.mean(covered)
    is_valid = abs(empirical_coverage - target_coverage) <= 0.05

    return empirical_coverage, is_valid
