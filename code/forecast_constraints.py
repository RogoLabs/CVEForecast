"""
Forecast constraint utilities to ensure realistic predictions.
Implements growth floors, trend adjustments, and sanity checks.
"""

import datetime as _dt
import logging
from typing import Dict, Optional

import numpy as np


class ForecastConstraints:
    """Apply realistic constraints to forecast outputs based on historical trends."""

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        """
        Initialize constraints with configuration parameters.

        Args:
            config: Dictionary with constraint parameters
            logger: Optional logger for output
        """
        self.min_growth = config.get('min_annual_growth_rate', 0.05)  # 5% minimum
        self.max_growth = config.get('max_annual_growth_rate', 0.40)  # 40% maximum
        self.historical_avg = config.get('historical_avg_growth', 0.18)  # 18% average
        self.adjustment_confidence = config.get('trend_adjustment_confidence', 0.7)
        self.adjustment_threshold = config.get('trend_adjustment_threshold', 0.75)  # When to trigger
        self.ytd_min_factor = config.get('ytd_minimum_factor', 0.85)  # YTD-based floor
        self.enable_floor = config.get('enable_growth_floor', True)
        self.enable_adjustment = config.get('enable_trend_adjustment', True)
        self.enable_ytd_floor = config.get('enable_ytd_floor', True)
        self.logger = logger or logging.getLogger(__name__)

    def apply_growth_floor(self, forecast_value: int, previous_year_total: int) -> int:
        """
        Ensure forecast meets minimum realistic growth.

        Args:
            forecast_value: Raw model prediction
            previous_year_total: Actual total from previous year

        Returns:
            Adjusted forecast meeting minimum growth constraint
        """
        if not self.enable_floor:
            return forecast_value

        min_forecast = int(previous_year_total * (1 + self.min_growth))
        max_forecast = int(previous_year_total * (1 + self.max_growth))

        # Clamp to realistic range
        adjusted = max(min_forecast, min(max_forecast, forecast_value))

        if adjusted != forecast_value:
            growth_rate = (adjusted - previous_year_total) / previous_year_total * 100
            self.logger.info(f'Growth floor applied: {forecast_value:,} → {adjusted:,} (growth: {growth_rate:.1f}%)')

        return adjusted

    def trend_adjusted_forecast(
        self, base_forecast: int, previous_year_total: int, ytd_growth: Optional[float] = None
    ) -> int:
        """
        Adjust conservative forecasts towards historical or YTD trend.

        Args:
            base_forecast: Model's raw prediction
            previous_year_total: Last year's total
            ytd_growth: Current year-to-date growth rate (optional, uses historical avg if None)

        Returns:
            Trend-adjusted forecast
        """
        if not self.enable_adjustment:
            return base_forecast

        current_growth = (base_forecast - previous_year_total) / previous_year_total

        # Use YTD growth if available and reasonable, else historical average
        if ytd_growth is not None and 0 < ytd_growth < self.max_growth:
            target_growth = ytd_growth
            growth_source = f'YTD ({ytd_growth * 100:.1f}%)'
        else:
            target_growth = self.historical_avg
            growth_source = f'historical ({self.historical_avg * 100:.1f}%)'

        # Check if forecast is below threshold (default 75% of target)
        threshold = target_growth * self.adjustment_threshold

        if current_growth < threshold:
            # Blend current with target using confidence parameter
            adjusted_growth = (
                current_growth * (1 - self.adjustment_confidence) + target_growth * self.adjustment_confidence
            )

            adjusted_forecast = int(previous_year_total * (1 + adjusted_growth))

            self.logger.info(
                f'Trend adjustment applied (using {growth_source}): '
                f'{current_growth * 100:.1f}% → {adjusted_growth * 100:.1f}% '
                f'(threshold: {threshold * 100:.1f}%, forecast: {base_forecast:,} → {adjusted_forecast:,})'
            )

            return adjusted_forecast
        else:
            self.logger.info(
                f'No trend adjustment needed: {current_growth * 100:.1f}% >= {threshold * 100:.1f}% threshold'
            )

        return base_forecast

    def apply_constraints(
        self,
        yearly_totals: Dict[int, Dict[str, int]],
        ytd_growth: Optional[float] = None,
        previous_year_actuals: Optional[Dict[int, int]] = None,
    ) -> Dict[int, Dict[str, int]]:
        """
        Apply all constraints to yearly forecast totals.

        Args:
            yearly_totals: Dictionary of {year: {model: total}}
            ytd_growth: Year-to-date growth rate (optional)
            previous_year_actuals: Actual yearly CVE totals from historical data

        Returns:
            Constrained yearly totals
        """
        if not yearly_totals:
            return yearly_totals

        current_year = _dt.datetime.now(_dt.timezone.utc).year
        constrained = {}
        years = sorted(yearly_totals.keys())

        for year in years:
            constrained[year] = {}
            previous_year = year - 1

            # Get previous year total from actuals or forecast data
            if previous_year in yearly_totals:
                # Use average of previous year models as baseline
                prev_totals = list(yearly_totals[previous_year].values())
                prev_baseline = int(np.mean(prev_totals))
            elif previous_year_actuals and previous_year in previous_year_actuals:
                prev_baseline = previous_year_actuals[previous_year]
            else:
                # No baseline available, skip constraints
                constrained[year] = yearly_totals[year].copy()
                continue

            self.logger.info(f'\nApplying constraints for {year} (baseline: {prev_baseline:,})')

            # Apply YTD-based floor for current year if enabled
            ytd_floor = None
            if year == current_year and ytd_growth is not None and self.enable_ytd_floor:
                ytd_floor = max(self.min_growth, ytd_growth * self.ytd_min_factor)
                self.logger.info(
                    f'YTD-based floor for {current_year}: {ytd_floor * 100:.1f}% ({self.ytd_min_factor * 100:.0f}% of YTD {ytd_growth * 100:.1f}%)'
                )

            for model_name, forecast_value in yearly_totals[year].items():
                # Apply growth floor (with YTD override for current year)
                if ytd_floor is not None:
                    # Use YTD-based floor for current year
                    ytd_based_forecast = int(prev_baseline * (1 + ytd_floor))
                    adjusted = max(ytd_based_forecast, forecast_value)
                    if adjusted != forecast_value:
                        self.logger.info(
                            f'YTD floor applied to {model_name}: {forecast_value:,} → {adjusted:,} '
                            f'(growth: {ytd_floor * 100:.1f}%)'
                        )
                else:
                    adjusted = self.apply_growth_floor(forecast_value, prev_baseline)

                # Apply trend adjustment (always for current year with YTD data)
                if year == current_year and ytd_growth is not None:
                    adjusted = self.trend_adjusted_forecast(adjusted, prev_baseline, ytd_growth)
                else:
                    adjusted = self.trend_adjusted_forecast(adjusted, prev_baseline)

                constrained[year][model_name] = adjusted

        return constrained
