"""
Forecast tracking module - capture prediction snapshots over time.

This module enables temporal tracking of CVE forecasts to measure:
- How predictions evolve as months approach
- Forecast stability and convergence
- Model accuracy improvements over time
- Systematic biases in predictions

Author: CVE Forecast Team
Date: 2025-10-05
Version: 1.0
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ForecastTracker:
    """
    Track forecast predictions over time for accuracy analysis.
    
    This class manages a history file that accumulates forecast snapshots
    from each run, allowing analysis of prediction evolution, stability,
    and convergence.
    
    Attributes:
        history_path (Path): Path to forecast_history.json
        history (Dict): Loaded history data with snapshots and metrics
    """
    
    def __init__(self, history_path: str = "web/forecast_history.json"):
        """
        Initialize forecast tracker.
        
        Args:
            history_path: Path to history JSON file (default: web/forecast_history.json)
        """
        self.history_path = Path(history_path)
        self.history = self._load_history()
        logger.info(f"Initialized ForecastTracker with {len(self.history.get('forecast_snapshots', []))} existing snapshots")
    
    def _load_history(self) -> Dict:
        """
        Load existing forecast history or create new structure.
        
        Returns:
            Dictionary with forecast_snapshots, accuracy_tracking, and stability_metrics
        """
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded forecast history with {len(data.get('forecast_snapshots', []))} snapshots")
                return data
            except Exception as e:
                logger.error(f"Failed to load forecast history: {e}")
                logger.info("Creating new history structure")
        
        return self._create_empty_history()
    
    def _create_empty_history(self) -> Dict:
        """Create empty history structure."""
        return {
            "version": "1.0",
            "last_updated": datetime.utcnow().isoformat(),
            "forecast_snapshots": [],
            "accuracy_tracking": {},
            "stability_metrics": {}
        }
    
    def add_snapshot(
        self,
        forecasts: Dict[str, Dict[str, float]],
        actuals: Dict[str, float],
        model_performance: Dict[str, Dict],
        snapshot_date: datetime,
        metadata: Optional[Dict] = None
    ):
        """
        Add new forecast snapshot to history.
        
        Args:
            forecasts: Forecasts by month and model
                      Format: {"2025-08": {"Prophet": 4500, "LightGBM": 4450}}
            actuals: Actual CVE counts for completed months
                    Format: {"2025-07": 4325, "2025-06": 4210}
            model_performance: Model metrics
                             Format: {"Prophet": {"mape": 1.12, "mae": 41.58}}
            snapshot_date: Date of this snapshot (when forecast was generated)
            metadata: Optional metadata (dataset_size, forecast_horizon, etc.)
        """
        snapshot_id = snapshot_date.strftime("%Y-%m-%d_%H%M%S")
        
        # Determine latest data month
        data_through = max(actuals.keys()) if actuals else None
        
        snapshot = {
            "snapshot_id": snapshot_id,
            "snapshot_date": snapshot_date.isoformat(),
            "generation_time": datetime.utcnow().isoformat(),
            "data_through": data_through,
            "forecasts": forecasts,
            "model_performance": model_performance,
            "metadata": metadata or {}
        }
        
        # Only include actuals if we have previous snapshots
        # (first run has no new actuals to report)
        if actuals and len(self.history["forecast_snapshots"]) > 0:
            snapshot["actuals"] = actuals
        
        self.history["forecast_snapshots"].append(snapshot)
        self.history["last_updated"] = datetime.utcnow().isoformat()
        
        # Update derived analytics
        self._update_accuracy_tracking()
        self._update_stability_metrics()
        
        # Save to disk
        self._save_history()
        
        logger.info(f"Added forecast snapshot {snapshot_id} (total: {len(self.history['forecast_snapshots'])})")
    
    def _update_accuracy_tracking(self):
        """
        Calculate accuracy metrics for months that now have actuals.
        
        For each completed month, this method:
        1. Finds all forecasts made for that month
        2. Calculates errors and error percentages
        3. Tracks how predictions evolved over time
        4. Measures prediction volatility
        """
        snapshots = self.history["forecast_snapshots"]
        
        # Collect all months that now have actuals
        months_with_actuals = {}
        for snapshot in snapshots:
            if "actuals" in snapshot:
                months_with_actuals.update(snapshot["actuals"])
        
        # For each completed month, track all forecasts made for it
        for month, actual in months_with_actuals.items():
            forecast_evolution = []
            
            for snapshot in snapshots:
                if month not in snapshot["forecasts"]:
                    continue
                
                snapshot_date = datetime.fromisoformat(snapshot["snapshot_date"])
                month_date = datetime.strptime(month, "%Y-%m")
                weeks_ahead = (month_date - snapshot_date).days / 7
                
                for model, forecast_value in snapshot["forecasts"][month].items():
                    # Skip ensemble average for individual model tracking
                    if model == "all_models_avg":
                        continue
                    
                    error = forecast_value - actual
                    error_pct = (error / actual * 100) if actual != 0 else 0
                    
                    forecast_evolution.append({
                        "snapshot_date": snapshot["snapshot_date"],
                        "model": model,
                        "forecast": int(forecast_value),
                        "error": int(error),
                        "error_pct": round(error_pct, 2),
                        "weeks_ahead": round(weeks_ahead, 1)
                    })
            
            if forecast_evolution:
                # Calculate prediction volatility (how much did forecasts change?)
                all_forecasts = [item["forecast"] for item in forecast_evolution]
                volatility = (np.std(all_forecasts) / np.mean(all_forecasts) * 100 
                             if len(all_forecasts) > 0 and np.mean(all_forecasts) != 0 else 0)
                
                # Determine if predictions converged (improved as month approached)
                convergence_quality = self._assess_convergence(forecast_evolution)
                
                self.history["accuracy_tracking"][month] = {
                    "actual": int(actual),
                    "forecasts_over_time": forecast_evolution,
                    "prediction_volatility": round(volatility, 2),
                    "convergence_quality": convergence_quality
                }
    
    def _assess_convergence(self, forecast_evolution: List[Dict]) -> str:
        """
        Assess if forecasts improved (converged) as month approached.
        
        Args:
            forecast_evolution: List of forecasts over time with error_pct
            
        Returns:
            "improved", "degraded", or "stable"
        """
        if len(forecast_evolution) < 2:
            return "insufficient_data"
        
        # Sort by weeks_ahead (descending - furthest first)
        sorted_forecasts = sorted(forecast_evolution, key=lambda x: x["weeks_ahead"], reverse=True)
        
        # Compare early vs late errors
        early_errors = [abs(f["error_pct"]) for f in sorted_forecasts[:len(sorted_forecasts)//2]]
        late_errors = [abs(f["error_pct"]) for f in sorted_forecasts[len(sorted_forecasts)//2:]]
        
        if not early_errors or not late_errors:
            return "insufficient_data"
        
        early_avg = np.mean(early_errors)
        late_avg = np.mean(late_errors)
        
        # If late errors are 10%+ better, it improved
        if late_avg < early_avg * 0.9:
            return "improved"
        elif late_avg > early_avg * 1.1:
            return "degraded"
        else:
            return "stable"
    
    def _update_stability_metrics(self):
        """
        Calculate forecast stability metrics per model.
        
        Stability measures how much forecasts change between runs.
        A stable model makes consistent predictions; an unstable model
        revises predictions significantly between runs.
        """
        snapshots = self.history["forecast_snapshots"]
        
        if len(snapshots) < 2:
            return
        
        model_revisions = {}
        
        # Compare consecutive snapshots
        for i in range(1, len(snapshots)):
            prev_snapshot = snapshots[i-1]
            curr_snapshot = snapshots[i]
            
            # Find months forecast in both snapshots
            prev_months = set(prev_snapshot["forecasts"].keys())
            curr_months = set(curr_snapshot["forecasts"].keys())
            common_months = prev_months & curr_months
            
            for month in common_months:
                prev_forecasts = prev_snapshot["forecasts"][month]
                curr_forecasts = curr_snapshot["forecasts"][month]
                
                # Find models in both snapshots
                common_models = set(prev_forecasts.keys()) & set(curr_forecasts.keys())
                
                for model in common_models:
                    if model == "all_models_avg":
                        continue
                    
                    prev_val = prev_forecasts[model]
                    curr_val = curr_forecasts[model]
                    
                    if prev_val != 0:
                        revision_pct = abs((curr_val - prev_val) / prev_val * 100)
                        
                        if model not in model_revisions:
                            model_revisions[model] = []
                        model_revisions[model].append(revision_pct)
        
        # Calculate stability scores for each model
        for model, revisions in model_revisions.items():
            if revisions:
                mean_rev = np.mean(revisions)
                max_rev = np.max(revisions)
                
                # Stability score: 1.0 = perfect (no changes), 0.0 = highly unstable
                # Formula: 1 / (1 + mean_revision/10)
                stability = 1 / (1 + mean_rev / 10)
                
                self.history["stability_metrics"][model] = {
                    "mean_revision_pct": round(mean_rev, 2),
                    "max_revision_pct": round(max_rev, 2),
                    "stability_score": round(stability, 3),
                    "num_revisions": len(revisions)
                }
    
    def _save_history(self):
        """Save forecast history to disk."""
        try:
            # Ensure directory exists
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with pretty printing
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            logger.debug(f"Saved forecast history to {self.history_path}")
        except Exception as e:
            logger.error(f"Failed to save forecast history: {e}")
            raise
    
    def get_forecast_evolution(self, target_month: str) -> List[Dict]:
        """
        Get all forecasts made for a specific month over time.
        
        Args:
            target_month: Month in YYYY-MM format (e.g., "2025-08")
            
        Returns:
            List of snapshots containing forecasts for that month
        """
        evolution = []
        for snapshot in self.history["forecast_snapshots"]:
            if target_month in snapshot["forecasts"]:
                evolution.append({
                    "date": snapshot["snapshot_date"],
                    "forecasts": snapshot["forecasts"][target_month],
                    "data_through": snapshot.get("data_through")
                })
        return evolution
    
    def get_model_stability_ranking(self) -> List[tuple]:
        """
        Get models ranked by stability score.
        
        Returns:
            List of (model_name, stability_score) tuples, sorted by stability (high to low)
        """
        metrics = self.history.get("stability_metrics", {})
        ranking = [(model, data["stability_score"]) 
                  for model, data in metrics.items()]
        return sorted(ranking, key=lambda x: x[1], reverse=True)
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics about forecast history.
        
        Returns:
            Dictionary with counts and date ranges
        """
        snapshots = self.history["forecast_snapshots"]
        
        return {
            "total_snapshots": len(snapshots),
            "months_tracked": len(self.history.get("accuracy_tracking", {})),
            "models_tracked": len(self.history.get("stability_metrics", {})),
            "date_range": {
                "first": snapshots[0]["snapshot_date"] if snapshots else None,
                "last": snapshots[-1]["snapshot_date"] if snapshots else None
            }
        }
    
    def get_accuracy_summary(self) -> Dict:
        """
        Get summary of forecast accuracy across all tracked months.
        
        Returns:
            Dictionary with average errors, convergence stats, etc.
        """
        tracking = self.history.get("accuracy_tracking", {})
        
        if not tracking:
            return {"message": "No completed months tracked yet"}
        
        all_errors = []
        convergence_counts = {"improved": 0, "degraded": 0, "stable": 0}
        
        for month_data in tracking.values():
            # Get final forecast error for each month
            forecasts = month_data.get("forecasts_over_time", [])
            if forecasts:
                # Most recent forecast
                final_forecast = min(forecasts, key=lambda x: abs(x["weeks_ahead"]))
                all_errors.append(abs(final_forecast["error_pct"]))
            
            # Count convergence quality
            quality = month_data.get("convergence_quality", "unknown")
            if quality in convergence_counts:
                convergence_counts[quality] += 1
        
        return {
            "months_completed": len(tracking),
            "mean_absolute_error_pct": round(np.mean(all_errors), 2) if all_errors else None,
            "median_absolute_error_pct": round(np.median(all_errors), 2) if all_errors else None,
            "convergence_breakdown": convergence_counts
        }
