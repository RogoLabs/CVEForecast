#!/usr/bin/env python3
"""
Hyperparameter tuning module for CVE Forecast application.
Analyzes performance history to optimize model hyperparameters.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
from collections import defaultdict
import statistics
import numpy as np

from utils import setup_logging

logger = setup_logging()


class HyperparameterTuner:
    """Handles hyperparameter optimization using historical performance data."""
    
    def __init__(self, performance_history_path: str = "../web/performance_history.json"):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            performance_history_path: Path to the performance history file
        """
        self.history_path = performance_history_path
        self.performance_history = []
        self.model_performance_db = defaultdict(list)
        self.optimal_hyperparameters = {}
        
    def load_performance_history(self) -> bool:
        """
        Load performance history from JSON file.
        
        Returns:
            True if history was loaded successfully, False otherwise
        """
        try:
            if Path(self.history_path).exists():
                with open(self.history_path, 'r') as f:
                    self.performance_history = json.load(f)
                logger.info(f"Loaded {len(self.performance_history)} performance history records")
                return True
            else:
                logger.warning(f"Performance history file {self.history_path} not found")
                return False
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
            return False
    
    def analyze_model_performances(self) -> None:
        """Analyze historical model performances to build performance database."""
        logger.info("Analyzing historical model performances...")
        
        self.model_performance_db.clear()
        
        for run in self.performance_history:
            run_timestamp = run.get('timestamp', 'unknown')
            
            for model_perf in run.get('model_performances', []):
                model_name = model_perf.get('model_name')
                if not model_name:
                    continue
                
                # Store performance record with all relevant data
                performance_record = {
                    'timestamp': run_timestamp,
                    'metrics': model_perf.get('metrics', {}),
                    'hyperparameters': model_perf.get('hyperparameters', {}),
                    'model_category': model_perf.get('model_category', 'unknown')
                }
                
                self.model_performance_db[model_name].append(performance_record)
        
        logger.info(f"Analyzed performances for {len(self.model_performance_db)} unique models")
        
        # Log model performance counts
        for model_name, records in self.model_performance_db.items():
            logger.debug(f"  {model_name}: {len(records)} performance records")
    
    def find_optimal_hyperparameters(self, primary_metric: str = 'mape') -> Dict[str, Dict[str, Any]]:
        """
        Find optimal hyperparameters for each model based on historical performance.
        
        Args:
            primary_metric: Primary metric to optimize for (default: 'mape', lower is better)
            
        Returns:
            Dictionary mapping model names to their optimal hyperparameters
        """
        logger.info(f"Finding optimal hyperparameters using {primary_metric} as primary metric...")
        
        self.optimal_hyperparameters.clear()
        optimization_summary = []
        
        for model_name, performance_records in self.model_performance_db.items():
            if not performance_records:
                continue
                
            logger.debug(f"Optimizing hyperparameters for {model_name}...")
            
            # Find best performing configuration
            best_record = None
            best_metric_value = float('inf')  # Assuming lower is better for primary metric
            
            valid_records = []
            for record in performance_records:
                metrics = record.get('metrics', {})
                metric_value = metrics.get(primary_metric)
                
                if metric_value is not None and not (np.isnan(metric_value) or np.isinf(metric_value)):
                    valid_records.append(record)
                    if metric_value < best_metric_value:
                        best_metric_value = metric_value
                        best_record = record
            
            if best_record:
                optimal_params = best_record['hyperparameters'].copy()
                best_metrics = best_record['metrics'].copy()
                
                self.optimal_hyperparameters[model_name] = {
                    'hyperparameters': optimal_params,
                    'expected_performance': best_metrics,
                    'model_category': best_record.get('model_category', 'unknown'),
                    'based_on_runs': len(valid_records),
                    'optimization_timestamp': best_record.get('timestamp')
                }
                
                # Calculate performance statistics if multiple records exist
                if len(valid_records) > 1:
                    metric_values = [r['metrics'].get(primary_metric) for r in valid_records 
                                   if r['metrics'].get(primary_metric) is not None]
                    if metric_values:
                        self.optimal_hyperparameters[model_name]['performance_stats'] = {
                            'mean': statistics.mean(metric_values),
                            'median': statistics.median(metric_values),
                            'std': statistics.stdev(metric_values) if len(metric_values) > 1 else 0,
                            'min': min(metric_values),
                            'max': max(metric_values),
                            'count': len(metric_values)
                        }
                
                optimization_summary.append({
                    'model': model_name,
                    'best_mape': best_metric_value,
                    'runs_analyzed': len(valid_records),
                    'category': best_record.get('model_category', 'unknown')
                })
                
                logger.info(f"✓ {model_name}: Best {primary_metric}={best_metric_value:.4f} "
                           f"(from {len(valid_records)} runs)")
            else:
                logger.warning(f"⚠ {model_name}: No valid performance records found")
        
        # Log optimization summary
        if optimization_summary:
            logger.info(f"\n=== HYPERPARAMETER OPTIMIZATION SUMMARY ===")
            logger.info(f"Optimized parameters for {len(optimization_summary)} models")
            
            # Sort by performance and show top performers
            optimization_summary.sort(key=lambda x: x['best_mape'])
            logger.info("Top performing model configurations:")
            for i, summary in enumerate(optimization_summary[:5]):
                logger.info(f"  {i+1}. {summary['model']}: {primary_metric}={summary['best_mape']:.4f} "
                           f"({summary['runs_analyzed']} runs, {summary['category']})")
            
            logger.info(f"============================================\n")
        
        return self.optimal_hyperparameters
    
    def get_tuned_hyperparameters(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get optimized hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Optimized hyperparameters dict or None if not available
        """
        if model_name in self.optimal_hyperparameters:
            return self.optimal_hyperparameters[model_name]['hyperparameters']
        else:
            logger.debug(f"No optimized hyperparameters available for {model_name}")
            return None
    
    def get_expected_performance(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Get expected performance metrics for a model with optimal hyperparameters.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Expected performance metrics dict or None if not available
        """
        if model_name in self.optimal_hyperparameters:
            return self.optimal_hyperparameters[model_name]['expected_performance']
        else:
            return None
    
    def should_retune(self, model_name: str, runs_threshold: int = 5) -> bool:
        """
        Determine if a model should be retuned based on the number of runs.
        
        Args:
            model_name: Name of the model
            runs_threshold: Minimum number of runs before considering retuning
            
        Returns:
            True if model should be retuned, False otherwise
        """
        if model_name not in self.model_performance_db:
            return False
            
        run_count = len(self.model_performance_db[model_name])
        return run_count >= runs_threshold
    
    def export_optimization_report(self, output_path: str = "../web/optimization_report.json") -> bool:
        """
        Export detailed optimization report to JSON file.
        
        Args:
            output_path: Path to save the optimization report
            
        Returns:
            True if report was exported successfully, False otherwise
        """
        try:
            report = {
                'generation_timestamp': self.performance_history[-1].get('timestamp') if self.performance_history else None,
                'total_runs_analyzed': len(self.performance_history),
                'models_optimized': len(self.optimal_hyperparameters),
                'optimization_details': self.optimal_hyperparameters,
                'model_performance_summary': {}
            }
            
            # Add performance summary for each model
            for model_name, records in self.model_performance_db.items():
                if records:
                    mape_values = [r['metrics'].get('mape') for r in records 
                                 if r['metrics'].get('mape') is not None]
                    if mape_values:
                        report['model_performance_summary'][model_name] = {
                            'runs_count': len(records),
                            'best_mape': min(mape_values),
                            'worst_mape': max(mape_values),
                            'average_mape': statistics.mean(mape_values),
                            'performance_trend': 'improving' if len(mape_values) > 1 and mape_values[-1] < mape_values[0] else 'stable'
                        }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Optimization report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export optimization report: {e}")
            return False
    
    def tune_models(self) -> bool:
        """
        Complete hyperparameter tuning workflow.
        
        Returns:
            True if tuning completed successfully, False otherwise
        """
        logger.info("🔧 Starting hyperparameter tuning workflow...")
        
        # Load performance history
        if not self.load_performance_history():
            logger.error("Cannot proceed without performance history")
            return False
        
        if not self.performance_history:
            logger.warning("No performance history available for tuning")
            return False
        
        # Analyze performances and find optimal parameters
        self.analyze_model_performances()
        self.find_optimal_hyperparameters()
        
        if not self.optimal_hyperparameters:
            logger.warning("No optimal hyperparameters found")
            return False
        
        # Export detailed report
        self.export_optimization_report()
        
        logger.info(f"🎯 Hyperparameter tuning completed successfully!")
        logger.info(f"   Optimized {len(self.optimal_hyperparameters)} models")
        logger.info(f"   Based on {len(self.performance_history)} historical runs")
        
        return True


def main():
    """Main function for testing the tuner module."""
    tuner = HyperparameterTuner()
    success = tuner.tune_models()
    
    if success:
        print("✅ Hyperparameter tuning completed successfully!")
        
        # Display some results
        print("\nOptimized models:")
        for model_name in tuner.optimal_hyperparameters:
            expected_perf = tuner.get_expected_performance(model_name)
            mape = expected_perf.get('mape') if expected_perf else None
            print(f"  - {model_name}: Expected MAPE = {mape:.4f}" if mape else f"  - {model_name}")
    else:
        print("❌ Hyperparameter tuning failed!")


if __name__ == "__main__":
    main()
