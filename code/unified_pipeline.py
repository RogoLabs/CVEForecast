"""
Unified Forecasting Pipeline

Coordinates CVE and CNA forecasting with shared validation and diagnostics.
Single entry point for all forecast generation.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

from adapters.cve_adapter import CVEForecaster
from adapters.cna_adapter import CNAForecaster


class UnifiedForecastPipeline:
    """
    Unified pipeline for CVE and CNA forecasting.
    
    Provides single interface for running both forecast types
    with shared validation, diagnostics, and reporting.
    """
    
    def __init__(self, 
                 cve_config: str = 'config.json',
                 cna_config: str = 'cna_config.json',
                 cvelist_dir: str = 'cvelistV5',
                 min_cves: int = 100):
        """
        Initialize unified pipeline.
        
        Args:
            cve_config: Path to CVE configuration
            cna_config: Path to CNA configuration
            cvelist_dir: Path to cvelistV5 repository
            min_cves: Minimum CVEs for CNA inclusion
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize forecasters
        self.cve_forecaster = CVEForecaster(config_path=cve_config)
        self.cna_forecaster = CNAForecaster(
            config_path=cna_config,
            cvelist_dir=cvelist_dir,
            min_cves=min_cves
        )
        
        self.results = {
            'cve': {},
            'cna': {}
        }
        
        self.logger.info("Unified Forecast Pipeline initialized")
    
    def run_cve_pipeline(self, 
                        train_ratio: float = 0.8,
                        run_validation: bool = True,
                        run_diagnostics: bool = False) -> Dict[str, Any]:
        """
        Execute CVE forecasting pipeline.
        
        Args:
            train_ratio: Train/test split ratio
            run_validation: Whether to run validation
            run_diagnostics: Whether to run diagnostics
            
        Returns:
            CVE pipeline results
        """
        self.logger.info("=" * 70)
        self.logger.info("EXECUTING CVE PIPELINE")
        self.logger.info("=" * 70)
        
        results = self.cve_forecaster.run_full_pipeline(
            train_ratio=train_ratio,
            run_validation=run_validation,
            run_diagnostics=run_diagnostics
        )
        
        self.results['cve'] = results
        
        return results
    
    def run_cna_pipeline(self) -> Dict[str, Any]:
        """
        Execute CNA forecasting pipeline.
        
        Returns:
            CNA pipeline results
        """
        self.logger.info("=" * 70)
        self.logger.info("EXECUTING CNA PIPELINE")
        self.logger.info("=" * 70)
        
        results = self.cna_forecaster.run_full_pipeline()
        
        self.results['cna'] = results
        
        return results
    
    def run_all(self,
                run_cve: bool = True,
                run_cna: bool = True,
                cve_train_ratio: float = 0.8,
                cve_validation: bool = True,
                cve_diagnostics: bool = False,
                retune_models: bool = False) -> Dict[str, Any]:
        """
        Execute complete unified pipeline (both CVE and CNA).
        
        Args:
            run_cve: Whether to run CVE forecasting
            run_cna: Whether to run CNA forecasting
            cve_train_ratio: Train/test split for CVE
            cve_validation: Run CVE validation
            cve_diagnostics: Run CVE diagnostics
            retune_models: Re-optimize hyperparameters (SLOW - takes hours!)
            
        Returns:
            Combined results
        """
        self.logger.info("=" * 70)
        self.logger.info("UNIFIED FORECASTING PIPELINE - STARTING")
        self.logger.info("=" * 70)
        
        # Optional: Re-tune models (quarterly maintenance)
        if retune_models:
            self.logger.warning("⚠️  MODEL RE-TUNING ENABLED")
            self.logger.warning("⚠️  This will take HOURS to complete!")
            self.logger.warning("⚠️  Expected duration: 2-4 hours for all models")
            self.logger.info("")
            self._run_comprehensive_tuning()
        
        if run_cve:
            cve_results = self.run_cve_pipeline(
                train_ratio=cve_train_ratio,
                run_validation=cve_validation,
                run_diagnostics=cve_diagnostics
            )
        else:
            self.logger.info("Skipping CVE pipeline")
            cve_results = {'skipped': True}
        
        if run_cna:
            cna_results = self.run_cna_pipeline()
        else:
            self.logger.info("Skipping CNA pipeline")
            cna_results = {'skipped': True}
        
        self.results = {
            'cve': cve_results,
            'cna': cna_results
        }
        
        self.logger.info("=" * 70)
        self.logger.info("UNIFIED PIPELINE - COMPLETE")
        self.logger.info("=" * 70)
        
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print unified pipeline summary."""
        self.logger.info("\nPIPELINE SUMMARY")
        self.logger.info("-" * 70)
        
        # CVE summary
        cve_results = self.results.get('cve', {})
        if not cve_results.get('skipped'):
            self.logger.info("CVE Forecasting:")
            self.logger.info(f"  ✓ Data: {cve_results.get('data_periods', 0)} periods")
            self.logger.info(f"  ✓ Models: {cve_results.get('models_trained', 0)} trained")
            self.logger.info(f"  ✓ Output: {cve_results.get('output_path', 'N/A')}")
            if cve_results.get('cv_completed'):
                self.logger.info(f"  ✓ Validation: {cve_results.get('cv_models', 0)} models")
            if cve_results.get('diagnostics_completed'):
                self.logger.info(f"  ✓ Diagnostics: {cve_results.get('diagnostics_models', 0)} models")
        
        # CNA summary
        cna_results = self.results.get('cna', {})
        if not cna_results.get('skipped'):
            self.logger.info("\nCNA Forecasting:")
            self.logger.info(f"  ✓ CNAs: {cna_results.get('cnas_loaded', 0)}")
            self.logger.info(f"  ✓ Forecasts: {cna_results.get('forecasts_generated', 0)}")
            self.logger.info(f"  ✓ Output: {cna_results.get('output_path', 'N/A')}")
        
        self.logger.info("-" * 70)
    
    def save_summary(self, output_path: str = 'pipeline_results.json'):
        """
        Save pipeline results to JSON.
        
        Args:
            output_path: Where to save results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"✓ Pipeline results saved to {output_file}")
    
    def _run_comprehensive_tuning(self):
        """
        Run comprehensive hyperparameter tuning.
        
        This re-optimizes all models and updates config.json.
        WARNING: This takes 2-4 hours to complete!
        """
        import subprocess
        import sys
        from datetime import datetime
        
        self.logger.info("=" * 70)
        self.logger.info("COMPREHENSIVE HYPERPARAMETER TUNING")
        self.logger.info("=" * 70)
        self.logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")
        
        # Run comprehensive tuner
        tuner_path = Path(__file__).parent / 'tuner' / 'comprehensive_tuner.py'
        
        try:
            result = subprocess.run(
                [sys.executable, str(tuner_path)],
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("")
                self.logger.info("✅ Model tuning completed successfully")
                self.logger.info("✅ config.json updated with optimized hyperparameters")
                self.logger.info("")
            else:
                self.logger.error(f"❌ Tuning failed with exit code {result.returncode}")
                raise RuntimeError("Hyperparameter tuning failed")
                
        except Exception as e:
            self.logger.error(f"❌ Tuning failed: {e}")
            raise
        
        self.logger.info("=" * 70)
        self.logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 70)
        self.logger.info("")
    
    def get_cve_forecaster(self) -> CVEForecaster:
        """Get CVE forecaster instance."""
        return self.cve_forecaster
    
    def get_cna_forecaster(self) -> CNAForecaster:
        """Get CNA forecaster instance."""
        return self.cna_forecaster
