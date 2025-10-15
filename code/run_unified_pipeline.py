#!/usr/bin/env python3
"""
Unified Pipeline Execution Script

Runs both CVE and CNA forecasting with shared validation framework.

Usage:
    python3 code/run_unified_pipeline.py [--cve-only] [--cna-only] [--with-validation] [--with-diagnostics]
"""

import sys
import argparse
import logging
from pathlib import Path

# Add code directory to path
code_dir = Path(__file__).parent
sys.path.insert(0, str(code_dir))

# Suppress non-critical warnings
from suppress_warnings import suppress_production_warnings
suppress_production_warnings()

from unified_pipeline import UnifiedForecastPipeline
from utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description='Run unified CVE + CNA forecasting pipeline')
    parser.add_argument('--cve-only', action='store_true', help='Run only CVE forecasting')
    parser.add_argument('--cna-only', action='store_true', help='Run only CNA forecasting')
    parser.add_argument('--with-validation', action='store_true', help='Run cross-validation')
    parser.add_argument('--with-diagnostics', action='store_true', help='Run diagnostics')
    parser.add_argument('--retune-models', action='store_true', help='Re-optimize hyperparameters (SLOW - hours!)')
    parser.add_argument('--cvelist-dir', default='cvelistV5', help='Path to cvelistV5')
    parser.add_argument('--min-cves', type=int, default=100, help='Minimum CVEs for CNA')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("  UNIFIED FORECASTING PIPELINE")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    pipeline = UnifiedForecastPipeline(
        cve_config=str(code_dir / 'config.json'),
        cna_config=str(code_dir / 'cna_config.json'),
        cvelist_dir=args.cvelist_dir,
        min_cves=args.min_cves
    )
    
    # Determine what to run
    run_cve = not args.cna_only
    run_cna = not args.cve_only
    
    print(f"Configuration:")
    print(f"  CVE Forecasting: {'Yes' if run_cve else 'No'}")
    print(f"  CNA Forecasting: {'Yes' if run_cna else 'No'}")
    print(f"  Validation: {'Yes' if args.with_validation else 'No'}")
    print(f"  Diagnostics: {'Yes' if args.with_diagnostics else 'No'}")
    print(f"  Model Re-tuning: {'Yes ⚠️  (SLOW - hours!)' if args.retune_models else 'No'}")
    print()
    
    # Run pipeline
    results = pipeline.run_all(
        run_cve=run_cve,
        run_cna=run_cna,
        cve_train_ratio=1.0,  # ✅ Use ALL data for production forecasts
        cve_validation=args.with_validation,
        cve_diagnostics=args.with_diagnostics,
        retune_models=args.retune_models
    )
    
    # Save results
    pipeline.save_summary('pipeline_results.json')
    
    print()
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    # Exit status
    cve_success = results.get('cve', {}).get('output_path') is not None or results.get('cve', {}).get('skipped')
    cna_success = results.get('cna', {}).get('output_path') is not None or results.get('cna', {}).get('skipped')
    
    if cve_success and cna_success:
        print("✅ All pipelines completed successfully")
        return 0
    else:
        print("❌ Some pipelines failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
