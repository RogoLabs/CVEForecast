#!/usr/bin/env python3
"""
Production Forecast Generation

Quick forecast generation without validation/diagnostics for production use.

Usage:
    python3 code/run_production_forecast.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add code directory to path
code_dir = Path(__file__).parent
sys.path.insert(0, str(code_dir))

# Suppress non-critical warnings
from suppress_warnings import suppress_production_warnings
suppress_production_warnings()

from unified_pipeline import UnifiedForecastPipeline


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("  PRODUCTION FORECAST GENERATION")
    print("=" * 80)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = UnifiedForecastPipeline(
            cve_config=str(code_dir / 'config.json'),
            cna_config=str(code_dir / 'cna_config.json')
        )
        
        # Run both CVE and CNA forecasting (no validation/diagnostics)
        logger.info("Running production forecasts...")
        results = pipeline.run_all(
            run_cve=True,
            run_cna=True,
            cve_train_ratio=1.0,   # ✅ Use ALL data for production forecasts
            cve_validation=False,  # Skip for speed
            cve_diagnostics=False  # Skip for speed
        )
        
        # Save results summary
        pipeline.save_summary('web/pipeline_results.json')
        
        print()
        print("=" * 80)
        print("PRODUCTION FORECAST COMPLETE")
        print("=" * 80)
        print()
        
        # Print outputs
        cve_output = results.get('cve', {}).get('output_path')
        cna_output = results.get('cna', {}).get('output_path')
        
        if cve_output:
            print(f"✅ CVE forecasts: {cve_output}")
        else:
            print(f"❌ CVE forecasts: FAILED")
        
        if cna_output:
            print(f"✅ CNA forecasts: {cna_output}")
        else:
            print(f"❌ CNA forecasts: FAILED")
        
        print()
        
        # Return success if both completed
        if cve_output and cna_output:
            print("🎉 All forecasts generated successfully!")
            return 0
        else:
            print("⚠️  Some forecasts failed - check logs")
            return 1
    
    except Exception as e:
        logger.error(f"Fatal error in production forecast: {e}", exc_info=True)
        print()
        print(f"❌ FATAL ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
