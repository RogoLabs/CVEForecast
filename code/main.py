#!/usr/bin/env python3
"""
Main entry point for the CVE Forecast application.
Orchestrates the complete workflow from data processing to forecast generation.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd

from config import DEFAULT_CVE_DATA_PATH, DEFAULT_OUTPUT_PATH
from utils import setup_logging
from data_processor import CVEDataProcessor
from analysis import CVEForecastAnalyzer
from file_io import FileIOManager

logger = setup_logging()


class CVEForecastEngine:
    """Main engine that orchestrates the CVE forecasting workflow."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the CVE forecast engine.
        
        Args:
            data_path: Optional path to CVE data directory
        """
        self.data_path = data_path
        self.data_processor = CVEDataProcessor(data_path)
        self.analyzer = CVEForecastAnalyzer()
        self.file_manager = FileIOManager()
        
    def run(self, output_path: str = DEFAULT_OUTPUT_PATH) -> None:
        """
        Execute the CVE forecasting workflow with fresh forecast generation.
        
        Args:
            output_path: Path to save the generated data file
        """
        try:
            logger.info("🔄 Starting CVE Forecast generation with fresh forecasts...")
            self._run_fresh_forecast_mode(output_path)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Process interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Error in CVE forecast generation: {e}")
            logger.exception("Full error details:")
            sys.exit(1)
    
    def _run_standard_mode(self, output_path: str) -> None:
        """
        Execute the standard CVE forecasting workflow.
        
        Args:
            output_path: Path to save the generated data file
        """
        # Step 1: Load existing performance history
        logger.info("📊 Loading performance history...")
        performance_history = self.file_manager.load_performance_history()
        logger.info(f"Loaded {len(performance_history)} previous runs from performance history")
        
        # Step 2: Parse CVE data
        logger.info("🔍 Parsing CVE data...")
        data = self.data_processor.parse_cve_data()
        
        # Validate data quality
        quality_metrics = self.data_processor.validate_data_quality(data)
        
        if quality_metrics['total_records'] == 0:
            logger.error("No valid CVE data found. Cannot proceed with forecasting.")
            sys.exit(1)
        
        # Step 3: Evaluate models with comprehensive hyperparameter tracking
        logger.info("🤖 Evaluating models with hyperparameter tracking...")
        model_results = self.analyzer.evaluate_models(data)
        
        if not model_results:
            logger.error("No models successfully evaluated. Cannot proceed with forecasting.")
            sys.exit(1)
            
        # Step 4: Create performance record for this run
        logger.info("📈 Creating performance record for this run...")
        run_record = self.analyzer.create_run_record(data, model_results)
        logger.info(f"Created run record with {len(run_record['model_performances'])} model performances")
        
        # Step 5: Update performance history
        performance_history.append(run_record)
        logger.info(f"Added new run to history. Total runs: {len(performance_history)}")
        
        # Step 6: Save updated performance history
        logger.info("💾 Saving updated performance history...")
        self.file_manager.save_performance_history(performance_history)
        
        # Step 7: Generate forecasts
        logger.info("🔮 Generating forecasts...")
        forecasts = self.analyzer.generate_forecasts(data, model_results)
        
        # Step 8: Save data file for web interface
        logger.info("📄 Saving forecast data file...")
        self.file_manager.save_data_file(data, model_results, forecasts, output_path)
        
        # Step 9: Report completion
        logger.info("\n✅ CVE Forecast generation completed successfully!")
        logger.info(f"📊 Performance history now contains {len(performance_history)} runs")
        logger.info(f"🎯 Best model this run: {model_results[0]['model_name']} (MAPE: {model_results[0]['mape']:.4f})")
        logger.info(f"💾 Data file saved to: {output_path}")
        
        # Step 10: Display summary statistics
        self._display_summary_statistics(data, model_results, forecasts)
    
    def _run_fresh_forecast_mode(self, output_path: str) -> None:
        """
        Execute the fresh forecast generation workflow.
        
        Args:
            output_path: Path to save the updated data file
        """
        # Step 1: Load existing data.json to get model rankings
        logger.info("📂 Loading existing data.json to get model rankings...")
        existing_data = self.file_manager.load_existing_data_file(output_path)
        
        if not existing_data or 'model_rankings' not in existing_data:
            logger.error("No existing data.json found or missing model_rankings. Run standard mode first.")
            sys.exit(1)
        
        model_rankings = existing_data['model_rankings']
        logger.info(f"Found {len(model_rankings)} models in existing rankings")
        
        # Step 2: Extract historical data from existing data.json (CRITICAL FIX for baseline bug)
        logger.info("📊 Using historical data from existing data.json...")
        data = self._extract_historical_data_from_json(existing_data)
        
        # Validate data quality
        if len(data) == 0:
            logger.error("No valid historical data found in data.json. Cannot proceed with forecasting.")
            sys.exit(1)
        
        # Step 3: Generate fresh forecasts by retraining models
        logger.info("🔄 Generating fresh forecasts by retraining models...")
        fresh_forecast_data = self.analyzer.generate_fresh_forecasts(data, model_rankings)
        
        # Step 4: Update existing data.json with fresh forecasts
        logger.info("📝 Updating data.json with fresh forecast results...")
        existing_data['new_forecast_runs'] = fresh_forecast_data
        
        # Step 5: Save updated data file
        logger.info("💾 Saving updated data file with fresh forecasts...")
        self.file_manager.save_updated_data_file(existing_data, output_path)
        
        # Step 6: Report completion
        logger.info("\n✅ Fresh CVE forecast generation completed successfully!")
        
        successful_models = len(fresh_forecast_data.get('yearly_forecast_totals', {}))
        total_models = len(model_rankings)
        
        logger.info(f"🔄 Fresh forecasts generated for {successful_models}/{total_models} models")
        logger.info(f"💾 Updated data file saved to: {output_path}")
        
        # Display fresh forecast summary
        if fresh_forecast_data.get('yearly_forecast_totals'):
            logger.info("\n📊 Fresh Forecast Summary (Top 5 by Total):")
            sorted_forecasts = sorted(
                fresh_forecast_data['yearly_forecast_totals'].items(), 
                key=lambda x: x[1], reverse=True
            )
            for i, (model, total) in enumerate(sorted_forecasts[:5]):
                logger.info(f"  {i+1}. {model}: {total:,} CVEs")
        
        logger.info("\n🎉 Fresh forecast data is now available in the new_forecast_runs section!")
    
    def _extract_historical_data_from_json(self, existing_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract historical data from existing data.json and convert to DataFrame.
        
        Args:
            existing_data: Loaded data.json content
            
        Returns:
            DataFrame with historical data
        """
        if 'historical_data' not in existing_data:
            logger.error("No historical_data found in existing data.json. Run standard mode first.")
            sys.exit(1)
        
        # Convert historical data back to DataFrame format for model training
        historical_data = existing_data['historical_data']
        data_records = []
        
        for record in historical_data:
            data_records.append({
                'date': pd.to_datetime(record['date']),
                'cve_count': record['cve_count']
            })
        
        data = pd.DataFrame(data_records)
        logger.info(f"✅ Loaded {len(data)} historical data records from existing data.json")
        
        # Show the data range for debugging
        if len(data) > 0:
            logger.info(f"📅 Data range: {data['date'].min().strftime('%Y-%m')} to {data['date'].max().strftime('%Y-%m')}")
            logger.info(f"📊 Total CVEs: {data['cve_count'].sum():,}")
        
        return data
    
    def _display_summary_statistics(self, data: pd.DataFrame, 
                                   model_results: List[Dict[str, Any]], 
                                   forecasts: Dict[str, Any]) -> None:
        """
        Display summary statistics of the forecasting results.
        
        Args:
            data: Historical CVE data DataFrame
            model_results: List of model evaluation results
            forecasts: Dictionary of forecasts by model
        """
        logger.info("\n📊 Summary Statistics:")
        logger.info(f"  Historical data points: {len(data):,}")
        logger.info(f"  Date range: {data['date'].min().strftime('%Y-%m')} to {data['date'].max().strftime('%Y-%m')}")
        logger.info(f"  Total CVEs in dataset: {data['cve_count'].sum():,}")
        logger.info(f"  Average CVEs per month: {data['cve_count'].mean():.1f}")
        logger.info(f"  Successful models: {len(model_results)}")
        logger.info(f"  Total forecasts generated: {sum(len(forecasts[model]) for model in forecasts)}")
        
        # Model performance
        if model_results:
            best_model = model_results[0]
            worst_model = model_results[-1]
            
            logger.info(f"🏆 Best model: {best_model['model_name']} (MAPE: {best_model['mape']:.4f})")
            logger.info(f"📉 Worst model: {worst_model['model_name']} (MAPE: {worst_model['mape']:.4f})")
            logger.info(f"🤖 Total models evaluated: {len(model_results)}")
        
        # Forecast statistics
        if forecasts:
            total_forecasts = sum(len(f) for f in forecasts.values())
            logger.info(f"🔮 Total forecasts generated: {total_forecasts}")
            logger.info(f"📝 Models with forecasts: {len(forecasts)}")
        
        logger.info("-" * 50)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Generate CVE forecasts by retraining models on complete historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Generate fresh forecasts with default settings
  python main.py --data-path ./cvelistV5            # Specify CVE data path
  python main.py --output ./output/data.json        # Specify output file
  python main.py --data-path ./data --output ./web/data.json  # Custom paths
        """
    )
    
    parser.add_argument(
        '--data-path', 
        help='Path to the CVE data directory (default: cvelistV5 in project root)',
        default=None
    )
    
    parser.add_argument(
        '--output', 
        default=DEFAULT_OUTPUT_PATH,
        help=f'Output path for the generated data file (default: {DEFAULT_OUTPUT_PATH})'
    )
    

    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='CVE Forecast Engine 2.0.0'
    )
    
    args = parser.parse_args()
    
    # Setup logging with specified level
    logger = setup_logging(args.log_level)
    
    # Validate output directory
    output_path = Path(args.output)
    if not output_path.parent.exists():
        logger.info(f"Creating output directory: {output_path.parent}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run the engine
    logger.info("Initializing CVE Forecast Engine...")
    engine = CVEForecastEngine(data_path=args.data_path)
    engine.run(output_path=str(output_path))


if __name__ == "__main__":
    main()
