#!/usr/bin/env python3
"""
CVE Forecast Validation Script
This script validates the forecast functionality using real CVE data from the shared cvelistV5 directory.
"""

import json
import os
import sys
from pathlib import Path


def check_cve_data():
    """Check if CVE data exists in the shared directory"""
    # Get the project root directory (parent of tests directory)
    project_root = Path(__file__).parent.parent
    cve_data_dir = project_root / "cvelistV5"
    cves_dir = cve_data_dir / "cves"
    
    if not cve_data_dir.exists() or not cves_dir.exists():
        print("❌ CVE data not found!")
        print(f"Expected CVE data directory: {cve_data_dir}")
        print("\nPlease run the following command from the project root to download CVE data:")
        print("    python download_data.py")
        print("\nThis will download the CVE data repository to the project root.")
        return None
    
    # Count available CVE files for verification
    json_files = list(cves_dir.rglob("*.json"))
    print(f"✓ Found CVE data with {len(json_files)} CVE files")
    
    return str(cve_data_dir)


def run_forecast_validation():
    """Run the forecast script with real CVE data"""
    print("\nRunning forecast validation with real CVE data...")
    
    # Create output file path
    output_file = "test_output.json"
    
    # Run the forecast script (it will automatically use cvelistV5)
    cmd = f"python ../cve_forecast.py --output {output_file}"
    
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Forecast script failed:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    # Validate output file was created
    if not os.path.exists(output_file):
        print(f"Error: Output file {output_file} was not created")
        return False
    
    # Validate output file contents
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        required_keys = ['generated_at', 'model_rankings', 'historical_data', 'forecasts', 'summary']
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            print(f"Error: Missing required keys in output: {missing_keys}")
            return False
        
        print("✓ Output file structure is valid")
        print(f"✓ Historical data points: {len(data.get('historical_data', []))}")
        print(f"✓ Model rankings: {len(data.get('model_rankings', []))}")
        print(f"✓ Forecast models: {len(data.get('forecasts', {}))}")
        print(f"✓ Total historical CVEs: {data.get('summary', {}).get('total_historical_cves', 0)}")
        
        # Cleanup
        os.remove(output_file)
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in output file: {e}")
        return False
    except Exception as e:
        print(f"Error validating output: {e}")
        return False


def main():
    """Main validation function"""
    print("=" * 60)
    print("CVE Forecast Validation Script")
    print("=" * 60)
    
    # Change to tests directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Step 1: Check for CVE data in shared directory
        cve_data_dir = check_cve_data()
        if not cve_data_dir:
            sys.exit(1)
        
        # Step 2: Run forecast validation
        if run_forecast_validation():
            print("\n✅ Forecast validation completed successfully!")
            print("The forecast script is working correctly with real CVE data.")
        else:
            print("\n❌ Forecast validation failed!")
            print("There are issues with the forecast script.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nUnexpected error during validation: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Validation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
