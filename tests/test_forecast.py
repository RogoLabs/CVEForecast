#!/usr/bin/env python3
"""
CVE Forecast Test Script
This script downloads CVE data locally and runs the forecast for testing purposes.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{description}...")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {description} failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f"Success: {description} completed")
    if result.stdout:
        print(f"Output: {result.stdout}")
    
    return True


def ensure_cve_data():
    """Ensure CVE data is available in the project root"""
    # Get project root directory (parent of tests directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "cvelistV5")
    
    if os.path.exists(data_dir):
        print(f"CVE data already exists at: {data_dir}")
        return data_dir
    else:
        print(f"CVE data not found. Please run 'python download_data.py' from the project root first.")
        print(f"Expected location: {data_dir}")
        return None


def run_forecast(output_path):
    """Run the CVE forecast script"""
    cmd = f"python ../cve_forecast.py --output {output_path}"
    return run_command(cmd, "Running CVE forecast")


def start_local_server():
    """Start the local development server"""
    cmd = "python ../serve.py"
    print(f"\nStarting local server...")
    print(f"Running: {cmd}")
    print("The server will start and open your browser automatically.")
    print("Press Ctrl+C to stop the server when done.")
    
    try:
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")


def main():
    """Main test function"""
    print("=" * 60)
    print("CVE Forecast Local Test Script")
    print("=" * 60)
    
    # Change to tests directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Step 1: Ensure CVE data is available
        data_dir = ensure_cve_data()
        if not data_dir:
            print("\nTo download CVE data, run from the project root:")
            print("  python download_data.py")
            sys.exit(1)
        
        # Verify data exists
        cves_path = Path(data_dir) / "cves"
        if not cves_path.exists():
            print(f"Error: CVE data directory not found at {cves_path}")
            sys.exit(1)
        
        json_files = list(cves_path.rglob("*.json"))
        print(f"Found {len(json_files)} CVE JSON files in the data")
        
        # Step 2: Run the forecast
        output_file = "../web/data.json"
        if not run_forecast(output_file):
            print("Failed to run CVE forecast. Exiting.")
            sys.exit(1)
        
        # Verify output was created
        if not os.path.exists(output_file):
            print(f"Error: Output file not created at {output_file}")
            sys.exit(1)
        
        print(f"\nForecast completed successfully!")
        print(f"Output saved to: {output_file}")
        
        # Step 3: Ask if user wants to start the server
        while True:
            response = input("\nDo you want to start the local development server? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                start_local_server()
                break
            elif response in ['n', 'no']:
                print("Skipping server startup.")
                break
            else:
                print("Please enter 'y' or 'n'")
        
        print("\nTest completed successfully!")
        print(f"Note: You can manually start the server later with: python serve.py")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
