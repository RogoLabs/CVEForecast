#!/usr/bin/env python3
"""
CVE Data Download and Management Script
This script downloads real CVE data to the project root and manages updates.
"""

import json
import os
import sys
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def download_cve_data():
    """Download or update the CVE repository in the project root"""
    print("=" * 60)
    print("CVE Data Download and Management")
    print("=" * 60)
    
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    cve_repo_url = "https://github.com/CVEProject/cvelistV5.git"
    data_dir = "cvelistV5"
    
    if os.path.exists(data_dir):
        print(f"CVE data directory '{data_dir}' already exists.")
        
        # Check if it's a git repository
        if os.path.exists(os.path.join(data_dir, '.git')):
            print("Updating existing CVE data repository...")
            
            # Change to the CVE data directory
            os.chdir(data_dir)
            
            # Pull latest changes
            result = subprocess.run("git pull origin master", shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error updating CVE data: {result.stderr}")
                print("Attempting to re-clone the repository...")
                
                # Go back to project root and remove the directory
                os.chdir(project_root)
                shutil.rmtree(data_dir)
                return clone_fresh_repo(cve_repo_url, data_dir)
            else:
                print("✓ CVE data repository updated successfully")
                os.chdir(project_root)  # Return to project root
                return data_dir
        else:
            print(f"Directory '{data_dir}' exists but is not a git repository.")
            print("Removing and re-cloning...")
            shutil.rmtree(data_dir)
            return clone_fresh_repo(cve_repo_url, data_dir)
    else:
        return clone_fresh_repo(cve_repo_url, data_dir)


def clone_fresh_repo(repo_url, data_dir):
    """Clone the CVE repository fresh"""
    print(f"Cloning CVE repository to '{data_dir}' (this may take a few minutes)...")
    
    # Use shallow clone to save space and time
    cmd = f"git clone --depth 1 {repo_url} {data_dir}"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error downloading CVE data: {result.stderr}")
        return None
    
    print(f"✓ CVE data cloned to '{data_dir}'")
    return data_dir


def create_recent_subset(source_dir, days_back=180):
    """Create a subset of CVE data from recent months for faster processing"""
    print(f"\nCreating subset of CVE data from the last {days_back} days...")
    
    cves_dir = Path(source_dir) / "cves"
    if not cves_dir.exists():
        print(f"Error: CVE data directory not found at {cves_dir}")
        return None
    
    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=days_back)
    cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
    
    # Find all JSON files
    all_files = list(cves_dir.rglob("*.json"))
    recent_files = []
    processed = 0
    
    print(f"Scanning {len(all_files)} CVE files for entries after {cutoff_date_str}...")
    
    for file_path in all_files:
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed}/{len(all_files)} files, found {len(recent_files)} recent CVEs")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cve_data = json.load(f)
            
            # Check if CVE has publication date
            if 'cveMetadata' in cve_data and 'published' in cve_data['cveMetadata']:
                published_date = cve_data['cveMetadata']['published']
                # Parse ISO format date
                pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00')).date()
                
                if pub_date >= cutoff_date.date():
                    recent_files.append(file_path)
        
        except (json.JSONDecodeError, KeyError, ValueError):
            # Skip malformed or incomplete CVE entries
            continue
    
    print(f"✓ Found {len(recent_files)} CVEs published after {cutoff_date_str}")
    return recent_files, cutoff_date_str


def run_forecast_on_subset(data_dir):
    """Run the forecast analysis on the CVE data subset"""
    print(f"\nRunning forecast analysis on CVE data...")
    
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    try:
        # Run the forecast script
        cmd = f"python cve_forecast.py --data-path {data_dir} --output web/data.json"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Forecast script failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        print("✓ Forecast analysis completed successfully")
        print("✓ Dashboard data generated at web/data.json")
        return True
    
    except Exception as e:
        print(f"Error running forecast: {e}")
        return False


def main():
    """Main function"""
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    try:
        # Step 1: Download CVE data
        data_dir = download_cve_data()
        if not data_dir:
            print("Failed to download CVE data. Exiting.")
            sys.exit(1)
        
        # Step 2: Create recent subset for analysis
        recent_files, cutoff_date = create_recent_subset(data_dir, days_back=180)
        if not recent_files:
            print("No recent CVE data found. Exiting.")
            sys.exit(1)
        
        print(f"\nUsing subset of {len(recent_files)} CVEs from {cutoff_date} onwards")
        print("This provides sufficient data for model training while keeping processing time reasonable.")
        
        # Step 3: Run forecast analysis
        if not run_forecast_on_subset(data_dir):
            print("Failed to run forecast analysis. Exiting.")
            sys.exit(1)
        
        # Step 4: Ask about server startup
        while True:
            response = input("\nDo you want to start the local development server? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                print("Starting development server...")
                try:
                    subprocess.run("python serve.py", shell=True)
                except KeyboardInterrupt:
                    print("\nServer stopped by user.")
                break
            elif response in ['n', 'no']:
                print("Skipping server startup.")
                break
            else:
                print("Please enter 'y' or 'n'")
        
        print("\n" + "=" * 60)
        print("CVE data processing completed!")
        print("Dashboard data is ready at web/data.json")
        print("You can start the server anytime with: python serve.py")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
