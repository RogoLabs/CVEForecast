#!/usr/bin/env python3
"""
Data fetcher module for CVE Forecast application.
Handles external data retrieval and API calls (future extensibility).
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from utils import setup_logging, is_file_readable

logger = setup_logging()


class CVEDataFetcher:
    """Handles external data fetching operations (placeholder for future extensibility)."""
    
    def __init__(self):
        """Initialize the data fetcher."""
        pass
    
    def verify_local_data_availability(self, data_path: str) -> bool:
        """
        Verify that local CVE data is available and accessible.
        
        Args:
            data_path: Path to the CVE data directory
            
        Returns:
            True if data is available, False otherwise
        """
        cves_path = Path(data_path) / "cves"
        
        if not cves_path.exists():
            logger.error(f"CVE data directory not found: {cves_path}")
            return False
        
        # Check for JSON files
        json_files = list(cves_path.rglob("*.json"))
        if len(json_files) == 0:
            logger.error(f"No JSON files found in CVE data directory: {cves_path}")
            return False
        
        logger.info(f"Found {len(json_files)} CVE JSON files in {cves_path}")
        return True
    
    def get_data_download_instructions(self) -> str:
        """
        Get instructions for downloading CVE data.
        
        Returns:
            String with download instructions
        """
        instructions = """
CVE Data Download Instructions:
==============================

To download the CVE data repository, run:

    python download_data.py

This will download the CVE data repository to the project root.

Alternatively, you can manually clone the repository:

    git clone https://github.com/CVEProject/cvelistV5.git

The CVE data should be placed in the 'cvelistV5' directory relative to the project root.
        """
        return instructions.strip()
    
    def validate_data_structure(self, data_path: str) -> Dict[str, Any]:
        """
        Validate the structure of the CVE data directory.
        
        Args:
            data_path: Path to the CVE data directory
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': False,
            'issues': [],
            'statistics': {}
        }
        
        base_path = Path(data_path)
        cves_path = base_path / "cves"
        
        # Check directory structure
        if not base_path.exists():
            validation_results['issues'].append(f"Base directory does not exist: {base_path}")
            return validation_results
        
        if not cves_path.exists():
            validation_results['issues'].append(f"CVEs directory does not exist: {cves_path}")
            return validation_results
        
        # Count files and directories
        json_files = list(cves_path.rglob("*.json"))
        year_dirs = [d for d in cves_path.iterdir() if d.is_dir() and d.name.isdigit()]
        
        validation_results['statistics'] = {
            'total_json_files': len(json_files),
            'year_directories': len(year_dirs),
            'years_covered': sorted([d.name for d in year_dirs])
        }
        
        # Validate minimum requirements
        if len(json_files) < 100:  # Arbitrary minimum threshold
            validation_results['issues'].append(f"Too few JSON files found: {len(json_files)}")
        
        if len(year_dirs) == 0:
            validation_results['issues'].append("No year directories found")
        
        # Mark as valid if no issues
        validation_results['is_valid'] = len(validation_results['issues']) == 0
        
        if validation_results['is_valid']:
            logger.info(f"Data validation passed: {validation_results['statistics']}")
        else:
            logger.warning(f"Data validation issues: {validation_results['issues']}")
        
        return validation_results
    
    # Future methods for external data fetching could be added here:
    # - fetch_latest_cve_data()
    # - update_local_repository()
    # - fetch_epss_scores()
    # - etc.
