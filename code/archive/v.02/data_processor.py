#!/usr/bin/env python3
"""
Data processing module for CVE Forecast application.
Handles parsing CVE data from JSON files and preparing it for analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from config import DEFAULT_CVE_DATA_PATH
from utils import setup_logging, validate_date_format

logger = setup_logging()


class CVEDataProcessor:
    """Handles CVE data parsing and processing operations."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the CVE data processor.
        
        Args:
            data_path: Path to CVE data directory (defaults to config value)
        """
        self.data_path = data_path or DEFAULT_CVE_DATA_PATH
        
    def parse_cve_data(self, repo_path: Optional[str] = None) -> pd.DataFrame:
        """
        Parse CVE JSON files and extract publication dates.
        
        Args:
            repo_path: Optional path to CVE repository (overrides instance path)
            
        Returns:
            DataFrame with CVE publication dates and counts
            
        Raises:
            FileNotFoundError: If CVE data directory is not found
        """
        if repo_path is None:
            repo_path = self.data_path
            
        if repo_path is None:
            # Default to cvelistV5 directory in project root
            script_dir = Path(__file__).parent
            repo_path = str(script_dir / "cvelistV5")
            
        # Check if the CVE data directory exists
        cves_path = Path(repo_path) / "cves"
        if not cves_path.exists():
            logger.error(f"CVE data not found at: {repo_path}")
            logger.error("\nPlease run the following command to download CVE data:")
            logger.error("    python download_data.py")
            logger.error("\nThis will download the CVE data repository to the project root.")
            raise FileNotFoundError(f"CVE data directory not found: {cves_path}")
            
        logger.info(f"Parsing CVE data from {repo_path}...")
        cve_dates = []
        
        cves_path = Path(repo_path) / "cves"
        # Recursively find all JSON files
        json_files = list(cves_path.rglob("*.json"))
        logger.info(f"Found {len(json_files)} CVE JSON files")
        
        for i, json_file in enumerate(json_files):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(json_files)} files")
            
            try:
                cve_date = self._extract_cve_date(json_file)
                if cve_date:
                    cve_dates.append(cve_date)
                    
            except (json.JSONDecodeError, KeyError, ValueError):
                # Skip malformed or incomplete CVE entries
                logger.debug(f"Skipping malformed CVE file: {json_file}")
                continue
        
        logger.info(f"Successfully parsed {len(cve_dates)} CVE publication dates")
        
        # Check if we found any valid dates
        if len(cve_dates) == 0:
            self._handle_no_dates_found(json_files)
            
        # Convert to DataFrame with monthly aggregation
        return self._aggregate_monthly_data(cve_dates)
    
    def _extract_cve_date(self, json_file: Path) -> Optional[datetime.date]:
        """
        Extract publication date from a single CVE JSON file.
        
        Args:
            json_file: Path to the CVE JSON file
            
        Returns:
            Publication date if found, None otherwise
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            cve_data = json.load(f)
        
        # Filter out rejected CVEs - check state array for "REJECTED"
        if self._is_cve_rejected(cve_data):
            return None
        
        # Try multiple methods to extract publication date
        published_date = self._find_publication_date(cve_data)
        
        if published_date:
            return self._parse_date_string(published_date)
        
        return None
    
    def _is_cve_rejected(self, cve_data: Dict[str, Any]) -> bool:
        """
        Check if a CVE is in rejected state.
        
        Args:
            cve_data: Parsed CVE data dictionary
            
        Returns:
            True if CVE is rejected, False otherwise
        """
        if 'cveMetadata' in cve_data and 'state' in cve_data['cveMetadata']:
            state = cve_data['cveMetadata']['state']
            if isinstance(state, list) and 'REJECTED' in state:
                return True
            elif isinstance(state, str) and state == 'REJECTED':
                return True
        return False
    
    def _find_publication_date(self, cve_data: Dict[str, Any]) -> Optional[str]:
        """
        Find publication date from CVE data using multiple fallback methods.
        
        Args:
            cve_data: Parsed CVE data dictionary
            
        Returns:
            Publication date string if found, None otherwise
        """
        # Method 1: cveMetadata.datePublished (primary field in schema)
        if 'cveMetadata' in cve_data and 'datePublished' in cve_data['cveMetadata']:
            return cve_data['cveMetadata']['datePublished']
        
        # Method 2: cveMetadata.dateReserved (fallback - reservation date)
        elif 'cveMetadata' in cve_data and 'dateReserved' in cve_data['cveMetadata']:
            return cve_data['cveMetadata']['dateReserved']
        
        # Method 3: containers.cna.datePublic (alternative publication date)
        elif ('containers' in cve_data and 'cna' in cve_data['containers'] and 
              'datePublic' in cve_data['containers']['cna']):
            return cve_data['containers']['cna']['datePublic']
        
        # Method 4: Legacy formats (for older CVE files)
        elif 'publishedDate' in cve_data:
            return cve_data['publishedDate']
        elif 'Published_Date' in cve_data:
            return cve_data['Published_Date']
        
        return None
    
    def _parse_date_string(self, date_string: str) -> Optional[datetime.date]:
        """
        Parse date string using multiple format attempts.
        
        Args:
            date_string: Date string to parse
            
        Returns:
            Parsed date object if successful, None otherwise
        """
        try:
            # Handle different date formats
            if 'T' in date_string:
                # ISO format with time (RFC3339/ISO8601)
                date_obj = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
            else:
                # Date only format
                date_obj = datetime.strptime(date_string, '%Y-%m-%d')
            
            return date_obj.date()
            
        except ValueError:
            # Try other common date formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    date_obj = datetime.strptime(date_string, fmt)
                    return date_obj.date()
                except ValueError:
                    continue
        
        logger.debug(f"Could not parse date string: {date_string}")
        return None
    
    def _handle_no_dates_found(self, json_files: List[Path]) -> None:
        """
        Handle the case where no valid CVE dates were found.
        
        Args:
            json_files: List of JSON files that were processed
        """
        logger.error("No valid CVE publication dates found!")
        logger.error("This could be due to:")
        logger.error("1. CVE files don't contain 'cveMetadata.datePublished' fields")
        logger.error("2. Date format is different than expected")
        logger.error("3. CVE files are malformed or empty")
        
        # Examine a few files to understand the structure
        logger.info("\nExamining first few CVE files for debugging...")
        for i, json_file in enumerate(json_files[:3]):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    cve_data = json.load(f)
                logger.info(f"\nFile {i+1}: {json_file.name}")
                logger.info(f"Keys: {list(cve_data.keys())}")
                
                if 'cveMetadata' in cve_data:
                    logger.info(f"cveMetadata keys: {list(cve_data['cveMetadata'].keys())}")
                    
            except Exception as e:
                logger.error(f"Error examining file {json_file}: {e}")
    
    def _aggregate_monthly_data(self, cve_dates: List[datetime.date]) -> pd.DataFrame:
        """
        Aggregate CVE dates into monthly counts.
        
        Args:
            cve_dates: List of CVE publication dates
            
        Returns:
            DataFrame with monthly CVE counts
        """
        # Create DataFrame from dates
        df = pd.DataFrame({'date': cve_dates})
        
        # Convert to datetime and extract year-month
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter to only include CVEs published after December 31st, 2016
        cutoff_date = pd.Timestamp('2017-01-01')
        original_count = len(df)
        df = df[df['date'] >= cutoff_date]
        filtered_count = len(df)
        
        logger.info(f"Filtered dataset: {original_count} -> {filtered_count} CVEs (removed {original_count - filtered_count} pre-2017 CVEs)")
        
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Group by month and count CVEs
        monthly_counts = df.groupby('year_month').size().reset_index(name='cve_count')
        monthly_counts['date'] = monthly_counts['year_month'].dt.to_timestamp()
        
        # Sort by date
        monthly_counts = monthly_counts.sort_values('date').reset_index(drop=True)
        
        # Keep only date and cve_count columns
        monthly_counts = monthly_counts[['date', 'cve_count']]
        
        logger.info(f"Aggregated data into {len(monthly_counts)} monthly periods")
        logger.info(f"Date range: {monthly_counts['date'].min()} to {monthly_counts['date'].max()}")
        
        return monthly_counts
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the quality of parsed CVE data.
        
        Args:
            data: CVE data DataFrame
            
        Returns:
            Dictionary with data quality metrics
        """
        quality_metrics = {
            'total_records': len(data),
            'date_range': {
                'start': data['date'].min().strftime('%Y-%m-%d') if len(data) > 0 else None,
                'end': data['date'].max().strftime('%Y-%m-%d') if len(data) > 0 else None
            },
            'total_cves': data['cve_count'].sum() if len(data) > 0 else 0,
            'average_monthly_cves': data['cve_count'].mean() if len(data) > 0 else 0,
            'missing_dates': data['date'].isnull().sum(),
            'zero_count_months': (data['cve_count'] == 0).sum()
        }
        
        # Log quality metrics
        logger.info("Data Quality Metrics:")
        logger.info(f"  Total records: {quality_metrics['total_records']}")
        logger.info(f"  Date range: {quality_metrics['date_range']['start']} to {quality_metrics['date_range']['end']}")
        logger.info(f"  Total CVEs: {quality_metrics['total_cves']}")
        logger.info(f"  Average monthly CVEs: {quality_metrics['average_monthly_cves']:.2f}")
        
        return quality_metrics
