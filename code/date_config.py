#!/usr/bin/env python3
"""
Centralized date configuration loader for CVE Forecast system.
Ensures consistent date handling across all Python scripts.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DateConfig:
    """Centralized date configuration manager."""
    
    def __init__(self, config_path: str = "dates.json"):
        """
        Initialize date configuration loader.
        
        Args:
            config_path: Path to dates.json configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load date configuration from JSON file."""
        try:
            # Try multiple path resolution strategies
            config_path = Path(self.config_path)
            
            # Strategy 1: Direct path
            if not config_path.exists():
                config_path = Path(__file__).parent / self.config_path
                
            # Strategy 2: Try in code subdirectory
            if not config_path.exists():
                config_path = Path('code') / self.config_path
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"✅ Loaded centralized date configuration from {config_path}")
                    return config
            else:
                logger.error(f"❌ Date configuration file not found: {self.config_path}")
                return self._get_fallback_config()
                
        except Exception as e:
            logger.error(f"Failed to load date configuration: {e}")
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Provide fallback configuration if dates.json not found."""
        return {
            "validation_periods": {
                "full_validation_start": "2017-01-01",
                "full_validation_end": "2025-03-31",
                "validation_months_count": 99
            },
            "data_cutoffs": {
                "historical_data_start": "2017-01-01",
                "last_complete_month": "2025-03-31"
            },
            "evaluation_settings": {
                "use_full_validation_period": True,
                "exclude_incomplete_months": True
            },
            "temporal_filters": {
                "filter_current_year_only": False,
                "include_partial_months": False
            }
        }
    
    def get_validation_period(self) -> Dict[str, str]:
        """Get validation period configuration."""
        return self.config.get("validation_periods", {})
    
    def get_data_cutoffs(self) -> Dict[str, str]:
        """Get data cutoff configuration."""
        return self.config.get("data_cutoffs", {})
    
    def get_evaluation_settings(self) -> Dict[str, Any]:
        """Get evaluation settings configuration."""
        return self.config.get("evaluation_settings", {})
    
    def get_temporal_filters(self) -> Dict[str, Any]:
        """Get temporal filtering configuration."""
        return self.config.get("temporal_filters", {})
    
    def should_use_full_validation_period(self) -> bool:
        """Check if full validation period should be used."""
        eval_settings = self.get_evaluation_settings()
        return eval_settings.get("use_full_validation_period", True)
    
    def should_filter_current_year_only(self) -> bool:
        """Check if validation should be filtered to current year only."""
        temporal_filters = self.get_temporal_filters()
        return temporal_filters.get("filter_current_year_only", False)
    
    def get_validation_end_date(self) -> Optional[datetime]:
        """Get validation end date as datetime object."""
        try:
            validation_config = self.get_validation_period()
            end_date_str = validation_config.get("full_validation_end")
            if end_date_str:
                return datetime.strptime(end_date_str, "%Y-%m-%d")
        except Exception as e:
            logger.error(f"Error parsing validation end date: {e}")
        return None
    
    def get_validation_start_date(self) -> Optional[datetime]:
        """Get validation start date as datetime object."""
        try:
            validation_config = self.get_validation_period()
            start_date_str = validation_config.get("full_validation_start")
            if start_date_str:
                return datetime.strptime(start_date_str, "%Y-%m-%d")
        except Exception as e:
            logger.error(f"Error parsing validation start date: {e}")
        return None

# Global instance for easy import
date_config = DateConfig()

def get_date_config() -> DateConfig:
    """Get global date configuration instance."""
    return date_config
