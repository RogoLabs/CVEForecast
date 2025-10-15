"""
CNA trend data utilities for external indicators.
Fetches official CNA list and extracts growth trends over time.
"""
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import json
import os


class CNATrendData:
    """Fetch and process CNA count trends from official CVE Project data."""
    
    OFFICIAL_CNA_LIST_URL = "https://raw.githubusercontent.com/CVEProject/cve-website/dev/src/assets/data/CNAsList.json"
    CACHE_FILE = "cna_count_cache.json"
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize CNA trend data fetcher."""
        self.logger = logger or logging.getLogger(__name__)
        self.cna_data = None
        
    def fetch_cna_list(self, use_cache: bool = True) -> List[Dict]:
        """
        Fetch official CNA list from CVE Project.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            List of CNA dictionaries
        """
        cache_path = os.path.join(os.path.dirname(__file__), self.CACHE_FILE)
        
        # Try cache first
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                    # Check if cache is less than 24 hours old
                    cache_time = datetime.fromisoformat(cache_data['timestamp'])
                    if (datetime.now() - cache_time).total_seconds() < 86400:
                        self.logger.info("Using cached CNA list data")
                        self.cna_data = cache_data['data']
                        return self.cna_data
            except Exception as e:
                self.logger.warning(f"Cache read failed: {e}")
        
        # Fetch from official source
        try:
            self.logger.info(f"Fetching CNA list from {self.OFFICIAL_CNA_LIST_URL}")
            response = requests.get(self.OFFICIAL_CNA_LIST_URL, timeout=10)
            response.raise_for_status()
            self.cna_data = response.json()
            
            # Cache the data
            try:
                with open(cache_path, 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'data': self.cna_data
                    }, f)
                self.logger.info(f"Cached CNA list data ({len(self.cna_data)} CNAs)")
            except Exception as e:
                self.logger.warning(f"Cache write failed: {e}")
                
            return self.cna_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch CNA list: {e}")
            return []
    
    def extract_join_dates(self) -> pd.DataFrame:
        """
        Extract CNA join dates from cnaID field.
        
        Returns:
            DataFrame with columns: shortName, cnaID, join_year, join_quarter
        """
        if not self.cna_data:
            self.fetch_cna_list()
            
        records = []
        for cna in self.cna_data:
            cna_id = cna.get('cnaID', '')
            short_name = cna.get('shortName', '')
            
            # Extract year from cnaID (format: CNA-YYYY-NNNN)
            if cna_id.startswith('CNA-'):
                try:
                    parts = cna_id.split('-')
                    year = int(parts[1])
                    sequence = int(parts[2])
                    
                    # Estimate quarter based on sequence number (rough approximation)
                    # Assuming ~100-150 CNAs per year, divide into quarters
                    quarter = min(4, (sequence // 25) + 1)
                    
                    records.append({
                        'shortName': short_name,
                        'cnaID': cna_id,
                        'join_year': year,
                        'join_quarter': quarter,
                        'organizationName': cna.get('organizationName', ''),
                        'country': cna.get('country', 'Unknown')
                    })
                except (IndexError, ValueError) as e:
                    self.logger.warning(f"Could not parse cnaID: {cna_id} - {e}")
        
        return pd.DataFrame(records)
    
    def get_monthly_cna_counts(self, start_year: int = 2015) -> pd.DataFrame:
        """
        Generate monthly cumulative CNA counts.
        
        Args:
            start_year: Year to start counting from
            
        Returns:
            DataFrame with columns: date, cna_count, new_cnas
        """
        df = self.extract_join_dates()
        
        if df.empty:
            self.logger.warning("No CNA data available")
            return pd.DataFrame()
        
        # Generate monthly date range
        start_date = pd.Timestamp(f'{start_year}-01-01')
        end_date = pd.Timestamp.now().replace(day=1)
        months = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        counts = []
        for date in months:
            year = date.year
            month = date.month
            quarter = (month - 1) // 3 + 1
            
            # Count CNAs that joined before or during this month
            # For simplicity, assume all CNAs in a quarter joined in the first month
            if month % 3 == 1:  # First month of quarter
                mask = (df['join_year'] < year) | \
                       ((df['join_year'] == year) & (df['join_quarter'] <= quarter))
            else:  # Other months in quarter, use previous quarter
                prev_quarter = quarter if month % 3 != 1 else quarter - 1
                mask = (df['join_year'] < year) | \
                       ((df['join_year'] == year) & (df['join_quarter'] <= prev_quarter))
            
            cumulative_count = mask.sum()
            counts.append({
                'date': date,
                'cna_count': cumulative_count
            })
        
        result = pd.DataFrame(counts)
        
        # Calculate new CNAs per month
        result['new_cnas'] = result['cna_count'].diff().fillna(0)
        
        return result
    
    def get_cna_count_for_date(self, date: pd.Timestamp) -> int:
        """
        Get CNA count for a specific date.
        
        Args:
            date: Date to get count for
            
        Returns:
            Number of CNAs active at that date
        """
        monthly_counts = self.get_monthly_cna_counts()
        if monthly_counts.empty:
            return 0
        
        # Find closest date
        mask = monthly_counts['date'] <= date
        if mask.any():
            return int(monthly_counts.loc[mask, 'cna_count'].iloc[-1])
        
        return 0
    
    def get_growth_rate(self, months: int = 12) -> float:
        """
        Calculate CNA growth rate over recent period.
        
        Args:
            months: Number of months to calculate growth over
            
        Returns:
            Growth rate as decimal (e.g., 0.15 = 15%)
        """
        monthly_counts = self.get_monthly_cna_counts()
        if len(monthly_counts) < months:
            return 0.0
        
        current = monthly_counts.iloc[-1]['cna_count']
        past = monthly_counts.iloc[-months]['cna_count']
        
        if past == 0:
            return 0.0
        
        return (current - past) / past
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics about CNA growth.
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.extract_join_dates()
        monthly_counts = self.get_monthly_cna_counts()
        
        if df.empty or monthly_counts.empty:
            return {}
        
        current_count = int(monthly_counts.iloc[-1]['cna_count'])
        growth_12m = self.get_growth_rate(12)
        growth_24m = self.get_growth_rate(24)
        
        return {
            'current_cna_count': current_count,
            'total_cnas_ever': len(df),
            'cnas_by_year': df['join_year'].value_counts().sort_index().to_dict(),
            'growth_rate_12m': round(growth_12m * 100, 2),
            'growth_rate_24m': round(growth_24m * 100, 2),
            'avg_new_cnas_per_month': round(monthly_counts['new_cnas'].mean(), 1),
            'countries': df['country'].value_counts().head(10).to_dict()
        }


def calculate_cna_momentum(logger: logging.Logger = None) -> Tuple[float, Dict]:
    """
    Calculate CNA growth momentum as an external indicator.
    
    Returns:
        Tuple of (momentum_score, summary_dict)
        momentum_score: 0.0-1.0 indicating CNA expansion velocity
    """
    cna_data = CNATrendData(logger)
    cna_data.fetch_cna_list()
    
    stats = cna_data.get_summary_stats()
    
    if not stats:
        return 0.5, {}  # Neutral momentum if no data
    
    # Calculate momentum based on recent growth
    growth_12m = stats.get('growth_rate_12m', 0) / 100
    growth_24m = stats.get('growth_rate_24m', 0) / 100
    
    # Weighted average favoring recent growth
    momentum = (growth_12m * 0.7 + growth_24m * 0.3)
    
    # Normalize to 0-1 scale (assume 0-30% growth range)
    momentum_score = min(1.0, max(0.0, momentum / 0.30))
    
    return momentum_score, stats


if __name__ == '__main__':
    # Test the CNA trend data fetcher
    logging.basicConfig(level=logging.INFO)
    
    cna_data = CNATrendData()
    cna_data.fetch_cna_list()
    
    print("\n" + "="*60)
    print("CNA TREND DATA TEST")
    print("="*60)
    
    stats = cna_data.get_summary_stats()
    print(f"\n📊 Current CNA Count: {stats['current_cna_count']}")
    print(f"📈 12-Month Growth: {stats['growth_rate_12m']}%")
    print(f"📈 24-Month Growth: {stats['growth_rate_24m']}%")
    print(f"➕ Avg New CNAs/Month: {stats['avg_new_cnas_per_month']}")
    
    print("\n📅 CNAs by Year:")
    for year, count in sorted(stats['cnas_by_year'].items())[-10:]:
        print(f"  {year}: {count} new CNAs")
    
    print("\n🌍 Top 10 Countries:")
    for country, count in list(stats['countries'].items())[:10]:
        print(f"  {country}: {count} CNAs")
    
    momentum, _ = calculate_cna_momentum()
    print(f"\n🚀 CNA Momentum Score: {momentum:.2f} (0.0-1.0 scale)")
    
    print("\n✅ CNA trend data test complete!")
