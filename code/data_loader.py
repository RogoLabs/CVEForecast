import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

def load_cve_data(config: dict) -> pd.DataFrame:
    """
    Parses CVE JSON files, aggregates them by month, and returns a DataFrame.

    Args:
        config: The application configuration dictionary.

    Returns:
        A pandas DataFrame with monthly CVE counts.
    """
    cve_data_path = Path(config['file_paths']['cve_data'])
    data_processing_config = config.get('data_processing', {})
    filter_by_date = data_processing_config.get('filter_by_date', False)
    start_date_filter = data_processing_config.get('start_date_filter')
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting CVE data processing from: {cve_data_path}")

    if not cve_data_path.exists():
        logger.error(f"CVE data directory not found at '{cve_data_path}'. Please check your config.json.")
        raise FileNotFoundError(f"Directory not found: {cve_data_path}")

    json_files = list(cve_data_path.rglob("cves/**/*.json"))
    logger.info(f"Found {len(json_files)} CVE JSON files to process.")

    cve_dates = []
    for i, json_file in enumerate(json_files):
        if (i + 1) % 50000 == 0:
            logger.info(f"  ... processed {i + 1}/{len(json_files)} files")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cve_items = data if isinstance(data, list) else [data]

            for cve_item in cve_items:
                if not isinstance(cve_item, dict):
                    continue

                # Skip rejected CVEs
                cve_metadata = cve_item.get('cveMetadata', {})
                if cve_metadata:
                    state_info = cve_metadata.get('state')
                    if isinstance(state_info, str) and state_info == 'REJECTED':
                        continue
                    elif isinstance(state_info, dict) and state_info.get('state') == 'REJECTED':
                        continue
                    elif isinstance(state_info, list) and any(s.get('state') == 'REJECTED' for s in state_info):
                        continue

                # Extract publication date
                published_date_str = cve_metadata.get('datePublished')
                if published_date_str:
                    try:
                        dt_obj = datetime.fromisoformat(published_date_str.replace('Z', '+00:00'))
                        cve_dates.append(dt_obj.date())
                    except ValueError:
                        logger.debug(f"Could not parse date '{published_date_str}' in {json_file.name}")

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.warning(f"Skipping malformed file {json_file.name}: {e}")
            continue
    
    logger.info(f"Successfully extracted dates from {len(cve_dates)} valid CVEs.")

    if not cve_dates:
        logger.error("No valid CVE publication dates were found. Aborting.")
        return pd.DataFrame()

    # Aggregate data monthly
    df = pd.DataFrame({'date': cve_dates})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    monthly_counts = df.resample('ME').size().to_frame('cve_count')

    # Ensure all months are present in the range, filling missing ones with 0
    if not monthly_counts.empty:
        start_date, end_date = monthly_counts.index.min(), monthly_counts.index.max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
        monthly_counts = monthly_counts.reindex(full_date_range, fill_value=0)

    # Conditionally filter by start date
    if filter_by_date and start_date_filter:
        logger.info(f"Applying date filter. Keeping data from {start_date_filter} onwards.")
        monthly_counts = monthly_counts[monthly_counts.index >= pd.to_datetime(start_date_filter)]
    else:
        logger.info("No date filter applied. Processing all historical data.")
    
    if not monthly_counts.empty:
        logger.info(f"Aggregated data into {len(monthly_counts)} monthly periods from {monthly_counts.index.min().strftime('%Y-%m')} to {monthly_counts.index.max().strftime('%Y-%m')}.")
    else:
        logger.warning("Resulting DataFrame is empty after processing and filtering.")

    return monthly_counts
