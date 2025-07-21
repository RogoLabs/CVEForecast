import os
import json
from datetime import datetime
from pathlib import Path
from collections import Counter

def extract_publish_dates(cve_dir):
    json_files = list(Path(cve_dir).rglob("*.json"))
    publish_dates = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cve_items = data if isinstance(data, list) else [data]
            for cve_item in cve_items:
                if not isinstance(cve_item, dict):
                    continue
                cve_metadata = cve_item.get('cveMetadata', {})
                if cve_metadata.get('state') == 'REJECTED':
                    continue
                published_date_str = cve_metadata.get('datePublished')
                if published_date_str:
                    try:
                        dt_obj = datetime.fromisoformat(published_date_str.replace('Z', '+00:00'))
                        publish_dates.append(dt_obj.date())
                    except Exception:
                        continue
        except Exception:
            continue
    return publish_dates

def cumulative_counts_by_month(publish_dates):
    counts = Counter()
    for d in publish_dates:
        ym = d.replace(day=1)
        counts[ym] += 1
    months = sorted(counts.keys())
    cumulative = []
    total = 0
    last_year = None
    for m in months:
        if last_year is not None and m.year != last_year:
            total = 0  # Reset at Jan 1 each year
        total += counts[m]
        cumulative.append({'date': m.strftime('%Y-%m-%d'), 'cumulative_total': total, 'monthly_total': counts[m]})
        last_year = m.year
    return cumulative

if __name__ == "__main__":
    cve_dir = os.path.join(os.path.dirname(__file__), '../cvelistV5/cves')
    publish_dates = extract_publish_dates(cve_dir)
    cumulative = cumulative_counts_by_month(publish_dates)
    print(f"Total CVEs published: {len(publish_dates)}")
    print("\nMonth-by-month counts for 2025:")
    print("  Month     | Monthly | Cumulative")
    print("------------|---------|-----------")
    for entry in cumulative:
        if entry['date'].startswith('2025-'):
            print(f"{entry['date'][:7]}   | {entry['monthly_total']:7d} | {entry['cumulative_total']:10d}")
