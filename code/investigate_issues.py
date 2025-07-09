#!/usr/bin/env python3
"""
Investigate restored performance and February 2025 data cutoff bug.
"""

import json

def investigate_issues():
    print('🔍 INVESTIGATING RESTORED PERFORMANCE + FEBRUARY 2025 DATA BUG')
    print('=' * 70)

    # Check current performance metrics
    with open('../web/data.json', 'r') as f:
        data = json.load(f)

    print('📊 CURRENT MODEL PERFORMANCE:')
    model_rankings = data['model_rankings']
    for i, model in enumerate(model_rankings[:5], 1):
        name = model['model_name']
        mape = model['mape']
        print(f'  {i}. {name}: {mape:.4f}% MAPE')
        if i <= 3:  # Check if top 3 models show optimized performance
            if mape < 5.0:  # Should be ~2.67% if optimized
                print(f'     ✅ OPTIMIZED performance detected!')
            else:
                print(f'     ❌ Still showing non-optimized performance')

    print(f'\n🗓️ INVESTIGATING FEBRUARY 2025 DATA CUTOFF BUG:')
    print('=' * 50)

    # Check historical data range
    if 'historical_data' in data:
        historical = data['historical_data']
        print(f'📅 Historical data points: {len(historical)}')
        if historical:
            first_date = historical[0]['date']
            last_date = historical[-1]['date']
            print(f'📅 Date range: {first_date} to {last_date}')
            
            # Check if data stops at February 2025
            feb_2025_found = any(item['date'] == '2025-02' for item in historical)
            mar_2025_found = any(item['date'] == '2025-03' for item in historical)
            apr_2025_found = any(item['date'] == '2025-04' for item in historical)
            may_2025_found = any(item['date'] == '2025-05' for item in historical)
            jun_2025_found = any(item['date'] == '2025-06' for item in historical)
            
            print(f'📅 February 2025 data: {"✅ Found" if feb_2025_found else "❌ Missing"}')
            print(f'📅 March 2025 data: {"✅ Found" if mar_2025_found else "❌ Missing"}')
            print(f'📅 April 2025 data: {"✅ Found" if apr_2025_found else "❌ Missing"}')
            print(f'📅 May 2025 data: {"✅ Found" if may_2025_found else "❌ Missing"}')
            print(f'📅 June 2025 data: {"✅ Found" if jun_2025_found else "❌ Missing"}')
            
            # Show last few data points
            print(f'\n📋 Last 5 historical data points:')
            for item in historical[-5:]:
                print(f'   {item["date"]}: {item["cve_count"]:,} CVEs')
                
        else:
            print('❌ No historical data found')
    else:
        print('❌ No historical_data section found')
        
    # Check current month data
    if 'current_month_actual' in data:
        current_month = data['current_month_actual']
        print(f'\n📅 Current month data: {current_month.get("date", "Unknown")}')
        print(f'📊 Current month CVEs: {current_month.get("cve_count", "Unknown")}')
    else:
        print('\n❌ No current_month_actual section found')

if __name__ == '__main__':
    investigate_issues()
