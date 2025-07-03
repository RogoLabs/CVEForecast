# CVE Forecast Script Updates - cvelistV5 Integration

## Summary of Changes

Successfully updated the `cve_forecast.py` script and test scripts to automatically use the `cvelistV5` directory in the project root as the default data source.

## Changes Made

### 1. Main CVE Forecast Script (`cve_forecast.py`)

**Updated `parse_cve_data()` method:**
- Now automatically defaults to `cvelistV5` directory in the project root when no data path is provided
- Improved error handling with helpful instructions when CVE data is not found
- Provides clear guidance to run `python download_data.py` if data is missing

**Updated command-line arguments:**
- Made `--data-path` argument optional (was previously required)
- Updated help text to reflect that it defaults to `cvelistV5` in project root
- Users can still override the data path if needed

### 2. Test Scripts Updates

**Updated `tests/test_forecast.py`:**
- Removed explicit `--data-path` argument from forecast command
- Simplified the script to rely on the default behavior

**Updated `tests/validate_forecast.py`:**
- Removed explicit `--data-path` argument from validation command
- Simplified the script to rely on the default behavior

## Key Benefits

1. **Simplified Usage**: Users no longer need to specify `--data-path` manually
2. **Consistent Behavior**: All scripts now use the same default data location
3. **Better Error Messages**: Clear instructions when data is missing
4. **Backward Compatibility**: Users can still override with `--data-path` if needed

## New Usage Examples

### Simple Usage (Recommended)
```bash
# Download CVE data (one time)
python download_data.py

# Run forecast with default settings
python cve_forecast.py

# Run with custom output location
python cve_forecast.py --output my_output.json
```

### Advanced Usage (Override data path)
```bash
# Use custom data location
python cve_forecast.py --data-path /path/to/other/cve/data

# Use custom data location and output
python cve_forecast.py --data-path /path/to/other/cve/data --output my_output.json
```

### Test Scripts
```bash
# Test scripts now work automatically
python tests/test_forecast.py
python tests/validate_forecast.py
```

## File Structure

```
CVEForecast/
├── cve_forecast.py          # Updated: Auto-uses cvelistV5/
├── download_data.py         # Downloads data to cvelistV5/
├── tests/
│   ├── test_forecast.py     # Updated: Simplified commands
│   └── validate_forecast.py # Updated: Simplified commands
├── cvelistV5/               # Default data directory
│   └── cves/                # CVE JSON files
└── web/
    └── data.json            # Default output location
```

## Status: ✅ COMPLETE

The CVE forecast script and all test scripts have been successfully updated to use the `cvelistV5` directory as the default data source. The integration is complete and working correctly.
