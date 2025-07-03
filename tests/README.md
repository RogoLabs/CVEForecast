# CVE Forecast Testing Scripts

This directory contains various testing and development scripts for the CVE Forecast project. All scripts use real CVE data from the official CVElistV5 repository.

## Scripts Overview

### 1. `test_forecast.py` - Full Integration Test
**Purpose**: Downloads complete CVE data and runs the full forecast pipeline locally.

**Usage**:
```bash
cd tests
python test_forecast.py
```

**Features**:
- Downloads the complete CVE data repository (shallow clone)
- Runs the full forecast analysis on all available data
- Offers to start the development server
- Provides cleanup options for downloaded data
- Interactive prompts for user control

**When to use**: 
- Testing the complete pipeline with full real data
- Final validation before deployment
- Performance testing with large datasets

---

### 2. `generate_sample_data.py` - CVE Data Subset Generator
**Purpose**: Downloads real CVE data and creates a focused subset for faster development.

**Usage**:
```bash
cd tests
python generate_sample_data.py
```

**Features**:
- Downloads the complete CVE repository
- Creates a subset using CVEs from the last 6 months (configurable)
- Runs forecast analysis on the subset
- Significantly faster than full data processing
- Uses real CVE data patterns and distributions

**When to use**:
- Daily development and testing
- Faster iteration cycles
- Testing with realistic data patterns
- When you need real data but not the full processing time

---

### 3. `validate_forecast.py` - Quick Validation Test
**Purpose**: Downloads real CVE data and performs quick validation of the forecast system.

**Usage**:
```bash
cd tests
python validate_forecast.py
```

**Features**:
- Downloads complete CVE data repository
- Validates forecast script execution
- Checks output file structure and content
- Uses real CVE data for authentic testing
- Automatic cleanup of test data

**When to use**:
- Quick validation of code changes
- Automated testing in CI/CD
- Debugging forecast script issues
- Ensuring system works with real data structure

---

## Usage Recommendations

### For Local Development
1. **Start with subset**: Run `generate_sample_data.py` for development with real data but faster processing
2. **Full testing**: Use `test_forecast.py` when you need complete validation with all data
3. **Quick validation**: Run `validate_forecast.py` to ensure changes don't break core functionality

### For CI/CD
- Use `validate_forecast.py` in automated testing pipelines
- Real data validation ensures production compatibility
- Comprehensive but reasonably fast execution

### For Development Workflow
1. **Daily work**: Use `generate_sample_data.py` (6-month subset)
2. **Pre-commit**: Run `validate_forecast.py` 
3. **Pre-deployment**: Run `test_forecast.py` (full data)

## Data Characteristics

All scripts work with real CVE data:
- **Authentic patterns**: Real publication frequency and distribution
- **Actual format**: Official CVE JSON structure from CVElistV5
- **Current data**: Latest available CVE records
- **Representative samples**: Subset scripts maintain data authenticity

## Performance Notes

| Script | Data Scope | Typical Runtime | Use Case |
|--------|------------|----------------|----------|
| `validate_forecast.py` | Full repo | 10-20 minutes | Validation |
| `generate_sample_data.py` | 6-month subset | 3-8 minutes | Development |
| `test_forecast.py` | Full repo | 15-30 minutes | Integration |

*Runtime varies based on internet speed and system performance*

## Requirements

All scripts require the same dependencies as the main project:
```bash
pip install -r ../requirements.txt
```

Additional requirements for sample data generation:
- `numpy` and `pandas` (included in main requirements)

## Directory Structure

```
tests/
├── README.md                    # This file
├── test_forecast.py            # Full integration test
├── generate_sample_data.py     # Sample data generator  
├── validate_forecast.py        # Unit testing script
└── cve_data_local/            # Created by test_forecast.py (gitignored)
```

## Notes

- All scripts are designed to be run from within the `tests/` directory
- Test data and temporary files are automatically cleaned up
- Scripts provide interactive prompts for user control
- All output goes to the main `web/data.json` file for consistency

## Troubleshooting

**Import errors**: Make sure you've installed the requirements with `pip install -r ../requirements.txt`

**Permission errors**: Ensure you have write permissions in the project directory

**Git clone failures**: Check your internet connection and git installation for `test_forecast.py`

**Path issues**: Always run scripts from within the `tests/` directory
