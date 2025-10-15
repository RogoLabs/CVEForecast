#!/bin/bash
# Verify Production Run Quality
# 
# This script runs the production forecast and checks for warnings/errors
# Expected: 0 warnings, 0 errors

# Get to project root (two directories up from code/utils)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

echo "=================================================="
echo "  CVE FORECAST PRODUCTION RUN VERIFICATION"
echo "=================================================="
echo ""

# Create temp log file
LOG_FILE="verify_run_$(date +%Y%m%d_%H%M%S).log"

echo "Running production forecast..."
python3 code/run_production_forecast.py > "$LOG_FILE" 2>&1

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ FAILED: Exit code $EXIT_CODE"
    exit 1
fi

# Count warnings and errors
WARNING_COUNT=$(grep -c "WARNING" "$LOG_FILE" || echo 0)
ERROR_COUNT=$(grep -c "ERROR" "$LOG_FILE" || echo 0)

echo ""
echo "=================================================="
echo "  RESULTS"
echo "=================================================="
echo "Exit Code: $EXIT_CODE"
echo "Warnings:  $WARNING_COUNT"
echo "Errors:    $ERROR_COUNT"
echo "Log File:  $LOG_FILE"
echo ""

# Verify outputs exist
if [ -f "web/data.json" ] && [ -f "web/cna_data.json" ]; then
    echo "✅ Output files created successfully"
    echo "   - web/data.json: $(wc -l < web/data.json) lines"
    echo "   - web/cna_data.json: $(wc -l < web/cna_data.json) lines"
else
    echo "❌ Output files missing!"
    exit 1
fi

echo ""

# Final verdict
if [ $WARNING_COUNT -eq 0 ] && [ $ERROR_COUNT -eq 0 ]; then
    echo "🎉 SUCCESS: Clean run with zero warnings and zero errors!"
    exit 0
else
    echo "⚠️  QUALITY ISSUE: Found $WARNING_COUNT warnings and $ERROR_COUNT errors"
    echo ""
    echo "Recent warnings/errors:"
    grep -E "WARNING|ERROR" "$LOG_FILE" | tail -10
    exit 1
fi
