# CVE Forecast Development Guide

**Version**: 0.10 "Phoenix" 🔥🐦  
**Last Updated**: October 2025

## Table of Contents
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Contributing](#contributing)
- [Release Process](#release-process)

## Development Setup

### Prerequisites
- Python 3.10 or higher
- Git
- 8GB+ RAM recommended
- Code editor (VS Code, PyCharm, etc.)

### Initial Setup

```bash
# 1. Fork and clone repository
git clone https://github.com/[your-username]/CVEForecast.git
cd CVEForecast

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# 5. Clone CVE data (for testing)
git clone --depth 1 https://github.com/CVEProject/cvelistV5.git

# 6. Run tests
pytest tests/

# 7. Verify installation
python code/run_production_forecast.py --help
```

### IDE Configuration

**VS Code** (`.vscode/settings.json`):
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true
}
```

**PyCharm**:
- File → Settings → Tools → Python Integrated Tools
- Default test runner: pytest
- Code Style → Python → Use Black formatter

## Project Structure

```
CVEForecast/
├── code/
│   ├── adapters/
│   │   ├── cve_adapter.py       # CVE forecasting implementation
│   │   └── cna_adapter.py       # CNA forecasting implementation
│   ├── core/
│   │   ├── base_forecaster.py   # Abstract base class
│   │   └── validation_mixin.py  # Validation functionality
│   ├── tuner/
│   │   └── comprehensive_tuner.py  # Hyperparameter optimization
│   ├── data_loader.py           # CVE data loading
│   ├── forecast_tracker.py      # Accuracy tracking
│   ├── unified_pipeline.py      # Pipeline orchestration
│   ├── run_production_forecast.py  # Main entry point
│   └── config.json              # Configuration
├── web/
│   ├── index.html               # Main dashboard
│   ├── script.js                # Dashboard JavaScript
│   ├── cna_forecast.html        # CNA dashboard
│   └── technical_details.html   # Documentation
├── docs/
│   ├── ARCHITECTURE.md          # System architecture
│   ├── API_REFERENCE.md         # API documentation
│   ├── DEPLOYMENT.md            # Deployment guide
│   ├── DEVELOPMENT.md           # This file
│   └── TUNING_GUIDE.md          # Tuning instructions
├── tests/
│   ├── test_data_loader.py
│   ├── test_forecasters.py
│   └── test_pipeline.py
├── .github/
│   └── workflows/
│       ├── main.yml             # Daily forecast workflow
│       └── monthly_tuning.yml   # Monthly tuning workflow
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```

## Coding Standards

### Python Style Guide

Follow PEP 8 with these specifics:

**Formatting**:
```python
# Use Black formatter (line length: 100)
black code/ --line-length 100

# Check with flake8
flake8 code/ --max-line-length=100
```

**Type Hints**:
```python
from typing import Dict, List, Optional, Tuple

def forecast_cves(data: pd.DataFrame, horizon: int) -> Dict[str, float]:
    """
    Generate CVE forecasts.
    
    Args:
        data: Historical CVE data
        horizon: Number of periods to forecast
        
    Returns:
        Dictionary mapping dates to predicted values
    """
    pass
```

**Docstrings**:
```python
def complex_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description if needed. Explain what the function does,
    any important details, and edge cases.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        
    Example:
        >>> complex_function("test", 5)
        True
    """
    pass
```

**Naming Conventions**:
```python
# Classes: PascalCase
class CVEForecaster:
    pass

# Functions/methods: snake_case
def load_cve_data():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3

# Private methods: _leading_underscore
def _internal_helper():
    pass
```

### Error Handling

```python
# Specific exceptions
try:
    data = load_cve_data(config)
except FileNotFoundError as e:
    logger.error(f"CVE data not found: {e}")
    raise DataLoadError("Cannot load CVE data") from e
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise

# Graceful degradation
try:
    model.fit(train_data)
except ModelTrainingError:
    logger.warning(f"Model {model_name} failed, continuing with others")
    continue  # Don't fail entire pipeline
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed diagnostic information")
logger.info("General informational messages")
logger.warning("Warning messages for recoverable issues")
logger.error("Error messages for serious problems")
logger.critical("Critical errors that may cause system failure")

# Include context
logger.info(f"Training {model_name} on {len(data)} data points")
logger.error(f"Model {model_name} failed with error: {e}")
```

## Testing

### Test Structure

```python
# tests/test_forecasters.py
import pytest
from adapters.cve_adapter import CVEForecaster

class TestCVEForecaster:
    """Test suite for CVE forecaster."""
    
    @pytest.fixture
    def forecaster(self):
        """Create forecaster instance for testing."""
        config = {...}
        logger = logging.getLogger("test")
        return CVEForecaster(config, logger)
    
    def test_load_data(self, forecaster):
        """Test data loading functionality."""
        data = forecaster.load_data()
        assert len(data) > 0
        assert data.index.is_monotonic_increasing
    
    def test_create_model(self, forecaster):
        """Test model creation."""
        model = forecaster.create_model("Prophet", {})
        assert model is not None
    
    @pytest.mark.slow
    def test_full_pipeline(self, forecaster):
        """Test complete forecasting pipeline."""
        results = forecaster.run_full_pipeline()
        assert results["models_trained"] > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_forecasters.py

# Run specific test
pytest tests/test_forecasters.py::TestCVEForecaster::test_load_data

# Run with coverage
pytest --cov=code --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run with debug output
pytest -s
```

### Test Categories

```python
# Mark slow tests
@pytest.mark.slow
def test_full_pipeline():
    pass

# Mark integration tests
@pytest.mark.integration
def test_end_to_end():
    pass

# Skip tests conditionally
@pytest.mark.skipif(not has_gpu(), reason="Requires GPU")
def test_deep_learning():
    pass
```

### Mocking

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test using mocked dependencies."""
    with patch('data_loader.load_cve_data') as mock_load:
        mock_load.return_value = pd.DataFrame(...)
        
        forecaster = CVEForecaster(config, logger)
        result = forecaster.load_data()
        
        mock_load.assert_called_once()
        assert len(result) > 0
```

## Contributing

### Workflow

1. **Create Issue**
   - Describe the problem or feature
   - Add appropriate labels
   - Discuss approach if needed

2. **Create Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

3. **Make Changes**
   - Write code following style guide
   - Add tests for new functionality
   - Update documentation

4. **Test Locally**
   ```bash
   # Run tests
   pytest
   
   # Check formatting
   black code/ --check
   flake8 code/
   
   # Type checking
   mypy code/
   
   # Run full pipeline
   python code/run_production_forecast.py
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: Add new forecasting model"
   # or
   git commit -m "fix: Correct data loading bug"
   ```

   **Commit Message Format**:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `perf:` Performance improvements
   - `chore:` Maintenance tasks

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   
   Then create Pull Request on GitHub with:
   - Clear description of changes
   - Link to related issue
   - Screenshots if UI changes
   - Test results

7. **Code Review**
   - Address review comments
   - Update PR as needed
   - Ensure CI passes

8. **Merge**
   - Squash and merge (preferred)
   - Delete branch after merge

### Code Review Checklist

**For Authors**:
- [ ] Tests pass locally
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No unnecessary changes
- [ ] Commit messages clear

**For Reviewers**:
- [ ] Code is readable and maintainable
- [ ] Tests are comprehensive
- [ ] No security issues
- [ ] Performance considerations addressed
- [ ] Documentation is accurate

## Release Process

### Version Numbering

Follow Semantic Versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Steps

1. **Update Version**
   ```python
   # code/__init__.py
   __version__ = "1.1.0"
   ```

2. **Update CHANGELOG**
   ```markdown
   ## [1.1.0] - 2025-11-01
   
   ### Added
   - New forecasting model: LSTM
   - API endpoint for programmatic access
   
   ### Changed
   - Improved data loading performance
   
   ### Fixed
   - Bug in CNA forecasting for sparse data
   ```

3. **Update Documentation**
   - README.md version badge
   - Technical details page
   - API documentation

4. **Create Release Branch**
   ```bash
   git checkout -b release/v1.1.0
   ```

5. **Final Testing**
   ```bash
   pytest
   python code/run_production_forecast.py
   ```

6. **Create Tag**
   ```bash
   git tag -a v1.1.0 -m "Release v1.1.0: Feature description"
   git push origin v1.1.0
   ```

7. **Create GitHub Release**
   - Go to Releases → Draft new release
   - Select tag v1.1.0
   - Add release notes
   - Attach artifacts if needed
   - Publish release

8. **Deploy**
   - Automated via GitHub Actions
   - Monitor deployment
   - Verify production

### Hotfix Process

For critical bugs in production:

```bash
# 1. Create hotfix branch from main
git checkout -b hotfix/critical-bug main

# 2. Fix bug and test
# ...

# 3. Update version (patch)
# 1.1.0 → 1.1.1

# 4. Merge to main
git checkout main
git merge hotfix/critical-bug

# 5. Tag and release
git tag -a v1.1.1 -m "Hotfix: Critical bug description"
git push origin v1.1.1

# 6. Delete hotfix branch
git branch -d hotfix/critical-bug
```

## Troubleshooting

### Common Development Issues

**Import Errors**:
```bash
# Ensure code/ is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

# Or use absolute imports
from code.adapters.cve_adapter import CVEForecaster
```

**Test Failures**:
```bash
# Run with verbose output
pytest -v -s

# Run specific failing test
pytest tests/test_file.py::test_name -v

# Check test logs
cat pytest.log
```

**Memory Issues**:
```bash
# Reduce enabled models during development
# Edit config.json: disable heavy models

# Use smaller data subset
# Edit data_loader.py: limit date range
```

---

**Next**: [Architecture Guide](ARCHITECTURE.md) | [API Reference](API_REFERENCE.md)
