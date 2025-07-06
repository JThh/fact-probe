# Test Suite Documentation

This directory contains the test suite for the Long-Form Fact Probe project.

## Overview

The test suite is designed to ensure the reliability and correctness of the hallucination detection framework. It includes unit tests, integration tests, and utilities for testing the core components.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                # Pytest configuration and fixtures
├── test_utils.py              # Tests for utility functions
├── test_integration.py        # Integration tests for main workflow
└── README.md                 # This file
```

## Running Tests

### Prerequisites

Install the test dependencies:
```bash
pip install pytest pytest-cov pytest-mock
```

### Running All Tests

```bash
# From the repository root
python run_tests.py

# Or directly with pytest
pytest tests/ --verbose --cov=long_fact_probes --cov=scripts
```

### Running Specific Tests

```bash
# Run only utility tests
pytest tests/test_utils.py -v

# Run only integration tests
pytest tests/test_integration.py -v

# Run tests with specific markers
pytest -m "not slow" -v
```

### Coverage Reports

Generate HTML coverage reports:
```bash
pytest tests/ --cov=long_fact_probes --cov-report=html
# Open htmlcov/index.html in browser
```

## Test Components

### Unit Tests (`test_utils.py`)

Tests individual functions and classes:
- `auroc()` function correctness
- `bootstrap_func()` statistical reliability
- Edge case handling
- Input validation

### Integration Tests (`test_integration.py`)

Tests component interactions:
- Data processing pipeline
- Classifier training workflow
- Evaluation pipeline
- Model configuration consistency
- Cross-format data handling

### Fixtures (`conftest.py`)

Shared test data and utilities:
- `sample_factoids`: Sample fact data for testing
- `sample_predictions`: Mock prediction results
- `temp_directory`: Temporary file system for tests
- `mock_model`/`mock_tokenizer`: Mock objects for model testing

## Test Data

Test data is generated programmatically to ensure:
- Reproducibility across environments
- Coverage of edge cases
- Independence from external data sources
- Fast test execution

## Continuous Integration

The test suite runs automatically on:
- Push to main/develop branches
- Pull requests
- Multiple Python versions (3.8, 3.9, 3.10)

See `.github/workflows/tests.yml` for CI configuration.

## Writing New Tests

### Guidelines

1. **Test Naming**: Use descriptive names with `test_` prefix
2. **Fixtures**: Use shared fixtures for common test data
3. **Mocking**: Mock external dependencies (models, APIs)
4. **Assertions**: Use clear, specific assertions
5. **Documentation**: Include docstrings explaining test purpose

### Example Test

```python
def test_new_function():
    """Test description of what this verifies."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result.shape == expected_shape
    assert all(x > 0 for x in result)
```

### Adding Integration Tests

For new components, add integration tests that verify:
- Component interactions
- End-to-end workflows
- Data flow consistency
- Error handling

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `long_fact_probes` is in Python path
2. **Missing Dependencies**: Install test requirements
3. **Model Loading**: Tests should mock model loading for speed
4. **Random Seeds**: Use fixed seeds for reproducible tests

### Debug Mode

Run tests with detailed output:
```bash
pytest tests/ -v -s --tb=long
```

### Performance

For slow tests, use markers:
```python
@pytest.mark.slow
def test_expensive_operation():
    # Test that takes significant time
    pass
```

Run only fast tests:
```bash
pytest -m "not slow"
```

## Contributing

When adding new functionality:
1. Write tests first (TDD approach)
2. Ensure >90% code coverage
3. Test both success and failure cases
4. Add integration tests for new workflows
5. Update this documentation if needed 