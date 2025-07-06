"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import tempfile
import os
import sys
from unittest.mock import Mock

# Add source directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'long_fact_probes'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


@pytest.fixture
def sample_factoids():
    """Sample factoid data for testing."""
    return [
        {'atom': 'The sky is blue', 'is_supported': 1},
        {'atom': 'The grass is purple', 'is_supported': 0},
        {'atom': 'Water boils at 100°C', 'is_supported': 1},
        {'atom': 'Elephants can fly', 'is_supported': 0}
    ]


@pytest.fixture
def sample_predictions():
    """Sample prediction data for testing."""
    return {
        'y_true': [1, 0, 1, 0, 1, 1, 0, 0],
        'y_pred_proba': [0.8, 0.2, 0.9, 0.1, 0.7, 0.6, 0.3, 0.4]
    }


@pytest.fixture
def temp_directory():
    """Temporary directory fixture."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.device = 'cpu'
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.eos_token_id = 2
    return tokenizer 