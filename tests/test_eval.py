"""Unit tests for eval.py module."""

import pytest
import numpy as np
import tempfile
import os
import sys
import pickle
from unittest.mock import Mock, patch

# Add the long_fact_probes directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'long_fact_probes'))

from eval import (
    flatten_scores, load_fact_scores, load_probes,
    MODEL_CONFIGS, setup_logging
)


def test_load_probes_empty_directory():
    """Test load_probes with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_probes(tmpdir)
        assert result == {}


class MockProbe:
    """Simple mock probe class that can be pickled."""
    def __init__(self):
        self.test_attribute = "test_value"


def test_load_probes_with_valid_files():
    """Test load_probes with valid pickle files."""
    test_data = {'probe': MockProbe(), 'layer_group': (0, 5)}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test pickle file
        test_file = os.path.join(tmpdir, 'test_probe.pkl')
        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        result = load_probes(tmpdir)
        assert 'test_probe.pkl' in result
        assert result['test_probe.pkl']['layer_group'] == (0, 5)


def test_load_probes_with_invalid_files():
    """Test load_probes handles invalid files gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an invalid file
        invalid_file = os.path.join(tmpdir, 'invalid.pkl')
        with open(invalid_file, 'w') as f:
            f.write('not a pickle file')
        
        result = load_probes(tmpdir)
        assert result == {}


def test_model_configs_consistency():
    """Test that MODEL_CONFIGS is consistent across modules."""
    expected_keys = ['name', 'hf_name', 'num_layers']
    for model, config in MODEL_CONFIGS.items():
        for key in expected_keys:
            assert key in config, f"Missing {key} in {model} config"
        assert isinstance(config['num_layers'], int)
        assert config['num_layers'] > 0 