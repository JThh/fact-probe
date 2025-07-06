"""Unit tests for train.py module."""

import pytest
import numpy as np
import pickle
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add the long_fact_probes directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'long_fact_probes'))

from train import (
    flatten_scores, load_fact_scores, get_classifier, 
    MODEL_CONFIGS, setup_logging
)


def test_model_configs():
    """Test that MODEL_CONFIGS contains expected models."""
    expected_models = ['gemma2-9b', 'llama3.1-8b', 'llama3.2-3b', 'llama3.1-70b', 'llama3.1-405b']
    for model in expected_models:
        assert model in MODEL_CONFIGS
        assert 'name' in MODEL_CONFIGS[model]
        assert 'hf_name' in MODEL_CONFIGS[model]
        assert 'num_layers' in MODEL_CONFIGS[model]


def test_flatten_scores_dict_format():
    """Test flatten_scores with dict format input."""
    scores = {
        'decisions': [
            [{'atom': 'fact1', 'is_supported': 1}, {'atom': 'fact2', 'is_supported': 0}],
            [{'atom': 'fact3', 'is_supported': 1}]
        ]
    }
    result = flatten_scores(scores)
    assert len(result) == 3
    assert result[0]['atom'] == 'fact1'
    assert result[0]['is_supported'] == 1


def test_flatten_scores_list_format():
    """Test flatten_scores with list of tuples format."""
    scores = [['fact1', True], ['fact2', False], ['fact3', True]]
    result = flatten_scores(scores)
    assert len(result) == 3
    assert result[0]['atom'] == 'fact1'
    assert result[0]['is_supported'] == 1
    assert result[1]['is_supported'] == 0


def test_get_classifier_logistic():
    """Test get_classifier for logistic regression."""
    classifier = get_classifier('logistic_regression', C=1.0)
    assert classifier.__class__.__name__ == 'LogisticRegression'
    assert classifier.C == 1.0


def test_get_classifier_xgboost():
    """Test get_classifier for XGBoost."""
    classifier = get_classifier('xgboost')
    assert classifier.__class__.__name__ == 'XGBClassifier'


def test_get_classifier_invalid():
    """Test get_classifier with invalid classifier name."""
    with pytest.raises(ValueError, match="Unsupported classifier"):
        get_classifier('invalid_classifier')


def test_setup_logging():
    """Test logging setup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logging(tmpdir)
        assert logger is not None
        assert os.path.exists(tmpdir)


def test_load_fact_scores():
    """Test loading fact scores from pickle file."""
    test_data = {'test': 'data'}
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        pickle.dump(test_data, f)
        temp_path = f.name
    
    try:
        result = load_fact_scores(temp_path)
        assert result == test_data
    finally:
        os.unlink(temp_path) 