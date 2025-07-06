"""Unit tests for predict.py module."""

import pytest
import numpy as np
import tempfile
import os
import sys
import json
from unittest.mock import Mock, patch

# Add the long_fact_probes directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'long_fact_probes'))

from predict import (
    NumpyJSONEncoder, compute_rejection_accuracy_curve,
    plot_rejection_accuracy_curve, MODEL_CONFIGS
)


class TestNumpyJSONEncoder:
    """Test cases for NumpyJSONEncoder."""
    
    def test_encode_numpy_int(self):
        """Test encoding numpy integers."""
        encoder = NumpyJSONEncoder()
        data = {'value': np.int32(42)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        assert '"value": 42' in result
    
    def test_encode_numpy_float(self):
        """Test encoding numpy floats."""
        encoder = NumpyJSONEncoder()
        data = {'value': np.float32(3.14)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        assert '"value": 3.14' in result
    
    def test_encode_numpy_array(self):
        """Test encoding numpy arrays."""
        encoder = NumpyJSONEncoder()
        data = {'array': np.array([1, 2, 3])}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        assert '"array": [1, 2, 3]' in result
    
    def test_encode_numpy_bool(self):
        """Test encoding numpy booleans."""
        encoder = NumpyJSONEncoder()
        data = {'bool': np.bool_(True)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        assert '"bool": true' in result


class TestRejectionAccuracyCurve:
    """Test cases for rejection-accuracy curve computation."""
    
    def test_compute_rejection_accuracy_curve_basic(self):
        """Test basic rejection-accuracy curve computation."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred_proba = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]
        
        result = compute_rejection_accuracy_curve(y_true, y_pred_proba, num_thresholds=5)
        
        assert 'rejection_ratios' in result
        assert 'accuracies' in result 
        assert 'thresholds' in result
        assert len(result['rejection_ratios']) == len(result['accuracies'])
        assert len(result['thresholds']) == len(result['accuracies'])
    
    def test_compute_rejection_accuracy_curve_perfect_classifier(self):
        """Test rejection-accuracy curve for perfect classifier."""
        y_true = [0, 0, 1, 1]
        y_pred_proba = [0.1, 0.2, 0.8, 0.9]
        
        result = compute_rejection_accuracy_curve(y_true, y_pred_proba, num_thresholds=3)
        
        # First point should be 0% rejection with 100% accuracy for perfect classifier
        assert result['rejection_ratios'][0] == 0.0
        assert result['accuracies'][0] == 1.0
    
    def test_compute_rejection_accuracy_curve_random_classifier(self):
        """Test rejection-accuracy curve for random classifier."""
        y_true = [0, 1, 0, 1] * 10
        y_pred_proba = [0.5] * 40  # All predictions at decision boundary
        
        result = compute_rejection_accuracy_curve(y_true, y_pred_proba, num_thresholds=3)
        
        # With all predictions at 0.5, accuracy should be around 50%
        assert 0.4 <= result['accuracies'][0] <= 0.6
    
    def test_plot_rejection_accuracy_curve(self):
        """Test plotting rejection-accuracy curve."""
        curve_data = {
            'rejection_ratios': [0.0, 0.2, 0.5, 0.8],
            'accuracies': [0.7, 0.8, 0.9, 1.0],
            'thresholds': [0.0, 0.25, 0.5, 0.75]
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_plot.png')
            
            # Should not raise an exception
            plot_rejection_accuracy_curve(curve_data, 'Test Probe', output_path)
            
            # Check that file was created
            assert os.path.exists(output_path)


def test_model_configs_structure():
    """Test MODEL_CONFIGS structure."""
    assert isinstance(MODEL_CONFIGS, dict)
    assert len(MODEL_CONFIGS) > 0
    
    for model_name, config in MODEL_CONFIGS.items():
        assert 'name' in config
        assert 'hf_name' in config
        assert 'num_layers' in config
        assert isinstance(config['num_layers'], int)
        assert config['num_layers'] > 0 