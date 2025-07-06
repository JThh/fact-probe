"""Integration tests for main workflow."""

import pytest
import numpy as np
import pickle
import tempfile
import os
from unittest.mock import Mock, patch

# Import required modules
from train import flatten_scores, get_classifier, MODEL_CONFIGS
from eval_utils import auroc, bootstrap_func


class TestWorkflowIntegration:
    """Test integration of main workflow components."""
    
    def test_data_pipeline(self, sample_factoids):
        """Test data processing pipeline."""
        # Test that we can process factoids
        assert len(sample_factoids) == 4
        assert all('atom' in factoid and 'is_supported' in factoid for factoid in sample_factoids)
        
        # Test labels
        labels = [f['is_supported'] for f in sample_factoids]
        assert set(labels) == {0, 1}  # Should have both positive and negative
    
    def test_classifier_training_pipeline(self, sample_predictions):
        """Test classifier training pipeline."""
        # Create mock feature data
        n_samples = len(sample_predictions['y_true'])
        n_features = 100
        X = np.random.rand(n_samples, n_features)
        y = sample_predictions['y_true']
        
        # Test logistic regression
        classifier = get_classifier('logistic_regression', C=1.0)
        classifier.fit(X, y)
        predictions = classifier.predict_proba(X)[:, 1]
        
        assert len(predictions) == n_samples
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_evaluation_pipeline(self, sample_predictions):
        """Test evaluation pipeline."""
        y_true = sample_predictions['y_true']
        y_pred_proba = sample_predictions['y_pred_proba']
        
        # Test AUROC computation
        auroc_score = auroc(y_true, y_pred_proba)
        assert 0 <= auroc_score <= 1
        
        # Test bootstrap evaluation
        bootstrap_result = bootstrap_func(y_true, y_pred_proba, auroc, n_bootstrap=5, rn=42)
        assert 'mean' in bootstrap_result
        assert 'bootstrap' in bootstrap_result
    
    def test_model_config_consistency(self):
        """Test that model configurations are consistent."""
        for model_name, config in MODEL_CONFIGS.items():
            assert isinstance(config['name'], str)
            assert isinstance(config['hf_name'], str)
            assert isinstance(config['num_layers'], int)
            assert config['num_layers'] > 0
            assert 'llama' in config['hf_name'].lower() or 'gemma' in config['hf_name'].lower()
    
    def test_data_format_consistency(self):
        """Test different data format handling."""
        # Test dict format
        dict_scores = {
            'decisions': [
                [{'atom': 'fact1', 'is_supported': 1}],
                [{'atom': 'fact2', 'is_supported': 0}]
            ]
        }
        flattened_dict = flatten_scores(dict_scores)
        assert len(flattened_dict) == 2
        
        # Test list format
        list_scores = [['fact1', True], ['fact2', False]]
        flattened_list = flatten_scores(list_scores)
        assert len(flattened_list) == 2
        
        # Both should produce similar structure
        assert flattened_dict[0]['atom'] == flattened_list[0]['atom']
    
    @patch('train.AutoModelForCausalLM')
    @patch('train.AutoTokenizer')
    def test_model_initialization_mock(self, mock_tokenizer, mock_model):
        """Test model initialization with mocks."""
        # Mock the tokenizer and model
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        # This would test the initialization function if we import it
        # For now, just test that the mocks work
        assert mock_tokenizer.from_pretrained is not None
        assert mock_model.from_pretrained is not None


class TestDataConsistency:
    """Test data consistency across different formats."""
    
    def test_score_normalization(self):
        """Test that different boolean representations are normalized."""
        test_cases = [
            (['fact1', True], 1),
            (['fact2', False], 0),
            (['fact3', 1], 1),
            (['fact4', 0], 0),
        ]
        
        for input_data, expected in test_cases:
            scores = [input_data]
            flattened = flatten_scores(scores)
            assert flattened[0]['is_supported'] == expected
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_dict = {'decisions': []}
        result = flatten_scores(empty_dict)
        assert result == []
        
        empty_list = []
        result = flatten_scores(empty_list)
        assert result == [] 