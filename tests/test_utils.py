"""Test utility functions."""

import pytest
import numpy as np
from eval_utils import auroc, bootstrap_func


def test_auroc_basic():
    """Test basic AUROC functionality."""
    y_true = [0, 0, 1, 1]
    y_pred_proba = [0.1, 0.2, 0.8, 0.9]
    result = auroc(y_true, y_pred_proba)
    assert result == 1.0


def test_auroc_edge_case():
    """Test AUROC with edge case."""
    y_true = [1, 1, 1, 1]  # All positive
    y_pred_proba = [0.1, 0.5, 0.8, 0.9]
    result = auroc(y_true, y_pred_proba)
    assert result == 0.5  # Should handle gracefully


def test_bootstrap_func():
    """Test bootstrap function."""
    y_true = [0, 0, 1, 1] * 5
    y_pred_proba = [0.1, 0.2, 0.8, 0.9] * 5
    
    result = bootstrap_func(y_true, y_pred_proba, auroc, n_bootstrap=5, rn=42)
    
    assert 'mean' in result
    assert 'original' in result
    assert 'bootstrap' in result
    assert 0 <= result['mean'] <= 1 