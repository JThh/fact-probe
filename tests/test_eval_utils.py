"""Unit tests for eval_utils module."""

import pytest
import numpy as np
import sys
import os

# Add the long_fact_probes directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'long_fact_probes'))

from eval_utils import auroc, bootstrap_func


def test_auroc_perfect():
    """Test AUROC for perfect classifier."""
    y_true = [0, 0, 1, 1]
    y_pred_proba = [0.1, 0.2, 0.8, 0.9]
    result = auroc(y_true, y_pred_proba)
    assert result == 1.0


def test_bootstrap_func_basic():
    """Test basic bootstrap functionality."""
    y_true = [0, 0, 1, 1] * 10
    y_pred_proba = [0.1, 0.2, 0.8, 0.9] * 10
    
    result = bootstrap_func(y_true, y_pred_proba, auroc, n_bootstrap=5, rn=42)
    
    assert 'mean' in result
    assert 'original' in result
    assert 'bootstrap' in result
    assert 0 <= result['mean'] <= 1 