"""
Evaluation utilities for fact probe experiments.
Common functions used across train/eval/predict scripts.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred_proba):
    """Compute AUROC score."""
    try:
        result = roc_auc_score(y_true, y_pred_proba)
        # Handle edge cases where all labels are the same (returns NaN)
        if np.isnan(result):
            return 0.5
        return result
    except ValueError:
        # Handle other edge cases
        return 0.5

def bootstrap_func(y_true, y_pred_proba, metric_func, n_bootstrap=1000, rn=42):
    """
    Compute bootstrap statistics for a metric.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric_func: Function to compute metric (e.g., auroc)
        n_bootstrap: Number of bootstrap samples
        rn: Random seed
        
    Returns:
        Dict with mean, std_err, and bootstrap samples
    """
    np.random.seed(rn)
    
    # Compute original metric
    original_metric = metric_func(y_true, y_pred_proba)
    
    # Bootstrap sampling
    n_samples = len(y_true)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred_proba[i] for i in indices]
        
        # Compute metric on bootstrap sample
        try:
            metric_boot = metric_func(y_true_boot, y_pred_boot)
            bootstrap_metrics.append(metric_boot)
        except:
            # Skip if metric computation fails
            continue
    
    if bootstrap_metrics:
        bootstrap_mean = np.mean(bootstrap_metrics)
        bootstrap_std = np.std(bootstrap_metrics)
        bootstrap_se = bootstrap_std / np.sqrt(len(bootstrap_metrics))
    else:
        bootstrap_mean = original_metric
        bootstrap_std = 0.0
        bootstrap_se = 0.0
    
    return {
        'mean': bootstrap_mean,
        'original': original_metric,
        'bootstrap': {
            'samples': bootstrap_metrics,
            'std': bootstrap_std,
            'std_err': bootstrap_se,
            'n_samples': len(bootstrap_metrics)
        }
    } 