#!/usr/bin/env python3
"""
Unified prediction script for fact probes across different language models.
Consolidates all model-specific prediction scripts with visualization capabilities.
"""

import os
import pickle
import argparse
import logging
import json
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_utils import auroc, bootstrap_func

# Model configurations
MODEL_CONFIGS = {
    'gemma2-9b': {
        'name': 'Gemma2-9B',
        'hf_name': 'google/gemma-2-9b-it',
        'num_layers': 42
    },
    'llama3.1-8b': {
        'name': 'Llama3.1-8B',
        'hf_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'num_layers': 32
    },
    'llama3.2-3b': {
        'name': 'Llama3.2-3B',
        'hf_name': 'meta-llama/Llama-3.2-3B-Instruct',
        'num_layers': 28
    },
    'llama3.1-70b': {
        'name': 'Llama3.1-70B',
        'hf_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'num_layers': 80
    },
    'llama3.1-405b': {
        'name': 'Llama3.1-405B',
        'hf_name': 'meta-llama/Meta-Llama-3.1-405B-Instruct',
        'num_layers': 126
    }
}

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f'predict_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def load_fact_scores(filepath):
    """Load fact scores from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def flatten_scores(scores):
    """Flatten scores into list of factoids."""
    if isinstance(scores, dict) and 'decisions' in scores:
        flat_scores = [factoid for decision_group in scores['decisions'] for factoid in decision_group]
        for factoid in flat_scores:
            if 'is_supported' not in factoid:
                factoid['is_supported'] = 0
        return flat_scores
    elif isinstance(scores, list):
        if all(isinstance(inner, list) and len(inner) == 2 for inner in scores):
            return [{'atom': atom, 'is_supported': int(bool(is_supported))} 
                   for atom, is_supported in scores]
        elif all(isinstance(inner, dict) for inner in scores):
            return scores
        else:
            flattened = []
            for datum in scores:
                extracted_facts = datum[4]
                truth_fact_level = datum[5]
                for fact, truth in zip(extracted_facts, truth_fact_level):
                    flattened.append({'atom': fact, 'is_supported': int(bool(truth))})
            return flattened
    else:
        raise ValueError("Unsupported score format.")

def initialize_model(config):
    """Initialize tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(config['hf_name'])
    model = AutoModelForCausalLM.from_pretrained(
        config['hf_name'], device_map="auto", torch_dtype=torch.float16
    )
    return tokenizer, model

def compute_hidden_states(tokenizer, model, scores, layer_range):
    """Compute hidden states for factoids."""
    X_all = {layer: [] for layer in layer_range}
    y_all = []
    
    for fact in tqdm(scores, desc='Computing hidden states'):
        text = fact['atom']
        label = fact['is_supported']
        inputs = tokenizer(text, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
        hidden_states = outputs.hidden_states
        for layer in layer_range:
            if layer < len(hidden_states):
                X_all[layer].append(hidden_states[layer][0, -1].cpu().numpy())
        y_all.append(label)
    
    return X_all, y_all

def compute_rejection_accuracy_curve(y_true, y_pred_proba, num_thresholds=100):
    """Compute rejection-accuracy curve."""
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    rejection_ratios = []
    accuracies = []
    
    # Add point for 0% rejection
    all_preds = [1 if p >= 0.5 else 0 for p in y_pred_proba]
    all_acc = np.mean([1 if p == t else 0 for p, t in zip(all_preds, y_true)])
    rejection_ratios.append(0.0)
    accuracies.append(all_acc)
    
    # Add points for confidence-based rejection
    for threshold in thresholds:
        if threshold == 0.0:
            continue
            
        confident_indices = []
        for i, pred_score in enumerate(y_pred_proba):
            confidence = abs(pred_score - 0.5)
            if confidence >= threshold / 2:
                confident_indices.append(i)
        
        rejection_ratio = 1.0 - (len(confident_indices) / len(y_true))
        
        if not confident_indices:
            rejection_ratios.append(1.0)
            accuracies.append(1.0)
            continue
        
        retained_true = [y_true[i] for i in confident_indices]
        retained_pred = [1 if y_pred_proba[i] >= 0.5 else 0 for i in confident_indices]
        accuracy = np.mean([1 if p == t else 0 for p, t in zip(retained_pred, retained_true)])
        
        rejection_ratios.append(rejection_ratio)
        accuracies.append(accuracy)
    
    return {
        'rejection_ratios': rejection_ratios,
        'accuracies': accuracies,
        'thresholds': [0.0] + list(thresholds[1:])
    }

def plot_rejection_accuracy_curve(curve_data, probe_name, output_path, dpi=300):
    """Plot rejection-accuracy curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(curve_data['rejection_ratios'], curve_data['accuracies'], 'b-', linewidth=2)
    plt.xlabel('Rejection Ratio', fontsize=12)
    plt.ylabel('Accuracy on Retained Data', fontsize=12)
    plt.title(f'Rejection-Accuracy Curve: {probe_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add annotations for key points
    max_acc_idx = np.argmax(curve_data['accuracies'])
    plt.annotate(f'Max Acc: {curve_data["accuracies"][max_acc_idx]:.3f}',
                xy=(curve_data['rejection_ratios'][max_acc_idx], curve_data['accuracies'][max_acc_idx]),
                xytext=(0.3, 0.7), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def predict_with_probe(probe_data, test_scores, tokenizer, model, logger):
    """Make predictions with a single probe."""
    probe = probe_data.get('probe')
    layer_group = probe_data.get('layer_group')
    
    if probe is None or layer_group is None:
        return None
    
    # Compute hidden states
    layers = list(range(layer_group[0], layer_group[1] + 1))
    X_all, y_true = compute_hidden_states(tokenizer, model, test_scores, layers)
    
    # Concatenate features
    try:
        X_concat = np.concatenate([np.stack(X_all[layer]) for layer in layers], axis=-1)
    except Exception as e:
        logger.error(f'Error concatenating hidden states: {e}')
        return None
    
    # Predict
    try:
        y_pred_proba = probe.predict_proba(X_concat)[:, 1]
        y_pred = probe.predict(X_concat)
    except Exception as e:
        logger.error(f'Prediction failed: {e}')
        return None
    
    # Compute metrics
    try:
        raw_auroc = roc_auc_score(y_true, y_pred_proba)
        metric_dict = bootstrap_func(y_true, y_pred_proba, auroc, rn=42)
        test_auroc_mean = metric_dict['mean']
        test_auroc_std = metric_dict['bootstrap']['std_err']
        
        # Compute rejection-accuracy curve
        curve_data = compute_rejection_accuracy_curve(y_true, y_pred_proba)
        
    except Exception as e:
        logger.error(f'Metrics computation failed: {e}')
        return None
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'raw_auroc': raw_auroc,
        'test_auroc_mean': test_auroc_mean,
        'test_auroc_std': test_auroc_std,
        'curve_data': curve_data
    }

def main():
    parser = argparse.ArgumentParser(description='Unified fact probe prediction script')
    parser.add_argument('--model', required=True, choices=MODEL_CONFIGS.keys(),
                       help='Model to use for prediction')
    parser.add_argument('--test_data_dir', default='./test_data/',
                       help='Test data directory')
    parser.add_argument('--probe_file', required=True,
                       help='Path to specific probe file (.pkl)')
    parser.add_argument('--results_dir', default='./test_results/',
                       help='Results output directory')
    parser.add_argument('--plots_dir', default='./plots/',
                       help='Plots output directory')
    parser.add_argument('--log_dir', default='./logs/',
                       help='Logs directory')
    parser.add_argument('--plot_dpi', type=int, default=300,
                       help='DPI for saved plots')
    
    args = parser.parse_args()
    
    # Setup
    config = MODEL_CONFIGS[args.model]
    logger = setup_logging(args.log_dir)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    logger.info(f'Starting prediction for {config["name"]}')
    logger.info(f'Probe file: {args.probe_file}')
    logger.info(f'Test data: {args.test_data_dir}')
    
    # Load probe
    try:
        with open(args.probe_file, 'rb') as f:
            probe_data = pickle.load(f)
        logger.info('Probe loaded successfully')
    except Exception as e:
        logger.error(f'Failed to load probe: {e}')
        return
    
    # Initialize model
    try:
        tokenizer, model = initialize_model(config)
        logger.info(f'Model {config["name"]} loaded successfully')
    except Exception as e:
        logger.error(f'Model initialization failed: {e}')
        return
    
    # Process test files
    test_files = [f for f in os.listdir(args.test_data_dir) if f.endswith('.pkl')]
    if not test_files:
        logger.error(f'No test files found in {args.test_data_dir}')
        return
    
    probe_name = os.path.splitext(os.path.basename(args.probe_file))[0]
    all_results = []
    
    for test_file in test_files:
        logger.info(f'Processing test file: {test_file}')
        test_filepath = os.path.join(args.test_data_dir, test_file)
        
        try:
            # Load test data
            scores = load_fact_scores(test_filepath)
            flat_scores = flatten_scores(scores)
            logger.info(f'Loaded {len(flat_scores)} test samples from {test_file}')
            
            # Make predictions
            result = predict_with_probe(probe_data, flat_scores, tokenizer, model, logger)
            
            if result is not None:
                logger.info(f'AUROC: {result["test_auroc_mean"]:.4f} ± {result["test_auroc_std"]:.4f}')
                
                # Save detailed results
                test_identifier = os.path.splitext(test_file)[0]
                
                # Save predictions
                predictions = {
                    'probe_name': probe_name,
                    'test_file': test_file,
                    'model': config['name'],
                    'metrics': {
                        'raw_auroc': result['raw_auroc'],
                        'test_auroc_mean': result['test_auroc_mean'],
                        'test_auroc_std': result['test_auroc_std']
                    },
                    'predictions': {
                        'y_true': result['y_true'],
                        'y_pred': result['y_pred'].tolist() if hasattr(result['y_pred'], 'tolist') else result['y_pred'],
                        'y_pred_proba': result['y_pred_proba'].tolist() if hasattr(result['y_pred_proba'], 'tolist') else result['y_pred_proba']
                    },
                    'curve_data': result['curve_data']
                }
                
                pred_filepath = os.path.join(args.results_dir, f'{probe_name}_{test_identifier}_predictions.json')
                with open(pred_filepath, 'w') as f:
                    json.dump(predictions, f, cls=NumpyJSONEncoder, indent=2)
                logger.info(f'Predictions saved to {pred_filepath}')
                
                # Create plot
                plot_filepath = os.path.join(args.plots_dir, f'{probe_name}_{test_identifier}_curve.png')
                plot_rejection_accuracy_curve(
                    result['curve_data'], 
                    f'{probe_name} on {test_identifier}',
                    plot_filepath,
                    dpi=args.plot_dpi
                )
                logger.info(f'Plot saved to {plot_filepath}')
                
                # Record summary result
                all_results.append({
                    'model': config['name'],
                    'probe_name': probe_name,
                    'test_file': test_file,
                    'raw_auroc': result['raw_auroc'],
                    'test_auroc_mean': result['test_auroc_mean'],
                    'test_auroc_std': result['test_auroc_std'],
                    'max_accuracy': max(result['curve_data']['accuracies']),
                    'predictions_file': pred_filepath,
                    'plot_file': plot_filepath
                })
            else:
                logger.warning(f'Failed to generate predictions for {test_file}')
                
        except Exception as e:
            logger.error(f'Error processing {test_file}: {e}')
            continue
    
    # Save summary results
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_filepath = os.path.join(args.results_dir, f'{probe_name}_prediction_summary.csv')
        results_df.to_csv(summary_filepath, index=False)
        logger.info(f'Summary results saved to {summary_filepath}')
        
        # Print summary
        print(f'\nPrediction Summary for {probe_name}:')
        print(f'Processed {len(all_results)} test files')
        print(f'Average AUROC: {results_df["test_auroc_mean"].mean():.4f}')
        print(f'Average Max Accuracy: {results_df["max_accuracy"].mean():.4f}')
        
        print('\nResults by test file:')
        for _, row in results_df.iterrows():
            print(f'  {row["test_file"]}: AUROC = {row["test_auroc_mean"]:.4f}, Max Acc = {row["max_accuracy"]:.4f}')
    else:
        logger.warning('No prediction results generated')
    
    logger.info('Prediction completed')

if __name__ == '__main__':
    main() 