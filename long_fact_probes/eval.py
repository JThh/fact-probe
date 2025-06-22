#!/usr/bin/env python3
"""
Unified evaluation script for fact probes across different language models.
Consolidates all model-specific evaluation scripts into a single configurable script.
"""

import os
import pickle
import argparse
import logging
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

def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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

def load_probes(probes_dir):
    """Load trained probes from directory."""
    probe_files = [f for f in os.listdir(probes_dir) if f.endswith('.pkl')]
    if not probe_files:
        return {}
    
    probes = {}
    for file in probe_files:
        path = os.path.join(probes_dir, file)
        try:
            with open(path, 'rb') as f:
                probe_data = pickle.load(f)
            probes[file] = probe_data
        except Exception as e:
            print(f'Failed to load {file}: {e}')
    
    return probes

def evaluate_probe(probe_data, test_scores, tokenizer, model, logger):
    """Evaluate a single probe on test data."""
    probe = probe_data.get('probe')
    layer_group = probe_data.get('layer_group')
    
    if probe is None or layer_group is None:
        return None
    
    # Compute hidden states for the required layers
    layers = list(range(layer_group[0], layer_group[1] + 1))
    X_all, y_true = compute_hidden_states(tokenizer, model, test_scores, layers)
    
    # Concatenate features across layers
    try:
        X_concat = np.concatenate([np.stack(X_all[layer]) for layer in layers], axis=-1)
    except Exception as e:
        logger.error(f'Error concatenating hidden states: {e}')
        return None
    
    # Predict
    try:
        y_pred_proba = probe.predict_proba(X_concat)[:, 1]
    except Exception as e:
        logger.error(f'Prediction failed: {e}')
        return None
    
    # Compute AUROC with bootstrap
    try:
        raw_auroc = roc_auc_score(y_true, y_pred_proba)
        metric_dict = bootstrap_func(y_true, y_pred_proba, auroc, rn=42)
        test_auroc_mean = metric_dict['mean']
        test_auroc_std = metric_dict['bootstrap']['std_err']
    except Exception as e:
        logger.error(f'AUROC computation failed: {e}')
        return None
    
    return {
        'raw_auroc': raw_auroc,
        'test_auroc_mean': test_auroc_mean,
        'test_auroc_std': test_auroc_std
    }

def main():
    parser = argparse.ArgumentParser(description='Unified fact probe evaluation script')
    parser.add_argument('--model', required=True, choices=MODEL_CONFIGS.keys(),
                       help='Model to use for evaluation')
    parser.add_argument('--test_data_dir', default='./test_data/',
                       help='Test data directory')
    parser.add_argument('--probes_dir', default='./results/',
                       help='Directory containing trained probes')
    parser.add_argument('--results_dir', default='./test_results/',
                       help='Results output directory')
    parser.add_argument('--log_dir', default='./logs/',
                       help='Logs directory')
    
    args = parser.parse_args()
    
    # Setup
    config = MODEL_CONFIGS[args.model]
    logger = setup_logging(args.log_dir)
    os.makedirs(args.results_dir, exist_ok=True)
    
    logger.info(f'Starting evaluation for {config["name"]}')
    logger.info(f'Test data: {args.test_data_dir}')
    logger.info(f'Probes directory: {args.probes_dir}')
    
    # Initialize model
    try:
        tokenizer, model = initialize_model(config)
        logger.info(f'Model {config["name"]} loaded successfully')
    except Exception as e:
        logger.error(f'Model initialization failed: {e}')
        return
    
    # Load probes
    probes = load_probes(args.probes_dir)
    if not probes:
        logger.error(f'No probes found in {args.probes_dir}')
        return
    
    logger.info(f'Loaded {len(probes)} probes')
    
    # Process test files
    test_files = [f for f in os.listdir(args.test_data_dir) if f.endswith('.pkl')]
    if not test_files:
        logger.error(f'No test files found in {args.test_data_dir}')
        return
    
    all_results = []
    
    for test_file in test_files:
        logger.info(f'Processing test file: {test_file}')
        test_filepath = os.path.join(args.test_data_dir, test_file)
        
        try:
            # Load and preprocess test data
            scores = load_fact_scores(test_filepath)
            flat_scores = flatten_scores(scores)
            logger.info(f'Loaded {len(flat_scores)} test samples from {test_file}')
            
            # Evaluate each probe
            for probe_name, probe_data in probes.items():
                logger.info(f'Evaluating probe: {probe_name}')
                
                result = evaluate_probe(probe_data, flat_scores, tokenizer, model, logger)
                
                if result is not None:
                    all_results.append({
                        'model': config['name'],
                        'test_file': test_file,
                        'probe_name': probe_name,
                        'classifier': probe_data.get('classifier_name', 'unknown'),
                        'C': probe_data.get('C', None),
                        'group_size': probe_data.get('group_size', None),
                        'layer_group': probe_data.get('layer_group', None),
                        'raw_auroc': result['raw_auroc'],
                        'test_auroc_mean': result['test_auroc_mean'],
                        'test_auroc_std': result['test_auroc_std']
                    })
                    
                    logger.info(f'Probe {probe_name}: AUROC = {result["test_auroc_mean"]:.4f} ± {result["test_auroc_std"]:.4f}')
                else:
                    logger.warning(f'Failed to evaluate probe: {probe_name}')
                    
        except Exception as e:
            logger.error(f'Error processing {test_file}: {e}')
            continue
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_filepath = os.path.join(args.results_dir, f'{config["name"]}_evaluation_results.csv')
        results_df.to_csv(results_filepath, index=False)
        logger.info(f'Evaluation results saved to {results_filepath}')
        
        # Print summary
        print(f'\nEvaluation Summary for {config["name"]}:')
        print(f'Evaluated {len(all_results)} probe-test combinations')
        print(f'Average AUROC: {results_df["test_auroc_mean"].mean():.4f}')
        print(f'Best AUROC: {results_df["test_auroc_mean"].max():.4f}')
        
        # Show top 5 probes
        top_probes = results_df.nlargest(5, 'test_auroc_mean')
        print('\nTop 5 Probes:')
        for _, row in top_probes.iterrows():
            print(f'  {row["probe_name"]}: {row["test_auroc_mean"]:.4f} ± {row["test_auroc_std"]:.4f}')
    else:
        logger.warning('No evaluation results generated')
    
    logger.info('Evaluation completed')

if __name__ == '__main__':
    main() 