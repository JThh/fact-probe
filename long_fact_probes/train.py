#!/usr/bin/env python3
"""
Unified training script for fact probes across different language models.
Consolidates all model-specific training scripts into a single configurable script.
"""

import os
import pickle
import argparse
import time
import logging
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

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
    log_filename = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
    path = snapshot_download(
        repo_id=config['hf_name'],
        allow_patterns=['*.json', '*.model', '*.safetensors'],
        ignore_patterns=['pytorch_model.bin.index.json']
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", torch_dtype=torch.float16
    )
    return tokenizer, model

def compute_hidden_states(tokenizer, model, scores, layer_range):
    """Compute hidden states for factoids."""
    X_all = defaultdict(list)
    y = []
    
    for factoid in tqdm(scores, desc='Processing factoids'):
        inputs = tokenizer([factoid['atom']], return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hiddens = outputs.hidden_states
        
        for layer_idx in layer_range:
            if layer_idx < len(hiddens):
                hidden_state = hiddens[layer_idx][:, -1, :].squeeze(0)
                X_all[layer_idx].append(hidden_state.cpu())
        y.append(factoid['is_supported'])
    
    return X_all, y

def get_classifier(classifier_name, C=None, use_gpu=False, random_state=42):
    """Get classifier instance."""
    if classifier_name == 'logistic_regression':
        return LogisticRegression(
            penalty='l1', solver='liblinear', C=C, 
            max_iter=1000, random_state=random_state
        )
    elif classifier_name == 'xgboost':
        params = {
            'n_estimators': 1000, 'learning_rate': 0.1, 'max_depth': 6,
            'objective': 'binary:logistic', 'use_label_encoder': False,
            'eval_metric': 'auc', 'random_state': random_state
        }
        if use_gpu:
            params.update({'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'})
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

def run_cv_probing(X_all, y, layer_range, group_size, classifier_name, C=None, use_gpu=False):
    """Run cross-validation probing."""
    best_val_auroc = -np.inf
    best_probe = None
    best_layer_group = None
    
    # Create layer groups
    for start_layer in range(0, len(layer_range) - group_size + 1, group_size):
        end_layer = min(start_layer + group_size - 1, len(layer_range) - 1)
        layer_group = (start_layer, end_layer)
        layers = list(range(start_layer, end_layer + 1))
        
        # Concatenate features from selected layers
        try:
            X_concat = np.concatenate([np.stack(X_all[layer]) for layer in layers], axis=-1)
        except:
            continue
            
        # Cross-validation
        cv_aurocs = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(X_concat, y):
            X_train, X_val = X_concat[train_idx], X_concat[val_idx]
            y_train, y_val = np.array(y)[train_idx], np.array(y)[val_idx]
            
            classifier = get_classifier(classifier_name, C=C, use_gpu=use_gpu)
            classifier.fit(X_train, y_train)
            
            try:
                y_pred_proba = classifier.predict_proba(X_val)[:, 1]
                auroc_score = roc_auc_score(y_val, y_pred_proba)
                cv_aurocs.append(auroc_score)
            except:
                continue
        
        if cv_aurocs:
            mean_auroc = np.mean(cv_aurocs)
            if mean_auroc > best_val_auroc:
                best_val_auroc = mean_auroc
                best_layer_group = layer_group
                # Train final probe on full data
                classifier = get_classifier(classifier_name, C=C, use_gpu=use_gpu)
                classifier.fit(X_concat, y)
                best_probe = classifier
    
    return {
        'probe': best_probe,
        'layer_group': best_layer_group,
        'val_auroc': best_val_auroc,
        'classifier_name': classifier_name,
        'C': C,
        'group_size': group_size
    }

def save_probe(probe_data, filepath):
    """Save trained probe."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(probe_data, f)

def main():
    parser = argparse.ArgumentParser(description='Unified fact probe training script')
    parser.add_argument('--model', required=True, choices=MODEL_CONFIGS.keys(),
                       help='Model to use for training')
    parser.add_argument('--train_data_dir', default='./train_data/',
                       help='Training data directory')
    parser.add_argument('--test_data_dir', default='./test_data/',
                       help='Test data directory')  
    parser.add_argument('--results_dir', default='./results/',
                       help='Results output directory')
    parser.add_argument('--log_dir', default='./logs/',
                       help='Logs directory')
    parser.add_argument('--group_sizes', nargs='+', type=int, default=[1, 5],
                       help='Layer group sizes')
    parser.add_argument('--c_values', nargs='+', type=float, default=[0.1, 0.5],
                       help='Regularization C values')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU for XGBoost')
    parser.add_argument('--classifiers', nargs='+', 
                       choices=['logistic_regression', 'xgboost'], 
                       default=['logistic_regression', 'xgboost'],
                       help='Classifiers to train')
    
    args = parser.parse_args()
    
    # Setup
    config = MODEL_CONFIGS[args.model]
    logger = setup_logging(args.log_dir)
    os.makedirs(args.results_dir, exist_ok=True)
    
    logger.info(f'Starting training for {config["name"]}')
    logger.info(f'Training data: {args.train_data_dir}')
    logger.info(f'Test data: {args.test_data_dir}')
    
    # Initialize model
    try:
        tokenizer, model = initialize_model(config)
        logger.info(f'Model {config["name"]} loaded successfully')
    except Exception as e:
        logger.error(f'Model initialization failed: {e}')
        return
    
    # Process training files
    training_files = [f for f in os.listdir(args.train_data_dir) if f.endswith('.pkl')]
    if not training_files:
        logger.error(f'No training files found in {args.train_data_dir}')
        return
    
    all_results = []
    
    for train_file in training_files:
        logger.info(f'Processing {train_file}')
        train_filepath = os.path.join(args.train_data_dir, train_file)
        identifier = os.path.splitext(train_file)[0]
        
        try:
            # Load and preprocess data
            scores = load_fact_scores(train_filepath)
            flat_scores = flatten_scores(scores)
            logger.info(f'Loaded {len(flat_scores)} samples from {train_file}')
            
            # Compute hidden states
            layer_range = list(range(config['num_layers']))
            X_all, y = compute_hidden_states(tokenizer, model, flat_scores, layer_range)
            
            # Train probes
            for classifier_name in args.classifiers:
                for group_size in args.group_sizes:
                    if classifier_name == 'logistic_regression':
                        for C in args.c_values:
                            logger.info(f'Training {classifier_name} C={C} group_size={group_size}')
                            result = run_cv_probing(
                                X_all, y, layer_range, group_size, 
                                classifier_name, C=C, use_gpu=args.use_gpu
                            )
                            
                            if result['probe'] is not None:
                                # Save probe
                                probe_filename = f'{config["name"]}_{identifier}_{classifier_name}_C{C}_group{group_size}.pkl'
                                probe_filepath = os.path.join(args.results_dir, probe_filename)
                                save_probe(result, probe_filepath)
                                
                                # Record result
                                all_results.append({
                                    'model': config['name'],
                                    'train_file': train_file,
                                    'classifier': classifier_name,
                                    'C': C,
                                    'group_size': group_size,
                                    'layer_group': result['layer_group'],
                                    'val_auroc': result['val_auroc'],
                                    'probe_file': probe_filename
                                })
                                
                                logger.info(f'Saved probe: {probe_filename}, AUROC: {result["val_auroc"]:.4f}')
                    
                    elif classifier_name == 'xgboost':
                        logger.info(f'Training {classifier_name} group_size={group_size}')
                        result = run_cv_probing(
                            X_all, y, layer_range, group_size,
                            classifier_name, use_gpu=args.use_gpu
                        )
                        
                        if result['probe'] is not None:
                            # Save probe
                            probe_filename = f'{config["name"]}_{identifier}_{classifier_name}_group{group_size}.pkl'
                            probe_filepath = os.path.join(args.results_dir, probe_filename)
                            save_probe(result, probe_filepath)
                            
                            # Record result
                            all_results.append({
                                'model': config['name'],
                                'train_file': train_file,
                                'classifier': classifier_name,
                                'C': None,
                                'group_size': group_size,
                                'layer_group': result['layer_group'],
                                'val_auroc': result['val_auroc'],
                                'probe_file': probe_filename
                            })
                            
                            logger.info(f'Saved probe: {probe_filename}, AUROC: {result["val_auroc"]:.4f}')
            
        except Exception as e:
            logger.error(f'Error processing {train_file}: {e}')
            continue
    
    # Save results summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_filepath = os.path.join(args.results_dir, f'{config["name"]}_training_results.csv')
        results_df.to_csv(results_filepath, index=False)
        logger.info(f'Training results saved to {results_filepath}')
        logger.info(f'Trained {len(all_results)} probes successfully')
    else:
        logger.warning('No probes were trained successfully')
    
    logger.info('Training completed')

if __name__ == '__main__':
    main() 