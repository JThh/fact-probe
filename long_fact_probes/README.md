# Long Fact Probes - Unified Framework

Modular, extensible framework for training, evaluating, and deploying fact verification probes across language models. Consolidates model-specific implementations into a unified architecture with configurable components.

## Architecture

### Core Components
```
long_fact_probes/
├── train.py              # Training orchestrator with CV pipeline
├── eval.py               # Batch evaluation engine
├── predict.py            # Inference engine with visualization
├── eval_utils.py         # Shared evaluation primitives
└── README.md            # This documentation
```

### Data Flow
```
Training Data (.pkl) → flatten_scores() → compute_hidden_states() → run_cv_probing() → save_probe()
Test Data (.pkl) → evaluate_probe() → bootstrap_func() → Results (CSV/JSON)
```

## Quick Start

```bash
# Basic training
python train.py --model llama3.1-8b --train_data_dir ./data/train/ --results_dir ./models/

# Batch evaluation
python eval.py --model llama3.1-8b --probes_dir ./models/ --test_data_dir ./data/test/

# Inference with visualization
python predict.py --model llama3.1-8b --probe_file ./models/best_probe.pkl --test_data_dir ./data/test/
```

## Configuration

### Model Configuration Schema
```python
MODEL_CONFIGS = {
    'model_id': {
        'name': str,           # Display name
        'hf_name': str,        # HuggingFace model identifier
        'num_layers': int      # Number of transformer layers
    }
}
```

### Adding New Models
```python
MODEL_CONFIGS['new-model'] = {
    'name': 'NewModel-7B',
    'hf_name': 'org/new-model-7b-instruct',
    'num_layers': 32
}

def initialize_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config['hf_name'])
    # Add model-specific tokenizer settings
    if 'new-model' in config['hf_name']:
        tokenizer.pad_token = tokenizer.eos_token
    # ... rest of initialization
```

## API Reference

### Training Pipeline

#### Core Functions
```python
def run_cv_probing(X_all, y, layer_range, group_size, classifier_name, C=None, use_gpu=False):
    """
    Cross-validation probing with layer grouping.
    
    Args:
        X_all: Dict[int, List[torch.Tensor]] - Hidden states by layer
        y: List[int] - Binary labels  
        layer_range: List[int] - Layer indices to probe
        group_size: int - Number of consecutive layers to group
        classifier_name: str - 'logistic_regression' or 'xgboost'
        C: float - Regularization parameter for LogReg
        use_gpu: bool - Enable GPU acceleration for XGBoost
        
    Returns:
        Dict containing trained probe, layer_group, validation AUROC
    """

def flatten_scores(scores):
    """
    Normalize diverse fact score formats to standard schema.
    
    Supported formats:
    - Dict with 'decisions' key (nested factoids)
    - List of [atom, is_supported] pairs  
    - List of dicts with factoid metadata
    - Custom tuple format: (_, _, _, _, facts, truth_levels)
    
    Returns:
        List[Dict] with keys: 'atom', 'is_supported'
    """

def compute_hidden_states(tokenizer, model, scores, layer_range):
    """
    Extract transformer hidden states for fact verification.
    
    Implementation details:
    - Uses last token representation ([:, -1, :])
    - Processes on GPU with torch.no_grad()
    - Returns CPU tensors to manage memory
    
    Returns:
        Tuple[Dict[int, List], List[int]] - (hidden_states, labels)
    """
```

### Evaluation System

#### Bootstrap Statistics
```python
def bootstrap_func(y_true, y_pred_proba, metric_func, n_bootstrap=1000, rn=42):
    """
    Robust statistical estimation with bootstrap resampling.
    
    Returns:
        {
            'mean': float,
            'original': float, 
            'bootstrap': {
                'samples': List[float],
                'std': float,
                'std_err': float,
                'n_samples': int
            }
        }
    """
```

#### Performance Metrics
```python
def compute_rejection_accuracy_curve(y_true, y_pred_proba, num_thresholds=100):
    """
    Generate rejection-accuracy tradeoff curve for confidence-based filtering.
    
    Algorithm:
    1. Compute confidence = |pred_proba - 0.5|
    2. For each threshold, retain high-confidence predictions
    3. Calculate accuracy on retained subset
    
    Returns:
        Dict with 'rejection_ratios', 'accuracies', 'thresholds'
    """
```

## Extension Patterns

### Custom Classifiers
```python
def get_classifier(classifier_name, **kwargs):
    if classifier_name == 'custom_svm':
        from sklearn.svm import SVC
        return SVC(probability=True, **kwargs)
    elif classifier_name == 'neural_probe':
        return CustomNeuralProbe(**kwargs)
    # ... existing classifiers
```

### Custom Data Loaders
```python
class CustomDataLoader:
    @staticmethod
    def load_fact_scores(filepath):
        # Implement custom format parsing
        with open(filepath, 'rb') as f:
            data = custom_deserialize(f)
        return CustomDataLoader.to_standard_format(data)
    
    @staticmethod 
    def to_standard_format(data):
        # Convert to List[Dict] with 'atom', 'is_supported' keys
        pass
```

### Layer Selection Strategies
```python
def custom_layer_grouping(layer_range, group_size, strategy='consecutive'):
    """
    Advanced layer grouping strategies.
    
    Strategies:
    - 'consecutive': Standard adjacent layers
    - 'skip': Every nth layer  
    - 'attention': Attention layer subset
    - 'residual': Residual stream layers
    """
    if strategy == 'skip':
        return [layer_range[i::group_size] for i in range(group_size)]
    # ... implement other strategies
```

## Advanced Configuration

### Training Hyperparameters
```bash
# Extensive hyperparameter search
python train.py \
    --model llama3.1-8b \
    --group_sizes 1 3 5 7 \
    --c_values 0.001 0.01 0.1 1.0 10.0 \
    --classifiers logistic_regression xgboost \
    --use_gpu
```

### Distributed Training
```python
def distributed_cv_probing(X_all, y, n_gpus=4):
    """Distribute cross-validation folds across GPUs."""
    import torch.multiprocessing as mp
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_gpus) as pool:
        results = pool.map(train_fold, cv_splits)
    return aggregate_results(results)
```

### Memory Optimization
```python
# Gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Streaming data processing
def compute_hidden_states_streaming(tokenizer, model, scores, batch_size=32):
    """Process data in batches to reduce memory usage."""
    for i in range(0, len(scores), batch_size):
        batch = scores[i:i+batch_size]
        yield process_batch(batch)
```

## Performance Tuning

### GPU Optimization
```python
# Mixed precision training
from torch.cuda.amp import autocast
with autocast():
    outputs = model(**inputs, output_hidden_states=True)
```

### Caching Strategies
```python
# Hidden state caching
CACHE_DIR = os.environ.get('PROBE_CACHE_DIR', './cache/')

def cached_hidden_states(cache_key, compute_fn):
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    
    result = compute_fn()
    torch.save(result, cache_path)
    return result
```

## Debugging

### Logging Configuration
```python
import logging

def setup_debug_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler()
        ]
    )
    
    # Suppress noisy libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Large model + batch size | Reduce batch size, use gradient checkpointing |
| Slow inference | CPU-only execution | Ensure CUDA available, use `device_map="auto"` |
| Poor AUROC | Data imbalance | Check label distribution, use stratified CV |
| Missing files | Incorrect paths | Verify file permissions, use absolute paths |

## Development Workflow

### Testing
```bash
# Unit tests
python -m pytest tests/ -v

# Integration tests  
python -m pytest tests/integration/ -x

# Performance benchmarks
python benchmarks/probe_timing.py
```

### Code Quality
```bash
# Linting
flake8 long_fact_probes/ --max-line-length=100

# Type checking
mypy long_fact_probes/ --ignore-missing-imports

# Formatting
black long_fact_probes/ --line-length=100
```

## Environment Setup

### Development Environment
```bash
# Create conda environment
conda create -n fact-probes python=3.9
conda activate fact-probes

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Editable install for development

# GPU setup (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Environment Variables
```bash
# Core directories
export PROBE_CACHE_DIR="/path/to/cache"
export FACTSCORE_CACHE_DIR="/path/to/factscore_cache" 
export HF_DATASETS_CACHE="/path/to/datasets_cache"

# Model configuration
export CUDA_VISIBLE_DEVICES="0,1"
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/path/to/huggingface_cache"
```

## Programmatic Usage

```python
from long_fact_probes.train import run_cv_probing
from long_fact_probes.eval_utils import bootstrap_func

# Initialize model
config = MODEL_CONFIGS['llama3.1-8b']
tokenizer, model = initialize_model(config)

# Load and process data
scores = load_fact_scores('data.pkl')
flat_scores = flatten_scores(scores)
X_all, y = compute_hidden_states(tokenizer, model, flat_scores, range(32))

# Train probe
result = run_cv_probing(
    X_all, y, 
    layer_range=range(32), 
    group_size=5,
    classifier_name='logistic_regression',
    C=0.1
)

# Evaluate with bootstrap
metrics = bootstrap_func(y_true, y_pred_proba, auroc)
print(f"AUROC: {metrics['mean']:.4f} ± {metrics['bootstrap']['std_err']:.4f}")
```

## File Structure

```
long_fact_probes/
├── train.py              # Unified training script
├── eval.py               # Unified evaluation script
├── predict.py            # Unified prediction script
├── eval_utils.py         # Common evaluation utilities
├── README.md             # This file
├── train_data/           # Training data directory
├── test_data/            # Test data directory
├── results/              # Output directory for trained probes
├── test_results/         # Output directory for evaluation results
├── plots/                # Output directory for visualizations
└── logs/                 # Log files directory
```

## Supported Models

| Model ID | Model Name | HuggingFace ID | Layers |
|----------|------------|----------------|--------|
| `gemma2-9b` | Gemma2-9B | `google/gemma-2-9b-it` | 42 |
| `llama3.1-8b` | Llama3.1-8B | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 32 |
| `llama3.2-3b` | Llama3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | 28 |
| `llama3.1-70b` | Llama3.1-70B | `meta-llama/Meta-Llama-3.1-70B-Instruct` | 80 |
| `llama3.1-405b` | Llama3.1-405B | `meta-llama/Meta-Llama-3.1-405B-Instruct` | 126 |

## Migration from Old Structure

Replace model-specific directories with unified scripts:

1. **Training**: `python train_probe_<model>.py` → `python train.py --model <model-id>`
2. **Evaluation**: `python eval.py` → `python eval.py --model <model-id>`
3. **Prediction**: `python pred.py` → `python predict.py --model <model-id> --probe_file <probe>`

## Benefits

- **Reduced code duplication**: ~80% reduction in code size
- **Easier maintenance**: Single point of updates for all models
- **Consistent behavior**: Same functionality across all models
- **Better error handling**: Comprehensive error checking and logging
- **Improved usability**: Clear command-line interface with help text
- **Enhanced flexibility**: Easy to add new models or modify behavior 