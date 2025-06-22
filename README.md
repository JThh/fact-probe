# Simple Probes Detect Long-Form Hallucinations

A research framework for detecting hallucinations in long-form LLM generations using lightweight probes on hidden states. This codebase implements methods from the paper "Simple Probes Detect Long-Form Hallucinations" and provides tools for training and evaluating hallucination detection probes.

## Abstract

Large language models (LLMs) often mislead users with confident hallucinations. Current approaches to detect hallucination require many samples from the LLM generator, which is computationally infeasible as frontier model sizes and generation lengths continue to grow. We present a remarkably simple baseline for detecting hallucinations in long-form LLM generations, with performance comparable to expensive multi-sample approaches while drawing only a single sample from the LLM generator. Our key observation is that LLM hidden states are highly predictive of long-form factuality and that this information may be efficiently extracted at inference time using a lightweight probe.

## Key Contributions

- **Efficient Hallucination Detection**: Achieves competitive performance with up to 100x fewer FLOPs compared to multi-sample approaches
- **Single-Sample Inference**: Requires only one forward pass through the LLM, making it practical for large models
- **Cross-Model Generalization**: Probes trained on smaller models generalize to larger out-of-distribution models
- **Comprehensive Evaluation**: Benchmarked across open-source models up to 405B parameters

## Repository Structure

```
long-form-fact-probe/
├── long_fact_probes/          # Unified probe training framework
├── baselines/                # Baseline hallucination detection methods
├── benchmarks/               # Evaluation frameworks (FActScore, longfact)
├── scripts/                  # Experiment orchestration
├── plotting/                 # Visualization and analysis scripts
├── notebooks/               # Jupyter analysis notebooks
├── misc/                    # Experimental components
└── requirements.txt         # Python dependencies
```

## Quick Start

```bash
# Setup environment
git clone https://github.com/your-org/long-form-fact-probe.git
cd long-form-fact-probe
conda create -n hallucination-probe python=3.9
conda activate hallucination-probe
pip install -r requirements.txt

# Configure cache directories
export PROBE_CACHE_DIR="./cache"
export FACTSCORE_CACHE_DIR="./factscore_cache"
export HF_DATASETS_CACHE="./datasets_cache"

# Train hallucination detection probes
cd long_fact_probes
python train.py --model llama3.1-8b --train_data_dir ./train_data/

# Evaluate probes
python eval.py --model llama3.1-8b --probes_dir ./results/

# Generate predictions
python predict.py --model llama3.1-8b --probe_file ./results/best_probe.pkl
```

## Supported Models

- **Llama 3.1**: 8B, 70B, 405B parameter variants
- **Llama 3.2**: 3B parameter model  
- **Gemma 2**: 9B parameter model
- Extensible to any HuggingFace transformer

## Core Components

### Probe Training Framework (`long_fact_probes/`)
- Cross-validation with stratified k-fold and bootstrap statistics
- Configurable layer grouping strategies
- Multiple classifiers (Logistic Regression, XGBoost)
- GPU acceleration and memory optimization

### Baseline Methods (`baselines/`)
- Multi-sample consistency approaches
- Confidence-based hallucination scoring
- Retrieval-augmented classification
- Statistical baseline methods

### Benchmarking (`benchmarks/`)
- FActScore integration with atomic fact extraction
- Long-form generation evaluation
- Cross-model performance comparison

## Method Overview

Our approach leverages the key insight that LLM hidden states contain rich information about factuality that can be extracted using lightweight probes:

1. **Single Forward Pass**: Extract hidden states during standard LLM generation
2. **Lightweight Probe**: Train simple linear classifiers on hidden state representations
3. **Efficient Inference**: Detect hallucinations without additional LLM samples
4. **Cross-Model Transfer**: Probes generalize across different model architectures

## Development

### Adding New Models

```python
# Update MODEL_CONFIGS in long_fact_probes/train.py
MODEL_CONFIGS['new-model'] = {
    'name': 'NewModel-7B',
    'hf_name': 'org/new-model-7b-instruct', 
    'num_layers': 32
}
```

### Custom Classifiers

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomProbe(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        # Implementation
        return self
        
    def predict_proba(self, X):
        # Return probabilities
        return probabilities

# Register in get_classifier()
def get_classifier(classifier_name, **kwargs):
    if classifier_name == 'custom_probe':
        return CustomProbe(**kwargs)
```

## Performance Optimization

### Memory Management
```python
# Gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Streaming data processing
def process_large_dataset(data_path, batch_size=32):
    for batch in stream_batches(data_path, batch_size):
        yield process_batch(batch)
```

### GPU Utilization
```bash
# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model llama3.1-70b --use_gpu

# Enable caching
export PROBE_CACHE_DIR="/fast/ssd/cache"
export HF_HOME="/shared/models"
```

## Experimental Features

### Sparse Probing
```bash
cd misc/sparse_probe
python train_linear_probe_sp.py --sparsity 0.1 --model llama3.1-8b
```

### Activation Clamping
```bash
cd misc/clamp_exp  
python run_sentence_clamped.py --clamp_layers 10,15,20
```

### Cross-Model Transfer
```bash
# Train on one model, evaluate on another
python train.py --model llama3.1-8b --train_data_dir ./data/
python eval.py --model gemma2-9b --probes_dir ./results/llama/
```

## Evaluation Metrics

- **AUROC**: Area under ROC curve with bootstrap confidence intervals
- **Accuracy**: Classification accuracy on retained predictions
- **Efficiency**: FLOPs comparison vs multi-sample baselines
- **Transfer Performance**: Cross-model generalization analysis

## Debugging

### Common Issues
| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce batch size, enable gradient checkpointing |
| Slow loading | Use local model cache, SSD storage |
| Poor AUROC | Check data quality, increase regularization |
| Memory leak | Enable garbage collection, check tensor cleanup |

### Profiling
```bash
# Memory profiling
python -m memory_profiler train.py --model llama3.1-8b

# Execution profiling  
python -m cProfile -s time train.py --model llama3.1-8b
```

## Testing

```bash
# Unit tests
python -m pytest tests/ -v

# Integration tests
python -m pytest tests/integration/ --slow

# Performance benchmarks
python benchmarks/probe_speed.py
```

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{han2024simple,
    title={Simple Probes Detect Long-Form Hallucinations},
    author={Jiatong Han and Neil Band and Mohammed Razzak and Jannik Kossen and Tim G.J. Rudner and Yarin Gal},
    year={2024},
    journal={arXiv preprint},
    note={Under review}
}
```

## Authors

- **Jiatong Han** - Independent Researcher (julius.han@outlook.com)
- **Neil Band** - New York University
- **Mohammed Razzak** - University of Oxford
- **Jannik Kossen** - University of Oxford  
- **Tim G.J. Rudner** - New York University
- **Yarin Gal** - University of Oxford

## Contributing

### Development Workflow
1. Fork repository and create feature branch
2. Install dev dependencies: `pip install -e .[dev]`
3. Run tests: `pytest tests/`
4. Submit pull request with detailed description

### Code Standards
- Black formatting, flake8 linting
- Type hints for all public functions
- Google-style docstrings
- >80% test coverage

### Issue Reporting
Include: Python version, GPU info, minimal reproduction case, full error traceback

## License

MIT License - see LICENSE file for details.