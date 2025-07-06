# Simple Probes Detect Long-Form Hallucinations

A framework for detecting hallucinations in long-form LLM generations using lightweight probes on hidden states. This codebase implements methods from the paper "Simple Probes Detect Long-Form Hallucinations" and provides tools for training and evaluating hallucination detection probes.

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
├── long_fact_probes/          # Core probe training and evaluation framework
│   ├── train.py              # Train hallucination detection probes
│   ├── eval.py               # Evaluate trained probes
│   ├── predict.py            # Generate predictions with probes
│   ├── eval_utils.py         # Evaluation utilities
│   └── README.md             # Detailed usage instructions
├── baselines/                # Baseline hallucination detection methods
│   ├── graph-base-uncertainty/  # Graph-based uncertainty estimation
│   └── long_hallucinations/     # Long-form hallucination detection baselines
├── factuality_benchmarks/   # Evaluation frameworks
│   ├── FActScore/           # Atomic fact evaluation framework
│   └── long-form-factuality/ # Long-form factuality evaluation
├── scripts/                 # Experiment orchestration scripts
├── tests/                   # Unit and integration tests
└── requirements.txt         # Python dependencies
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/long-form-fact-probe.git
cd long-form-fact-probe

# Setup environment
conda create -n hallucination-probe python=3.9
conda activate hallucination-probe
pip install -r requirements.txt
```

### Configure Environment Variables

```bash
# Set cache directories for efficient model loading
export PROBE_CACHE_DIR="./cache"
export FACTSCORE_CACHE_DIR="./factscore_cache"  
export HF_DATASETS_CACHE="./datasets_cache"
export HF_HOME="./models_cache"
```

### Train and Evaluate Probes

```bash
# Navigate to probe training directory
cd long_fact_probes

# Train hallucination detection probes
python train.py --model llama3.1-8b --train_data_dir ./train_data/

# Evaluate trained probes
python eval.py --model llama3.1-8b --probes_dir ./results/

# Generate predictions on new data
python predict.py --model llama3.1-8b --probe_file ./results/best_probe.pkl
```

### Run Tests

```bash
# Run the complete test suite
python run_tests.py

# Or run specific tests
pytest tests/test_utils.py -v
pytest tests/test_integration.py -v
```

## Supported Models

- **Llama 3.1**: 8B, 70B, 405B parameter variants
- **Llama 3.2**: 3B parameter model  
- **Gemma 2**: 9B parameter model
- Extensible to any HuggingFace transformer model

## Probe Training Framework (`long_fact_probes/`)

The core framework for training and evaluating hallucination detection probes:

- **`train.py`**: Main training script with cross-validation and model selection
- **`eval.py`**: Comprehensive evaluation with bootstrap confidence intervals
- **`predict.py`**: Inference script for generating predictions
- **`eval_utils.py`**: Shared evaluation utilities and metrics

Features:
- Cross-validation with stratified k-fold and bootstrap statistics
- Configurable layer grouping strategies
- Multiple classifiers (Logistic Regression, XGBoost)
- GPU acceleration and memory optimization


## Usage Examples

### Basic Probe Training

```bash
cd long_fact_probes

# Train on Llama 3.1 8B with default settings
python train.py \
    --model llama3.1-8b \
    --train_data_dir ./data/factscore_train/ \
    --output_dir ./results/llama3.1-8b/ \
    --num_folds 5

# Evaluate the trained probes
python eval.py \
    --model llama3.1-8b \
    --probes_dir ./results/llama3.1-8b/ \
    --test_data_dir ./data/factscore_test/ \
    --output_file ./results/evaluation_results.json
```

### Cross-Model Evaluation

```bash
# Train probe on smaller model
python train.py --model llama3.1-8b --train_data_dir ./data/

# Test generalization to larger model  
python eval.py \
    --model llama3.1-70b \
    --probes_dir ./results/llama3.1-8b/ \
    --test_data_dir ./data/ \
    --cross_model_eval
```

### Running Baselines

```bash
# Graph-based uncertainty baseline
cd baselines/graph-base-uncertainty
python main.py --model llama3.1-8b --dataset factscore

# Long-form hallucination baselines
cd baselines/long_hallucinations  
# Note: See baselines/long_hallucinations/README.md for specific usage
python hallucination.py --model QADebertaEntailment
python hallucination.py --model SelfCheckBaseline
```

### Data Generation

```bash
# Generate biographical text and compute FactScores
cd scripts
python run_sentence.py \
    --hf-model-name meta-llama/Llama-3.1-8B-Instruct \
    --model-name Llama3.1-8B \
    --entities-file entities.txt \
    --max-entities 50
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

## Contributing

We welcome contributions! Please see individual component READMEs for specific contribution guidelines:

- [`long_fact_probes/README.md`](long_fact_probes/README.md) - Core probe framework
- [`tests/README.md`](tests/README.md) - Testing framework
- [`scripts/README.md`](scripts/README.md) - Utility scripts
