## Scripts

### `run_sentence.py`

**Purpose**: Generate long-form biographical text and compute FactScore evaluations.

**Usage**:
```bash
python run_sentence.py \
    --hf-model-name meta-llama/Llama-3.1-70B-Instruct \
    --model-name Llama3.1-70B \
    --entities-file ./FActScore/data/prompt_entities.txt \
    --entity-range "100:" \
    --max-entities 30 \
    --verbose
```


**Parameters**:
- `--hf-model-name`: HuggingFace model identifier
- `--model-name`: Human-readable model name for output files
- `--entities-file`: Path to file containing prompt entities
- `--entity-range`: Slice notation for entity selection (e.g., "100:200")
- `--max-entities`: Maximum number of entities to process
- `--temperature`: Sampling temperature for generation
- `--max-new-tokens`: Maximum tokens to generate per prompt
- `--verbose`: Enable detailed logging

### Usage Examples

Current workflow with existing scripts:

```bash
# Step 1: Generate and evaluate text
python scripts/run_sentence.py \
    --hf-model-name meta-llama/Llama-3.1-8B-Instruct \
    --model-name Llama3.1-8B \
    --entities-file entities.txt \
    --max-entities 50

# Step 2: Train probes (from long_fact_probes/)
cd long_fact_probes
python train.py --model llama3.1-8b --train_data_dir ../data/

# Step 3: Evaluate probes
python eval.py --model llama3.1-8b --probes_dir ./results/
```

## Configuration

### Environment Variables

Set these environment variables for optimal performance:

```bash
export PROBE_CACHE_DIR="./cache"
export FACTSCORE_CACHE_DIR="./factscore_cache"
export HF_DATASETS_CACHE="./datasets_cache"
export HF_HOME="./models_cache"
export OPENAI_API_KEY="your-api-key-here"  # For OpenAI models
```

### Supported Models

The scripts support the following model architectures:

- **Llama 3.1**: 8B, 70B, 405B parameter variants
- **Llama 3.2**: 3B parameter model
- **Gemma 2**: 9B parameter model
- **OpenAI**: GPT-4 variants (requires API key)

### Data Formats

Scripts handle multiple input formats:
- FactScore format (JSON with decisions)
- Simple list format (atom-label pairs)
- Structured dictionaries with metadata
