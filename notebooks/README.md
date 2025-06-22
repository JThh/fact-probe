
Interactive analysis and visualization notebooks for fact probe experiments. Contains exploratory data analysis, statistical testing, and experimental workflows.

## Structure

```
notebooks/
├── analysis_and_plots.ipynb          # Main analysis workflow
├── shreshth_results_viz_for_fp.ipynb # Specialized result visualization  
├── draft_plots.ipynb                 # Experimental plotting
├── experimental_analysis_and_plots.ipynb # Extended analysis
└── README.md                         # This file
```

## Quick Start

```bash
# Start Jupyter
jupyter notebook
# or
jupyter lab

# Execute programmatically
jupyter nbconvert --to notebook --execute analysis_and_plots.ipynb
```

## Notebook Overview

### `analysis_and_plots.ipynb`
Primary analysis workflow with model performance comparison, statistical testing, and publication figures.

### `shreshth_results_viz_for_fp.ipynb` 
Specialized visualization for fact probe results and layer-wise analysis.

### `draft_plots.ipynb`
Experimental plotting sandbox for prototyping visualizations.

### `experimental_analysis_and_plots.ipynb`
Extended analysis with multi-model comparisons and advanced statistics.

## Development Patterns

### Standard Cell Structure
```python
# Cell 1: Imports and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Cell 2: Configuration
CONFIG = {
    'data_dir': '../data/',
    'output_dir': '../figures/',
    'random_seed': 42
}

# Cell 3: Data loading
def load_experimental_data():
    """Load and preprocess data."""
    pass

# Cell 4: Analysis functions
def statistical_analysis():
    """Perform analysis."""
    pass
```

### Data Integration
```python
# Load probe results
def load_probe_results(results_dir='../long_fact_probes/results/'):
    """Load training/evaluation results."""
    return pd.concat([pd.read_csv(f) for f in Path(results_dir).glob('*.csv')])

# Load plotting data
def load_plotting_data():
    """Load extracted plotting data."""
    data_dir = Path('../plotting/data/')
    return {f.stem: pd.read_csv(f) for f in data_dir.glob('*.csv')}
```

### Statistical Analysis Template
```python
def compare_models(results_df, model1, model2, metric='auroc'):
    """Statistical comparison between models."""
    group1 = results_df[results_df['model'] == model1][metric]
    group2 = results_df[results_df['model'] == model2][metric]
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    effect_size = (group1.mean() - group2.mean()) / np.sqrt((group1.var() + group2.var()) / 2)
    
    return {'p_value': p_value, 'effect_size': effect_size}
```

### Visualization Template
```python
def create_publication_figure(data, title, output_path=None):
    """Generate publication-ready figure."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.barplot(data=data, x='model', y='auroc', ax=ax)
    ax.set_title(title, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
```

## Integration with Codebase

### Using Long Fact Probes
```python
import sys
sys.path.append('../long_fact_probes/')

from train import run_cv_probing
from eval_utils import bootstrap_func
```

### Using Plotting Scripts
```python
sys.path.append('../plotting/')
from model_performance_comparison import plot_model_comparison
```

## Best Practices

### Reproducibility
```python
# Set seeds
np.random.seed(42)

# Document environment
import sys
print(f"Python: {sys.version}")
print(f"Pandas: {pd.__version__}")
```

### Code Organization
- Import libraries at top
- Define configuration early
- Create utility functions
- Load data once
- Organize analysis in logical sections
- Document findings in markdown cells

### Output Management
```python
# Save intermediate results
def save_analysis_results(data, filename):
    """Save for cross-notebook sharing."""
    output_path = Path('../data/intermediate/') / filename
    output_path.parent.mkdir(exist_ok=True)
    data.to_csv(output_path, index=False)
```
# Analysis Notebooks
 