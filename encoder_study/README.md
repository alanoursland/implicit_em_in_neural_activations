# Encoder Study: Decoder-Free Sparse Autoencoders from Implicit EM

This codebase validates decoder-free sparse autoencoders derived from implicit EM with InfoMax regularization.

## Paper Claim

Decoder-free sparse autoencoders arise naturally from implicit EM when log-sum-exp (LSE) objectives are combined with InfoMax regularization. The gradient of the LSE loss equals the responsibility: `∂L_LSE/∂E_j = r_j`.

## Installation

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch numpy pandas matplotlib seaborn pyyaml tqdm pytest

# For MNIST (included with torchvision)
pip install torchvision

# Optional: for LLM activations
pip install transformers datasets
```

## Running Tests

Tests are located in the `tests/` directory and use pytest.

### Run all tests

```bash
cd encoder_study
pytest tests/ -v
```

### Run specific test modules

```bash
# Test model implementations
pytest tests/test_model.py -v

# Test loss functions (including gradient=responsibility theorem)
pytest tests/test_losses.py -v

# Test metrics
pytest tests/test_metrics.py -v

# Test data loading (uses synthetic data, MNIST tests are skipped by default)
pytest tests/test_data.py -v

# Test training loop
pytest tests/test_training.py -v
```

### Run tests with coverage

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Key tests to verify

1. **test_losses.py::TestLSELoss::test_gradient_equals_responsibility** - Verifies the core theorem
2. **test_model.py::TestEncoder** - Verifies encoder shapes and gradients
3. **test_training.py::TestIntegration** - Full pipeline tests

### Expected test output

All tests should pass. The key theorem verification test confirms:
- Correlation between gradient and responsibility > 0.9999
- This is the mathematical foundation of the paper

## Data Configuration

MNIST data is loaded from `E:/ml_datasets` by default. To change this:

1. Update `src/data.py` - modify the `data_dir` default parameter
2. Update config files in `config/` - modify `data.data_dir` field
3. Or pass `data_dir` parameter when calling `get_mnist()`

## Running Experiments

### Experiment 1: Verify the Theorem

Verifies that `∂L_LSE/∂E_j = r_j` (gradient equals responsibility).

```bash
cd encoder_study
python scripts/verify_theorem.py --output figures/fig1_theorem.pdf
```

**Expected output:**
- Scatter plot with perfect correlation (~1.0) between gradient and responsibility
- Points should lie exactly on the y=x identity line
- This confirms the key theoretical result

### Experiment 2: Ablation Study

Shows that InfoMax regularization prevents collapse.

```bash
python scripts/run_ablation.py --config config/ablation.yaml --output-dir results/ablation
```

**Configurations tested:**
| Config | LSE | Variance | Decorrelation | Expected Behavior |
|--------|-----|----------|---------------|-------------------|
| A: lse_only | ✓ | | | Collapse (one component wins) |
| B: lse_var | ✓ | ✓ | | No dead units, some redundancy |
| C: lse_var_tc | ✓ | ✓ | ✓ | Stable, diverse features |
| D: var_tc_only | | ✓ | ✓ | Different behavior (no EM structure) |

**Expected output:**
- Config A should show many dead units (collapse)
- Config C should show fewest dead units and lowest redundancy
- Results saved to `results/ablation/ablation_results.json`

### Experiment 3: Benchmark

Compares our method to standard sparse autoencoders.

```bash
python scripts/run_benchmark.py --config config/benchmark.yaml --output-dir results/benchmark
```

**Expected output:**
- Comparable reconstruction MSE
- Similar sparsity (L0)
- ~50% fewer parameters (no decoder)
- Results saved to `results/benchmark/benchmark_results.json`

### Generate Paper Figures

After running experiments:

```bash
python scripts/generate_figures.py --results-dir results --output-dir figures
```

**Outputs:**
- `figures/fig1_theorem.pdf` — Gradient vs responsibility scatter plot
- `figures/fig2_ablation.pdf` — Ablation results (similarity matrices + metrics)
- `figures/fig3_benchmark.pdf` — Benchmark comparison charts

## Directory Structure

```
encoder_study/
├── config/
│   ├── default.yaml       # Default hyperparameters
│   ├── ablation.yaml      # Ablation study configs
│   └── benchmark.yaml     # Benchmark configs
├── src/
│   ├── __init__.py
│   ├── model.py           # Encoder and StandardSAE classes
│   ├── losses.py          # LSE, variance, correlation losses
│   ├── metrics.py         # Evaluation metrics
│   ├── data.py            # Data loaders (MNIST, synthetic, LLM)
│   └── training.py        # Training and evaluation loops
├── scripts/
│   ├── verify_theorem.py  # Experiment 1
│   ├── run_ablation.py    # Experiment 2
│   ├── run_benchmark.py   # Experiment 3
│   └── generate_figures.py
├── tests/
│   ├── conftest.py        # Pytest fixtures
│   ├── test_model.py      # Model tests
│   ├── test_losses.py     # Loss function tests
│   ├── test_metrics.py    # Metrics tests
│   ├── test_data.py       # Data loading tests
│   └── test_training.py   # Training loop tests
├── results/               # Experiment outputs
│   ├── ablation/          # Ablation study results
│   └── benchmark/         # Benchmark results
└── figures/               # Generated figures
```

## Key Components

### Encoder (model.py)

Single-layer encoder with no decoder:
```python
from src.model import Encoder

model = Encoder(input_dim=784, hidden_dim=64, activation="relu")
a, z = model(x)  # a: activations, z: pre-activations
W = model.W      # Weight matrix for analysis
```

### LSE Loss (losses.py)

Log-sum-exp loss that returns responsibilities:
```python
from src.losses import lse_loss

loss, responsibilities = lse_loss(a)
# responsibilities = softmax(-a)
# Key theorem: gradient of loss w.r.t. a equals responsibilities
```

### Combined Loss

```python
from src.losses import combined_loss

config = {
    "lambda_lse": 1.0,   # LSE (EM structure)
    "lambda_var": 1.0,   # Variance (prevents dead units)
    "lambda_tc": 1.0,    # Decorrelation (prevents redundancy)
}
losses = combined_loss(a, W, config)
# losses["total"] for backprop
# losses["responsibilities"] for analysis
```

## Success Criteria

1. **Figure 1:** Scatter plot shows identity line (correlation > 0.999)
2. **Figure 2:** Config A shows collapse; Config C is stable
3. **Figure 3:** Competitive MSE with ~50% fewer parameters

If all three hold, the paper validates the theoretical derivation.

## Hyperparameters

Key hyperparameters in `config/default.yaml`:

```yaml
model:
  hidden_dim: 64      # Number of features
  activation: relu    # Activation function

loss:
  lambda_lse: 1.0     # LSE weight
  lambda_var: 1.0     # Variance weight
  lambda_tc: 1.0      # Total correlation weight

training:
  epochs: 100
  batch_size: 128
  lr: 0.001

data:
  dataset: mnist
  data_dir: "E:/ml_datasets"
```

## Troubleshooting

### MNIST not found
Ensure MNIST data exists at `E:/ml_datasets/MNIST/`. The first run will attempt to download if not present.

### Tests fail with import error
Run tests from the `encoder_study` directory:
```bash
cd encoder_study
pytest tests/ -v
```

### CUDA out of memory
Reduce batch size in config files or use CPU:
```bash
python scripts/run_ablation.py --device cpu
```

## Citation

If you use this code, please cite the associated paper.
