# Encoder Study: Implementation Design

## Overview

Minimal codebase to validate decoder-free sparse autoencoders derived from implicit EM. Three experiments, three figures, one clean result.

## Directory Structure

```
encoder_study/
├── config/
│   ├── default.yaml
│   ├── ablation.yaml
│   └── benchmark.yaml
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── data.py
│   └── training.py
├── scripts/
│   ├── verify_theorem.py
│   ├── run_ablation.py
│   ├── run_benchmark.py
│   └── generate_figures.py
├── results/
│   └── [outputs]
└── figures/
    └── [paper figures]
```

## Core Components

### model.py

**Class: Encoder**

```
Encoder(input_dim, hidden_dim, activation)
```

Single layer encoder. No decoder.

- `input_dim`: Size of input (e.g., 784 for MNIST, activation dim for LLM)
- `hidden_dim`: Number of features/components
- `activation`: Activation function after linear layer (relu, identity, etc.)

Methods:
- `forward(x)` → returns activations `a` and pre-activations `z`
- `get_weights()` → returns W matrix for analysis

Properties:
- `W`: Weight matrix (hidden_dim × input_dim)
- `b`: Bias vector (hidden_dim)

### losses.py

**Function: lse_loss**

```
lse_loss(a) → scalar, responsibilities
```

Computes -log Σ_j exp(-a_j) and returns both the loss and the responsibilities r_j = softmax(-a).

Returns responsibilities separately for Experiment 1 verification.

**Function: variance_loss**

```
variance_loss(a) → scalar
```

Computes -Σ_j log(Var(a_j) + ε) over batch.

**Function: correlation_loss**

```
correlation_loss(a) → scalar
```

Computes ||Corr(A) - I||² over batch.

**Function: weight_redundancy_loss**

```
weight_redundancy_loss(W) → scalar
```

Computes ||W^T W - I||² on normalized rows.

**Function: combined_loss**

```
combined_loss(a, W, config) → dict
```

Combines all losses with configurable lambdas. Returns dict with:
- `total`: Combined loss for backprop
- `lse`: LSE component
- `var`: Variance component
- `tc`: Correlation component
- `wr`: Weight redundancy component
- `responsibilities`: For analysis

### metrics.py

**Function: dead_units**

```
dead_units(a, threshold=0.01) → int
```

Count units with variance below threshold.

**Function: redundancy_score**

```
redundancy_score(a) → float
```

||Corr(A) - I||² (off-diagonal only).

**Function: weight_redundancy**

```
weight_redundancy(W) → float
```

||W^T W - I||² on normalized rows.

**Function: effective_rank**

```
effective_rank(W) → float
```

Effective dimensionality of weight matrix.

**Function: responsibility_entropy**

```
responsibility_entropy(r) → float
```

Average H(r) per sample. Measures competition sharpness.

**Function: usage_distribution**

```
usage_distribution(r) → array
```

E_x[r(x)] — average responsibility per component.

**Function: reconstruction_mse**

```
reconstruction_mse(x, a, W) → float
```

||x - W^T a||² — reconstruction using transposed weights.

### data.py

**Function: get_mnist**

```
get_mnist(batch_size, flatten=True) → train_loader, test_loader
```

Standard MNIST, flattened to 784.

**Function: get_synthetic**

```
get_synthetic(n_samples, n_features, n_components) → data, true_components
```

Synthetic data with known ground truth for sanity checks.

**Function: get_llm_activations**

```
get_llm_activations(model_name, layer, cache_path) → train_loader, test_loader
```

Cached activations from transformer. For benchmark experiment.

### training.py

**Function: train_epoch**

```
train_epoch(model, loader, optimizer, loss_fn) → metrics_dict
```

Single epoch. Returns losses and metrics.

**Function: train**

```
train(model, train_loader, test_loader, config) → history
```

Full training loop. Logs metrics, saves checkpoints.

**Function: evaluate**

```
evaluate(model, loader, loss_fn) → metrics_dict
```

Evaluation pass. Computes all metrics.

## Scripts

### verify_theorem.py

**Purpose:** Experiment 1 — Verify ∂L_LSE/∂E_j = r_j

**Method:**
1. Create random input batch
2. Forward pass, compute activations
3. Compute responsibilities r = softmax(-a)
4. Backward pass on LSE loss only
5. Extract gradients ∂L/∂a
6. Scatter plot: r vs gradient

**Output:** `figures/theorem_verification.pdf`

No training needed. Single forward/backward pass.

### run_ablation.py

**Purpose:** Experiment 2 — Show collapse without InfoMax

**Method:**
1. Define four configurations:
   - A: LSE only (λ_var=0, λ_tc=0)
   - B: LSE + Variance (λ_var>0, λ_tc=0)
   - C: LSE + Variance + Correlation (λ_var>0, λ_tc>0)
   - D: Variance + Correlation only (no LSE)
2. Train each on MNIST
3. Compute metrics: dead units, redundancy, usage distribution
4. Save results

**Output:** `results/ablation/` with metrics per configuration

### run_benchmark.py

**Purpose:** Experiment 3 — Compare to standard SAE

**Method:**
1. Train our model (LSE + InfoMax)
2. Train baseline SAE (encoder + decoder + L1)
3. Compare:
   - Reconstruction MSE (ours uses W^T)
   - Sparsity (L0)
   - Parameters

**Output:** `results/benchmark/` with comparison metrics

### generate_figures.py

**Purpose:** Generate paper figures from results

**Outputs:**
- `figures/fig1_theorem.pdf` — Gradient vs responsibility scatter
- `figures/fig2_ablation.pdf` — Similarity matrices or bar chart
- `figures/fig3_benchmark.pdf` — Comparison table/chart

## Configuration

### default.yaml

```yaml
model:
  input_dim: 784
  hidden_dim: 64
  activation: relu

loss:
  lambda_lse: 1.0
  lambda_var: 1.0
  lambda_tc: 1.0
  lambda_wr: 0.0
  variance_eps: 1e-6

training:
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: adam
  seed: 42

data:
  dataset: mnist
```

### ablation.yaml

```yaml
configurations:
  - name: lse_only
    lambda_lse: 1.0
    lambda_var: 0.0
    lambda_tc: 0.0
    
  - name: lse_var
    lambda_lse: 1.0
    lambda_var: 1.0
    lambda_tc: 0.0
    
  - name: lse_var_tc
    lambda_lse: 1.0
    lambda_var: 1.0
    lambda_tc: 1.0
    
  - name: var_tc_only
    lambda_lse: 0.0
    lambda_var: 1.0
    lambda_tc: 1.0

seeds: [1, 2, 3]
```

### benchmark.yaml

```yaml
models:
  - name: ours
    type: encoder_lse_infomax
    
  - name: baseline_sae
    type: standard_sae
    l1_weight: 0.01

dataset: mnist  # or llm_activations
epochs: 100
seeds: [1, 2, 3, 4, 5]
```

## Baseline Implementation

**Class: StandardSAE**

```
StandardSAE(input_dim, hidden_dim)
```

Standard sparse autoencoder for comparison.

- Encoder: Linear + ReLU
- Decoder: Linear
- Loss: MSE reconstruction + L1 sparsity

Methods:
- `forward(x)` → returns reconstruction, activations
- `encode(x)` → returns activations only

## Key Design Decisions

**1. Responsibilities computed inside loss function**

The LSE loss returns both the scalar loss and the responsibilities. This allows Experiment 1 verification without modifying the training loop.

**2. Modular loss components**

Each loss term is a separate function. Combined loss assembles them based on config. Easy to ablate.

**3. No decoder in main model**

The Encoder class has no decoder. Reconstruction (for metrics) uses W^T explicitly. This enforces the "decoder-free" constraint.

**4. Metrics separate from loss**

Metrics are computed for logging/analysis but don't affect gradients. Clear separation.

**5. Config-driven experiments**

All hyperparameters in YAML. Scripts read config. Reproducible.

## Dependencies

```
torch
numpy
pandas
matplotlib
seaborn
pyyaml
tqdm
```

Optional for LLM activations:
```
transformers
datasets
```

## Execution Order

```bash
# 1. Verify theorem (no training)
python scripts/verify_theorem.py

# 2. Run ablation (4 configs × 3 seeds)
python scripts/run_ablation.py --config config/ablation.yaml

# 3. Run benchmark (2 models × 5 seeds)
python scripts/run_benchmark.py --config config/benchmark.yaml

# 4. Generate figures
python scripts/generate_figures.py --results-dir results/ --output-dir figures/
```

## Expected Outputs

### Figure 1: Theorem Verification
- Scatter plot
- X-axis: Computed responsibility r_j
- Y-axis: Gradient ∂L/∂a_j
- Should show perfect y=x line
- One plot, ~1000 points (batch × hidden_dim)

### Figure 2: Ablation Results
Option A: 2×2 grid of feature similarity matrices
Option B: Bar chart of dead units / redundancy by config
Option C: Table with metrics

### Figure 3: Benchmark Comparison
Table or bar chart:
- Reconstruction MSE
- Sparsity (L0)
- Parameter count
- Training time (optional)

## Code Size Estimate

| File | Lines |
|------|-------|
| model.py | 50 |
| losses.py | 80 |
| metrics.py | 60 |
| data.py | 50 |
| training.py | 80 |
| verify_theorem.py | 40 |
| run_ablation.py | 60 |
| run_benchmark.py | 80 |
| generate_figures.py | 100 |
| **Total** | ~600 |

Minimal codebase. Three experiments. Three figures. One clean paper.