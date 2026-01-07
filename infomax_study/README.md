# InfoMax Activation Study

This project implements experiments to study how different activation functions affect information maximization in single-layer neural networks.

## Project Structure

```
infomax_study/
├── config/
│   ├── default.yaml      # Default experiment configuration
│   └── sweep.yaml        # Parameter sweep configuration
├── src/
│   ├── __init__.py
│   ├── activations.py    # Activation function implementations
│   ├── model.py          # Single layer model
│   ├── losses.py         # InfoMax loss functions
│   ├── metrics.py        # Evaluation metrics
│   ├── data.py           # Data loading utilities
│   └── training.py       # Training loop
├── scripts/
│   ├── run_experiment.py    # Run single experiment
│   ├── run_sweep.py         # Run parameter sweep
│   └── analyze_results.py   # Analyze and visualize results
├── tests/
│   ├── test_activations.py
│   ├── test_losses.py
│   └── test_metrics.py
├── results/              # Experiment outputs (created at runtime)
└── notebooks/            # Analysis notebooks (optional)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run a Single Experiment

```bash
cd infomax_study
python scripts/run_experiment.py --config config/default.yaml --output results/test
```

### Run a Full Parameter Sweep

The sweep configuration generates 180 experiments (6 activations × 3 hidden dims × 10 seeds):

```bash
cd infomax_study
python scripts/run_sweep.py --base-config config/default.yaml --sweep-config config/sweep.yaml --output results/sweep_v1
```

### Analyze Results

```bash
python scripts/analyze_results.py --results-dir results/sweep_v1 --output-dir analysis/
```

This will generate:
- `analysis/summary.csv` - Aggregated metrics by activation
- `analysis/effective_rank.png` - Box plot of effective rank
- `analysis/redundancy_vs_dead.png` - Scatter plot
- `analysis/training_curves_*.png` - Training curves per activation

## Run Tests

```bash
pytest tests/
```

## Configuration

Edit `config/default.yaml` to change:
- Dataset (mnist, synthetic_2d)
- Model architecture (hidden_dim, activation)
- Training hyperparameters (epochs, lr, batch_size)
- Loss function weights (lambda_tc)

Edit `config/sweep.yaml` to modify parameter sweep ranges.

## Activations Tested

- Identity (linear)
- ReLU
- Leaky ReLU
- Softmax
- Tanh
- Softplus

## Metrics Computed

- **Effective Rank**: Measure of weight matrix diversity
- **Weight Redundancy**: Similarity between learned weight vectors
- **Dead Units**: Count of inactive neurons
- **Output Correlation**: Decorrelation quality
- **Variance Ratio**: Balance of activation magnitudes

## Compute Requirements

- **Per experiment**: ~2 minutes on CPU, ~30 seconds on GPU
- **Full sweep (180 experiments)**: ~6 hours on CPU
- **Storage**: ~200 MB for all results

## Citation

Based on the InfoMax activation study implementation design.
