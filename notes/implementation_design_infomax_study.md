Filename: `implementation_design_infomax_study.md`

---

# Implementation Design: InfoMax Activation Study

## Overview

This document specifies the implementation for the InfoMax activation study experiment. The goal is clean, reproducible code that allows systematic comparison across activations, hyperparameters, and seeds.

## Language and Framework

**Python 3.10+**

**PyTorch** for:
- Model definition
- Automatic differentiation
- Optimization

**NumPy** for:
- Metric computation
- Data manipulation

**Pandas** for:
- Results aggregation
- Analysis

**Matplotlib / Seaborn** for:
- Visualization

## Project Structure

```
infomax_study/
├── config/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── activations.py
│   ├── losses.py
│   ├── metrics.py
│   ├── data.py
│   └── training.py
├── scripts/
│   ├── run_experiment.py
│   ├── run_sweep.py
│   └── analyze_results.py
├── notebooks/
│   └── visualize_results.ipynb
├── results/
│   └── [experiment outputs]
└── tests/
    ├── test_losses.py
    ├── test_metrics.py
    └── test_activations.py
```

## Configuration

Use YAML for experiment configuration. Single source of truth.

```yaml
# config/default.yaml

data:
  dataset: "mnist"  # or "synthetic_2d"
  batch_size: 128
  num_workers: 4

model:
  input_dim: 784  # auto-set from dataset
  hidden_dim: 16
  activation: "relu"  # identity, relu, softmax, tanh, leaky_relu, softplus

loss:
  lambda_tc: 1.0
  variance_eps: 1e-6

training:
  epochs: 100
  lr: 1e-3
  optimizer: "adam"
  seed: 42

logging:
  save_weights: true
  save_metrics_every: 10  # epochs
  output_dir: "results/"
```

## Core Components

### activations.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Activation(nn.Module):
    """Base class for activations."""
    pass

class Identity(Activation):
    def forward(self, z):
        return z

class ReLU(Activation):
    def forward(self, z):
        return F.relu(z)

class LeakyReLU(Activation):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, z):
        return F.leaky_relu(z, self.negative_slope)

class Softmax(Activation):
    def forward(self, z):
        return F.softmax(z, dim=-1)

class Tanh(Activation):
    def forward(self, z):
        return torch.tanh(z)

class Softplus(Activation):
    def forward(self, z):
        return F.softplus(z)

def get_activation(name: str) -> Activation:
    """Factory function."""
    activations = {
        "identity": Identity,
        "relu": ReLU,
        "leaky_relu": LeakyReLU,
        "softmax": Softmax,
        "tanh": Tanh,
        "softplus": Softplus,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]()
```

### model.py

```python
import torch
import torch.nn as nn
from .activations import get_activation

class SingleLayer(nn.Module):
    """
    Single layer model: z = Wx + b, a = activation(z)
    """
    def __init__(self, input_dim: int, hidden_dim: int, activation: str):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = get_activation(activation)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        z = self.linear(x)
        a = self.activation(z)
        return a, z
    
    @property
    def weight_matrix(self):
        return self.linear.weight.detach()
```

### losses.py

```python
import torch

def batch_variance(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute variance of each column (output dimension) across batch.
    
    Args:
        a: (N, K) activations
        eps: small constant for numerical stability
    
    Returns:
        (K,) variances
    """
    return a.var(dim=0) + eps

def batch_correlation(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute correlation matrix of outputs across batch.
    
    Args:
        a: (N, K) activations
    
    Returns:
        (K, K) correlation matrix
    """
    # Center
    a_centered = a - a.mean(dim=0, keepdim=True)
    
    # Standardize
    std = a.std(dim=0, keepdim=True) + eps
    a_standardized = a_centered / std
    
    # Correlation
    N = a.shape[0]
    corr = (a_standardized.T @ a_standardized) / (N - 1)
    
    return corr

def infomax_loss(a: torch.Tensor, lambda_tc: float = 1.0, eps: float = 1e-6) -> dict:
    """
    Compute InfoMax loss: -Σ log(Var(aⱼ)) + λ ||Corr(A) - I||²
    
    Args:
        a: (N, K) activations
        lambda_tc: weight on total correlation term
        eps: numerical stability
    
    Returns:
        dict with 'total', 'entropy', 'tc' losses
    """
    K = a.shape[1]
    
    # Entropy term (Gaussian approximation): maximize variance
    var = batch_variance(a, eps)
    entropy_loss = -torch.log(var).sum()
    
    # TC term (decorrelation proxy): minimize off-diagonal correlation
    corr = batch_correlation(a, eps)
    identity = torch.eye(K, device=a.device)
    tc_loss = ((corr - identity) ** 2).sum()
    
    total_loss = entropy_loss + lambda_tc * tc_loss
    
    return {
        "total": total_loss,
        "entropy": entropy_loss,
        "tc": tc_loss,
    }
```

### metrics.py

```python
import torch
import numpy as np

def weight_redundancy(W: torch.Tensor) -> float:
    """
    Measure redundancy in weight matrix via normalized Gram matrix.
    
    ||WᵀW / ||W||² - I/K||²
    
    Lower is better.
    """
    W = W.cpu().numpy()
    K = W.shape[0]
    
    # Normalize rows
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    W_normalized = W / norms
    
    # Gram matrix
    gram = W_normalized @ W_normalized.T
    
    # Compare to identity
    identity = np.eye(K)
    redundancy = ((gram - identity) ** 2).sum()
    
    return float(redundancy)

def effective_rank(W: torch.Tensor) -> float:
    """
    Compute effective rank of weight matrix.
    
    eff_rank = (Σσᵢ)² / Σσᵢ²
    
    Higher is better.
    """
    W = W.cpu().numpy()
    
    _, s, _ = np.linalg.svd(W, full_matrices=False)
    s = s + 1e-10  # stability
    
    eff_rank = (s.sum() ** 2) / (s ** 2).sum()
    
    return float(eff_rank)

def dead_units(a: torch.Tensor, threshold: float = 0.01) -> int:
    """
    Count units with near-zero activation across batch.
    
    Args:
        a: (N, K) activations
        threshold: activation rate below this = dead
    
    Returns:
        count of dead units
    """
    activation_rate = (a.abs() > 1e-6).float().mean(dim=0)
    dead = (activation_rate < threshold).sum()
    return int(dead)

def activation_rate_per_unit(a: torch.Tensor) -> np.ndarray:
    """
    Fraction of samples where each unit is active.
    
    Returns:
        (K,) array of activation rates
    """
    rate = (a.abs() > 1e-6).float().mean(dim=0)
    return rate.cpu().numpy()

def max_avg_responsibility(a: torch.Tensor) -> float:
    """
    For softmax outputs, measure collapse via max average responsibility.
    
    Close to 1 = collapse. Close to 1/K = balanced.
    """
    # Assumes a is already softmax output (sums to 1)
    avg_responsibility = a.mean(dim=0)
    return float(avg_responsibility.max())

def variance_ratio(a: torch.Tensor) -> float:
    """
    Ratio of max to min variance across units.
    
    High = some units much more active than others.
    """
    var = a.var(dim=0)
    ratio = var.max() / (var.min() + 1e-10)
    return float(ratio)

def output_correlation(a: torch.Tensor) -> float:
    """
    ||Corr(A) - I||²
    
    Lower is better (more independent).
    """
    K = a.shape[1]
    
    a_centered = a - a.mean(dim=0, keepdim=True)
    std = a.std(dim=0, keepdim=True) + 1e-6
    a_standardized = a_centered / std
    
    N = a.shape[0]
    corr = (a_standardized.T @ a_standardized) / (N - 1)
    
    identity = torch.eye(K, device=a.device)
    return float(((corr - identity) ** 2).sum())

def compute_all_metrics(model, dataloader, device) -> dict:
    """
    Compute all metrics over full dataset.
    """
    model.eval()
    
    all_a = []
    all_z = []
    
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            a, z = model(x)
            all_a.append(a.cpu())
            all_z.append(z.cpu())
    
    all_a = torch.cat(all_a, dim=0)
    all_z = torch.cat(all_z, dim=0)
    
    W = model.weight_matrix
    
    metrics = {
        "weight_redundancy": weight_redundancy(W),
        "effective_rank": effective_rank(W),
        "dead_units": dead_units(all_a),
        "variance_ratio": variance_ratio(all_a),
        "output_correlation": output_correlation(all_a),
        "activation_mean": float(all_a.mean()),
        "activation_std": float(all_a.std()),
    }
    
    # Softmax-specific
    if all_a.min() >= 0 and torch.allclose(all_a.sum(dim=1), torch.ones(all_a.shape[0]), atol=1e-5):
        metrics["max_avg_responsibility"] = max_avg_responsibility(all_a)
    
    return metrics
```

### data.py

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def get_mnist(batch_size: int, num_workers: int = 4):
    """
    Load MNIST, flatten to vectors.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])
    
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader, 784

def get_synthetic_2d(batch_size: int, n_samples: int = 10000, n_clusters: int = 8):
    """
    Generate 2D Gaussian clusters.
    """
    import numpy as np
    
    np.random.seed(42)
    
    # Cluster centers on a circle
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 2
    
    # Generate samples
    samples_per_cluster = n_samples // n_clusters
    X = []
    y = []
    
    for i, center in enumerate(centers):
        samples = np.random.randn(samples_per_cluster, 2) * 0.3 + center
        X.append(samples)
        y.extend([i] * samples_per_cluster)
    
    X = np.concatenate(X, axis=0).astype(np.float32)
    y = np.array(y)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    
    # Split
    split = int(0.8 * len(X))
    train_dataset = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    test_dataset = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, 2

def get_data(name: str, batch_size: int, **kwargs):
    """Factory function."""
    if name == "mnist":
        return get_mnist(batch_size, **kwargs)
    elif name == "synthetic_2d":
        return get_synthetic_2d(batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
```

### training.py

```python
import torch
import torch.optim as optim
from typing import Callable
import time

def train_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn: Callable,
    device: torch.device,
) -> dict:
    """
    Train for one epoch.
    
    Returns:
        dict of average losses
    """
    model.train()
    
    total_losses = {"total": 0, "entropy": 0, "tc": 0}
    n_batches = 0
    
    for x, _ in dataloader:
        x = x.to(device)
        
        optimizer.zero_grad()
        
        a, z = model(x)
        losses = loss_fn(a)
        
        losses["total"].backward()
        optimizer.step()
        
        for k, v in losses.items():
            total_losses[k] += v.item()
        n_batches += 1
    
    return {k: v / n_batches for k, v in total_losses.items()}

def train(
    model,
    train_loader,
    test_loader,
    loss_fn: Callable,
    epochs: int,
    lr: float,
    device: torch.device,
    metrics_fn: Callable = None,
    log_every: int = 10,
) -> dict:
    """
    Full training loop.
    
    Returns:
        dict with training history and final metrics
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "train_entropy": [],
        "train_tc": [],
        "epoch_time": [],
        "metrics": [],
    }
    
    for epoch in range(epochs):
        start = time.time()
        
        losses = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        elapsed = time.time() - start
        
        history["train_loss"].append(losses["total"])
        history["train_entropy"].append(losses["entropy"])
        history["train_tc"].append(losses["tc"])
        history["epoch_time"].append(elapsed)
        
        if metrics_fn is not None and (epoch + 1) % log_every == 0:
            metrics = metrics_fn(model, test_loader, device)
            metrics["epoch"] = epoch + 1
            history["metrics"].append(metrics)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {losses['total']:.4f} | "
                  f"Eff Rank: {metrics['effective_rank']:.2f} | "
                  f"Dead: {metrics['dead_units']}")
    
    # Final metrics
    if metrics_fn is not None:
        final_metrics = metrics_fn(model, test_loader, device)
        history["final_metrics"] = final_metrics
    
    return history
```

### scripts/run_experiment.py

```python
import argparse
import yaml
import torch
import json
import os
from pathlib import Path

from src.model import SingleLayer
from src.losses import infomax_loss
from src.metrics import compute_all_metrics
from src.data import get_data
from src.training import train

def run(config_path: str, output_dir: str = None):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    
    # Data
    train_loader, test_loader, input_dim = get_data(
        config["data"]["dataset"],
        config["data"]["batch_size"],
    )
    
    # Model
    model = SingleLayer(
        input_dim=input_dim,
        hidden_dim=config["model"]["hidden_dim"],
        activation=config["model"]["activation"],
    ).to(device)
    
    # Loss
    lambda_tc = config["loss"]["lambda_tc"]
    eps = config["loss"]["variance_eps"]
    loss_fn = lambda a: infomax_loss(a, lambda_tc=lambda_tc, eps=eps)
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        epochs=config["training"]["epochs"],
        lr=config["training"]["lr"],
        device=device,
        metrics_fn=compute_all_metrics,
        log_every=config["logging"]["save_metrics_every"],
    )
    
    # Save results
    if output_dir is None:
        output_dir = config["logging"]["output_dir"]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Save weights
    if config["logging"]["save_weights"]:
        torch.save(model.state_dict(), output_dir / "model.pt")
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    run(args.config, args.output)
```

### scripts/run_sweep.py

```python
import argparse
import yaml
import itertools
import subprocess
from pathlib import Path
import copy

def generate_configs(base_config_path: str, sweep_config_path: str, output_dir: str):
    """
    Generate all config files for sweep.
    """
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)
    
    with open(sweep_config_path) as f:
        sweep = yaml.safe_load(f)
    
    # Example sweep config:
    # activation: [identity, relu, softmax, tanh, leaky_relu, softplus]
    # hidden_dim: [16, 32, 64]
    # lambda_tc: [0.1, 1.0, 10.0]
    # seed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    
    configs = []
    
    for combo in itertools.product(*values):
        config = copy.deepcopy(base_config)
        name_parts = []
        
        for key, val in zip(keys, combo):
            # Navigate to correct config location
            if key == "activation":
                config["model"]["activation"] = val
            elif key == "hidden_dim":
                config["model"]["hidden_dim"] = val
            elif key == "lambda_tc":
                config["loss"]["lambda_tc"] = val
            elif key == "seed":
                config["training"]["seed"] = val
            
            name_parts.append(f"{key}={val}")
        
        name = "_".join(name_parts)
        config_path = output_dir / f"{name}.yaml"
        
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        configs.append({
            "name": name,
            "config_path": str(config_path),
            "output_path": str(output_dir / name),
        })
    
    return configs

def run_sweep(configs: list, parallel: int = 1):
    """
    Run all experiments.
    """
    for config in configs:
        print(f"Running: {config['name']}")
        
        subprocess.run([
            "python", "scripts/run_experiment.py",
            "--config", config["config_path"],
            "--output", config["output_path"],
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--sweep-config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    
    configs = generate_configs(args.base_config, args.sweep_config, args.output)
    print(f"Generated {len(configs)} configurations")
    
    run_sweep(configs, args.parallel)
```

### scripts/analyze_results.py

```python
import argparse
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_dir: str) -> pd.DataFrame:
    """
    Load all experiment results into a DataFrame.
    """
    results_dir = Path(results_dir)
    
    rows = []
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        history_path = exp_dir / "history.json"
        config_path = exp_dir / "config.yaml"
        
        if not history_path.exists():
            continue
        
        with open(history_path) as f:
            history = json.load(f)
        
        with open(config_path) as f:
            import yaml
            config = yaml.safe_load(f)
        
        row = {
            "name": exp_dir.name,
            "activation": config["model"]["activation"],
            "hidden_dim": config["model"]["hidden_dim"],
            "lambda_tc": config["loss"]["lambda_tc"],
            "seed": config["training"]["seed"],
            "final_loss": history["train_loss"][-1],
        }
        
        if "final_metrics" in history:
            row.update(history["final_metrics"])
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by activation.
    """
    agg = df.groupby("activation").agg({
        "effective_rank": ["mean", "std"],
        "weight_redundancy": ["mean", "std"],
        "dead_units": ["mean", "std"],
        "output_correlation": ["mean", "std"],
        "final_loss": ["mean", "std"],
    })
    
    return agg

def plot_effective_rank(df: pd.DataFrame, output_path: str = None):
    """
    Box plot of effective rank by activation.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="activation", y="effective_rank")
    plt.title("Effective Rank by Activation")
    plt.ylabel("Effective Rank")
    plt.xlabel("Activation")
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

def plot_redundancy_vs_dead(df: pd.DataFrame, output_path: str = None):
    """
    Scatter plot: redundancy vs dead units, colored by activation.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="weight_redundancy",
        y="dead_units",
        hue="activation",
        style="activation",
        s=100,
    )
    plt.title("Weight Redundancy vs Dead Units")
    plt.xlabel("Weight Redundancy (lower = better)")
    plt.ylabel("Dead Units (lower = better)")
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

def plot_training_curves(results_dir: str, activation: str, output_path: str = None):
    """
    Plot training curves for one activation across seeds.
    """
    results_dir = Path(results_dir)
    
    plt.figure(figsize=(12, 4))
    
    for exp_dir in results_dir.iterdir():
        if activation not in exp_dir.name:
            continue
        
        history_path = exp_dir / "history.json"
        if not history_path.exists():
            continue
        
        with open(history_path) as f:
            history = json.load(f)
        
        plt.subplot(1, 3, 1)
        plt.plot(history["train_loss"], alpha=0.5)
        plt.title("Total Loss")
        
        plt.subplot(1, 3, 2)
        plt.plot(history["train_entropy"], alpha=0.5)
        plt.title("Entropy Loss")
        
        plt.subplot(1, 3, 3)
        plt.plot(history["train_tc"], alpha=0.5)
        plt.title("TC Loss")
    
    plt.suptitle(f"Training Curves: {activation}")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="analysis/")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_results(args.results_dir)
    
    # Summary table
    summary = summary_table(df)
    print(summary)
    summary.to_csv(output_dir / "summary.csv")
    
    # Plots
    plot_effective_rank(df, output_dir / "effective_rank.png")
    plot_redundancy_vs_dead(df, output_dir / "redundancy_vs_dead.png")
    
    for activation in df["activation"].unique():
        plot_training_curves(
            args.results_dir,
            activation,
            output_dir / f"training_curves_{activation}.png"
        )
```

## Sweep Configuration

```yaml
# config/sweep.yaml

activation:
  - identity
  - relu
  - softmax
  - tanh
  - leaky_relu
  - softplus

hidden_dim:
  - 16
  - 32
  - 64

lambda_tc:
  - 1.0

seed:
  - 1
  - 2
  - 3
  - 4
  - 5
  - 6
  - 7
  - 8
  - 9
  - 10
```

Total experiments: 6 activations × 3 hidden dims × 10 seeds = 180 runs.

## Tests

### tests/test_losses.py

```python
import torch
import pytest
from src.losses import batch_variance, batch_correlation, infomax_loss

def test_batch_variance_shape():
    a = torch.randn(32, 16)
    var = batch_variance(a)
    assert var.shape == (16,)

def test_batch_variance_positive():
    a = torch.randn(32, 16)
    var = batch_variance(a)
    assert (var > 0).all()

def test_batch_correlation_shape():
    a = torch.randn(32, 16)
    corr = batch_correlation(a)
    assert corr.shape == (16, 16)

def test_batch_correlation_diagonal():
    a = torch.randn(32, 16)
    corr = batch_correlation(a)
    diag = torch.diag(corr)
    assert torch.allclose(diag, torch.ones(16), atol=1e-5)

def test_infomax_loss_keys():
    a = torch.randn(32, 16)
    losses = infomax_loss(a)
    assert "total" in losses
    assert "entropy" in losses
    assert "tc" in losses

def test_infomax_loss_differentiable():
    a = torch.randn(32, 16, requires_grad=True)
    losses = infomax_loss(a)
    losses["total"].backward()
    assert a.grad is not None
```

### tests/test_metrics.py

```python
import torch
import numpy as np
import pytest
from src.metrics import weight_redundancy, effective_rank, dead_units

def test_effective_rank_identity():
    # Identity matrix should have full rank
    W = torch.eye(10)
    rank = effective_rank(W)
    assert abs(rank - 10) < 0.1

def test_effective_rank_rank_one():
    # Rank-1 matrix
    W = torch.ones(10, 5)
    rank = effective_rank(W)
    assert abs(rank - 1) < 0.1

def test_weight_redundancy_orthogonal():
    # Orthogonal rows should have low redundancy
    W = torch.eye(5, 10)
    redundancy = weight_redundancy(W)
    assert redundancy < 0.1

def test_weight_redundancy_identical():
    # Identical rows should have high redundancy
    W = torch.ones(5, 10)
    redundancy = weight_redundancy(W)
    assert redundancy > 1.0

def test_dead_units_none():
    a = torch.randn(100, 16)
    dead = dead_units(a)
    assert dead == 0

def test_dead_units_all():
    a = torch.zeros(100, 16)
    dead = dead_units(a)
    assert dead == 16
```

## Running the Experiments

```bash
# Single experiment
python scripts/run_experiment.py --config config/default.yaml --output results/test

# Full sweep
python scripts/run_sweep.py \
    --base-config config/default.yaml \
    --sweep-config config/sweep.yaml \
    --output results/sweep_v1

# Analyze results
python scripts/analyze_results.py --results-dir results/sweep_v1 --output-dir analysis/
```

## Compute Requirements

**Per experiment:**
- MNIST: ~2 minutes on CPU, ~30 seconds on GPU
- 100 epochs, 60k samples, batch size 128

**Full sweep:**
- 180 experiments × 2 minutes = 6 hours on CPU
- Parallelizable across GPUs

**Storage:**
- ~1 MB per experiment (history + weights)
- ~200 MB total

## Deliverables

After running sweep and analysis:

1. `analysis/summary.csv` — metrics by activation
2. `analysis/effective_rank.png` — box plot
3. `analysis/redundancy_vs_dead.png` — scatter plot
4. `analysis/training_curves_*.png` — per activation
5. All raw results in `results/sweep_v1/`