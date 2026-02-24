"""Experiment: Activation sign comparison.

Compares weight visualizations for four model variants:
1. Encoder with ReLU (standard: maximize activations)
2. Encoder with -ReLU (negative: minimize activations)
3. SAE with ReLU (standard)
4. SAE with -ReLU (negative)

The key question: Does the sign of activations (maximizing vs minimizing)
affect the structure of learned weights?

This experiment tests the paper's thesis about two geometric learning regimes:
- Cosine-like (ReLU): activation ≈ template similarity, weights look like data
- Mahalanobis-like (-ReLU): activation ≈ distance from boundary, weights are normals

Outputs weight visualizations and geometric metrics to results/activation_comparison/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data import get_mnist
from src.training import set_seed
from src.losses import combined_loss, sae_loss
from src.metrics import sparsity_l0
from src.geometric_metrics import (
    compute_all_geometric_metrics,
    print_metric_comparison,
    get_expected_ranges,
)


class NegativeReLU(nn.Module):
    """Negative ReLU: -max(0, x) = min(0, -x)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.relu(x)


class EncoderWithActivation(nn.Module):
    """Encoder with configurable activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        negative_relu: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.negative_relu = negative_relu

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = NegativeReLU() if negative_relu else nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.linear(x)
        a = self.activation(z)
        return a, z

    @property
    def W(self) -> torch.Tensor:
        return self.linear.weight


class SAEWithActivation(nn.Module):
    """SAE with configurable encoder activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        negative_relu: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.negative_relu = negative_relu

        self.encoder_linear = nn.Linear(input_dim, hidden_dim)
        self.encoder_activation = NegativeReLU() if negative_relu else nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder_linear(x)
        a = self.encoder_activation(z)
        recon = self.decoder(a)
        return recon, a

    def get_encoder_weights(self) -> torch.Tensor:
        return self.encoder_linear.weight.data


def train_encoder(
    model: EncoderWithActivation,
    train_loader,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Train encoder with LSE+InfoMax objective."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    epochs = config["training"]["epochs"]
    infomax_config = config["infomax"]

    final_loss = 0.0
    for epoch in tqdm(range(epochs), desc="Training Encoder"):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch[0]
            if x.device != device:
                x = x.to(device)

            optimizer.zero_grad()
            a, _ = model(x)
            losses = combined_loss(a, model.W, infomax_config)
            losses["total"].backward()
            optimizer.step()

            total_loss += losses["total"].item()
            n_batches += 1

        final_loss = total_loss / n_batches

    return {"final_loss": final_loss}


def train_sae(
    model: SAEWithActivation,
    train_loader,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Train SAE with MSE + L1 objective."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    epochs = config["training"]["epochs"]
    l1_weight = config["sae"]["l1_weight"]

    final_loss = 0.0
    final_mse = 0.0
    for epoch in tqdm(range(epochs), desc="Training SAE"):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch[0]
            if x.device != device:
                x = x.to(device)

            optimizer.zero_grad()
            recon, a = model(x)
            losses = sae_loss(x, recon, a, l1_weight=l1_weight)
            losses["total"].backward()
            optimizer.step()

            total_loss += losses["total"].item()
            total_mse += losses["mse"].item()
            n_batches += 1

        final_loss = total_loss / n_batches
        final_mse = total_mse / n_batches

    return {"final_loss": final_loss, "final_mse": final_mse}


def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
    is_sae: bool = False,
) -> Dict[str, float]:
    """Evaluate model metrics."""
    model.eval()
    all_a = []
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0]
            if x.device != device:
                x = x.to(device)

            if is_sae:
                recon, a = model(x)
                total_mse += torch.nn.functional.mse_loss(recon, x).item()
            else:
                a, _ = model(x)

            all_a.append(a)
            n_batches += 1

    all_a = torch.cat(all_a, dim=0)
    l0, density = sparsity_l0(all_a, threshold=0.01)

    # Compute activation statistics
    mean_activation = all_a.mean().item()
    std_activation = all_a.std().item()
    frac_positive = (all_a > 0).float().mean().item()
    frac_negative = (all_a < 0).float().mean().item()
    frac_zero = (all_a == 0).float().mean().item()

    result = {
        "l0_sparsity": l0,
        "l0_density": density,
        "mean_activation": mean_activation,
        "std_activation": std_activation,
        "frac_positive": frac_positive,
        "frac_negative": frac_negative,
        "frac_zero": frac_zero,
    }

    if is_sae:
        result["reconstruction_mse"] = total_mse / n_batches

    return result


def compute_geometric_metrics_for_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
    is_sae: bool = False,
    n_samples: int = 5000,
) -> Dict[str, float]:
    """
    Compute geometric metrics for a trained model.

    Args:
        model: Trained encoder or SAE
        data_loader: Data loader
        device: Compute device
        is_sae: Whether model is an SAE
        n_samples: Number of samples to use for metric computation

    Returns:
        Dictionary of geometric metrics
    """
    model.eval()

    # Collect data samples and activations
    all_x = []
    all_a = []
    all_z = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0]
            if x.device != device:
                x = x.to(device)

            if is_sae:
                # For SAE, get pre-activation from encoder
                z = model.encoder_linear(x)
                a = model.encoder_activation(z)
            else:
                a, z = model(x)

            all_x.append(x)
            all_a.append(a)
            all_z.append(z)

            if sum(x.shape[0] for x in all_x) >= n_samples:
                break

    X = torch.cat(all_x, dim=0)[:n_samples]
    a = torch.cat(all_a, dim=0)[:n_samples]
    z = torch.cat(all_z, dim=0)[:n_samples]  # Pre-activation

    # Get weights and biases
    if is_sae:
        W = model.encoder_linear.weight.data  # (K, D)
        b = model.encoder_linear.bias.data    # (K,)
    else:
        W = model.linear.weight.data  # (K, D)
        b = model.linear.bias.data    # (K,)

    # Compute responsibilities: softmax(-z) where z is pre-activation
    # This is the key: LSE gradient = softmax(-a) assigns responsibility
    # to components with small activation
    r = F.softmax(-z, dim=1)

    # Compute all geometric metrics
    metrics = compute_all_geometric_metrics(
        W=W,
        b=b,
        X=X,
        a=z,  # Use pre-activation for correlation metrics
        r=r,
        image_size=28
    )

    return metrics


def visualize_weights(
    weights: torch.Tensor,
    title: str,
    output_path: str,
):
    """Visualize encoder weights as 8x8 grid of 28x28 images."""
    W = weights.cpu().numpy()
    features = W.reshape(64, 28, 28)

    # Symmetric scaling around zero
    vmax = abs(features).max()

    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(features[i], cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def visualize_all_weights(
    all_weights: Dict[str, torch.Tensor],
    output_dir: str,
):
    """Create comparison figure with all four weight matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    titles = [
        ("encoder_relu", "Encoder + ReLU\n(maximize activations)"),
        ("encoder_neg_relu", "Encoder + (-ReLU)\n(minimize activations)"),
        ("sae_relu", "SAE + ReLU\n(maximize activations)"),
        ("sae_neg_relu", "SAE + (-ReLU)\n(minimize activations)"),
    ]

    # Find global vmax for consistent scaling
    global_vmax = max(abs(w.cpu().numpy()).max() for w in all_weights.values())

    for ax, (key, title) in zip(axes.flat, titles):
        W = all_weights[key].cpu().numpy()
        features = W.reshape(64, 28, 28)

        # Create 8x8 mosaic
        mosaic = np.zeros((8 * 28, 8 * 28))
        for i in range(8):
            for j in range(8):
                idx = i * 8 + j
                mosaic[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = features[idx]

        im = ax.imshow(mosaic, cmap="RdBu", vmin=-global_vmax, vmax=global_vmax)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()

    output_path = Path(output_dir) / "comparison_all.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saved comparison: {output_path}")

    # Also save PDF
    output_path_pdf = Path(output_dir) / "comparison_all.pdf"
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    for ax, (key, title) in zip(axes.flat, titles):
        W = all_weights[key].cpu().numpy()
        features = W.reshape(64, 28, 28)

        mosaic = np.zeros((8 * 28, 8 * 28))
        for i in range(8):
            for j in range(8):
                idx = i * 8 + j
                mosaic[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = features[idx]

        ax.imshow(mosaic, cmap="RdBu", vmin=-global_vmax, vmax=global_vmax)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison: {output_path_pdf}")


def run_experiment(config_path: str, output_dir: str, device: torch.device):
    """Run full activation comparison experiment."""
    print("=" * 80)
    print(" " * 20 + "ACTIVATION SIGN COMPARISON")
    print("=" * 80)
    print(f"Using device: {device}")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    # Load data
    print("\nLoading MNIST...")
    train_loader, test_loader = get_mnist(
        batch_size=config["training"]["batch_size"],
        use_gpu_cache=True,
        device=str(device),
    )
    print("MNIST loaded.\n")

    results = {}
    all_weights = {}
    geometric_results = {}

    # ==========================================================================
    # Variant 1: Encoder + ReLU
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Variant 1: Encoder + ReLU (maximize activations)")
    print("=" * 60)

    set_seed(seed)
    model_enc_relu = EncoderWithActivation(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        negative_relu=False,
    )

    train_results = train_encoder(model_enc_relu, train_loader, config, device)
    eval_results = evaluate_model(model_enc_relu, test_loader, device, is_sae=False)
    geo_metrics = compute_geometric_metrics_for_model(model_enc_relu, test_loader, device, is_sae=False)

    results["encoder_relu"] = {**train_results, **eval_results}
    geometric_results["encoder_relu"] = geo_metrics
    all_weights["encoder_relu"] = model_enc_relu.linear.weight.data.clone()

    print(f"Final loss: {train_results['final_loss']:.4f}")
    print(f"Mean activation: {eval_results['mean_activation']:.4f}")
    print(f"Frac positive: {eval_results['frac_positive']:.4f}")
    print(f"Frac zero: {eval_results['frac_zero']:.4f}")

    visualize_weights(
        all_weights["encoder_relu"],
        "Encoder + ReLU (maximize)",
        f"{output_dir}/encoder_relu.png",
    )

    # ==========================================================================
    # Variant 2: Encoder + (-ReLU)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Variant 2: Encoder + (-ReLU) (minimize activations)")
    print("=" * 60)

    set_seed(seed)
    model_enc_neg = EncoderWithActivation(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        negative_relu=True,
    )

    train_results = train_encoder(model_enc_neg, train_loader, config, device)
    eval_results = evaluate_model(model_enc_neg, test_loader, device, is_sae=False)
    geo_metrics = compute_geometric_metrics_for_model(model_enc_neg, test_loader, device, is_sae=False)

    results["encoder_neg_relu"] = {**train_results, **eval_results}
    geometric_results["encoder_neg_relu"] = geo_metrics
    all_weights["encoder_neg_relu"] = model_enc_neg.linear.weight.data.clone()

    print(f"Final loss: {train_results['final_loss']:.4f}")
    print(f"Mean activation: {eval_results['mean_activation']:.4f}")
    print(f"Frac negative: {eval_results['frac_negative']:.4f}")
    print(f"Frac zero: {eval_results['frac_zero']:.4f}")

    visualize_weights(
        all_weights["encoder_neg_relu"],
        "Encoder + (-ReLU) (minimize)",
        f"{output_dir}/encoder_neg_relu.png",
    )

    # ==========================================================================
    # Variant 3: SAE + ReLU
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Variant 3: SAE + ReLU (maximize activations)")
    print("=" * 60)

    set_seed(seed)
    model_sae_relu = SAEWithActivation(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        negative_relu=False,
    )

    train_results = train_sae(model_sae_relu, train_loader, config, device)
    eval_results = evaluate_model(model_sae_relu, test_loader, device, is_sae=True)
    geo_metrics = compute_geometric_metrics_for_model(model_sae_relu, test_loader, device, is_sae=True)

    results["sae_relu"] = {**train_results, **eval_results}
    geometric_results["sae_relu"] = geo_metrics
    all_weights["sae_relu"] = model_sae_relu.get_encoder_weights().clone()

    print(f"Final loss: {train_results['final_loss']:.4f}")
    print(f"Reconstruction MSE: {eval_results['reconstruction_mse']:.6f}")
    print(f"Mean activation: {eval_results['mean_activation']:.4f}")
    print(f"Frac positive: {eval_results['frac_positive']:.4f}")

    visualize_weights(
        all_weights["sae_relu"],
        "SAE + ReLU (maximize)",
        f"{output_dir}/sae_relu.png",
    )

    # ==========================================================================
    # Variant 4: SAE + (-ReLU)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Variant 4: SAE + (-ReLU) (minimize activations)")
    print("=" * 60)

    set_seed(seed)
    model_sae_neg = SAEWithActivation(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        negative_relu=True,
    )

    train_results = train_sae(model_sae_neg, train_loader, config, device)
    eval_results = evaluate_model(model_sae_neg, test_loader, device, is_sae=True)
    geo_metrics = compute_geometric_metrics_for_model(model_sae_neg, test_loader, device, is_sae=True)

    results["sae_neg_relu"] = {**train_results, **eval_results}
    geometric_results["sae_neg_relu"] = geo_metrics
    all_weights["sae_neg_relu"] = model_sae_neg.get_encoder_weights().clone()

    print(f"Final loss: {train_results['final_loss']:.4f}")
    print(f"Reconstruction MSE: {eval_results['reconstruction_mse']:.6f}")
    print(f"Mean activation: {eval_results['mean_activation']:.4f}")
    print(f"Frac negative: {eval_results['frac_negative']:.4f}")

    visualize_weights(
        all_weights["sae_neg_relu"],
        "SAE + (-ReLU) (minimize)",
        f"{output_dir}/sae_neg_relu.png",
    )

    # ==========================================================================
    # Combined visualization
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Creating combined comparison figure...")
    print("=" * 60)

    visualize_all_weights(all_weights, output_dir)

    # ==========================================================================
    # Save results
    # ==========================================================================
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Merge geometric results into main results
    for name in results:
        results[name]["geometric"] = geometric_results[name]

    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save geometric results separately for easy access
    with open(output_path / "geometric_metrics.json", "w") as f:
        json.dump(geometric_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print(" " * 30 + "SUMMARY")
    print("=" * 80)
    print(f"\n{'Variant':<25} | {'Mean Act':<10} | {'Frac +/-/0':<20} | {'Sparsity':<10}")
    print("-" * 80)

    for name, res in results.items():
        frac_str = f"{res['frac_positive']:.2f}/{res['frac_negative']:.2f}/{res['frac_zero']:.2f}"
        print(f"{name:<25} | {res['mean_activation']:>10.4f} | {frac_str:<20} | {res['l0_density']:.4f}")

    # ==========================================================================
    # Geometric Metrics Comparison
    # ==========================================================================
    print("\n" + "=" * 80)
    print(" " * 20 + "GEOMETRIC METRICS COMPARISON")
    print("=" * 80)

    # Print geometric metrics table
    print_metric_comparison(geometric_results)

    # Print interpretation guide
    print("\nMetric Interpretation:")
    print("-" * 60)
    print("Weight-Input Alignment: Cosine→+0.3-0.8, Mahalanobis→-0.2-0.2")
    print("Activation-Cosine Corr: Cosine→+0.7-0.9, Mahalanobis→-0.3-0.3")
    print("Frac Bias Below Mean:   Cosine→<0.5,     Mahalanobis→>0.5")
    print("Variance Norm Index:    Cosine→~0,       Mahalanobis→+0.3-0.7")
    print("Frac Positive Preact:   Cosine→<0.5,     Mahalanobis→~0.5")
    print("Responsibility Entropy: Cosine→Higher,   Mahalanobis→Lower")
    print("Template Correlation:   Cosine→High,     Mahalanobis→Low")
    print("Center-Surround Corr:   Cosine→Positive, Mahalanobis→Negative")

    print("\n" + "=" * 80)
    print(f"\nResults saved to {output_path}")
    print(f"  - results.json (all metrics)")
    print(f"  - geometric_metrics.json (geometric metrics only)")


def main():
    parser = argparse.ArgumentParser(description="Run activation sign comparison")
    parser.add_argument(
        "--config",
        type=str,
        default="config/activation_comparison.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/activation_comparison",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, auto-detected if not specified)",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_experiment(args.config, args.output_dir, device)


if __name__ == "__main__":
    main()
