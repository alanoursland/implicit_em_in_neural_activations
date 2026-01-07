"""Experiment 3: Benchmark comparison with standard SAE.

Compares:
- Our model: Encoder with LSE + InfoMax losses (decoder-free)
- Baseline: Standard SAE with encoder + decoder + L1 sparsity

Primary Metrics:
- Linear Probe Accuracy (feature quality for downstream tasks)
- L0 Sparsity (mean active features per input)
- Parameter Count

Secondary Metrics:
- Normalized Reconstruction MSE

Outputs results to results/benchmark/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import yaml
import json
import argparse
from typing import Dict, Any, Tuple, List
import numpy as np

from src.model import Encoder, StandardSAE
from src.data import get_mnist
from src.training import set_seed
from src.losses import combined_loss, sae_loss
from src.metrics import (
    sparsity_l0,
    linear_probe_accuracy,
    normalized_reconstruction_mse,
)


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_features(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    is_sae: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from a data loader."""
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            if x.device != device:
                x = x.to(device)

            if is_sae:
                _, a = model(x)
            else:
                a, _ = model(x)

            all_features.append(a.cpu().numpy())
            all_labels.append(y.cpu().numpy() if isinstance(y, torch.Tensor) else y)

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def train_encoder_with_logging(
    model: Encoder,
    train_loader,
    loss_config: Dict[str, Any],
    train_config: Dict[str, Any],
    device: torch.device,
    log_every: int = 10,
) -> Dict[str, List[float]]:
    """Train decoder-free encoder with detailed logging."""
    model = model.to(device)
    epochs = train_config.get("epochs", 100)
    lr = train_config.get("lr", 0.001)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "lse": [], "var": [], "tc": []}

    print("Training:")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_lse = 0.0
        total_var = 0.0
        total_tc = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch[0]
            if x.device != device:
                x = x.to(device)

            optimizer.zero_grad()
            a, _ = model(x)
            W = model.W
            losses = combined_loss(a, W, loss_config)
            loss = losses["total"]

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_lse += losses["lse"].item()
            total_var += losses["var"].item()
            total_tc += losses["tc"].item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_lse = total_lse / n_batches
        avg_var = total_var / n_batches
        avg_tc = total_tc / n_batches

        history["loss"].append(avg_loss)
        history["lse"].append(avg_lse)
        history["var"].append(avg_var)
        history["tc"].append(avg_tc)

        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {avg_loss:9.2f} | "
                f"LSE: {avg_lse:9.2f} | "
                f"Var: {avg_var:9.2f} | "
                f"TC: {avg_tc:7.2f}"
            )

    return history


def train_sae_with_logging(
    model: StandardSAE,
    train_loader,
    loss_config: Dict[str, Any],
    train_config: Dict[str, Any],
    device: torch.device,
    log_every: int = 10,
) -> Dict[str, List[float]]:
    """Train standard SAE with detailed logging."""
    model = model.to(device)
    epochs = train_config.get("epochs", 100)
    lr = train_config.get("lr", 0.001)
    l1_weight = loss_config.get("l1_weight", 0.01)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "mse": [], "l1": []}

    print("Training:")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_l1 = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch[0]
            if x.device != device:
                x = x.to(device)

            optimizer.zero_grad()
            recon, a = model(x)
            losses = sae_loss(x, recon, a, l1_weight=l1_weight)
            loss = losses["total"]

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += losses["mse"].item()
            total_l1 += losses["l1"].item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_mse = total_mse / n_batches
        avg_l1 = total_l1 / n_batches

        history["loss"].append(avg_loss)
        history["mse"].append(avg_mse)
        history["l1"].append(avg_l1)

        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {avg_loss:8.4f} | "
                f"MSE: {avg_mse:8.4f} | "
                f"L1: {avg_l1:6.2f}"
            )

    return history


def evaluate_model(
    model: torch.nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    is_sae: bool = False,
) -> Dict[str, Any]:
    """Evaluate model on all metrics."""
    model.eval()

    print("\nEvaluating:")
    print("  Extracting features...", end=" ", flush=True)
    train_features, train_labels = extract_features(model, train_loader, device, is_sae)
    test_features, test_labels = extract_features(model, test_loader, device, is_sae)
    print("done")

    print("  Training linear probe...", end=" ", flush=True)
    probe_acc = linear_probe_accuracy(train_features, train_labels, test_features, test_labels)
    print("done")

    print("  Computing sparsity...", end=" ", flush=True)
    all_a = []
    all_x = []
    all_recon = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0]
            if x.device != device:
                x = x.to(device)

            if is_sae:
                recon, a = model(x)
                all_recon.append(recon)
            else:
                a, _ = model(x)

            all_a.append(a)
            all_x.append(x)

    all_a = torch.cat(all_a, dim=0)
    all_x = torch.cat(all_x, dim=0)

    l0, density = sparsity_l0(all_a, threshold=0.01)
    hidden_dim = all_a.shape[1]
    print("done")

    # Reconstruction MSE
    if is_sae:
        all_recon = torch.cat(all_recon, dim=0)
        recon_mse = float(F.mse_loss(all_recon, all_x).item())
        W_dec = model.decoder.weight.T
        norm_recon_mse = normalized_reconstruction_mse(all_x, all_a, W_dec)
    else:
        W = model.W
        recon_mse = float(F.mse_loss(all_a @ W, all_x).item())
        norm_recon_mse = normalized_reconstruction_mse(all_x, all_a, W)

    return {
        "linear_probe_accuracy": probe_acc,
        "l0_sparsity": l0,
        "l0_density": density,
        "hidden_dim": hidden_dim,
        "parameters": count_parameters(model),
        "reconstruction_mse": recon_mse,
        "normalized_reconstruction_mse": norm_recon_mse,
    }


def run_benchmark(config_path: str, output_dir: str, device: torch.device):
    """Run full benchmark comparison."""
    # Print header
    print("=" * 80)
    print(" " * 15 + "BENCHMARK: DECODER-FREE VS STANDARD SAE")
    print("=" * 80)
    print(f"Using device: {device}")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seeds = config.get("seeds", [1, 2, 3])
    model_config = config["model"]
    train_config = config["training"]
    loss_config = config.get("loss", {})

    # Get baseline L1 weight
    baseline_l1_weight = 0.01
    for m in config.get("models", []):
        if m.get("type") == "standard_sae":
            baseline_l1_weight = m.get("l1_weight", 0.01)

    # Load data
    print("Loading MNIST...")
    train_loader, test_loader = get_mnist(
        batch_size=train_config.get("batch_size", 128),
        use_gpu_cache=True,
        device=str(device),
    )
    print("MNIST loaded and cached on GPU.")

    # Results storage
    ours_results = []
    baseline_results = []

    # ===== Train all decoder-free models first =====
    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"Model: Decoder-Free Encoder | Seed: {seed}")
        print("=" * 70)

        set_seed(seed)

        model = Encoder(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            activation=model_config.get("activation", "relu"),
        )

        print(f"Parameters: {count_parameters(model):,}")

        train_encoder_with_logging(
            model=model,
            train_loader=train_loader,
            loss_config=loss_config,
            train_config=train_config,
            device=device,
            log_every=10,
        )

        metrics = evaluate_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            is_sae=False,
        )
        metrics["seed"] = seed
        ours_results.append(metrics)

        print(f"\n--- Seed {seed} Results ---")
        print(f"Linear Probe Accuracy: {metrics['linear_probe_accuracy']*100:.2f}%")
        print(f"L0 Sparsity: {metrics['l0_sparsity']:.1f} / {metrics['hidden_dim']} ({metrics['l0_density']*100:.1f}% density)")
        print(f"Normalized Recon MSE: {metrics['normalized_reconstruction_mse']:.4f}")

    # ===== Train all standard SAE models =====
    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"Model: Standard SAE | Seed: {seed}")
        print("=" * 70)

        set_seed(seed)

        model = StandardSAE(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
        )

        print(f"Parameters: {count_parameters(model):,}")

        baseline_loss_config = {"l1_weight": baseline_l1_weight}
        train_sae_with_logging(
            model=model,
            train_loader=train_loader,
            loss_config=baseline_loss_config,
            train_config=train_config,
            device=device,
            log_every=10,
        )

        metrics = evaluate_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            is_sae=True,
        )
        metrics["seed"] = seed
        baseline_results.append(metrics)

        print(f"\n--- Seed {seed} Results ---")
        print(f"Linear Probe Accuracy: {metrics['linear_probe_accuracy']*100:.2f}%")
        print(f"L0 Sparsity: {metrics['l0_sparsity']:.1f} / {metrics['hidden_dim']} ({metrics['l0_density']*100:.1f}% density)")
        print(f"Normalized Recon MSE: {metrics['normalized_reconstruction_mse']:.4f}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def summarize_results(results):
        return {
            "linear_probe_accuracy": {
                "mean": float(np.mean([r["linear_probe_accuracy"] for r in results])),
                "std": float(np.std([r["linear_probe_accuracy"] for r in results])),
            },
            "l0_sparsity": {
                "mean": float(np.mean([r["l0_sparsity"] for r in results])),
                "std": float(np.std([r["l0_sparsity"] for r in results])),
            },
            "l0_density": {
                "mean": float(np.mean([r["l0_density"] for r in results])),
                "std": float(np.std([r["l0_density"] for r in results])),
            },
            "hidden_dim": results[0]["hidden_dim"],
            "parameters": results[0]["parameters"],
            "reconstruction_mse": {
                "mean": float(np.mean([r["reconstruction_mse"] for r in results])),
                "std": float(np.std([r["reconstruction_mse"] for r in results])),
            },
            "normalized_reconstruction_mse": {
                "mean": float(np.mean([r["normalized_reconstruction_mse"] for r in results])),
                "std": float(np.std([r["normalized_reconstruction_mse"] for r in results])),
            },
        }

    all_results = {
        "ours": summarize_results(ours_results),
        "standard_sae": summarize_results(baseline_results),
        "per_seed": {
            "ours": ours_results,
            "standard_sae": baseline_results,
        },
        "config": {
            "model": model_config,
            "training": train_config,
            "seeds": seeds,
        },
    }

    with open(output_path / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    ours_summary = all_results["ours"]
    sae_summary = all_results["standard_sae"]

    print("\n" + "=" * 80)
    print(" " * 25 + "BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'':24} | {'Decoder-Free':^20} | {'Standard SAE':^20}")
    print(f"{'':24} | {'Mean ± Std':^20} | {'Mean ± Std':^20}")
    print("-" * 80)

    # Linear Probe Accuracy
    ours_acc = ours_summary["linear_probe_accuracy"]
    sae_acc = sae_summary["linear_probe_accuracy"]
    print(
        f"{'Linear Probe Accuracy':<24} | "
        f"{ours_acc['mean']*100:5.2f}% ± {ours_acc['std']*100:4.2f}%    | "
        f"{sae_acc['mean']*100:5.2f}% ± {sae_acc['std']*100:4.2f}%"
    )

    # L0 Sparsity
    ours_l0 = ours_summary["l0_sparsity"]
    sae_l0 = sae_summary["l0_sparsity"]
    ours_density = ours_summary["l0_density"]
    sae_density = sae_summary["l0_density"]
    hidden_dim = ours_summary["hidden_dim"]
    print(
        f"{'L0 Sparsity':<24} | "
        f"{ours_l0['mean']:4.1f}/{hidden_dim} ({ours_density['mean']*100:4.1f}%)     | "
        f"{sae_l0['mean']:4.1f}/{hidden_dim} ({sae_density['mean']*100:4.1f}%)"
    )

    # Parameters
    print(
        f"{'Parameters':<24} | "
        f"{ours_summary['parameters']:>12,}        | "
        f"{sae_summary['parameters']:>12,}"
    )

    # Normalized Recon MSE
    ours_mse = ours_summary["normalized_reconstruction_mse"]
    sae_mse = sae_summary["normalized_reconstruction_mse"]
    print(
        f"{'Normalized Recon MSE':<24} | "
        f"{ours_mse['mean']:7.4f} ± {ours_mse['std']:6.4f}    | "
        f"{sae_mse['mean']:7.4f} ± {sae_mse['std']:6.4f}"
    )

    print("-" * 80)

    # Bottom line stats
    ours_params = ours_summary["parameters"]
    sae_params = sae_summary["parameters"]
    savings = (sae_params - ours_params) / sae_params * 100
    acc_diff = (ours_acc["mean"] - sae_acc["mean"]) * 100

    print(f"{'Parameter Reduction':<24} | {savings:.1f}%")
    print(f"{'Accuracy Difference':<24} | {acc_diff:+.2f}%")
    print("=" * 80)

    print(f"Results saved to {output_path / 'benchmark_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark comparison")
    parser.add_argument(
        "--config",
        type=str,
        default="config/benchmark.yaml",
        help="Path to benchmark config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark",
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

    run_benchmark(args.config, args.output_dir, device)


if __name__ == "__main__":
    main()
