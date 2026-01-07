"""Experiment 3: Benchmark comparison with standard SAE.

Compares:
- Our model: Encoder with LSE + InfoMax losses
- Baseline: Standard SAE with encoder + decoder + L1 sparsity

Metrics:
- Reconstruction MSE
- Sparsity (L0)
- Parameter count

Outputs results to results/benchmark/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import json
import argparse
from typing import Dict, Any
import numpy as np

from src.model import Encoder, StandardSAE
from src.data import get_mnist
from src.training import train, set_seed, evaluate
from src.losses import sae_loss
from src.metrics import reconstruction_mse, sparsity_l0


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_our_model(
    train_loader,
    test_loader,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    loss_config: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Train our LSE + InfoMax encoder."""
    set_seed(seed)

    model = Encoder(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        activation=model_config.get("activation", "relu"),
    )

    full_config = {
        "training": train_config,
        "loss": loss_config,
    }

    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=full_config,
        device=device,
        verbose=True,
    )

    # Final evaluation
    model.eval()
    model = model.to(device)

    all_a = []
    all_x = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            a, _ = model(x)
            all_a.append(a)
            all_x.append(x)

    all_a = torch.cat(all_a, dim=0)
    all_x = torch.cat(all_x, dim=0)
    W = model.W

    return {
        "reconstruction_mse": reconstruction_mse(all_x, all_a, W),
        "sparsity_l0": sparsity_l0(all_a),
        "parameters": count_parameters(model),
        "history": history,
    }


def train_baseline_sae(
    train_loader,
    test_loader,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    l1_weight: float,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Train standard SAE baseline."""
    set_seed(seed)

    model = StandardSAE(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
    )

    # Use same training config but with SAE loss
    loss_config = {"l1_weight": l1_weight}
    full_config = {
        "training": train_config,
        "loss": loss_config,
    }

    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=full_config,
        device=device,
        verbose=True,
    )

    # Final evaluation
    model.eval()
    model = model.to(device)

    all_a = []
    all_x = []
    all_recon = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            recon, a = model(x)
            all_a.append(a)
            all_x.append(x)
            all_recon.append(recon)

    all_a = torch.cat(all_a, dim=0)
    all_x = torch.cat(all_x, dim=0)
    all_recon = torch.cat(all_recon, dim=0)

    mse = float(((all_x - all_recon) ** 2).mean().item())

    return {
        "reconstruction_mse": mse,
        "sparsity_l0": sparsity_l0(all_a),
        "parameters": count_parameters(model),
        "history": history,
    }


def run_benchmark(config_path: str, output_dir: str, device: torch.device):
    """Run full benchmark comparison.

    Args:
        config_path: Path to benchmark.yaml config
        output_dir: Directory to save results
        device: Device to train on
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seeds = config.get("seeds", [1, 2, 3, 4, 5])
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
        batch_size=train_config.get("batch_size", 128)
    )

    # Results storage
    ours_results = []
    baseline_results = []

    # Run experiments
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print("=" * 60)

        # Train our model
        print("\nTraining our model (LSE + InfoMax)...")
        ours = train_our_model(
            train_loader=train_loader,
            test_loader=test_loader,
            model_config=model_config,
            train_config=train_config,
            loss_config=loss_config,
            seed=seed,
            device=device,
        )
        ours["seed"] = seed
        ours_results.append(ours)

        # Train baseline
        print("\nTraining baseline SAE...")
        baseline = train_baseline_sae(
            train_loader=train_loader,
            test_loader=test_loader,
            model_config=model_config,
            train_config=train_config,
            l1_weight=baseline_l1_weight,
            seed=seed,
            device=device,
        )
        baseline["seed"] = seed
        baseline_results.append(baseline)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert history to serializable format
    def make_serializable(results):
        for r in results:
            if "history" in r:
                # Keep only final values to reduce file size
                r["final_train_loss"] = r["history"]["train_loss"][-1]
                r["final_test_loss"] = r["history"]["test_loss"][-1]
                del r["history"]
        return results

    all_results = {
        "ours": make_serializable(ours_results.copy()),
        "baseline": make_serializable(baseline_results.copy()),
        "config": {
            "model": model_config,
            "training": train_config,
            "seeds": seeds,
        },
    }

    with open(output_path / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    def summarize(results, name):
        mse = [r["reconstruction_mse"] for r in results]
        l0 = [r["sparsity_l0"] for r in results]
        params = results[0]["parameters"]

        print(f"\n{name}:")
        print(f"  Reconstruction MSE: {np.mean(mse):.4f} ± {np.std(mse):.4f}")
        print(f"  Sparsity (L0):      {np.mean(l0):.2f} ± {np.std(l0):.2f}")
        print(f"  Parameters:         {params:,}")

    summarize(ours_results, "Ours (LSE + InfoMax)")
    summarize(baseline_results, "Baseline SAE")

    # Parameter savings
    ours_params = ours_results[0]["parameters"]
    baseline_params = baseline_results[0]["parameters"]
    savings = (baseline_params - ours_params) / baseline_params * 100
    print(f"\nParameter savings: {savings:.1f}%")

    print(f"\nResults saved to {output_path / 'benchmark_results.json'}")


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

    print(f"Using device: {device}")

    run_benchmark(args.config, args.output_dir, device)


if __name__ == "__main__":
    main()
