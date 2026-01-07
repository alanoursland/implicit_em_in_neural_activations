"""Experiment 2: Ablation study showing collapse without InfoMax.

Trains four configurations:
- A: LSE only (λ_var=0, λ_tc=0) - expected to collapse
- B: LSE + Variance (λ_var>0, λ_tc=0) - no dead units, some redundancy
- C: LSE + Variance + Correlation (full) - stable, diverse
- D: Variance + Correlation only (no LSE) - different behavior

Outputs metrics and figures to results/ablation/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import json
import argparse
from typing import Dict, Any, List
import numpy as np

from src.model import Encoder
from src.data import get_mnist
from src.training import train, set_seed, evaluate
from src.metrics import usage_distribution, feature_similarity_matrix


def run_single_config(
    config_name: str,
    loss_config: Dict[str, Any],
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    train_loader,
    test_loader,
    seed: int,
    device: torch.device,
    checkpoint_dir: str = None,
) -> Dict[str, Any]:
    """Run training for a single configuration.

    Args:
        config_name: Name of the configuration
        loss_config: Loss configuration
        model_config: Model configuration
        train_config: Training configuration
        train_loader: Training data loader
        test_loader: Test data loader
        seed: Random seed
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Dict with final metrics and training history
    """
    set_seed(seed)

    # Create model
    model = Encoder(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        activation=model_config.get("activation", "relu"),
    )

    # Create full config
    full_config = {
        "training": train_config,
        "loss": loss_config,
    }

    # Train
    print(f"\nTraining {config_name} (seed={seed})...")
    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=full_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        verbose=True,
    )

    # Final evaluation
    model.eval()
    final_metrics = evaluate(model, test_loader, loss_config, device)

    # Get usage distribution
    all_r = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            a, _ = model(x)
            neg_a = -a
            r = torch.softmax(neg_a, dim=1)
            all_r.append(r)

    all_r = torch.cat(all_r, dim=0)
    usage = usage_distribution(all_r)

    # Get feature similarity matrix
    W = model.get_weights()
    similarity = feature_similarity_matrix(W)

    return {
        "config_name": config_name,
        "seed": seed,
        "final_metrics": final_metrics,
        "history": history,
        "usage_distribution": usage.tolist(),
        "feature_similarity": similarity.tolist(),
    }


def run_ablation(config_path: str, output_dir: str, device: torch.device):
    """Run full ablation study.

    Args:
        config_path: Path to ablation.yaml config
        output_dir: Directory to save results
        device: Device to train on
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    configurations = config["configurations"]
    seeds = config.get("seeds", [1, 2, 3])
    model_config = config["model"]
    train_config = config["training"]
    base_loss_config = config.get("loss", {})

    # Load data
    print("Loading MNIST...")
    train_loader, test_loader = get_mnist(
        batch_size=train_config.get("batch_size", 128)
    )

    # Results storage
    all_results = []

    # Run each configuration with each seed
    for cfg in configurations:
        config_name = cfg["name"]

        # Build loss config for this configuration
        loss_config = base_loss_config.copy()
        loss_config["lambda_lse"] = cfg.get("lambda_lse", 0.0)
        loss_config["lambda_var"] = cfg.get("lambda_var", 0.0)
        loss_config["lambda_tc"] = cfg.get("lambda_tc", 0.0)

        for seed in seeds:
            checkpoint_dir = f"{output_dir}/checkpoints/{config_name}_seed{seed}"

            results = run_single_config(
                config_name=config_name,
                loss_config=loss_config,
                model_config=model_config,
                train_config=train_config,
                train_loader=train_loader,
                test_loader=test_loader,
                seed=seed,
                device=device,
                checkpoint_dir=checkpoint_dir,
            )

            all_results.append(results)

    # Save all results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)

    # Aggregate by config
    config_names = list(set(r["config_name"] for r in all_results))
    for cfg_name in sorted(config_names):
        cfg_results = [r for r in all_results if r["config_name"] == cfg_name]

        dead_units = [r["final_metrics"]["dead_units"] for r in cfg_results]
        redundancy = [r["final_metrics"]["redundancy_score"] for r in cfg_results]
        mse = [r["final_metrics"]["reconstruction_mse"] for r in cfg_results]

        print(f"\n{cfg_name}:")
        print(f"  Dead units:  {np.mean(dead_units):.1f} ± {np.std(dead_units):.1f}")
        print(f"  Redundancy:  {np.mean(redundancy):.2f} ± {np.std(redundancy):.2f}")
        print(f"  Recon MSE:   {np.mean(mse):.4f} ± {np.std(mse):.4f}")

    print(f"\nResults saved to {output_path / 'ablation_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--config",
        type=str,
        default="config/ablation.yaml",
        help="Path to ablation config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation",
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

    run_ablation(args.config, args.output_dir, device)


if __name__ == "__main__":
    main()
