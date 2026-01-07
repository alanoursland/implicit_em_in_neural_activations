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
from tqdm import tqdm

from src.model import Encoder
from src.data import get_mnist
from src.training import set_seed
from src.losses import combined_loss
from src.metrics import (
    usage_distribution,
    feature_similarity_matrix,
    dead_units,
    redundancy_score,
    weight_redundancy,
    responsibility_entropy,
    reconstruction_mse,
    sparsity_l0,
)


def train_with_logging(
    model: Encoder,
    train_loader,
    test_loader,
    loss_config: Dict[str, Any],
    train_config: Dict[str, Any],
    device: torch.device,
    log_every: int = 10,
) -> Dict[str, List[float]]:
    """Train model with detailed epoch logging.

    Args:
        model: Encoder model
        train_loader: Training data loader
        test_loader: Test data loader
        loss_config: Loss configuration
        train_config: Training configuration
        device: Device to train on
        log_every: Log metrics every N epochs

    Returns:
        History dict with metrics per epoch
    """
    model = model.to(device)
    epochs = train_config.get("epochs", 100)
    lr = train_config.get("lr", 0.001)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "lse": [],
        "var": [],
        "tc": [],
        "dead_units": [],
        "redundancy": [],
    }

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        total_loss = 0.0
        total_lse = 0.0
        total_var = 0.0
        total_tc = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
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

        history["train_loss"].append(avg_loss)
        history["lse"].append(avg_lse)
        history["var"].append(avg_var)
        history["tc"].append(avg_tc)

        # Compute evaluation metrics periodically
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            model.eval()
            all_a = []
            with torch.no_grad():
                for batch in test_loader:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    if x.device != device:
                        x = x.to(device)
                    a, _ = model(x)
                    all_a.append(a)

            all_a = torch.cat(all_a, dim=0)
            n_dead = dead_units(all_a)
            redund = redundancy_score(all_a)

            history["dead_units"].append(n_dead)
            history["redundancy"].append(redund)

            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {avg_loss:8.2f} | "
                f"LSE: {avg_lse:8.2f} | "
                f"Var: {avg_var:8.2f} | "
                f"TC: {avg_tc:8.2f}"
            )

    return history


def evaluate_final(
    model: Encoder,
    test_loader,
    loss_config: Dict[str, Any],
    device: torch.device,
    hidden_dim: int,
) -> Dict[str, Any]:
    """Compute final evaluation metrics.

    Args:
        model: Trained encoder model
        test_loader: Test data loader
        loss_config: Loss configuration
        device: Device
        hidden_dim: Number of hidden units

    Returns:
        Dict with all final metrics
    """
    model.eval()
    all_a = []
    all_x = []
    all_r = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x.device != device:
                x = x.to(device)
            a, _ = model(x)
            neg_a = -a
            r = torch.softmax(neg_a, dim=1)

            all_a.append(a)
            all_x.append(x)
            all_r.append(r)

    all_a = torch.cat(all_a, dim=0)
    all_x = torch.cat(all_x, dim=0)
    all_r = torch.cat(all_r, dim=0)
    W = model.get_weights()

    n_dead = dead_units(all_a)
    redund = redundancy_score(all_a)
    w_redund = weight_redundancy(W)
    resp_ent = responsibility_entropy(all_r)
    max_resp = all_r.max(dim=1).values.mean().item()
    recon_mse = reconstruction_mse(all_x, all_a, W)

    return {
        "dead_units": n_dead,
        "dead_units_pct": 100.0 * n_dead / hidden_dim,
        "redundancy_score": redund,
        "weight_redundancy": w_redund,
        "responsibility_entropy": resp_ent,
        "max_responsibility": max_resp,
        "reconstruction_mse": recon_mse,
    }


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
    hidden_dim = model_config["hidden_dim"]

    # Print configuration header
    lambda_lse = loss_config.get("lambda_lse", 0.0)
    lambda_var = loss_config.get("lambda_var", 0.0)
    lambda_tc = loss_config.get("lambda_tc", 0.0)

    print("\n" + "=" * 70)
    print(
        f"Configuration: {config_name} "
        f"(lambda_lse={lambda_lse}, lambda_var={lambda_var}, lambda_tc={lambda_tc})"
    )
    print(f"Seed: {seed}")
    print("=" * 70)

    # Create model
    model = Encoder(
        input_dim=model_config["input_dim"],
        hidden_dim=hidden_dim,
        activation=model_config.get("activation", "relu"),
    )

    # Train with logging
    history = train_with_logging(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_config=loss_config,
        train_config=train_config,
        device=device,
        log_every=10,
    )

    # Final evaluation
    final_metrics = evaluate_final(
        model=model,
        test_loader=test_loader,
        loss_config=loss_config,
        device=device,
        hidden_dim=hidden_dim,
    )

    # Print final metrics
    print("\n--- Final Metrics ---")
    print(
        f"Dead units (var < 1e-2): {final_metrics['dead_units']}/{hidden_dim} "
        f"({final_metrics['dead_units_pct']:.1f}%)"
    )
    print(f"Redundancy (||Corr - I||^2): {final_metrics['redundancy_score']:.2f}")
    print(f"Weight redundancy (||W^TW - I||^2): {final_metrics['weight_redundancy']:.2f}")
    print(f"Responsibility entropy (mean): {final_metrics['responsibility_entropy']:.2f}")
    print(f"Max responsibility (mean): {final_metrics['max_responsibility']:.2f}")
    print(f"Reconstruction MSE: {final_metrics['reconstruction_mse']:.4f}")

    # Get usage distribution and similarity matrix for saving
    all_r = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x.device != device:
                x = x.to(device)
            a, _ = model(x)
            neg_a = -a
            r = torch.softmax(neg_a, dim=1)
            all_r.append(r)

    all_r = torch.cat(all_r, dim=0)
    usage = usage_distribution(all_r)
    W = model.get_weights()
    similarity = feature_similarity_matrix(W)

    # Save checkpoint
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path / "model.pt")
        with open(checkpoint_path / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    return {
        "config_name": config_name,
        "seed": seed,
        "final_metrics": final_metrics,
        "history": history,
        "usage_distribution": usage.tolist(),
        "feature_similarity": similarity.tolist(),
    }


def print_summary_table(all_results: List[Dict[str, Any]], hidden_dim: int):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print(" " * 25 + "ABLATION SUMMARY")
    print("=" * 80)
    print(
        f"{'Config':<16} | {'Dead':>4} | {'Redundancy':>10} | "
        f"{'Weight Red':>10} | {'Resp Entropy':>12}"
    )
    print("-" * 80)

    # Get unique config names in order
    seen = set()
    config_names = []
    for r in all_results:
        if r["config_name"] not in seen:
            seen.add(r["config_name"])
            config_names.append(r["config_name"])

    for cfg_name in config_names:
        cfg_results = [r for r in all_results if r["config_name"] == cfg_name]

        dead = [r["final_metrics"]["dead_units"] for r in cfg_results]
        redund = [r["final_metrics"]["redundancy_score"] for r in cfg_results]
        w_redund = [r["final_metrics"]["weight_redundancy"] for r in cfg_results]
        resp_ent = [r["final_metrics"]["responsibility_entropy"] for r in cfg_results]

        print(
            f"{cfg_name:<16} | {np.mean(dead):4.0f} | {np.mean(redund):10.2f} | "
            f"{np.mean(w_redund):10.2f} | {np.mean(resp_ent):12.2f}"
        )

    print("=" * 80)


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
    hidden_dim = model_config["hidden_dim"]

    # Load data (with GPU caching for speed)
    print("Loading MNIST...")
    train_loader, test_loader = get_mnist(
        batch_size=train_config.get("batch_size", 128),
        use_gpu_cache=True,
        device=str(device),
    )
    print("MNIST loaded and cached on GPU.\n")

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

    # Print summary table
    print_summary_table(all_results, hidden_dim)

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
