"""Training utilities for encoder study."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import json
from tqdm import tqdm

from .model import Encoder, StandardSAE
from .losses import combined_loss, sae_loss, lse_loss
from .metrics import (
    dead_units,
    redundancy_score,
    weight_redundancy,
    reconstruction_mse,
    sparsity_l0,
    responsibility_entropy,
)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Encoder or StandardSAE model
        loader: Training data loader
        optimizer: Optimizer
        loss_config: Loss configuration dict
        device: Device to train on

    Returns:
        Dict with average losses and metrics for the epoch
    """
    model.train()

    total_loss = 0.0
    total_lse = 0.0
    total_var = 0.0
    total_tc = 0.0
    total_wr = 0.0
    n_batches = 0

    for batch in loader:
        # Handle different data formats
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        # Only move to device if not already there (GPU-cached loaders are already on device)
        if x.device != device:
            x = x.to(device)

        optimizer.zero_grad()

        if isinstance(model, Encoder):
            a, z = model(x)
            W = model.W
            losses = combined_loss(a, W, loss_config)
            loss = losses["total"]

            total_lse += losses["lse"].item()
            total_var += losses["var"].item()
            total_tc += losses["tc"].item()
            total_wr += losses["wr"].item()
        else:
            # StandardSAE
            recon, a = model(x)
            l1_weight = loss_config.get("l1_weight", 0.01)
            losses = sae_loss(x, recon, a, l1_weight=l1_weight)
            loss = losses["total"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    metrics = {
        "loss": total_loss / n_batches,
    }

    if isinstance(model, Encoder):
        metrics.update({
            "lse": total_lse / n_batches,
            "var": total_var / n_batches,
            "tc": total_tc / n_batches,
            "wr": total_wr / n_batches,
        })

    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on data.

    Args:
        model: Encoder or StandardSAE model
        loader: Data loader
        loss_config: Loss configuration dict
        device: Device to evaluate on

    Returns:
        Dict with losses and metrics
    """
    model.eval()

    all_a = []
    all_x = []
    all_r = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            # Only move to device if not already there
            if x.device != device:
                x = x.to(device)

            if isinstance(model, Encoder):
                a, z = model(x)
                W = model.W
                losses = combined_loss(a, W, loss_config)
                all_r.append(losses["responsibilities"])
            else:
                recon, a = model(x)
                l1_weight = loss_config.get("l1_weight", 0.01)
                losses = sae_loss(x, recon, a, l1_weight=l1_weight)

            total_loss += losses["total"].item()
            all_a.append(a)
            all_x.append(x)
            n_batches += 1

    # Concatenate all batches
    all_a = torch.cat(all_a, dim=0)
    all_x = torch.cat(all_x, dim=0)

    metrics = {
        "loss": total_loss / n_batches,
        "dead_units": dead_units(all_a),
        "redundancy_score": redundancy_score(all_a),
        "sparsity_l0": sparsity_l0(all_a),
    }

    if isinstance(model, Encoder):
        W = model.W
        metrics["weight_redundancy"] = weight_redundancy(W)
        metrics["reconstruction_mse"] = reconstruction_mse(all_x, all_a, W)

        if all_r:
            all_r = torch.cat(all_r, dim=0)
            metrics["responsibility_entropy"] = responsibility_entropy(all_r)
    else:
        # StandardSAE
        with torch.no_grad():
            recon, _ = model(all_x)
            metrics["reconstruction_mse"] = float(
                ((all_x - recon) ** 2).mean().item()
            )

    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Full training loop.

    Args:
        model: Encoder or StandardSAE model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Full configuration dict with 'training' and 'loss' keys
        device: Device to train on (auto-detected if None)
        checkpoint_dir: Directory to save checkpoints (optional)
        verbose: Whether to print progress

    Returns:
        History dict with lists of metrics per epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Extract config
    train_config = config.get("training", {})
    loss_config = config.get("loss", {})

    epochs = train_config.get("epochs", 100)
    lr = train_config.get("lr", 0.001)
    optimizer_name = train_config.get("optimizer", "adam")

    # Create optimizer
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Training history
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_dead_units": [],
        "test_redundancy_score": [],
        "test_reconstruction_mse": [],
        "test_sparsity_l0": [],
    }

    # Training loop
    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training")

    for epoch in iterator:
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_config, device
        )
        history["train_loss"].append(train_metrics["loss"])

        # Evaluate
        test_metrics = evaluate(model, test_loader, loss_config, device)
        history["test_loss"].append(test_metrics["loss"])
        history["test_dead_units"].append(test_metrics["dead_units"])
        history["test_redundancy_score"].append(test_metrics["redundancy_score"])
        history["test_reconstruction_mse"].append(test_metrics["reconstruction_mse"])
        history["test_sparsity_l0"].append(test_metrics["sparsity_l0"])

        if verbose:
            iterator.set_postfix({
                "train_loss": f"{train_metrics['loss']:.4f}",
                "test_mse": f"{test_metrics['reconstruction_mse']:.4f}",
            })

    # Save checkpoint if requested
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), checkpoint_path / "model.pt")
        with open(checkpoint_path / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    return history


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
