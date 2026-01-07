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

    # Accumulate scalars on GPU, convert once at end (avoid per-step list appends/stack).
    loss_sums = {
        "total": torch.zeros((), device=device),
        "entropy": torch.zeros((), device=device),
        "tc": torch.zeros((), device=device),
        "wr": torch.zeros((), device=device),
    }
    steps = 0

    for x, _ in dataloader:
        # Only move to device if not already there
        if x.device != device:
            x = x.to(device)

        optimizer.zero_grad()

        a, z = model(x)
        losses = loss_fn(a)

        losses["total"].backward()
        optimizer.step()

        # Keep on GPU; detach so we don't retain graphs.
        for k, v in losses.items():
            loss_sums[k] += v.detach()
        steps += 1

    # Convert to CPU only once at the end.
    return {k: (v / steps).item() for k, v in loss_sums.items()}

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
    print_every: int = 5,
    optimizer_name: str = "adam",
) -> dict:
    """
    Full training loop.

    Args:
        print_every: Print progress every N epochs (default: 5)
        optimizer_name: Optimizer to use ("adam" or "sgd", default: "adam")

    Returns:
        dict with training history and final metrics
    """
    # Create optimizer
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Supported: 'adam', 'sgd'")

    history = {
        "train_loss": [],
        "train_entropy": [],
        "train_tc": [],
        "train_wr": [],
        "epoch_time": [],
        "metrics": [],
    }

    print(f"Training with {len(train_loader)} batches per epoch")
    total_start = time.time()

    for epoch in range(epochs):
        start = time.time()

        losses = train_epoch(model, train_loader, optimizer, loss_fn, device)

        elapsed = time.time() - start

        history["train_loss"].append(losses["total"])
        history["train_entropy"].append(losses["entropy"])
        history["train_tc"].append(losses["tc"])
        history["train_wr"].append(losses["wr"])
        history["epoch_time"].append(elapsed)

        # Print progress at intervals, first epoch, and last epoch
        if (epoch + 1) % print_every == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {losses['total']:7.4f} | "
                  f"Entropy: {losses['entropy']:7.4f} | "
                  f"TC: {losses['tc']:7.4f} | "
                  f"WR: {losses['wr']:7.4f} | "
                  f"Time: {elapsed:.2f}s")

        # Compute and print metrics at intervals
        if metrics_fn is not None and (epoch + 1) % log_every == 0:
            print(f"  → Computing metrics...", end=" ", flush=True)
            metric_start = time.time()
            # Use only 10 batches for speed (1280 samples instead of 10000)
            metrics = metrics_fn(model, test_loader, device, max_batches=10)
            metric_time = time.time() - metric_start
            metrics["epoch"] = epoch + 1
            history["metrics"].append(metrics)

            print(f"done ({metric_time:.1f}s)")
            print(f"     Effective Rank: {metrics['effective_rank']:5.2f} | "
                  f"Dead Units: {metrics['dead_units']:2d} | "
                  f"Redundancy: {metrics['weight_redundancy']:.4f} | "
                  f"Output Corr: {metrics['output_correlation']:.4f}")

    total_elapsed = time.time() - total_start
    avg_epoch_time = total_elapsed / epochs

    print(f"\nTraining summary:")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Avg time per epoch: {avg_epoch_time:.2f}s")

    # Final metrics (use full dataset for final evaluation)
    if metrics_fn is not None:
        print(f"  Computing final metrics on full test set...", end=" ", flush=True)
        final_metrics = metrics_fn(model, test_loader, device, max_batches=None)
        history["final_metrics"] = final_metrics
        print("done")

    return history
