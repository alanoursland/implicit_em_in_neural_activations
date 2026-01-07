"""Experiment 4: Training dynamics across optimizers and learning rates.

Studies:
- Learning rate sensitivity for Adam vs SGD
- Convergence speed and stability
- Final performance across configurations

Outputs results to results/dynamics/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import json
import argparse
import time
from typing import Dict, Any, List
import numpy as np

from src.model import Encoder
from src.data import get_mnist
from src.training import set_seed
from src.losses import combined_loss
from src.metrics import linear_probe_accuracy


def extract_features(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple:
    """Extract features and labels from a data loader."""
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            if x.device != device:
                x = x.to(device)

            a, _ = model(x)
            all_features.append(a.cpu().numpy())
            all_labels.append(y.cpu().numpy() if isinstance(y, torch.Tensor) else y)

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def detect_convergence(
    loss_curve: List[float],
    threshold: float = 1.0,
    window: int = 5,
) -> int:
    """Detect convergence epoch.

    Converged when loss change < threshold for `window` consecutive epochs.

    Returns:
        Epoch at which convergence occurred (1-indexed), or len(loss_curve) if not converged.
    """
    if len(loss_curve) < window + 1:
        return len(loss_curve)

    for i in range(window, len(loss_curve)):
        # Check if all changes in the window are below threshold
        changes = [
            abs(loss_curve[j] - loss_curve[j - 1]) for j in range(i - window + 1, i + 1)
        ]
        if all(c < threshold for c in changes):
            return i - window + 1  # Return first epoch of the stable window

    return len(loss_curve)


def train_with_dynamics(
    model: Encoder,
    train_loader,
    loss_config: Dict[str, Any],
    epochs: int,
    optimizer_name: str,
    lr: float,
    device: torch.device,
    log_every: int = 10,
) -> Dict[str, Any]:
    """Train model and track full dynamics."""
    model = model.to(device)

    # Create optimizer
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Track metrics per epoch
    loss_curve = []
    lse_curve = []
    var_curve = []
    tc_curve = []

    start_time = time.time()

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

        loss_curve.append(avg_loss)
        lse_curve.append(avg_lse)
        var_curve.append(avg_var)
        tc_curve.append(avg_tc)

        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:9.2f}")

    training_time = time.time() - start_time
    convergence_epoch = detect_convergence(loss_curve)

    return {
        "loss_curve": loss_curve,
        "lse_curve": lse_curve,
        "var_curve": var_curve,
        "tc_curve": tc_curve,
        "final_loss": loss_curve[-1],
        "final_lse": lse_curve[-1],
        "final_var": var_curve[-1],
        "final_tc": tc_curve[-1],
        "convergence_epoch": convergence_epoch,
        "training_time": training_time,
    }


def run_dynamics(config_path: str, output_dir: str, device: torch.device):
    """Run full dynamics sweep."""
    # Print header
    print("=" * 80)
    print(" " * 15 + "EXPERIMENT 4: TRAINING DYNAMICS")
    print("=" * 80)
    print(f"Using device: {device}")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    loss_config = config.get("loss", {})
    train_config = config["training"]
    sweep_config = config["sweep"]

    epochs = train_config.get("epochs", 100)
    optimizers = sweep_config.get("optimizers", ["adam", "sgd"])
    learning_rates = sweep_config.get("learning_rates", [0.0001, 0.001, 0.01, 0.1])
    seeds = sweep_config.get("seeds", [1, 2, 3])

    # Load data
    print("Loading MNIST...")
    train_loader, test_loader = get_mnist(
        batch_size=train_config.get("batch_size", 128),
        use_gpu_cache=True,
        device=str(device),
    )
    print("MNIST loaded and cached on GPU.\n")

    # Results storage
    all_results = []

    # Main sweep loop
    for optimizer_name in optimizers:
        for lr in learning_rates:
            print("=" * 70)
            print(f"Optimizer: {optimizer_name.upper()} | Learning Rate: {lr}")
            print("=" * 70)

            runs = []

            for seed in seeds:
                print(f"\n--- Seed {seed} ---")
                set_seed(seed)

                model = Encoder(
                    input_dim=model_config["input_dim"],
                    hidden_dim=model_config["hidden_dim"],
                    activation=model_config.get("activation", "relu"),
                )

                # Train with dynamics tracking
                dynamics = train_with_dynamics(
                    model=model,
                    train_loader=train_loader,
                    loss_config=loss_config,
                    epochs=epochs,
                    optimizer_name=optimizer_name,
                    lr=lr,
                    device=device,
                    log_every=10,
                )

                # Evaluate probe accuracy
                train_features, train_labels = extract_features(
                    model, train_loader, device
                )
                test_features, test_labels = extract_features(
                    model, test_loader, device
                )
                probe_acc = linear_probe_accuracy(
                    train_features, train_labels, test_features, test_labels
                )

                dynamics["seed"] = seed
                dynamics["probe_accuracy"] = probe_acc

                print(
                    f"Converged: epoch {dynamics['convergence_epoch']} | "
                    f"Probe Acc: {probe_acc*100:.2f}% | "
                    f"Time: {dynamics['training_time']:.1f}s"
                )

                runs.append(dynamics)

            # Compute summary for this optimizer-lr combination
            final_losses = [r["final_loss"] for r in runs]
            conv_epochs = [r["convergence_epoch"] for r in runs]
            probe_accs = [r["probe_accuracy"] for r in runs]

            final_loss_mean = float(np.mean(final_losses))
            final_loss_std = float(np.std(final_losses))
            conv_epoch_mean = float(np.mean(conv_epochs))
            conv_epoch_std = float(np.std(conv_epochs))
            probe_acc_mean = float(np.mean(probe_accs))
            probe_acc_std = float(np.std(probe_accs))

            # Stable if std < 5% of |mean|
            stable = final_loss_std < 0.05 * abs(final_loss_mean)

            summary = {
                "final_loss_mean": final_loss_mean,
                "final_loss_std": final_loss_std,
                "convergence_epoch_mean": conv_epoch_mean,
                "convergence_epoch_std": conv_epoch_std,
                "probe_accuracy_mean": probe_acc_mean,
                "probe_accuracy_std": probe_acc_std,
                "stable": stable,
            }

            print(f"\nSummary ({optimizer_name.upper()}, lr={lr}):")
            print(f"  Final Loss: {final_loss_mean:.2f} +/- {final_loss_std:.2f}")
            print(f"  Convergence: epoch {conv_epoch_mean:.0f} +/- {conv_epoch_std:.0f}")
            print(f"  Probe Accuracy: {probe_acc_mean*100:.2f}% +/- {probe_acc_std*100:.2f}%")

            all_results.append(
                {
                    "optimizer": optimizer_name,
                    "lr": lr,
                    "runs": runs,
                    "summary": summary,
                }
            )

            print()

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert loss curves to lists for JSON serialization
    for result in all_results:
        for run in result["runs"]:
            run["loss_curve"] = [float(x) for x in run["loss_curve"]]
            run["lse_curve"] = [float(x) for x in run["lse_curve"]]
            run["var_curve"] = [float(x) for x in run["var_curve"]]
            run["tc_curve"] = [float(x) for x in run["tc_curve"]]

    output_data = {
        "sweep": all_results,
        "config": config,
    }

    with open(output_path / "dynamics_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary table
    print("=" * 80)
    print(" " * 25 + "DYNAMICS SUMMARY")
    print("=" * 80)
    print(
        f"{'Optimizer':<10}| {'LR':<8}| {'Final Loss':<16}| "
        f"{'Converged':<13}| {'Probe Acc':<16}| {'Stable':<6}"
    )
    print("-" * 80)

    for result in all_results:
        s = result["summary"]
        opt = result["optimizer"].upper()
        lr = result["lr"]
        stable_str = "Yes" if s["stable"] else "No"

        print(
            f"{opt:<10}| {lr:<8}| "
            f"{s['final_loss_mean']:7.1f} +/- {s['final_loss_std']:5.1f} | "
            f"epoch {s['convergence_epoch_mean']:3.0f}+/-{s['convergence_epoch_std']:2.0f} | "
            f"{s['probe_accuracy_mean']*100:5.1f}% +/- {s['probe_accuracy_std']*100:4.1f}% | "
            f"{stable_str:<6}"
        )

    print("=" * 80)
    print(f"Results saved to {output_path / 'dynamics_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run training dynamics experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dynamics.yaml",
        help="Path to dynamics config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dynamics",
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

    run_dynamics(args.config, args.output_dir, device)


if __name__ == "__main__":
    main()
