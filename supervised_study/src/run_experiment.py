"""Supervised implicit EM ablation experiment.

Trains five configurations × 3 seeds on MNIST, evaluates intermediate-layer
metrics, and produces an ablation table + weight visualizations.

Usage (from supervised_study/src/):
    python run_experiment.py
    python run_experiment.py --configs baseline nls_var_tc
    python run_experiment.py --seeds 42
"""

import argparse
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SupervisedModel
from configs import CONFIGS
from metrics import min_variance, redundancy_score, responsibility_entropy
from data import get_mnist
from utils import set_seed

_supervised_root = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(
    config_name: str,
    seed: int,
    train_loader,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    hidden_dim: int = 64,
    epochs: int = 100,
    lr: float = 0.001,
    lambda_reg: float = 1.0,
    log_interval: int = 10,
) -> tuple[SupervisedModel, dict]:
    """Train a single model for one config/seed combination.

    Returns:
        (model, curves) where curves is a dict of per-epoch metric lists.
    """

    set_seed(seed)
    cfg = CONFIGS[config_name]
    model = SupervisedModel(hidden_dim=hidden_dim, **cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    curves = {
        "epoch": [],
        "ce_loss": [],
        "reg_loss": [],
        "total_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_ce = 0.0
        epoch_reg = 0.0
        epoch_total = 0.0
        correct = 0
        total_samples = 0
        n_batches = 0

        for x, labels in train_loader:
            optimizer.zero_grad()

            h = model(x)
            ce_loss = F.cross_entropy(h, labels)
            reg_loss = model.regularization_loss()["total"]
            total = ce_loss + lambda_reg * reg_loss

            total.backward()
            optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_reg += reg_loss.item()
            epoch_total += total.item()
            n_batches += 1

            with torch.no_grad():
                correct += (h.argmax(dim=1) == labels).sum().item()
                total_samples += labels.shape[0]

        train_acc = correct / total_samples
        model.eval()
        with torch.no_grad():
            test_acc = (model(test_x).argmax(dim=1) == test_y).float().mean().item()
        model.train()

        curves["epoch"].append(epoch + 1)
        curves["ce_loss"].append(epoch_ce / n_batches)
        curves["reg_loss"].append(epoch_reg / n_batches)
        curves["total_loss"].append(epoch_total / n_batches)
        curves["train_acc"].append(train_acc)
        curves["test_acc"].append(test_acc)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(
                f"  [{config_name} seed={seed}] epoch {epoch+1:3d}  "
                f"CE={epoch_ce/n_batches:.4f}  "
                f"reg={epoch_reg/n_batches:.4f}  "
                f"total={epoch_total/n_batches:.4f}  "
                f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}"
            )

    return model, curves


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: SupervisedModel,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> dict:
    """Evaluate a trained model on the full test set (single forward pass)."""

    model.eval()
    with torch.no_grad():
        h = model(test_x)
        d = model.distances

    acc = (h.argmax(dim=1) == test_y).float().mean().item()
    r = torch.softmax(-d, dim=1)

    return {
        "min_variance": min_variance(d),
        "redundancy": redundancy_score(d),
        "resp_entropy": responsibility_entropy(r),
        "accuracy": acc,
    }


# ---------------------------------------------------------------------------
# Weight visualization
# ---------------------------------------------------------------------------

def visualize_weights(model: SupervisedModel, title: str, save_path: Path):
    """Visualize first-layer weights as an 8×8 grid of 28×28 images."""

    W = model.W1.detach().cpu()  # (64, 784)
    n_units = W.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_units)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    vmax = W.abs().max().item()

    for i in range(grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        if i < n_units:
            img = W[i].reshape(28, 28).numpy()
            ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved weights figure: {save_path}")


# ---------------------------------------------------------------------------
# Aggregation & table
# ---------------------------------------------------------------------------

def aggregate_results(all_results: dict) -> dict:
    """Compute mean ± std across seeds for each config."""

    summary = {}
    for config_name, seed_results in all_results.items():
        metrics = {}
        for key in seed_results[0]:
            vals = [r[key] for r in seed_results]
            metrics[key] = {"mean": np.mean(vals), "std": np.std(vals)}
        summary[config_name] = metrics
    return summary


def print_table(summary: dict):
    """Print ablation table to stdout."""

    header = f"{'Config':<20s} {'Min Variance':>14s} {'Redundancy':>14s} {'Resp Entropy':>14s} {'Accuracy':>14s}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for config_name in CONFIGS:
        if config_name not in summary:
            continue
        m = summary[config_name]
        print(
            f"{config_name:<20s} "
            f"{m['min_variance']['mean']:>6.4f}±{m['min_variance']['std']:<5.4f} "
            f"{m['redundancy']['mean']:>6.1f}±{m['redundancy']['std']:<5.1f} "
            f"{m['resp_entropy']['mean']:>6.3f}±{m['resp_entropy']['std']:<5.3f} "
            f"{m['accuracy']['mean']:>6.4f}±{m['accuracy']['std']:<5.4f}"
        )

    print("=" * len(header))


def save_csv(summary: dict, path: Path):
    """Save ablation table as CSV."""

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "Min Variance", "Redundancy", "Resp Entropy", "Accuracy"])
        for config_name in CONFIGS:
            if config_name not in summary:
                continue
            m = summary[config_name]
            writer.writerow([
                config_name,
                f"{m['min_variance']['mean']:.4f}±{m['min_variance']['std']:.4f}",
                f"{m['redundancy']['mean']:.1f}±{m['redundancy']['std']:.1f}",
                f"{m['resp_entropy']['mean']:.3f}±{m['resp_entropy']['std']:.3f}",
                f"{m['accuracy']['mean']:.4f}±{m['accuracy']['std']:.4f}",
            ])
    print(f"Saved CSV: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Supervised implicit EM ablation")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        help="Which configs to run (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                        help="Random seeds (default: 42 43 44)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Number of hidden units / mixture components")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print training loss every N epochs")
    parser.add_argument("--data-dir", type=str, default="E:/ml_datasets")
    parser.add_argument("--output-dir", type=str, default=str(_supervised_root / "results"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data once
    use_gpu_cache = device.type == "cuda"
    train_loader, test_loader = get_mnist(
        batch_size=args.batch_size,
        flatten=True,
        data_dir=args.data_dir,
        use_gpu_cache=use_gpu_cache,
        device=str(device),
    )

    # Full test tensors for single-pass evaluation
    test_x, test_y = test_loader.data, test_loader.labels

    # Run all configs × seeds
    all_results = {}  # config_name -> [result_per_seed]
    all_curves = {}   # config_name -> {seed -> curves}

    for config_name in args.configs:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")

        seed_results = []
        seed_curves = {}
        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            model, curves = train_one(
                config_name, seed, train_loader, test_x, test_y, device,
                hidden_dim=args.hidden_dim, epochs=args.epochs,
                lr=args.lr, lambda_reg=args.lambda_reg,
                log_interval=args.log_interval,
            )
            result = evaluate_model(model, test_x, test_y)
            seed_results.append(result)
            seed_curves[seed] = curves
            print(f"  Result: {result}")

            # Save model checkpoint
            ckpt_path = models_dir / f"{config_name}_seed{seed}.pt"
            torch.save({
                "config_name": config_name,
                "seed": seed,
                "hidden_dim": args.hidden_dim,
                "model_state_dict": model.state_dict(),
                "config_kwargs": CONFIGS[config_name],
            }, ckpt_path)
            print(f"  Saved model: {ckpt_path}")

            # Save weights for last seed
            if seed == args.seeds[-1]:
                visualize_weights(
                    model, config_name,
                    weights_dir / f"weights_{config_name}.png",
                )

        all_results[config_name] = seed_results
        all_curves[config_name] = seed_curves

    # Save training curves
    with open(curves_dir / "training_curves.json", "w") as f:
        json.dump(all_curves, f, indent=2)
    print(f"Saved training curves: {curves_dir / 'training_curves.json'}")

    # Aggregate and output
    summary = aggregate_results(all_results)
    print_table(summary)
    save_csv(summary, output_dir / "ablation_table.csv")

    # Save raw results as JSON
    json_results = {}
    for config_name, seed_results in all_results.items():
        json_results[config_name] = seed_results
    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved raw results: {output_dir / 'raw_results.json'}")


if __name__ == "__main__":
    main()
