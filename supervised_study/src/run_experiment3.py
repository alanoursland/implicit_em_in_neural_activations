"""Experiment 3: Optimization Dynamics.

1 config (nls_var_tc) × 2 optimizers × 4 learning rates × 3 seeds × 50 epochs.
Tests whether Paper 2's optimization anomalies survive with supervised loss.

Usage (from supervised_study/src/):
    python run_experiment3.py
    python run_experiment3.py --optimizers adam --lrs 0.001 --seeds 42
    python run_experiment3.py --epochs 10  # quick test
"""

import argparse
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from data import get_mnist
from run_experiment import train_one

_supervised_root = Path(__file__).resolve().parent.parent

DEFAULT_OPTIMIZERS = ["sgd", "adam"]
DEFAULT_LRS = [0.0001, 0.001, 0.01, 0.1]
DEFAULT_SEEDS = [42, 43, 44]

LR_COLORS = {
    0.0001: "#1f77b4",
    0.001: "#ff7f0e",
    0.01: "#2ca02c",
    0.1: "#d62728",
}

SUMMARY_METRICS = [
    "test_acc", "test_ce_loss", "test_reg_loss",
    "dead_units", "min_variance", "redundancy",
]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_optimizer(
    all_curves: dict,
    metrics: list[str] = SUMMARY_METRICS,
) -> dict:
    """Aggregate by (optimizer, lr), last-10-epoch averaging, mean±std across seeds.

    Returns:
        {(optimizer, lr): {metric: {mean, std}}}
    """

    summary = {}
    for (opt, lr), seed_curves in all_curves.items():
        per_seed = {m: [] for m in metrics}
        for seed, curves in seed_curves.items():
            for m in metrics:
                last10 = curves[m][-10:]
                per_seed[m].append(np.mean(last10))

        summary[(opt, lr)] = {
            m: {"mean": np.mean(vals), "std": np.std(vals)}
            for m, vals in per_seed.items()
        }

    return summary


def print_table(summary: dict, optimizers: list[str], lrs: list[float]):
    """Print optimizer table to stdout."""

    header = (f"{'Optimizer':<10s} {'LR':>8s} {'Accuracy':>12s} "
              f"{'CE Loss':>12s} {'Reg Loss':>12s} {'Dead':>8s}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for opt in optimizers:
        for lr in lrs:
            key = (opt, lr)
            if key not in summary:
                continue
            m = summary[key]
            print(
                f"{opt:<10s} {lr:>8.4f} "
                f"{m['test_acc']['mean']:>7.4f}±{m['test_acc']['std']:<5.4f}"
                f"{m['test_ce_loss']['mean']:>7.4f}±{m['test_ce_loss']['std']:<5.4f}"
                f"{m['test_reg_loss']['mean']:>7.4f}±{m['test_reg_loss']['std']:<5.4f}"
                f"{m['dead_units']['mean']:>5.1f}±{m['dead_units']['std']:<3.1f}"
            )
        print("-" * len(header))

    print("=" * len(header))


def save_csv(summary: dict, optimizers: list[str], lrs: list[float], path: Path):
    """Save optimizer table as CSV."""

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Optimizer", "LR"] + SUMMARY_METRICS)
        for opt in optimizers:
            for lr in lrs:
                key = (opt, lr)
                if key not in summary:
                    continue
                m = summary[key]
                row = [opt, lr]
                for metric in SUMMARY_METRICS:
                    row.append(f"{m[metric]['mean']:.4f}±{m[metric]['std']:.4f}")
                writer.writerow(row)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curves(all_curves: dict, optimizers: list[str], lrs: list[float],
                     save_path: Path):
    """2-panel figure: SGD and Adam loss curves, each with 4 lr lines."""

    fig, axes = plt.subplots(1, len(optimizers), figsize=(7 * len(optimizers), 5))
    if len(optimizers) == 1:
        axes = [axes]

    for ax, opt in zip(axes, optimizers):
        for lr in lrs:
            key = (opt, lr)
            if key not in all_curves:
                continue

            seed_curves = all_curves[key]
            arrays = []
            for seed, curves in seed_curves.items():
                arrays.append(curves["test_ce_loss"])
            epochs = np.array(next(iter(seed_curves.values()))["epoch"])
            data = np.array(arrays)
            mean = data.mean(axis=0)

            color = LR_COLORS.get(lr)
            ax.plot(epochs, mean, label=f"lr={lr}", color=color, linewidth=1.5)

        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test CE Loss")
        ax.set_title(f"{opt.upper()}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiment3(
    train_loader,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    optimizers: list[str] = DEFAULT_OPTIMIZERS,
    lrs: list[float] = DEFAULT_LRS,
    seeds: list[int] = DEFAULT_SEEDS,
    hidden_dim: int = 25,
    epochs: int = 50,
    lambda_reg: float = 0.001,
    log_interval: int = 10,
):
    """Run Experiment 3: Optimization Dynamics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_curves = {}  # (optimizer, lr) -> {seed -> curves}

    for opt in optimizers:
        for lr in lrs:
            print(f"\n{'='*60}")
            print(f"Experiment 3 — Optimizer: {opt}, lr={lr}")
            print(f"{'='*60}")

            key = (opt, lr)
            all_curves[key] = {}

            for seed in seeds:
                print(f"\n--- Seed {seed} ---")
                model, curves = train_one(
                    "nls_var_tc", seed, train_loader, test_x, test_y, device,
                    hidden_dim=hidden_dim, epochs=epochs, lr=lr,
                    lambda_reg=lambda_reg, log_interval=log_interval,
                    optimizer_name=opt,
                )
                all_curves[key][seed] = curves

                ckpt_path = models_dir / f"{opt}_lr{lr}_seed{seed}.pt"
                torch.save({
                    "config_name": "nls_var_tc",
                    "seed": seed,
                    "hidden_dim": hidden_dim,
                    "optimizer": opt,
                    "lr": lr,
                    "model_state_dict": model.state_dict(),
                }, ckpt_path)
                print(f"  Saved model: {ckpt_path}")

    # Save full curves (convert tuple keys to strings for JSON)
    json_curves = {}
    for (opt, lr), seed_curves in all_curves.items():
        json_curves[f"{opt}_lr{lr}"] = seed_curves
    with open(output_dir / "training_curves.json", "w") as f:
        json.dump(json_curves, f, indent=2)
    print(f"Saved: {output_dir / 'training_curves.json'}")

    # Aggregate and save summary
    summary = aggregate_optimizer(all_curves)
    print_table(summary, optimizers, lrs)
    save_csv(summary, optimizers, lrs, output_dir / "optimizer_table.csv")

    # Save raw summary as JSON
    json_summary = {
        f"{opt}_lr{lr}": metrics
        for (opt, lr), metrics in summary.items()
    }
    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved: {output_dir / 'raw_results.json'}")

    # Generate figures
    plot_loss_curves(all_curves, optimizers, lrs, figures_dir / "loss_curves.png")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Optimization Dynamics")
    parser.add_argument("--optimizers", nargs="+", default=DEFAULT_OPTIMIZERS)
    parser.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--hidden-dim", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda-reg", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="E:/ml_datasets")
    parser.add_argument("--output-dir", type=str,
                        default=str(_supervised_root / "results" / "experiment3"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_mnist(
        batch_size=args.batch_size, flatten=True, data_dir=args.data_dir,
        use_gpu_cache=(device.type == "cuda"), device=str(device),
    )
    test_x, test_y = test_loader.data, test_loader.labels

    run_experiment3(
        train_loader, test_x, test_y, device, Path(args.output_dir),
        optimizers=args.optimizers, lrs=args.lrs, seeds=args.seeds,
        hidden_dim=args.hidden_dim, epochs=args.epochs,
        lambda_reg=args.lambda_reg, log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
