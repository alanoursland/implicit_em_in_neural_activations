"""Experiment 1: Volume Control Ablation.

5 configs × 10 seeds × 50 epochs at hidden_dim=25, λ_reg=0.001.
Tests whether both components of volume control are needed at intermediate layers.

Usage (from supervised_study/src/):
    python run_experiment1.py
    python run_experiment1.py --configs baseline nls_var_tc --seeds 42 43
    python run_experiment1.py --epochs 10  # quick test
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
from collections import defaultdict

from configs import CONFIGS
from data import get_mnist
from run_experiment import train_one

_supervised_root = Path(__file__).resolve().parent.parent

DEFAULT_CONFIGS = list(CONFIGS.keys())
DEFAULT_SEEDS = list(range(42, 52))  # 42-51

CONFIG_COLORS = {
    "baseline": "#1f77b4",
    "nls_only": "#ff7f0e",
    "nls_var": "#2ca02c",
    "nls_var_tc": "#d62728",
    "var_tc_only": "#9467bd",
}

SUMMARY_METRICS = [
    "dead_units", "min_variance", "redundancy", "resp_entropy",
    "test_acc", "test_ce_loss", "test_reg_loss",
]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_last10(all_curves: dict, metrics: list[str] = SUMMARY_METRICS) -> dict:
    """For each config, average metrics over last 10 epochs, then mean±std across seeds."""

    summary = {}
    for config_name, seed_curves in all_curves.items():
        per_seed = {m: [] for m in metrics}
        for seed, curves in seed_curves.items():
            for m in metrics:
                last10 = curves[m][-10:]
                per_seed[m].append(np.mean(last10))
        summary[config_name] = {
            m: {"mean": np.mean(vals), "std": np.std(vals)}
            for m, vals in per_seed.items()
        }
    return summary


def print_table(summary: dict, configs: list[str]):
    """Print ablation table to stdout."""

    header = (f"{'Config':<15s} {'Dead':>6s} {'MinVar':>10s} {'Redund':>10s} "
              f"{'RespEnt':>10s} {'Accuracy':>12s} {'CE Loss':>12s}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for config_name in configs:
        if config_name not in summary:
            continue
        m = summary[config_name]
        print(
            f"{config_name:<15s} "
            f"{m['dead_units']['mean']:>4.1f}±{m['dead_units']['std']:<3.1f}"
            f"{m['min_variance']['mean']:>7.4f}±{m['min_variance']['std']:<5.4f}"
            f"{m['redundancy']['mean']:>7.1f}±{m['redundancy']['std']:<5.1f}"
            f"{m['resp_entropy']['mean']:>7.3f}±{m['resp_entropy']['std']:<5.3f}"
            f"{m['test_acc']['mean']:>7.4f}±{m['test_acc']['std']:<5.4f}"
            f"{m['test_ce_loss']['mean']:>7.4f}±{m['test_ce_loss']['std']:<5.4f}"
        )

    print("=" * len(header))


def save_csv(summary: dict, configs: list[str], path: Path):
    """Save ablation table as CSV."""

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Config"] + SUMMARY_METRICS)
        for config_name in configs:
            if config_name not in summary:
                continue
            m = summary[config_name]
            row = [config_name]
            for metric in SUMMARY_METRICS:
                row.append(f"{m[metric]['mean']:.4f}±{m[metric]['std']:.4f}")
            writer.writerow(row)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_dynamics(all_curves: dict, configs: list[str], save_path: Path):
    """3-panel figure: dead units, redundancy, log test error over epochs."""

    panels = [
        ("dead_units", "Dead Units"),
        ("redundancy", "Redundancy"),
        ("test_acc", "Log Test Classification Error"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (metric, title) in zip(axes, panels):
        for config_name in configs:
            if config_name not in all_curves:
                continue

            seed_curves = all_curves[config_name]
            arrays = []
            for seed, curves in seed_curves.items():
                arrays.append(curves[metric])
            epochs = np.array(next(iter(seed_curves.values()))["epoch"])
            data = np.array(arrays)

            if metric == "test_acc":
                data = np.log10(np.clip(1.0 - data, 1e-10, None))

            mean = data.mean(axis=0)
            color = CONFIG_COLORS.get(config_name)
            ax.plot(epochs, mean, label=config_name, color=color, linewidth=1.5)

        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_loss_curves(all_curves: dict, configs: list[str], save_path: Path):
    """2-panel figure: CE loss and reg loss over epochs."""

    panels = [
        ("test_ce_loss", "Cross-Entropy Loss"),
        ("test_reg_loss", "Regularization Loss"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (metric, title) in zip(axes, panels):
        for config_name in configs:
            if config_name not in all_curves:
                continue

            seed_curves = all_curves[config_name]
            arrays = []
            for seed, curves in seed_curves.items():
                arrays.append(curves[metric])
            epochs = np.array(next(iter(seed_curves.values()))["epoch"])
            data = np.array(arrays)
            mean = data.mean(axis=0)

            color = CONFIG_COLORS.get(config_name)
            ax.plot(epochs, mean, label=config_name, color=color, linewidth=1.5)

        if metric == "test_ce_loss":
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiment1(
    train_loader,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    configs: list[str] = DEFAULT_CONFIGS,
    seeds: list[int] = DEFAULT_SEEDS,
    hidden_dim: int = 25,
    epochs: int = 50,
    lr: float = 0.001,
    lambda_reg: float = 0.001,
    log_interval: int = 10,
):
    """Run Experiment 1: Volume Control Ablation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_curves = defaultdict(dict)

    for config_name in configs:
        print(f"\n{'='*60}")
        print(f"Experiment 1 — Config: {config_name}")
        print(f"{'='*60}")

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            model, curves = train_one(
                config_name, seed, train_loader, test_x, test_y, device,
                hidden_dim=hidden_dim, epochs=epochs, lr=lr,
                lambda_reg=lambda_reg, log_interval=log_interval,
            )
            all_curves[config_name][seed] = curves

            ckpt_path = models_dir / f"{config_name}_seed{seed}.pt"
            torch.save({
                "config_name": config_name,
                "seed": seed,
                "hidden_dim": hidden_dim,
                "model_state_dict": model.state_dict(),
            }, ckpt_path)
            print(f"  Saved model: {ckpt_path}")

    # Save full curves
    with open(output_dir / "training_curves.json", "w") as f:
        json.dump(dict(all_curves), f, indent=2)
    print(f"Saved: {output_dir / 'training_curves.json'}")

    # Aggregate and save summary
    summary = aggregate_last10(all_curves)
    print_table(summary, configs)
    save_csv(summary, configs, output_dir / "ablation_table.csv")

    # Save raw summary as JSON
    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {output_dir / 'raw_results.json'}")

    # Generate figures
    plot_training_dynamics(all_curves, configs, figures_dir / "training_dynamics.png")
    plot_loss_curves(all_curves, configs, figures_dir / "loss_curves.png")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Volume Control Ablation")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--hidden-dim", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda-reg", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="E:/ml_datasets")
    parser.add_argument("--output-dir", type=str,
                        default=str(_supervised_root / "results" / "experiment1"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_mnist(
        batch_size=args.batch_size, flatten=True, data_dir=args.data_dir,
        use_gpu_cache=(device.type == "cuda"), device=str(device),
    )
    test_x, test_y = test_loader.data, test_loader.labels

    run_experiment1(
        train_loader, test_x, test_y, device, Path(args.output_dir),
        configs=args.configs, seeds=args.seeds, hidden_dim=args.hidden_dim,
        epochs=args.epochs, lr=args.lr, lambda_reg=args.lambda_reg,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
