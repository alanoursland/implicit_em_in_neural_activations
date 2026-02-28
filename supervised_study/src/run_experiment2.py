"""Experiment 2: Capacity and Volume Control.

2 configs × 5 hidden_dims × 5 seeds × 50 epochs.
Shows that volume control's benefit depends on capacity.

Usage (from supervised_study/src/):
    python run_experiment2.py
    python run_experiment2.py --hidden-dims 16 25 --seeds 42 43
    python run_experiment2.py --epochs 10  # quick test
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

from data import get_mnist
from run_experiment import train_one

_supervised_root = Path(__file__).resolve().parent.parent

DEFAULT_CONFIGS = ["baseline", "nls_var_tc"]
DEFAULT_HIDDEN_DIMS = [16, 25, 36, 49, 64]
DEFAULT_SEEDS = list(range(42, 47))  # 42-46

CONFIG_COLORS = {
    "baseline": "#1f77b4",
    "nls_var_tc": "#d62728",
}

SUMMARY_METRICS = [
    "dead_units", "min_variance", "redundancy", "resp_entropy",
    "test_acc", "test_ce_loss", "test_reg_loss",
]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_capacity(
    all_curves: dict,
    hidden_dims: list[int],
    metrics: list[str] = SUMMARY_METRICS,
) -> dict:
    """Aggregate by (config, hidden_dim), last-10-epoch averaging, mean±std across seeds.

    Returns:
        {config_name: {hidden_dim: {metric: {mean, std}}}}
    """

    summary = {}
    for (config_name, hidden_dim), seed_curves in all_curves.items():
        if config_name not in summary:
            summary[config_name] = {}

        per_seed = {m: [] for m in metrics}
        for seed, curves in seed_curves.items():
            for m in metrics:
                last10 = curves[m][-10:]
                per_seed[m].append(np.mean(last10))

        result = {
            m: {"mean": np.mean(vals), "std": np.std(vals)}
            for m, vals in per_seed.items()
        }
        # Add derived metric
        result["dead_fraction"] = {
            "mean": result["dead_units"]["mean"] / hidden_dim,
            "std": result["dead_units"]["std"] / hidden_dim,
        }

        summary[config_name][hidden_dim] = result

    return summary


def print_table(summary: dict, hidden_dims: list[int], configs: list[str]):
    """Print capacity table to stdout."""

    header = (f"{'HidDim':>6s} {'Config':<15s} {'Dead':>8s} {'DeadFrac':>10s} "
              f"{'Redund':>10s} {'Accuracy':>12s}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for hd in hidden_dims:
        for config_name in configs:
            if config_name not in summary or hd not in summary[config_name]:
                continue
            m = summary[config_name][hd]
            print(
                f"{hd:>6d} {config_name:<15s} "
                f"{m['dead_units']['mean']:>5.1f}±{m['dead_units']['std']:<3.1f}"
                f"{m['dead_fraction']['mean']:>7.2f}±{m['dead_fraction']['std']:<4.2f}"
                f"{m['redundancy']['mean']:>7.1f}±{m['redundancy']['std']:<5.1f}"
                f"{m['test_acc']['mean']:>7.4f}±{m['test_acc']['std']:<5.4f}"
            )
        print("-" * len(header))

    print("=" * len(header))


def save_csv(summary: dict, hidden_dims: list[int], configs: list[str], path: Path):
    """Save capacity table as CSV."""

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Hidden Dim", "Config"] + SUMMARY_METRICS + ["dead_fraction"]
        writer.writerow(header)
        for hd in hidden_dims:
            for config_name in configs:
                if config_name not in summary or hd not in summary[config_name]:
                    continue
                m = summary[config_name][hd]
                row = [hd, config_name]
                for metric in SUMMARY_METRICS + ["dead_fraction"]:
                    row.append(f"{m[metric]['mean']:.4f}±{m[metric]['std']:.4f}")
                writer.writerow(row)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_accuracy_gain(summary: dict, hidden_dims: list[int], save_path: Path):
    """Accuracy gain (nls_var_tc - baseline) vs hidden dim."""

    gains_mean = []
    gains_std = []
    valid_dims = []

    for hd in hidden_dims:
        if ("baseline" in summary and hd in summary["baseline"] and
                "nls_var_tc" in summary and hd in summary["nls_var_tc"]):
            bl = summary["baseline"][hd]["test_acc"]["mean"]
            vc = summary["nls_var_tc"][hd]["test_acc"]["mean"]
            bl_std = summary["baseline"][hd]["test_acc"]["std"]
            vc_std = summary["nls_var_tc"][hd]["test_acc"]["std"]
            gains_mean.append(vc - bl)
            gains_std.append(np.sqrt(bl_std**2 + vc_std**2))
            valid_dims.append(hd)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(valid_dims, gains_mean, yerr=gains_std, fmt="o-",
                color="#d62728", capsize=4, linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Accuracy Gain (nls_var_tc − baseline)")
    ax.set_title("Volume Control Benefit vs Capacity")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_redundancy(summary: dict, hidden_dims: list[int], save_path: Path):
    """Redundancy vs hidden dim for both configs."""

    fig, ax = plt.subplots(figsize=(7, 5))

    for config_name in ["baseline", "nls_var_tc"]:
        if config_name not in summary:
            continue
        dims = []
        means = []
        stds = []
        for hd in hidden_dims:
            if hd in summary[config_name]:
                dims.append(hd)
                means.append(summary[config_name][hd]["redundancy"]["mean"])
                stds.append(summary[config_name][hd]["redundancy"]["std"])

        color = CONFIG_COLORS.get(config_name)
        ax.errorbar(dims, means, yerr=stds, fmt="o-", label=config_name,
                     color=color, capsize=4, linewidth=1.5)

    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Redundancy")
    ax.set_title("Redundancy vs Capacity")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiment2(
    train_loader,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    configs: list[str] = DEFAULT_CONFIGS,
    hidden_dims: list[int] = DEFAULT_HIDDEN_DIMS,
    seeds: list[int] = DEFAULT_SEEDS,
    epochs: int = 50,
    lr: float = 0.001,
    lambda_reg: float = 0.001,
    log_interval: int = 10,
):
    """Run Experiment 2: Capacity × Volume Control."""

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_curves = {}  # (config_name, hidden_dim) -> {seed -> curves}

    for config_name in configs:
        for hidden_dim in hidden_dims:
            print(f"\n{'='*60}")
            print(f"Experiment 2 — Config: {config_name}, hidden_dim={hidden_dim}")
            print(f"{'='*60}")

            key = (config_name, hidden_dim)
            all_curves[key] = {}

            for seed in seeds:
                print(f"\n--- Seed {seed} ---")
                model, curves = train_one(
                    config_name, seed, train_loader, test_x, test_y, device,
                    hidden_dim=hidden_dim, epochs=epochs, lr=lr,
                    lambda_reg=lambda_reg, log_interval=log_interval,
                )
                all_curves[key][seed] = curves

                ckpt_path = models_dir / f"{config_name}_h{hidden_dim}_seed{seed}.pt"
                torch.save({
                    "config_name": config_name,
                    "seed": seed,
                    "hidden_dim": hidden_dim,
                    "model_state_dict": model.state_dict(),
                }, ckpt_path)
                print(f"  Saved model: {ckpt_path}")

    # Save full curves (convert tuple keys to strings for JSON)
    json_curves = {}
    for (cfg, hd), seed_curves in all_curves.items():
        json_curves[f"{cfg}_h{hd}"] = seed_curves
    with open(output_dir / "training_curves.json", "w") as f:
        json.dump(json_curves, f, indent=2)
    print(f"Saved: {output_dir / 'training_curves.json'}")

    # Aggregate and save summary
    summary = aggregate_capacity(all_curves, hidden_dims)
    print_table(summary, hidden_dims, configs)
    save_csv(summary, hidden_dims, configs, output_dir / "capacity_table.csv")

    # Save raw summary as JSON
    json_summary = {
        cfg: {str(hd): metrics for hd, metrics in hd_dict.items()}
        for cfg, hd_dict in summary.items()
    }
    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved: {output_dir / 'raw_results.json'}")

    # Generate figures
    plot_accuracy_gain(summary, hidden_dims, figures_dir / "capacity_accuracy.png")
    plot_redundancy(summary, hidden_dims, figures_dir / "capacity_redundancy.png")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Capacity × Volume Control")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=DEFAULT_HIDDEN_DIMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda-reg", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="E:/ml_datasets")
    parser.add_argument("--output-dir", type=str,
                        default=str(_supervised_root / "results" / "experiment2"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_mnist(
        batch_size=args.batch_size, flatten=True, data_dir=args.data_dir,
        use_gpu_cache=(device.type == "cuda"), device=str(device),
    )
    test_x, test_y = test_loader.data, test_loader.labels

    run_experiment2(
        train_loader, test_x, test_y, device, Path(args.output_dir),
        configs=args.configs, hidden_dims=args.hidden_dims, seeds=args.seeds,
        epochs=args.epochs, lr=args.lr, lambda_reg=args.lambda_reg,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
