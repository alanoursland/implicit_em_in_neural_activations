"""Plot training dynamics figures.

Generates:
- figures/fig4a_dynamics_loss.png/pdf — Loss curves for SGD and Adam
- figures/fig4b_dynamics_convergence.png/pdf — Convergence epoch bar chart
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(data: dict, output_dir: Path):
    """Generate Figure 4a: Training loss curves.

    Two panels side-by-side showing SGD (left) and Adam (right).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Color scheme for learning rates
    colors = {0.0001: "C0", 0.001: "C1", 0.01: "C2", 0.1: "C3"}

    for item in data["sweep"]:
        optimizer = item["optimizer"]
        lr = item["lr"]
        ax = ax1 if optimizer.lower() == "sgd" else ax2

        # Average loss curves across seeds
        curves = [run["loss_curve"] for run in item["runs"]]
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)

        epochs = np.arange(len(mean_curve))
        ax.plot(epochs, mean_curve, color=colors[lr], label=f"lr={lr}", linewidth=1.5)

    # Styling
    ax1.set_title("SGD", fontsize=12)
    ax2.set_title("Adam", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save both formats
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "fig4a_dynamics_loss.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "fig4a_dynamics_loss.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir / 'fig4a_dynamics_loss.png'}")
    print(f"Saved: {output_dir / 'fig4a_dynamics_loss.pdf'}")


def plot_convergence_bars(data: dict, output_dir: Path):
    """Generate Figure 4b: Convergence epoch bar chart.

    Shows convergence epoch for each optimizer-lr configuration.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Collect data
    labels = []
    means = []
    stds = []
    colors_list = []

    # Color by optimizer
    opt_colors = {"sgd": "steelblue", "adam": "coral"}

    for item in data["sweep"]:
        optimizer = item["optimizer"]
        lr = item["lr"]

        conv_epochs = [run["convergence_epoch"] for run in item["runs"]]

        labels.append(f"{optimizer.upper()}\nlr={lr}")
        means.append(np.mean(conv_epochs))
        stds.append(np.std(conv_epochs))
        colors_list.append(opt_colors[optimizer.lower()])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors_list, edgecolor="black", linewidth=0.5)

    # Add horizontal line at 70 for reference
    ax.axhline(70, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax.text(len(labels) - 0.5, 72, "~EM convergence", fontsize=9, color="gray", ha="right")

    # Add horizontal line at 100 (max epochs)
    ax.axhline(100, color="lightgray", linestyle=":", alpha=0.7, linewidth=1)
    ax.text(len(labels) - 0.5, 97, "max epochs", fontsize=9, color="lightgray", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Convergence Epoch", fontsize=11)
    ax.set_xlabel("Configuration", fontsize=11)
    ax.set_title("Convergence Epoch by Optimizer and Learning Rate", fontsize=12)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", edgecolor="black", label="SGD"),
        Patch(facecolor="coral", edgecolor="black", label="Adam"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()

    # Save both formats
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "fig4b_dynamics_convergence.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "fig4b_dynamics_convergence.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir / 'fig4b_dynamics_convergence.png'}")
    print(f"Saved: {output_dir / 'fig4b_dynamics_convergence.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="Plot training dynamics figures")
    parser.add_argument(
        "--results",
        type=str,
        default="results/dynamics/dynamics_results.json",
        help="Path to dynamics results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--figure",
        type=str,
        choices=["all", "loss", "convergence"],
        default="all",
        help="Which figure to generate",
    )
    args = parser.parse_args()

    # Load data
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run dynamics experiment first.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    output_dir = Path(args.output_dir)

    if args.figure in ["all", "loss"]:
        plot_loss_curves(data, output_dir)

    if args.figure in ["all", "convergence"]:
        plot_convergence_bars(data, output_dir)


if __name__ == "__main__":
    main()
