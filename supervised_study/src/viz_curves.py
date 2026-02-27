"""Plot training curves from saved JSON.

Usage (from supervised_study/src/):
    python viz_curves.py
    python viz_curves.py --configs baseline nls_var_tc
    python viz_curves.py --metrics test_acc ce_loss
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from configs import CONFIGS

_supervised_root = Path(__file__).resolve().parent.parent

METRIC_LABELS = {
    "ce_loss": "Cross-Entropy Loss",
    "reg_loss": "Regularization Loss",
    "total_loss": "Total Loss",
    "train_acc": "Log Train Classification Error",
    "test_acc": "Log Test Classification Error",
}

ACC_METRICS = {"train_acc", "test_acc"}
LOG_SCALE_METRICS = {"ce_loss", "total_loss"}

CONFIG_COLORS = {
    "baseline": "#1f77b4",
    "nls_only": "#ff7f0e",
    "nls_var": "#2ca02c",
    "nls_var_tc": "#d62728",
    "var_tc_only": "#9467bd",
}


def plot_curves(all_curves: dict, configs: list[str], metrics: list[str],
                save_path: Path):
    """Plot training curves averaged across seeds with std shading."""

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 5),
                             squeeze=False)

    for col, metric in enumerate(metrics):
        ax = axes[0, col]

        for config_name in configs:
            if config_name not in all_curves:
                continue

            seed_curves = all_curves[config_name]
            arrays = []
            for seed, curves in seed_curves.items():
                arrays.append(curves[metric])

            epochs = np.array(all_curves[config_name][next(iter(seed_curves))]["epoch"])
            data = np.array(arrays)

            if metric in ACC_METRICS:
                data = np.log10(np.clip(1.0 - data, 1e-10, None))

            mean = data.mean(axis=0)
            std = data.std(axis=0)

            color = CONFIG_COLORS.get(config_name, None)
            ax.plot(epochs, mean, label=config_name, color=color, linewidth=1.5)

        if metric in LOG_SCALE_METRICS:
            ax.set_yscale("log")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--curves-file", type=str,
                        default=str(_supervised_root / "results" / "curves" /
                                    "training_curves.json"))
    parser.add_argument("--output-dir", type=str,
                        default=str(_supervised_root / "results" / "curves"))
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Which configs to plot (default: all)")
    parser.add_argument("--metrics", nargs="+",
                        default=["test_acc", "ce_loss", "reg_loss"],
                        help="Which metrics to plot (default: test_acc ce_loss reg_loss)")
    args = parser.parse_args()

    with open(args.curves_file) as f:
        all_curves = json.load(f)

    configs = args.configs or list(CONFIGS.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "training_curves.png"
    plot_curves(all_curves, configs, args.metrics, save_path)


if __name__ == "__main__":
    main()
