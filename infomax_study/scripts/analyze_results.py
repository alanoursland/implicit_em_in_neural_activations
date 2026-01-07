import argparse
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_dir: str) -> pd.DataFrame:
    """
    Load all experiment results into a DataFrame.
    """
    results_dir = Path(results_dir)

    rows = []

    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        history_path = exp_dir / "history.json"
        config_path = exp_dir / "config.yaml"

        if not history_path.exists():
            continue

        with open(history_path) as f:
            history = json.load(f)

        with open(config_path) as f:
            import yaml
            config = yaml.safe_load(f)

        row = {
            "name": exp_dir.name,
            "activation": config["model"]["activation"],
            "hidden_dim": config["model"]["hidden_dim"],
            "lambda_tc": config["loss"]["lambda_tc"],
            "seed": config["training"]["seed"],
            "final_loss": history["train_loss"][-1],
        }

        if "final_metrics" in history:
            row.update(history["final_metrics"])

        rows.append(row)

    return pd.DataFrame(rows)

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by activation.
    """
    agg = df.groupby("activation").agg({
        "effective_rank": ["mean", "std"],
        "weight_redundancy": ["mean", "std"],
        "dead_units": ["mean", "std"],
        "output_correlation": ["mean", "std"],
        "final_loss": ["mean", "std"],
    })

    return agg

def plot_effective_rank(df: pd.DataFrame, output_path: str = None):
    """
    Box plot of effective rank by activation.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="activation", y="effective_rank")
    plt.title("Effective Rank by Activation")
    plt.ylabel("Effective Rank")
    plt.xlabel("Activation")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

def plot_redundancy_vs_dead(df: pd.DataFrame, output_path: str = None):
    """
    Scatter plot: redundancy vs dead units, colored by activation.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="weight_redundancy",
        y="dead_units",
        hue="activation",
        style="activation",
        s=100,
    )
    plt.title("Weight Redundancy vs Dead Units")
    plt.xlabel("Weight Redundancy (lower = better)")
    plt.ylabel("Dead Units (lower = better)")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

def plot_training_curves(results_dir: str, activation: str, output_path: str = None):
    """
    Plot training curves for one activation across seeds.
    """
    results_dir = Path(results_dir)

    plt.figure(figsize=(12, 4))

    for exp_dir in results_dir.iterdir():
        if activation not in exp_dir.name:
            continue

        history_path = exp_dir / "history.json"
        if not history_path.exists():
            continue

        with open(history_path) as f:
            history = json.load(f)

        plt.subplot(1, 3, 1)
        plt.plot(history["train_loss"], alpha=0.5)
        plt.title("Total Loss")

        plt.subplot(1, 3, 2)
        plt.plot(history["train_entropy"], alpha=0.5)
        plt.title("Entropy Loss")

        plt.subplot(1, 3, 3)
        plt.plot(history["train_tc"], alpha=0.5)
        plt.title("TC Loss")

    plt.suptitle(f"Training Curves: {activation}")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="analysis/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_dir)

    # Summary table
    summary = summary_table(df)
    print(summary)
    summary.to_csv(output_dir / "summary.csv")

    # Plots
    plot_effective_rank(df, output_dir / "effective_rank.png")
    plot_redundancy_vs_dead(df, output_dir / "redundancy_vs_dead.png")

    for activation in df["activation"].unique():
        plot_training_curves(
            args.results_dir,
            activation,
            output_dir / f"training_curves_{activation}.png"
        )
