"""Experiment 6: SAE feature visualization.

Displays the 64 learned SAE encoder weights as 28x28 images in an 8x8 grid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import matplotlib.pyplot as plt
from typing import List

from src.model import StandardSAE
from src.data import get_mnist
from src.training import set_seed
from src.losses import sae_loss


def train_model(device: torch.device) -> StandardSAE:
    """Train a fresh SAE model."""
    print("No checkpoint provided. Training fresh SAE model...")

    set_seed(1)

    model = StandardSAE(input_dim=784, hidden_dim=64)
    model = model.to(device)

    train_loader, _ = get_mnist(batch_size=128, use_gpu_cache=True, device=str(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    l1_weight = 0.01

    for epoch in range(100):
        model.train()
        for batch in train_loader:
            x = batch[0]
            if x.device != device:
                x = x.to(device)

            optimizer.zero_grad()
            recon, a = model(x)
            losses = sae_loss(x, recon, a, l1_weight=l1_weight)
            losses["total"].backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/100")

    print("Training complete.")
    return model


def visualize_features(
    checkpoint_path: str, output_paths: List[str], device: torch.device
):
    """Load SAE model and visualize encoder features."""

    # Load or train model
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading SAE from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = StandardSAE(input_dim=784, hidden_dim=64)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
    else:
        model = train_model(device)

    # Get encoder weights - StandardSAE uses encoder[0].weight (first layer of Sequential)
    W = model.encoder[0].weight.data  # Shape: (64, 784)

    print(f"Visualizing {W.shape[0]} features (28x28 each)")

    # Reshape to images
    features = W.reshape(64, 28, 28).cpu().numpy()

    # Normalize for visualization
    vmax = abs(features).max()

    # Plot 8x8 grid
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(features[i], cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.axis("off")

    plt.tight_layout()

    # Save to all output paths
    for output_path in output_paths:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved to {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize SAE encoder features")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SAE model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        nargs="+",
        default=["figures/fig6_sae_features.pdf", "figures/fig6_sae_features.png"],
        help="Output figure path(s). Can specify multiple formats, e.g., --output fig.pdf fig.png",
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

    visualize_features(args.checkpoint, args.output, device)


if __name__ == "__main__":
    main()
