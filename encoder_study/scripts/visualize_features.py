"""Experiment 5: Feature visualization.

Displays the 64 learned weight vectors as 28x28 images in an 8x8 grid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import matplotlib.pyplot as plt

from src.model import Encoder
from src.data import get_mnist
from src.training import set_seed
from src.losses import combined_loss


def train_model(device: torch.device) -> Encoder:
    """Train a fresh model."""
    print("No checkpoint provided. Training fresh model...")

    set_seed(1)

    model = Encoder(input_dim=784, hidden_dim=64, activation="relu")
    model = model.to(device)

    train_loader, _ = get_mnist(batch_size=128, use_gpu_cache=True, device=str(device))

    loss_config = {"lambda_lse": 1.0, "lambda_var": 1.0, "lambda_tc": 1.0}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        for batch in train_loader:
            x = batch[0]
            if x.device != device:
                x = x.to(device)

            optimizer.zero_grad()
            a, _ = model(x)
            losses = combined_loss(a, model.W, loss_config)
            losses["total"].backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/100")

    print("Training complete.")
    return model


def visualize_features(checkpoint_path: str, output_path: str, device: torch.device):
    """Load model and visualize features."""

    # Load or train model
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = Encoder(input_dim=784, hidden_dim=64, activation="relu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
    else:
        model = train_model(device)

    # Get weights
    W = model.encoder.weight.data  # Shape: (64, 784)

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

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Figure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize learned features")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/fig5_features.pdf",
        help="Output figure path",
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
