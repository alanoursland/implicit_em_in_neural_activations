"""Experiment 1: Verify ∂L_LSE/∂E_j = r_j

This script verifies that the gradient of the LSE loss with respect to
activations equals the responsibilities (softmax of negative activations).

No training needed - single forward/backward pass.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from src.model import Encoder
from src.losses import lse_loss


def verify_theorem(
    batch_size: int = 64,
    input_dim: int = 784,
    hidden_dim: int = 64,
    seed: int = 42,
    output_path: str = None,
):
    """Verify that gradient of LSE loss equals responsibilities.

    Args:
        batch_size: Number of samples in batch
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        seed: Random seed
        output_path: Path to save figure (optional)

    Returns:
        Dict with verification results
    """
    torch.manual_seed(seed)

    # Create model and random input
    model = Encoder(input_dim, hidden_dim, activation="identity")
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    a, z = model(x)

    # Enable gradient tracking on activations (use retain_grad for non-leaf tensors)
    a.retain_grad()

    # Compute LSE loss and responsibilities
    loss, responsibilities = lse_loss(a)

    # Backward pass
    loss.backward()

    # Get gradients
    gradients = a.grad

    # Flatten for comparison
    r_flat = responsibilities.detach().cpu().numpy().flatten()
    g_flat = gradients.detach().cpu().numpy().flatten()

    # Compute correlation and max error
    correlation = np.corrcoef(r_flat, g_flat)[0, 1]
    max_error = np.abs(r_flat - g_flat).max()
    mean_error = np.abs(r_flat - g_flat).mean()

    print(f"Correlation between gradient and responsibility: {correlation:.6f}")
    print(f"Max absolute error: {max_error:.2e}")
    print(f"Mean absolute error: {mean_error:.2e}")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(r_flat, g_flat, alpha=0.3, s=10)

    # Add identity line
    lims = [
        min(r_flat.min(), g_flat.min()),
        max(r_flat.max(), g_flat.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=2, label="y = x")

    ax.set_xlabel("Responsibility r_j = softmax(-a)_j", fontsize=12)
    ax.set_ylabel("Gradient ∂L_LSE/∂a_j", fontsize=12)
    ax.set_title(f"Theorem Verification: Gradient = Responsibility\nCorr = {correlation:.4f}", fontsize=14)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    plt.close()

    return {
        "correlation": correlation,
        "max_error": max_error,
        "mean_error": mean_error,
        "responsibilities": r_flat,
        "gradients": g_flat,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify gradient = responsibility theorem")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--input-dim", type=int, default=784, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/fig1_theorem.pdf",
        help="Output path for figure",
    )
    args = parser.parse_args()

    results = verify_theorem(
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        output_path=args.output,
    )

    # Verify theorem holds
    if results["correlation"] > 0.9999:
        print("\n✓ Theorem verified: Gradient equals responsibility")
    else:
        print(f"\n✗ Theorem verification failed: correlation = {results['correlation']:.6f}")


if __name__ == "__main__":
    main()
