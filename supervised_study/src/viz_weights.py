"""Regenerate weight visualizations from saved model checkpoints.

Usage (from supervised_study/src/):
    python viz_weights.py
    python viz_weights.py --models-dir ../results/models --output-dir ../results/weights
    python viz_weights.py --configs baseline nls_var_tc --seed 44
    python viz_weights.py --gamma 0.4
    python viz_weights.py --hist-eq
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from model import SupervisedModel
from configs import CONFIGS

_supervised_root = Path(__file__).resolve().parent.parent


def rank_color(w: np.ndarray) -> np.ndarray:
    """Map weights to [0, 1] via separate rank-ordering of positive and negative values.

    Negative values get ranks mapped to [0, 0.5), zero maps to 0.5,
    positive values get ranks mapped to (0.5, 1.0].
    """
    result = np.full_like(w, 0.5, dtype=np.float64)

    neg_mask = w < 0
    pos_mask = w > 0

    if neg_mask.any():
        neg_vals = w[neg_mask]
        ranks = np.argsort(np.argsort(neg_vals)).astype(np.float64)
        n = len(neg_vals)
        result[neg_mask] = ranks / (2 * n)

    if pos_mask.any():
        pos_vals = w[pos_mask]
        ranks = np.argsort(np.argsort(pos_vals)).astype(np.float64)
        n = len(pos_vals)
        result[pos_mask] = 0.5 + (ranks + 1) / (2 * n)

    return result


def power_transform(w: np.ndarray, gamma: float) -> np.ndarray:
    """Signed power transform: sign(w) * |w|^gamma.

    With gamma < 1 this boosts small values toward the extremes of the
    colormap while compressing large values, increasing contrast without
    destroying relative magnitude information.
    """
    return np.sign(w) * np.abs(w) ** gamma


def visualize_weights(W: torch.Tensor, title: str, save_path: Path,
                      gamma: float | None = None, hist_eq: bool = False):
    """Visualize first-layer weights as a grid of 28x28 images.

    Args:
        gamma: If set, apply signed power transform (sign(w)*|w|^gamma)
               before coloring.  Values < 1 (e.g. 0.3-0.5) boost contrast
               on small weights.  None uses raw values.
        hist_eq: If True, use per-unit rank-ordered histogram coloring.
    """

    W = W.detach().cpu()  # (hidden_dim, 784)
    n_units = W.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_units)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    cmap = plt.cm.RdBu_r

    for i in range(grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        if i < n_units:
            img = W[i].reshape(28, 28).numpy()
            if gamma is not None:
                img = power_transform(img, gamma)
            if hist_eq:
                ax.imshow(cmap(rank_color(img)))
            else:
                vmax = np.abs(img).max()
                ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def load_model(ckpt_path: Path) -> SupervisedModel:
    """Load a SupervisedModel from a checkpoint file."""

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden_dim = ckpt["hidden_dim"]
    cfg_kwargs = ckpt["config_kwargs"]
    model = SupervisedModel(hidden_dim=hidden_dim, **cfg_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Regenerate weight visualizations")
    parser.add_argument("--models-dir", type=str,
                        default=str(_supervised_root / "results" / "models"))
    parser.add_argument("--output-dir", type=str,
                        default=str(_supervised_root / "results" / "weights"))
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Which configs to visualize (default: all found)")
    parser.add_argument("--seed", type=int, default=44,
                        help="Which seed's model to visualize (default: 44)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Signed power transform exponent (e.g. 0.4). "
                             "Values < 1 boost contrast on small weights.")
    parser.add_argument("--hist-eq", action="store_true",
                        help="Use rank-ordered histogram coloring for max contrast")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = args.configs or list(CONFIGS.keys())

    for config_name in configs:
        ckpt_path = models_dir / f"{config_name}_seed{args.seed}.pt"
        if not ckpt_path.exists():
            print(f"Skipping {config_name}: {ckpt_path} not found")
            continue

        model = load_model(ckpt_path)

        # Always generate raw version
        raw_path = output_dir / f"weights_{config_name}.png"
        visualize_weights(model.W1, config_name, raw_path)

        # Generate enhanced version(s) if requested
        if args.gamma is not None:
            gamma_path = output_dir / f"weights_{config_name}_gamma{args.gamma}.png"
            visualize_weights(model.W1, f"{config_name} (gamma={args.gamma})",
                              gamma_path, gamma=args.gamma)
        if args.hist_eq:
            eq_path = output_dir / f"weights_{config_name}_histeq.png"
            visualize_weights(model.W1, f"{config_name} (hist-eq)",
                              eq_path, gamma=args.gamma, hist_eq=True)


if __name__ == "__main__":
    main()
