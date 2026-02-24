"""Sweep lambda_reg for nls_var_tc configuration.

Usage (from supervised_study/src/):
    python run_sweep.py
    python run_sweep.py --epochs 50
"""

import argparse
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from model import SupervisedModel
from configs import CONFIGS
from metrics import min_variance, redundancy_score
from data import get_mnist
from utils import set_seed

_supervised_root = Path(__file__).resolve().parent.parent

LAMBDA_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0]


def train_and_evaluate(lambda_reg, train_loader, test_x, test_y, device, hidden_dim=64, epochs=100, lr=0.001, seed=42, log_interval=10):
    """Train nls_var_tc with a given lambda_reg and return final metrics + per-epoch history."""

    set_seed(seed)
    cfg = CONFIGS["nls_var_tc"]
    model = SupervisedModel(hidden_dim=hidden_dim, **cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "ce_loss": [],
        "reg_loss": [],
        "train_acc": [],
        "test_acc": [],
        "redundancy": [],
        "min_variance": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_ce = 0.0
        epoch_reg = 0.0
        correct = 0
        total_samples = 0
        n_batches = 0

        for x, labels in train_loader:
            optimizer.zero_grad()

            h = model(x)
            ce_loss = F.cross_entropy(h, labels)
            reg_loss = model.regularization_loss()["total"]
            total = ce_loss + lambda_reg * reg_loss

            total.backward()
            optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_reg += reg_loss.item()
            n_batches += 1

            with torch.no_grad():
                correct += (h.argmax(dim=1) == labels).sum().item()
                total_samples += labels.shape[0]

        train_acc = correct / total_samples
        model.eval()
        with torch.no_grad():
            test_h = model(test_x)
            test_d = model.distances
            test_acc = (test_h.argmax(dim=1) == test_y).float().mean().item()
        model.train()

        avg_ce = epoch_ce / n_batches
        avg_reg = epoch_reg / n_batches

        history["epoch"].append(epoch + 1)
        history["ce_loss"].append(avg_ce)
        history["reg_loss"].append(avg_reg)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["redundancy"].append(redundancy_score(test_d))
        history["min_variance"].append(min_variance(test_d))

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(
                f"  [lambda={lambda_reg}] epoch {epoch+1:3d}  "
                f"CE={avg_ce:.4f}  "
                f"reg={avg_reg:.4f}  "
                f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}"
            )

    final = {
        "lambda_reg": lambda_reg,
        "ce_loss": history["ce_loss"][-1],
        "reg_loss": history["reg_loss"][-1],
        "accuracy": history["test_acc"][-1],
        "redundancy": history["redundancy"][-1],
        "min_variance": history["min_variance"][-1],
    }

    return final, history


def main():
    parser = argparse.ArgumentParser(description="Sweep lambda_reg for nls_var_tc")
    parser.add_argument("--lambdas", nargs="+", type=float, default=LAMBDA_VALUES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Number of hidden units / mixture components")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print training loss every N epochs")
    parser.add_argument("--data-dir", type=str, default="E:/ml_datasets")
    parser.add_argument("--output-dir", type=str, default=str(_supervised_root / "results"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_gpu_cache = device.type == "cuda"
    train_loader, test_loader = get_mnist(
        batch_size=args.batch_size,
        flatten=True,
        data_dir=args.data_dir,
        use_gpu_cache=use_gpu_cache,
        device=str(device),
    )

    # Full test tensors for single-pass evaluation
    test_x, test_y = test_loader.data, test_loader.labels

    results = []
    all_history = {}
    for lam in args.lambdas:
        print(f"\n{'='*60}")
        print(f"lambda_reg = {lam}")
        print(f"{'='*60}")
        r, history = train_and_evaluate(lam, train_loader, test_x, test_y, device, args.hidden_dim, args.epochs, args.lr, args.seed, args.log_interval)
        results.append(r)
        all_history[str(lam)] = history
        print(f"  -> CE={r['ce_loss']:.4f}  reg={r['reg_loss']:.4f}  "
              f"acc={r['accuracy']:.4f}  redund={r['redundancy']:.1f}  "
              f"min_var={r['min_variance']:.4f}")

    # Print table
    header = f"{'lambda_reg':>12s} {'CE Loss':>10s} {'Reg Loss':>10s} {'Accuracy':>10s} {'Redundancy':>12s} {'Min Variance':>14s}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['lambda_reg']:>12.4f} "
            f"{r['ce_loss']:>10.4f} "
            f"{r['reg_loss']:>10.4f} "
            f"{r['accuracy']:>10.4f} "
            f"{r['redundancy']:>12.1f} "
            f"{r['min_variance']:>14.4f}"
        )
    print("=" * len(header))

    # Save CSV
    csv_path = output_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda_reg", "CE Loss", "Reg Loss", "Accuracy", "Redundancy", "Min Variance"])
        for r in results:
            writer.writerow([
                r["lambda_reg"],
                f"{r['ce_loss']:.4f}",
                f"{r['reg_loss']:.4f}",
                f"{r['accuracy']:.4f}",
                f"{r['redundancy']:.1f}",
                f"{r['min_variance']:.4f}",
            ])
    print(f"Saved: {csv_path}")

    # Save per-epoch history for all lambda values
    history_path = output_dir / "sweep_history.json"
    with open(history_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"Saved: {history_path}")


if __name__ == "__main__":
    main()
