"""Run all three experiments.

Loads MNIST once and runs experiments sequentially.

Usage (from supervised_study/src/):
    python run_all.py                    # all 3 experiments
    python run_all.py --experiments 1 2  # just experiments 1 and 2
    python run_all.py --experiments 3    # just experiment 3
"""

import argparse
import torch
from pathlib import Path

from data import get_mnist
from run_experiment1 import run_experiment1
from run_experiment2 import run_experiment2
from run_experiment3 import run_experiment3

_supervised_root = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--experiments", nargs="+", type=int, default=[1, 2, 3],
                        help="Which experiments to run (default: 1 2 3)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="E:/ml_datasets")
    parser.add_argument("--output-dir", type=str,
                        default=str(_supervised_root / "results"))
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data once
    train_loader, test_loader = get_mnist(
        batch_size=args.batch_size, flatten=True, data_dir=args.data_dir,
        use_gpu_cache=(device.type == "cuda"), device=str(device),
    )
    test_x, test_y = test_loader.data, test_loader.labels

    if 1 in args.experiments:
        print("\n" + "#" * 70)
        print("# EXPERIMENT 1: Volume Control Ablation")
        print("#" * 70)
        run_experiment1(
            train_loader, test_x, test_y, device,
            output_dir / "experiment1",
            log_interval=args.log_interval,
        )

    if 2 in args.experiments:
        print("\n" + "#" * 70)
        print("# EXPERIMENT 2: Capacity × Volume Control")
        print("#" * 70)
        run_experiment2(
            train_loader, test_x, test_y, device,
            output_dir / "experiment2",
            log_interval=args.log_interval,
        )

    if 3 in args.experiments:
        print("\n" + "#" * 70)
        print("# EXPERIMENT 3: Optimization Dynamics")
        print("#" * 70)
        run_experiment3(
            train_loader, test_x, test_y, device,
            output_dir / "experiment3",
            log_interval=args.log_interval,
        )

    print("\nAll requested experiments complete.")


if __name__ == "__main__":
    main()
