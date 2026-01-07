import argparse
import yaml
import torch
import json
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SingleLayer
from src.losses import infomax_loss
from src.metrics import compute_all_metrics
from src.data import get_data
from src.training import train

def run(config_path: str, output_dir: str = None, profile: bool = False, profile_epochs: int = 5):
    print("=" * 70)
    print("InfoMax Activation Study - Experiment Runner")
    if profile:
        print("PROFILING MODE - LIMITED EPOCHS")
    print("=" * 70)

    # Load config
    print(f"\n[1/7] Loading configuration from: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override epochs if profiling
    if profile:
        config['training']['epochs'] = profile_epochs
        print(f"  ⚡ Profiling mode: epochs overridden to {profile_epochs}")

    print(f"  ✓ Configuration loaded")
    print(f"    - Dataset: {config['data']['dataset']}")
    print(f"    - Activation: {config['model']['activation']}")
    print(f"    - Hidden dim: {config['model']['hidden_dim']}")
    print(f"    - Epochs: {config['training']['epochs']}")
    print(f"    - Learning rate: {config['training']['lr']}")
    print(f"    - Seed: {config['training']['seed']}")

    # Setup
    print(f"\n[2/7] Setting up device and random seed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  ✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA version: {torch.version.cuda}")

    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    print(f"  ✓ Random seed set to: {seed}")

    # Data
    print(f"\n[3/7] Loading dataset: {config['data']['dataset']}")
    train_loader, test_loader, input_dim = get_data(
        config["data"]["dataset"],
        config["data"]["batch_size"],
    )
    print(f"  ✓ Dataset loaded")
    print(f"    - Input dimension: {input_dim}")
    print(f"    - Batch size: {config['data']['batch_size']}")
    print(f"    - Training batches: {len(train_loader)}")
    print(f"    - Test batches: {len(test_loader)}")

    # Model
    print(f"\n[4/7] Creating model")
    model = SingleLayer(
        input_dim=input_dim,
        hidden_dim=config["model"]["hidden_dim"],
        activation=config["model"]["activation"],
    ).to(device)
    print(f"  ✓ Model created: {input_dim} → {config['model']['hidden_dim']} ({config['model']['activation']})")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    - Total parameters: {total_params:,}")

    # Loss
    print(f"\n[5/7] Setting up loss function")
    lambda_tc = config["loss"]["lambda_tc"]
    lambda_wr = config["loss"].get("lambda_wr", 0.0)
    eps = config["loss"]["variance_eps"]
    loss_fn = lambda a: infomax_loss(a, W=model.linear.weight, lambda_tc=lambda_tc, lambda_wr=lambda_wr, eps=eps)
    print(f"  ✓ InfoMax loss configured")
    print(f"    - lambda_tc: {lambda_tc}")
    print(f"    - lambda_wr: {lambda_wr}")
    print(f"    - variance_eps: {eps}")

    # Train
    print(f"\n[6/7] Starting training for {config['training']['epochs']} epochs")
    print("-" * 70)

    if profile:
        print("Starting PyTorch profiler...")
        import torch.cuda.profiler as cuda_profiler
        import torch.autograd.profiler as autograd_profiler

        # Enable CUDA profiling
        if torch.cuda.is_available():
            torch.cuda.profiler.start()

        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # Disable stack traces for faster profiling
        ) as prof:
            history = train(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                loss_fn=loss_fn,
                epochs=config["training"]["epochs"],
                lr=config["training"]["lr"],
                device=device,
                metrics_fn=compute_all_metrics,
                log_every=config["logging"]["save_metrics_every"],
                print_every=config["logging"].get("print_every", 5),
                optimizer_name=config["training"].get("optimizer", "adam"),
            )

        if torch.cuda.is_available():
            torch.cuda.profiler.stop()

        print("-" * 70)
        print(f"  ✓ Training complete!")

        # Save profiler trace
        if output_dir is None:
            output_dir = config["logging"]["output_dir"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_path = output_dir / "profile_trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"\n⚡ Profiler trace saved to: {trace_path}")
        print(f"   View in Chrome: chrome://tracing")

        # Print table summary
        print(f"\n{'='*70}")
        print("Top 10 operations by CUDA time:")
        print(f"{'='*70}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print(f"\n{'='*70}")
        print("Top 10 operations by CPU time:")
        print(f"{'='*70}")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    else:
        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            epochs=config["training"]["epochs"],
            lr=config["training"]["lr"],
            device=device,
            metrics_fn=compute_all_metrics,
            log_every=config["logging"]["save_metrics_every"],
            print_every=config["logging"].get("print_every", 5),
            optimizer_name=config["training"].get("optimizer", "adam"),
        )
        print("-" * 70)
        print(f"  ✓ Training complete!")

    # Save results
    print(f"\n[7/7] Saving results")
    if output_dir is None:
        output_dir = config["logging"]["output_dir"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {output_dir}")

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ Saved history.json")

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    print(f"  ✓ Saved config.yaml")

    # Save weights
    if config["logging"]["save_weights"]:
        torch.save(model.state_dict(), output_dir / "model.pt")
        print(f"  ✓ Saved model.pt")

    # Print final metrics
    if "final_metrics" in history:
        print(f"\nFinal Metrics:")
        print(f"  - Effective Rank: {history['final_metrics']['effective_rank']:.2f}")
        print(f"  - Weight Redundancy: {history['final_metrics']['weight_redundancy']:.4f}")
        print(f"  - Dead Units: {history['final_metrics']['dead_units']}")
        print(f"  - Output Correlation: {history['final_metrics']['output_correlation']:.4f}")
        print(f"  - Final Loss: {history['train_loss'][-1]:.4f}")

    print("\n" + "=" * 70)
    print("Experiment completed successfully!")
    print("=" * 70 + "\n")

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode")
    parser.add_argument("--profile-epochs", type=int, default=5, help="Number of epochs to run in profiling mode")
    args = parser.parse_args()

    run(args.config, args.output, args.profile, args.profile_epochs)
