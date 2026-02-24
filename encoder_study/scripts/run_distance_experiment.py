"""Experiment: Parameter distance analysis with LSE+InfoMax.

Measures L2 distance between initial and final parameters under different
training regimes to understand how LSE+InfoMax affects the optimization trajectory.

Conditions:
0. SAE baseline: Train SAE from random init. Measure distance(init, final).

1. Pretrain then SAE: Pretrain encoder with LSE+InfoMax, then train full SAE.
   Measure distance(random_init, pretrained), distance(pretrained, final),
   distance(random_init, final).

2. SAE + LSE+InfoMax regularizer: Train SAE from scratch with LSE+InfoMax
   added as a regularizer. Measure distance(init, final).

3. Alternating phases: Train SAE -> unsupervised LSE+InfoMax -> SAE again.
   Measure distances at each phase boundary.

Outputs results to results/distance_experiment/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import yaml
import json
import argparse
import copy
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm

from src.model import StandardSAE
from src.data import get_mnist
from src.training import set_seed
from src.losses import combined_loss, sae_loss
from src.metrics import sparsity_l0


def get_all_parameters(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


class ConvergenceTracker:
    """Track convergence via EMA of absolute loss deltas."""

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.ema = None
        self.prev_loss = None
        self.history = []

    def update(self, loss: float) -> float:
        """Update tracker with new loss value. Returns current EMA."""
        if self.prev_loss is not None:
            delta = abs(loss - self.prev_loss)
            if self.ema is None:
                self.ema = delta
            else:
                self.ema = self.decay * self.ema + (1 - self.decay) * delta
            self.history.append(self.ema)
        self.prev_loss = loss
        return self.ema if self.ema is not None else 0.0

    def get_final_ema(self) -> float:
        """Return final EMA value (lower = more converged)."""
        return self.ema if self.ema is not None else 0.0

    def reset(self):
        """Reset tracker for a new training phase."""
        self.ema = None
        self.prev_loss = None
        self.history = []


def parameter_l2_distance(params1: torch.Tensor, params2: torch.Tensor) -> float:
    """Compute L2 distance between two parameter vectors."""
    return float((params1 - params2).norm().item())


def parameter_cosine_similarity(params1: torch.Tensor, params2: torch.Tensor) -> float:
    """Compute cosine similarity between two parameter vectors."""
    return float(torch.nn.functional.cosine_similarity(
        params1.unsqueeze(0), params2.unsqueeze(0)
    ).item())


def evaluate_sae(
    model: StandardSAE,
    test_loader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate SAE on reconstruction metrics."""
    model.eval()

    all_a = []
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0]
            if x.device != device:
                x = x.to(device)
            recon, a = model(x)
            all_a.append(a)
            total_mse += torch.nn.functional.mse_loss(recon, x).item()
            n_batches += 1

    all_a = torch.cat(all_a, dim=0)
    l0, density = sparsity_l0(all_a, threshold=0.01)

    return {
        "reconstruction_mse": total_mse / n_batches,
        "l0_sparsity": l0,
        "l0_density": density,
    }


def train_sae_epoch(
    model: StandardSAE,
    train_loader,
    optimizer: torch.optim.Optimizer,
    l1_weight: float,
    device: torch.device,
) -> Dict[str, float]:
    """Train SAE for one epoch. Returns average losses."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    n_batches = 0

    for batch in train_loader:
        x = batch[0]
        if x.device != device:
            x = x.to(device)

        optimizer.zero_grad()
        recon, a = model(x)
        losses = sae_loss(x, recon, a, l1_weight=l1_weight)
        losses["total"].backward()
        optimizer.step()

        total_loss += losses["total"].item()
        total_mse += losses["mse"].item()
        total_l1 += losses["l1"].item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "l1": total_l1 / n_batches,
    }


def train_sae_with_regularizer_epoch(
    model: StandardSAE,
    train_loader,
    optimizer: torch.optim.Optimizer,
    l1_weight: float,
    infomax_config: Dict[str, Any],
    infomax_weight: float,
    device: torch.device,
) -> Dict[str, float]:
    """Train SAE for one epoch with LSE+InfoMax regularizer."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    total_infomax = 0.0
    n_batches = 0

    for batch in train_loader:
        x = batch[0]
        if x.device != device:
            x = x.to(device)

        optimizer.zero_grad()
        recon, a = model(x)

        # SAE loss
        sae_losses = sae_loss(x, recon, a, l1_weight=l1_weight)

        # InfoMax regularizer (applied to encoder weights)
        W_enc = model.encoder[0].weight
        infomax_losses = combined_loss(a, W_enc, infomax_config)

        # Combined loss
        total = sae_losses["total"] + infomax_weight * infomax_losses["total"]
        total.backward()
        optimizer.step()

        total_loss += total.item()
        total_mse += sae_losses["mse"].item()
        total_l1 += sae_losses["l1"].item()
        total_infomax += infomax_losses["total"].item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "l1": total_l1 / n_batches,
        "infomax": total_infomax / n_batches,
    }


def train_encoder_only_epoch(
    model: StandardSAE,
    train_loader,
    optimizer: torch.optim.Optimizer,
    infomax_config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Train only the encoder with LSE+InfoMax (freeze decoder)."""
    model.train()

    # Freeze decoder
    for p in model.decoder.parameters():
        p.requires_grad = False

    total_loss = 0.0
    total_lse = 0.0
    total_var = 0.0
    total_tc = 0.0
    n_batches = 0

    for batch in train_loader:
        x = batch[0]
        if x.device != device:
            x = x.to(device)

        optimizer.zero_grad()
        _, a = model(x)  # Only use activations
        W_enc = model.encoder[0].weight
        losses = combined_loss(a, W_enc, infomax_config)
        losses["total"].backward()
        optimizer.step()

        total_loss += losses["total"].item()
        total_lse += losses["lse"].item()
        total_var += losses["var"].item()
        total_tc += losses["tc"].item()
        n_batches += 1

    # Unfreeze decoder
    for p in model.decoder.parameters():
        p.requires_grad = True

    return {
        "loss": total_loss / n_batches,
        "lse": total_lse / n_batches,
        "var": total_var / n_batches,
        "tc": total_tc / n_batches,
    }


# =============================================================================
# Condition 0: SAE Baseline
# =============================================================================

def run_condition_0(
    train_loader,
    test_loader,
    config: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Condition 0: Train SAE from random init."""
    print("\n" + "=" * 70)
    print(f"Condition 0: SAE Baseline | Seed: {seed}")
    print("=" * 70)

    set_seed(seed)

    model = StandardSAE(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
    ).to(device)

    # Store initial parameters
    init_params = get_all_parameters(model).clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    l1_weight = config["sae"]["l1_weight"]
    epochs = config["training"]["epochs"]
    convergence = ConvergenceTracker(decay=config["training"].get("ema_decay", 0.9))

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training SAE"):
        epoch_losses = train_sae_epoch(model, train_loader, optimizer, l1_weight, device)
        convergence.update(epoch_losses["loss"])

    # Final parameters
    final_params = get_all_parameters(model)

    # Compute distances
    distance_init_final = parameter_l2_distance(init_params, final_params)
    cosine_init_final = parameter_cosine_similarity(init_params, final_params)
    convergence_ema = convergence.get_final_ema()

    # Evaluate
    metrics = evaluate_sae(model, test_loader, device)

    print(f"Distance (init -> final): {distance_init_final:.4f}")
    print(f"Cosine (init, final): {cosine_init_final:.4f}")
    print(f"Reconstruction MSE: {metrics['reconstruction_mse']:.6f}")
    print(f"Convergence EMA: {convergence_ema:.6f}")

    return {
        "condition": 0,
        "seed": seed,
        "distance_init_final": distance_init_final,
        "cosine_init_final": cosine_init_final,
        "convergence_ema": convergence_ema,
        "metrics": metrics,
    }


# =============================================================================
# Condition 1: Pretrain Encoder then SAE
# =============================================================================

def run_condition_1(
    train_loader,
    test_loader,
    config: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Condition 1: Pretrain encoder with LSE+InfoMax, then train full SAE."""
    print("\n" + "=" * 70)
    print(f"Condition 1: Pretrain Encoder -> SAE | Seed: {seed}")
    print("=" * 70)

    set_seed(seed)

    model = StandardSAE(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
    ).to(device)

    # Store initial parameters
    init_params = get_all_parameters(model).clone()

    infomax_config = config["infomax"]
    pretrain_epochs = config["training"]["pretrain_epochs"]
    sae_epochs = config["training"]["epochs"]
    l1_weight = config["sae"]["l1_weight"]
    ema_decay = config["training"].get("ema_decay", 0.9)

    # Phase 1: Pretrain encoder only
    print("Phase 1: Pretraining encoder with LSE+InfoMax...")
    encoder_params = list(model.encoder.parameters())
    optimizer = torch.optim.Adam(encoder_params, lr=config["training"]["lr"])
    pretrain_convergence = ConvergenceTracker(decay=ema_decay)

    for epoch in tqdm(range(pretrain_epochs), desc="Pretraining"):
        epoch_losses = train_encoder_only_epoch(model, train_loader, optimizer, infomax_config, device)
        pretrain_convergence.update(epoch_losses["loss"])

    pretrained_params = get_all_parameters(model).clone()
    pretrain_ema = pretrain_convergence.get_final_ema()

    # Phase 2: Train full SAE
    print("Phase 2: Training full SAE...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    sae_convergence = ConvergenceTracker(decay=ema_decay)

    for epoch in tqdm(range(sae_epochs), desc="SAE Training"):
        epoch_losses = train_sae_epoch(model, train_loader, optimizer, l1_weight, device)
        sae_convergence.update(epoch_losses["loss"])

    final_params = get_all_parameters(model)
    sae_ema = sae_convergence.get_final_ema()

    # Compute distances
    distance_init_pretrained = parameter_l2_distance(init_params, pretrained_params)
    distance_pretrained_final = parameter_l2_distance(pretrained_params, final_params)
    distance_init_final = parameter_l2_distance(init_params, final_params)

    cosine_init_pretrained = parameter_cosine_similarity(init_params, pretrained_params)
    cosine_pretrained_final = parameter_cosine_similarity(pretrained_params, final_params)
    cosine_init_final = parameter_cosine_similarity(init_params, final_params)

    # Evaluate
    metrics = evaluate_sae(model, test_loader, device)

    print(f"Distance (init -> pretrained): {distance_init_pretrained:.4f}")
    print(f"Distance (pretrained -> final): {distance_pretrained_final:.4f}")
    print(f"Distance (init -> final): {distance_init_final:.4f}")
    print(f"Reconstruction MSE: {metrics['reconstruction_mse']:.6f}")
    print(f"Convergence EMA (pretrain): {pretrain_ema:.6f}")
    print(f"Convergence EMA (SAE): {sae_ema:.6f}")

    return {
        "condition": 1,
        "seed": seed,
        "distance_init_pretrained": distance_init_pretrained,
        "distance_pretrained_final": distance_pretrained_final,
        "distance_init_final": distance_init_final,
        "cosine_init_pretrained": cosine_init_pretrained,
        "cosine_pretrained_final": cosine_pretrained_final,
        "cosine_init_final": cosine_init_final,
        "convergence_ema_pretrain": pretrain_ema,
        "convergence_ema_sae": sae_ema,
        "convergence_ema": sae_ema,  # Use final phase for summary
        "metrics": metrics,
    }


# =============================================================================
# Condition 2: SAE with LSE+InfoMax Regularizer
# =============================================================================

def run_condition_2(
    train_loader,
    test_loader,
    config: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Condition 2: Train SAE with LSE+InfoMax as regularizer."""
    print("\n" + "=" * 70)
    print(f"Condition 2: SAE + LSE+InfoMax Regularizer | Seed: {seed}")
    print("=" * 70)

    set_seed(seed)

    model = StandardSAE(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
    ).to(device)

    # Store initial parameters
    init_params = get_all_parameters(model).clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    l1_weight = config["sae"]["l1_weight"]
    infomax_config = config["infomax"]
    infomax_weight = config["training"]["infomax_weight"]
    epochs = config["training"]["epochs"]
    convergence = ConvergenceTracker(decay=config["training"].get("ema_decay", 0.9))

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training SAE+InfoMax"):
        epoch_losses = train_sae_with_regularizer_epoch(
            model, train_loader, optimizer, l1_weight,
            infomax_config, infomax_weight, device
        )
        convergence.update(epoch_losses["loss"])

    final_params = get_all_parameters(model)

    # Compute distances
    distance_init_final = parameter_l2_distance(init_params, final_params)
    cosine_init_final = parameter_cosine_similarity(init_params, final_params)
    convergence_ema = convergence.get_final_ema()

    # Evaluate
    metrics = evaluate_sae(model, test_loader, device)

    print(f"Distance (init -> final): {distance_init_final:.4f}")
    print(f"Cosine (init, final): {cosine_init_final:.4f}")
    print(f"Reconstruction MSE: {metrics['reconstruction_mse']:.6f}")
    print(f"Convergence EMA: {convergence_ema:.6f}")

    return {
        "condition": 2,
        "seed": seed,
        "distance_init_final": distance_init_final,
        "cosine_init_final": cosine_init_final,
        "convergence_ema": convergence_ema,
        "metrics": metrics,
    }


# =============================================================================
# Condition 3: Alternating Training Phases
# =============================================================================

def run_condition_3(
    train_loader,
    test_loader,
    config: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Condition 3: SAE -> Unsupervised -> SAE (alternating phases)."""
    print("\n" + "=" * 70)
    print(f"Condition 3: Alternating Phases | Seed: {seed}")
    print("=" * 70)

    set_seed(seed)

    model = StandardSAE(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
    ).to(device)

    # Store initial parameters
    init_params = get_all_parameters(model).clone()

    infomax_config = config["infomax"]
    l1_weight = config["sae"]["l1_weight"]
    phase_epochs = config["training"]["phase_epochs"]
    ema_decay = config["training"].get("ema_decay", 0.9)

    results = {
        "condition": 3,
        "seed": seed,
        "phases": [],
    }

    param_checkpoints = {"init": init_params.clone()}

    # Phase 1: Train SAE
    print("Phase 1: Training SAE...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    phase1_convergence = ConvergenceTracker(decay=ema_decay)

    for epoch in tqdm(range(phase_epochs), desc="SAE Phase 1"):
        epoch_losses = train_sae_epoch(model, train_loader, optimizer, l1_weight, device)
        phase1_convergence.update(epoch_losses["loss"])

    phase1_params = get_all_parameters(model).clone()
    param_checkpoints["phase1_sae"] = phase1_params
    phase1_ema = phase1_convergence.get_final_ema()

    metrics1 = evaluate_sae(model, test_loader, device)
    results["phases"].append({
        "phase": "sae_1",
        "epochs": phase_epochs,
        "metrics": metrics1,
        "distance_from_init": parameter_l2_distance(init_params, phase1_params),
        "cosine_with_init": parameter_cosine_similarity(init_params, phase1_params),
        "convergence_ema": phase1_ema,
    })
    print(f"Phase 1 Reconstruction MSE: {metrics1['reconstruction_mse']:.6f}, Convergence EMA: {phase1_ema:.6f}")

    # Phase 2: Unsupervised LSE+InfoMax (encoder only)
    print("Phase 2: Unsupervised LSE+InfoMax...")
    encoder_params = list(model.encoder.parameters())
    optimizer = torch.optim.Adam(encoder_params, lr=config["training"]["lr"])
    phase2_convergence = ConvergenceTracker(decay=ema_decay)

    for epoch in tqdm(range(phase_epochs), desc="Unsupervised Phase"):
        epoch_losses = train_encoder_only_epoch(model, train_loader, optimizer, infomax_config, device)
        phase2_convergence.update(epoch_losses["loss"])

    phase2_params = get_all_parameters(model).clone()
    param_checkpoints["phase2_unsup"] = phase2_params
    phase2_ema = phase2_convergence.get_final_ema()

    metrics2 = evaluate_sae(model, test_loader, device)
    results["phases"].append({
        "phase": "unsupervised",
        "epochs": phase_epochs,
        "metrics": metrics2,
        "distance_from_init": parameter_l2_distance(init_params, phase2_params),
        "distance_from_prev": parameter_l2_distance(phase1_params, phase2_params),
        "cosine_with_init": parameter_cosine_similarity(init_params, phase2_params),
        "cosine_with_prev": parameter_cosine_similarity(phase1_params, phase2_params),
        "convergence_ema": phase2_ema,
    })
    print(f"Phase 2 Reconstruction MSE: {metrics2['reconstruction_mse']:.6f}, Convergence EMA: {phase2_ema:.6f}")

    # Phase 3: Train SAE again
    print("Phase 3: Training SAE again...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    phase3_convergence = ConvergenceTracker(decay=ema_decay)

    for epoch in tqdm(range(phase_epochs), desc="SAE Phase 2"):
        epoch_losses = train_sae_epoch(model, train_loader, optimizer, l1_weight, device)
        phase3_convergence.update(epoch_losses["loss"])

    phase3_params = get_all_parameters(model).clone()
    param_checkpoints["phase3_sae"] = phase3_params
    phase3_ema = phase3_convergence.get_final_ema()

    metrics3 = evaluate_sae(model, test_loader, device)
    results["phases"].append({
        "phase": "sae_2",
        "epochs": phase_epochs,
        "metrics": metrics3,
        "distance_from_init": parameter_l2_distance(init_params, phase3_params),
        "distance_from_prev": parameter_l2_distance(phase2_params, phase3_params),
        "cosine_with_init": parameter_cosine_similarity(init_params, phase3_params),
        "cosine_with_prev": parameter_cosine_similarity(phase2_params, phase3_params),
        "convergence_ema": phase3_ema,
    })
    print(f"Phase 3 Reconstruction MSE: {metrics3['reconstruction_mse']:.6f}, Convergence EMA: {phase3_ema:.6f}")

    # Summary distances
    results["distance_init_final"] = parameter_l2_distance(init_params, phase3_params)
    results["cosine_init_final"] = parameter_cosine_similarity(init_params, phase3_params)
    results["convergence_ema"] = phase3_ema  # Use final phase for summary
    results["final_metrics"] = metrics3

    return results


# =============================================================================
# Main
# =============================================================================

def print_summary(all_results: List[Dict[str, Any]]):
    """Print summary table of all results."""
    print("\n" + "=" * 95)
    print(" " * 30 + "DISTANCE EXPERIMENT SUMMARY")
    print("=" * 95)

    # Group by condition
    conditions = {}
    for r in all_results:
        cond = r["condition"]
        if cond not in conditions:
            conditions[cond] = []
        conditions[cond].append(r)

    condition_names = {
        0: "SAE Baseline",
        1: "Pretrain -> SAE",
        2: "SAE + InfoMax Reg",
        3: "Alternating",
    }

    print(f"\n{'Condition':<20} | {'Distance':<12} | {'Cosine':<8} | {'Recon MSE':<16} | {'Conv EMA':<12}")
    print("-" * 95)

    for cond in sorted(conditions.keys()):
        results = conditions[cond]
        name = condition_names[cond]

        distances = [r["distance_init_final"] for r in results]
        cosines = [r["cosine_init_final"] for r in results]
        emas = [r["convergence_ema"] for r in results]

        if cond == 3:
            mses = [r["final_metrics"]["reconstruction_mse"] for r in results]
        else:
            mses = [r["metrics"]["reconstruction_mse"] for r in results]

        dist_mean = np.mean(distances)
        dist_std = np.std(distances)
        cos_mean = np.mean(cosines)
        mse_mean = np.mean(mses)
        mse_std = np.std(mses)
        ema_mean = np.mean(emas)
        ema_std = np.std(emas)

        print(f"{name:<20} | {dist_mean:>6.2f}±{dist_std:<4.2f} | {cos_mean:>6.4f} | {mse_mean:.6f}±{mse_std:.6f} | {ema_mean:.6f}±{ema_std:.4f}")

    print("=" * 95)


def run_experiment(config_path: str, output_dir: str, device: torch.device):
    """Run full distance experiment."""
    print("=" * 80)
    print(" " * 15 + "PARAMETER DISTANCE EXPERIMENT")
    print("=" * 80)
    print(f"Using device: {device}")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seeds = config.get("seeds", [1, 2, 3])

    # Load data
    print("Loading MNIST...")
    train_loader, test_loader = get_mnist(
        batch_size=config["training"]["batch_size"],
        use_gpu_cache=True,
        device=str(device),
    )
    print("MNIST loaded.\n")

    all_results = []

    # Run all conditions for all seeds
    for seed in seeds:
        # Condition 0: SAE Baseline
        result = run_condition_0(train_loader, test_loader, config, seed, device)
        all_results.append(result)

        # Condition 1: Pretrain then SAE
        result = run_condition_1(train_loader, test_loader, config, seed, device)
        all_results.append(result)

        # Condition 2: SAE + InfoMax Regularizer
        result = run_condition_2(train_loader, test_loader, config, seed, device)
        all_results.append(result)

        # Condition 3: Alternating Phases
        result = run_condition_3(train_loader, test_loader, config, seed, device)
        all_results.append(result)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "distance_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print_summary(all_results)

    print(f"\nResults saved to {output_path / 'distance_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run parameter distance experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="config/distance_experiment.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/distance_experiment",
        help="Output directory",
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

    run_experiment(args.config, args.output_dir, device)


if __name__ == "__main__":
    main()
