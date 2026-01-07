import torch
import numpy as np

# Cache for identity matrices (K -> identity matrix per device)
_identity_cache = {}

def _get_identity(K: int, device: torch.device) -> torch.Tensor:
    """Get cached identity matrix for given size and device."""
    key = (K, device)
    if key not in _identity_cache:
        _identity_cache[key] = torch.eye(K, device=device)
    return _identity_cache[key]

def weight_redundancy(W: torch.Tensor) -> float:
    """
    Measure redundancy in weight matrix via normalized Gram matrix.

    ||WᵀW / ||W||² - I/K||²

    Lower is better.
    """
    K = W.shape[0]

    # Normalize rows (keep on GPU if possible)
    norms = torch.norm(W, dim=1, keepdim=True) + 1e-8
    W_normalized = W / norms

    # Gram matrix
    gram = W_normalized @ W_normalized.T

    # Compare to cached identity
    identity = _get_identity(K, W.device)
    redundancy = ((gram - identity) ** 2).sum()

    return float(redundancy)

def effective_rank(W: torch.Tensor) -> float:
    """
    Compute effective rank of weight matrix.

    eff_rank = (Σσᵢ)² / Σσᵢ²

    Higher is better.
    """
    # Use PyTorch SVD (can run on GPU)
    try:
        _, s, _ = torch.linalg.svd(W, full_matrices=False)
    except:
        # Fallback to CPU if GPU SVD fails
        _, s, _ = torch.linalg.svd(W.cpu(), full_matrices=False)
        s = s.to(W.device)

    s = s + 1e-10  # stability

    eff_rank = (s.sum() ** 2) / (s ** 2).sum()

    return float(eff_rank)

def dead_units(a: torch.Tensor, threshold: float = 0.01) -> int:
    """
    Count units with near-zero activation across batch.

    Args:
        a: (N, K) activations
        threshold: activation rate below this = dead

    Returns:
        count of dead units
    """
    activation_rate = (a.abs() > 1e-6).float().mean(dim=0)
    dead = (activation_rate < threshold).sum()
    return int(dead)

def activation_rate_per_unit(a: torch.Tensor) -> np.ndarray:
    """
    Fraction of samples where each unit is active.

    Returns:
        (K,) array of activation rates
    """
    rate = (a.abs() > 1e-6).float().mean(dim=0)
    return rate.cpu().numpy()

def max_avg_responsibility(a: torch.Tensor) -> float:
    """
    For softmax outputs, measure collapse via max average responsibility.

    Close to 1 = collapse. Close to 1/K = balanced.
    """
    # Assumes a is already softmax output (sums to 1)
    avg_responsibility = a.mean(dim=0)
    return float(avg_responsibility.max())

def variance_ratio(a: torch.Tensor) -> float:
    """
    Ratio of max to min variance across units.

    High = some units much more active than others.
    """
    var = a.var(dim=0)
    ratio = var.max() / (var.min() + 1e-10)
    return float(ratio)

def output_correlation(a: torch.Tensor) -> float:
    """
    ||Corr(A) - I||²

    Lower is better (more independent).
    """
    K = a.shape[1]

    a_centered = a - a.mean(dim=0, keepdim=True)
    std = a.std(dim=0, keepdim=True) + 1e-6
    a_standardized = a_centered / std

    N = a.shape[0]
    corr = (a_standardized.T @ a_standardized) / (N - 1)

    identity = _get_identity(K, a.device)
    return float(((corr - identity) ** 2).sum())

def compute_all_metrics(model, dataloader, device, max_batches=None) -> dict:
    """
    Compute all metrics over full dataset (or subset if max_batches specified).

    Args:
        max_batches: If specified, only use first N batches (for speed)
    """
    model.eval()

    all_a = []

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            # Only move to device if not already there
            if x.device != device:
                x = x.to(device)
            a, z = model(x)
            # Keep on GPU for faster computation
            all_a.append(a)

    # Concatenate on GPU
    all_a = torch.cat(all_a, dim=0)

    W = model.weight_matrix

    metrics = {
        "weight_redundancy": weight_redundancy(W),
        "effective_rank": effective_rank(W),
        "dead_units": dead_units(all_a),
        "variance_ratio": variance_ratio(all_a),
        "output_correlation": output_correlation(all_a),
        "activation_mean": float(all_a.mean()),
        "activation_std": float(all_a.std()),
    }

    # Softmax-specific
    if all_a.min() >= 0 and torch.allclose(all_a.sum(dim=1), torch.ones(all_a.shape[0], device=all_a.device), atol=1e-5):
        metrics["max_avg_responsibility"] = max_avg_responsibility(all_a)

    return metrics
