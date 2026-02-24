"""Metrics for supervised implicit EM study."""

import torch


def min_variance(a: torch.Tensor) -> float:
    """Minimum per-component variance across the batch.

    Args:
        a: Activations (batch, hidden_dim)

    Returns:
        Minimum variance across components
    """
    return float(a.var(dim=0).min().item())


def redundancy_score(a: torch.Tensor) -> float:
    """||Corr(A) - I||^2_F (off-diagonal only).

    Args:
        a: Activations (batch, hidden_dim)

    Returns:
        Redundancy score
    """
    batch_size, hidden_dim = a.shape
    a_centered = a - a.mean(dim=0, keepdim=True)
    cov = (a_centered.T @ a_centered) / (batch_size - 1)
    std = a.std(dim=0) + 1e-8
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))

    identity = torch.eye(hidden_dim, device=a.device)
    off_diag = corr - identity
    mask = 1.0 - identity
    return float((off_diag * mask).pow(2).sum().item())


def responsibility_entropy(r: torch.Tensor) -> float:
    """Mean entropy H(r) per sample.

    Args:
        r: Responsibilities (batch, hidden_dim)

    Returns:
        Average entropy per sample
    """
    entropy = -(r * torch.log(r + 1e-8)).sum(dim=1)
    return float(entropy.mean().item())
