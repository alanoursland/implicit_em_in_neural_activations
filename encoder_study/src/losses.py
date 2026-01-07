"""Loss functions for encoder study."""

import torch
from typing import Tuple, Dict, Any


def lse_loss(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Log-sum-exp loss with responsibilities.

    Computes -log sum_j exp(-a_j) and returns responsibilities r_j = softmax(-a).

    Args:
        a: Activations of shape (batch, hidden_dim)

    Returns:
        loss: Scalar loss (mean over batch)
        responsibilities: Tensor of shape (batch, hidden_dim) with r_j = softmax(-a_j)
    """
    # LSE loss: -logsumexp(-a) so that gradient = softmax(-a) = responsibility
    # This ensures the key theorem: ∂loss/∂a = r
    neg_a = -a
    lse = torch.logsumexp(neg_a, dim=1)  # (batch,)
    loss = -lse.sum()  # Negative sign ensures gradient = responsibility

    # Responsibilities: r_j = exp(-a_j) / sum_k exp(-a_k) = softmax(-a)
    responsibilities = torch.softmax(neg_a, dim=1)  # (batch, hidden_dim)

    return loss, responsibilities


def variance_loss(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Variance regularization loss.

    Computes -sum_j log(Var(a_j) + eps) over batch.
    Encourages each unit to have non-zero variance (prevents dead units).

    Args:
        a: Activations of shape (batch, hidden_dim)
        eps: Small constant for numerical stability

    Returns:
        loss: Scalar loss
    """
    # Variance of each unit across the batch
    var = a.var(dim=0)  # (hidden_dim,)

    # Negative log variance (encourages high variance)
    loss = -torch.log(var + eps).sum()

    return loss


def correlation_loss(a: torch.Tensor) -> torch.Tensor:
    """Correlation regularization loss (Total Correlation proxy).

    Computes ||Corr(A) - I||^2 (off-diagonal elements only).
    Encourages decorrelated activations.

    Args:
        a: Activations of shape (batch, hidden_dim)

    Returns:
        loss: Scalar loss
    """
    batch_size, hidden_dim = a.shape

    # Center activations
    a_centered = a - a.mean(dim=0, keepdim=True)

    # Compute covariance
    cov = (a_centered.T @ a_centered) / (batch_size - 1)  # (hidden_dim, hidden_dim)

    # Compute standard deviations
    std = a.std(dim=0) + 1e-8  # (hidden_dim,)

    # Correlation matrix
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))

    # Off-diagonal penalty: ||Corr - I||^2 for off-diagonal elements
    identity = torch.eye(hidden_dim, device=a.device)
    off_diag = corr - identity

    # Only penalize off-diagonal elements
    mask = 1.0 - identity
    loss = (off_diag * mask).pow(2).sum()

    return loss


def weight_redundancy_loss(W: torch.Tensor) -> torch.Tensor:
    """Weight redundancy regularization.

    Computes ||W_norm^T @ W_norm - I||^2 where W_norm has normalized rows.
    Encourages orthogonal weight vectors.

    Args:
        W: Weight matrix of shape (hidden_dim, input_dim)

    Returns:
        loss: Scalar loss
    """
    # Normalize rows
    W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)

    # Gram matrix of normalized weights
    gram = W_norm @ W_norm.T  # (hidden_dim, hidden_dim)

    # Penalty: ||Gram - I||^2
    identity = torch.eye(W.shape[0], device=W.device)
    loss = (gram - identity).pow(2).sum()

    return loss


def combined_loss(
    a: torch.Tensor,
    W: torch.Tensor,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Combined loss with configurable components.

    Args:
        a: Activations of shape (batch, hidden_dim)
        W: Weight matrix of shape (hidden_dim, input_dim)
        config: Dict with keys:
            - lambda_lse: Weight for LSE loss
            - lambda_var: Weight for variance loss
            - lambda_tc: Weight for correlation loss
            - lambda_wr: Weight for weight redundancy loss
            - variance_eps: Epsilon for variance loss

    Returns:
        Dict with keys:
            - total: Combined loss for backprop
            - lse: LSE component (scalar)
            - var: Variance component (scalar)
            - tc: Correlation component (scalar)
            - wr: Weight redundancy component (scalar)
            - responsibilities: For analysis (tensor)
    """
    lambda_lse = config.get("lambda_lse", 1.0)
    lambda_var = config.get("lambda_var", 1.0)
    lambda_tc = config.get("lambda_tc", 1.0)
    lambda_wr = config.get("lambda_wr", 0.0)
    variance_eps = config.get("variance_eps", 1e-6)

    # Compute individual losses
    lse, responsibilities = lse_loss(a)
    var = variance_loss(a, eps=variance_eps)
    tc = correlation_loss(a)
    wr = weight_redundancy_loss(W)

    # Combine
    total = (
        lambda_lse * lse
        + lambda_var * var
        + lambda_tc * tc
        + lambda_wr * wr
    )

    return {
        "total": total,
        "lse": lse.detach(),
        "var": var.detach(),
        "tc": tc.detach(),
        "wr": wr.detach(),
        "responsibilities": responsibilities.detach(),
    }


def sae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    a: torch.Tensor,
    l1_weight: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Standard SAE loss: MSE reconstruction + L1 sparsity.

    Args:
        x: Original input (batch, input_dim)
        recon: Reconstruction (batch, input_dim)
        a: Activations (batch, hidden_dim)
        l1_weight: Weight for L1 sparsity penalty

    Returns:
        Dict with keys:
            - total: Combined loss
            - mse: Reconstruction MSE
            - l1: L1 sparsity term
    """
    mse = torch.nn.functional.mse_loss(recon, x)
    l1 = a.abs().mean()
    total = mse + l1_weight * l1

    return {
        "total": total,
        "mse": mse.detach(),
        "l1": l1.detach(),
    }
