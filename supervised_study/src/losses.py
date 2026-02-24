"""Loss functions for supervised implicit EM study."""

import torch
from typing import Tuple, Dict, Any


def lse_loss(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Log-sum-exp loss with responsibilities.

    Loss is -logsumexp(-a).sum(). Responsibilities are softmax(-a).

    Args:
        a: Activations (batch, hidden_dim)

    Returns:
        loss: Scalar loss (sum over batch)
        responsibilities: (batch, hidden_dim)
    """
    neg_a = -a
    lse = torch.logsumexp(neg_a, dim=1)  # (batch,)
    loss = -lse.sum()
    responsibilities = torch.softmax(neg_a, dim=1)
    return loss, responsibilities


def variance_loss(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Variance regularization: -sum_j log(Var(a_j) + eps).

    Prevents dead units by encouraging non-zero variance per feature.

    Args:
        a: Activations (batch, hidden_dim)
        eps: Numerical stability constant

    Returns:
        Scalar loss
    """
    var = a.var(dim=0)  # (hidden_dim,)
    return -torch.log(var + eps).sum()


def correlation_loss(a: torch.Tensor) -> torch.Tensor:
    """Total correlation proxy: ||Corr(A) - I||^2_F (off-diagonal).

    Encourages decorrelated activations.

    Args:
        a: Activations (batch, hidden_dim)

    Returns:
        Scalar loss
    """
    batch_size, hidden_dim = a.shape
    a_centered = a - a.mean(dim=0, keepdim=True)
    cov = (a_centered.T @ a_centered) / (batch_size - 1)
    std = a.std(dim=0) + 1e-8
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))

    identity = torch.eye(hidden_dim, device=a.device)
    off_diag = corr - identity
    mask = 1.0 - identity
    return (off_diag * mask).pow(2).sum()


def weight_redundancy_loss(W: torch.Tensor) -> torch.Tensor:
    """Weight redundancy: ||W_norm^T W_norm - I||^2.

    Args:
        W: Weight matrix (hidden_dim, input_dim)

    Returns:
        Scalar loss
    """
    W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    gram = W_norm @ W_norm.T
    identity = torch.eye(W.shape[0], device=W.device)
    return (gram - identity).pow(2).sum()


def combined_loss(
    a: torch.Tensor,
    W: torch.Tensor,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Combined auxiliary loss with configurable components.

    Args:
        a: Activations (batch, hidden_dim)
        W: Weight matrix (hidden_dim, input_dim)
        config: Dict with lambda_lse, lambda_var, lambda_tc, lambda_wr, variance_eps

    Returns:
        Dict with total, lse, var, tc, wr (scalars), and responsibilities (tensor)
    """
    lambda_lse = config.get("lambda_lse", 1.0)
    lambda_var = config.get("lambda_var", 1.0)
    lambda_tc = config.get("lambda_tc", 1.0)
    lambda_wr = config.get("lambda_wr", 0.0)
    variance_eps = config.get("variance_eps", 1e-6)

    lse, responsibilities = lse_loss(a)
    var = variance_loss(a, eps=variance_eps)
    tc = correlation_loss(a)
    wr = weight_redundancy_loss(W)

    total = lambda_lse * lse + lambda_var * var + lambda_tc * tc + lambda_wr * wr

    return {
        "total": total,
        "lse": lse.detach(),
        "var": var.detach(),
        "tc": tc.detach(),
        "wr": wr.detach(),
        "responsibilities": responsibilities.detach(),
    }
