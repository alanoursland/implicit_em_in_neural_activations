import torch

# Cache for identity matrices (K -> identity matrix)
_identity_cache = {}

def get_identity(K: int, device: torch.device) -> torch.Tensor:
    """Get cached identity matrix for given size and device."""
    key = (K, device)
    if key not in _identity_cache:
        _identity_cache[key] = torch.eye(K, device=device)
    return _identity_cache[key]

def batch_variance(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute variance of each column (output dimension) across batch.

    Args:
        a: (N, K) activations
        eps: small constant for numerical stability

    Returns:
        (K,) variances
    """
    return a.var(dim=0) + eps

def batch_correlation(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute correlation matrix of outputs across batch.

    Args:
        a: (N, K) activations

    Returns:
        (K, K) correlation matrix
    """
    # Center
    a_centered = a - a.mean(dim=0, keepdim=True)

    # Standardize
    std = a.std(dim=0, keepdim=True) + eps
    a_standardized = a_centered / std

    # Correlation
    N = a.shape[0]
    corr = (a_standardized.T @ a_standardized) / (N - 1)

    return corr

def weight_redundancy(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Measure redundancy in weight matrix via normalized Gram matrix.

    ||G - I||² where G is the Gram matrix of normalized weight rows.

    Args:
        W: (K, n) weight matrix - K units, n input dimensions
        eps: numerical stability for normalization

    Returns:
        scalar loss - lower means weight vectors point in more diverse directions
    """
    K = W.shape[0]

    # Normalize rows to unit length
    norms = W.norm(dim=1, keepdim=True) + eps
    W_norm = W / norms

    # Gram matrix: G[i,j] = cos(angle between row i and row j)
    G = W_norm @ W_norm.T  # (K, K)

    # Compare to identity (diagonal = 1, off-diagonal = 0)
    identity = get_identity(K, W.device)

    return ((G - identity) ** 2).sum()

def infomax_loss(a: torch.Tensor, W: torch.Tensor = None, lambda_tc: float = 1.0, lambda_wr: float = 0.0, eps: float = 1e-6) -> dict:
    """
    Compute InfoMax loss: -Σ log(Var(aⱼ)) + λ_tc ||Corr(A) - I||² + λ_wr ||G - I||²

    Optimized version that caches identity matrix and combines operations.

    Args:
        a: (N, K) activations
        W: (K, n) weight matrix (optional, required if lambda_wr > 0)
        lambda_tc: weight on total correlation term
        lambda_wr: weight on weight redundancy term
        eps: numerical stability

    Returns:
        dict with 'total', 'entropy', 'tc', 'wr' losses
    """
    N, K = a.shape

    # Center once for both variance and correlation
    a_centered = a - a.mean(dim=0, keepdim=True)

    # Variance for entropy term
    var = a_centered.var(dim=0) + eps
    entropy_loss = -torch.log(var).sum()

    # TC term: compute correlation and distance from identity
    # Standardize centered activations
    std = torch.sqrt(var)
    a_standardized = a_centered / std.unsqueeze(0)

    # Correlation matrix
    corr = (a_standardized.T @ a_standardized) / (N - 1)

    # Use cached identity matrix
    identity = get_identity(K, a.device)
    tc_loss = ((corr - identity) ** 2).sum()

    # Weight redundancy term
    if lambda_wr > 0 and W is not None:
        wr_loss = weight_redundancy(W, eps=eps)
    else:
        wr_loss = torch.tensor(0.0, device=a.device)

    total_loss = entropy_loss + lambda_tc * tc_loss + lambda_wr * wr_loss

    return {
        "total": total_loss,
        "entropy": entropy_loss,
        "tc": tc_loss,
        "wr": wr_loss,
    }
