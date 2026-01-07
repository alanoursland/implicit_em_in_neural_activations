"""Metrics for encoder study."""

import torch
import numpy as np
from typing import Tuple, Union


def dead_units(a: torch.Tensor, threshold: float = 0.01) -> int:
    """Count units with variance below threshold.

    Args:
        a: Activations of shape (batch, hidden_dim)
        threshold: Variance threshold below which a unit is considered dead

    Returns:
        Number of dead units
    """
    var = a.var(dim=0)  # (hidden_dim,)
    return int((var < threshold).sum().item())


def redundancy_score(a: torch.Tensor) -> float:
    """Compute ||Corr(A) - I||^2 (off-diagonal only).

    Measures how correlated the activations are.
    Lower is better (more independent features).

    Args:
        a: Activations of shape (batch, hidden_dim)

    Returns:
        Redundancy score (float)
    """
    batch_size, hidden_dim = a.shape

    # Center activations
    a_centered = a - a.mean(dim=0, keepdim=True)

    # Compute covariance
    cov = (a_centered.T @ a_centered) / (batch_size - 1)

    # Compute standard deviations
    std = a.std(dim=0) + 1e-8

    # Correlation matrix
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))

    # Off-diagonal penalty
    identity = torch.eye(hidden_dim, device=a.device)
    off_diag = corr - identity
    mask = 1.0 - identity

    return float((off_diag * mask).pow(2).sum().item())


def weight_redundancy(W: torch.Tensor) -> float:
    """Compute ||W_norm^T @ W_norm - I||^2 on normalized rows.

    Measures how redundant the weight vectors are.
    Lower is better (more orthogonal features).

    Args:
        W: Weight matrix of shape (hidden_dim, input_dim)

    Returns:
        Weight redundancy score (float)
    """
    # Normalize rows
    W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)

    # Gram matrix
    gram = W_norm @ W_norm.T

    # Penalty
    identity = torch.eye(W.shape[0], device=W.device)
    return float((gram - identity).pow(2).sum().item())


def effective_rank(W: torch.Tensor) -> float:
    """Compute effective dimensionality of weight matrix.

    Uses the entropy-based definition: exp(H(p)) where p_i = s_i / sum(s)
    and s_i are singular values.

    Args:
        W: Weight matrix of shape (hidden_dim, input_dim)

    Returns:
        Effective rank (float)
    """
    # Compute singular values
    s = torch.linalg.svdvals(W)

    # Normalize to probability distribution
    s_norm = s / (s.sum() + 1e-8)

    # Compute entropy
    entropy = -(s_norm * torch.log(s_norm + 1e-8)).sum()

    # Effective rank = exp(entropy)
    return float(torch.exp(entropy).item())


def responsibility_entropy(r: torch.Tensor) -> float:
    """Compute average entropy H(r) per sample.

    Measures competition sharpness. Lower entropy means sharper competition
    (one component clearly wins).

    Args:
        r: Responsibilities of shape (batch, hidden_dim)

    Returns:
        Average entropy per sample (float)
    """
    # Entropy per sample: -sum_j r_j log(r_j)
    entropy = -(r * torch.log(r + 1e-8)).sum(dim=1)  # (batch,)
    return float(entropy.mean().item())


def usage_distribution(r: torch.Tensor) -> np.ndarray:
    """Compute E_x[r(x)] - average responsibility per component.

    Shows which components are used most often.

    Args:
        r: Responsibilities of shape (batch, hidden_dim)

    Returns:
        Array of shape (hidden_dim,) with average usage per component
    """
    return r.mean(dim=0).cpu().numpy()


def reconstruction_mse(
    x: torch.Tensor,
    a: torch.Tensor,
    W: torch.Tensor,
) -> float:
    """Compute ||x - W^T a||^2 using transposed weights as decoder.

    Args:
        x: Original input (batch, input_dim)
        a: Activations (batch, hidden_dim)
        W: Weight matrix (hidden_dim, input_dim)

    Returns:
        Mean squared error (float)
    """
    # Reconstruction using transposed weights
    recon = a @ W  # (batch, input_dim)
    mse = ((x - recon) ** 2).mean()
    return float(mse.item())


def sparsity_l0(
    a: torch.Tensor, threshold: float = 0.01
) -> Tuple[float, float]:
    """Compute average L0 sparsity (number of active units per sample).

    Args:
        a: Activations of shape (batch, hidden_dim)
        threshold: Activation threshold below which a unit is considered inactive

    Returns:
        Tuple of (l0, density):
            - l0: Average number of active units per sample
            - density: Fraction of features active (l0 / hidden_dim)
    """
    hidden_dim = a.shape[1]
    active = (a.abs() > threshold).float()
    l0 = float(active.sum(dim=1).mean().item())
    density = l0 / hidden_dim
    return l0, density


def feature_similarity_matrix(W: torch.Tensor) -> np.ndarray:
    """Compute cosine similarity matrix between weight vectors.

    Args:
        W: Weight matrix of shape (hidden_dim, input_dim)

    Returns:
        Similarity matrix of shape (hidden_dim, hidden_dim)
    """
    W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    sim = (W_norm @ W_norm.T).cpu().numpy()
    return sim


def activation_correlation_matrix(a: torch.Tensor) -> np.ndarray:
    """Compute correlation matrix of activations.

    Args:
        a: Activations of shape (batch, hidden_dim)

    Returns:
        Correlation matrix of shape (hidden_dim, hidden_dim)
    """
    batch_size, hidden_dim = a.shape

    # Center activations
    a_centered = a - a.mean(dim=0, keepdim=True)

    # Compute covariance
    cov = (a_centered.T @ a_centered) / (batch_size - 1)

    # Compute standard deviations
    std = a.std(dim=0) + 1e-8

    # Correlation matrix
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))

    return corr.cpu().numpy()


def compute_all_metrics(
    a: torch.Tensor,
    W: torch.Tensor,
    x: torch.Tensor,
    r: torch.Tensor,
) -> dict:
    """Compute all metrics for a batch.

    Args:
        a: Activations (batch, hidden_dim)
        W: Weight matrix (hidden_dim, input_dim)
        x: Original input (batch, input_dim)
        r: Responsibilities (batch, hidden_dim)

    Returns:
        Dict with all metrics
    """
    return {
        "dead_units": dead_units(a),
        "redundancy_score": redundancy_score(a),
        "weight_redundancy": weight_redundancy(W),
        "effective_rank": effective_rank(W),
        "responsibility_entropy": responsibility_entropy(r),
        "reconstruction_mse": reconstruction_mse(x, a, W),
        "sparsity_l0": sparsity_l0(a),
    }


def linear_probe_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    """Train linear classifier on features and return test accuracy.

    Args:
        train_features: Training features (n_train, hidden_dim)
        train_labels: Training labels (n_train,)
        test_features: Test features (n_test, hidden_dim)
        test_labels: Test labels (n_test,)

    Returns:
        Test accuracy (float between 0 and 1)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Scale features (activations can have large magnitude)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    clf = LogisticRegression(max_iter=5000, solver="lbfgs", multi_class="multinomial")
    clf.fit(train_features, train_labels)
    accuracy = clf.score(test_features, test_labels)
    return float(accuracy)


def normalized_reconstruction_mse(
    x: torch.Tensor,
    a: torch.Tensor,
    W: torch.Tensor,
) -> float:
    """Reconstruction MSE with normalized activations.

    Normalizes activations to unit norm before reconstruction, then scales
    to match input norm. This handles scale differences between models.

    Args:
        x: Original input (batch, input_dim)
        a: Activations (batch, hidden_dim)
        W: Weight matrix (hidden_dim, input_dim)

    Returns:
        Normalized mean squared error (float)
    """
    # Normalize activations to unit norm per sample
    a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)

    # Reconstruct
    x_hat = a_norm @ W  # (batch, input_dim)

    # Scale to match input norm
    x_hat = x_hat * (x.norm(dim=1, keepdim=True) / (x_hat.norm(dim=1, keepdim=True) + 1e-8))

    # MSE
    mse = ((x - x_hat) ** 2).mean()
    return float(mse.item())
