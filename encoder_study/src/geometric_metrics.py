"""
Geometric metrics for comparing cosine-like vs Mahalanobis-like learned representations.

These metrics distinguish two learning regimes based on activation sign:
- Cosine regime (ReLU): activation ≈ template similarity, weights look like data
- Mahalanobis regime (-ReLU): activation ≈ distance from boundary, weights are normals

Reference: The LSE loss gradient equals softmax(-a), so:
- ReLU: responsibility → small activations → features with low similarity
- -ReLU: responsibility → small (negative) activations → features with high similarity
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


def corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation coefficient between two 1D tensors."""
    x = x.flatten().float()
    y = y.flatten().float()

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    num = (x_centered * y_centered).sum()
    denom = (x_centered.pow(2).sum() * y_centered.pow(2).sum()).sqrt()

    if denom < 1e-8:
        return 0.0
    return (num / denom).item()


def weight_input_alignment(W: torch.Tensor, X: torch.Tensor, r: torch.Tensor) -> float:
    """
    Responsibility-weighted cosine similarity between weights and inputs.

    Measures: Do high-responsibility inputs align with their assigned weight vector?

    Args:
        W: Weight matrix (K, D) - encoder weights
        X: Input data (N, D)
        r: Responsibility matrix (N, K) from softmax(-a)

    Returns:
        Mean responsibility-weighted cosine similarity

    Expected:
        - Cosine regime: +0.3 to +0.8 (weights point toward responsible inputs)
        - Mahalanobis regime: -0.2 to +0.2 (weights are boundary normals)
    """
    # Normalize weight vectors and inputs
    W_norm = F.normalize(W, dim=1)  # (K, D)
    X_norm = F.normalize(X, dim=1)  # (N, D)

    # Cosine similarity between each input and each weight
    cos_sim = X_norm @ W_norm.T  # (N, K)

    # Weight by responsibility and average
    weighted_cos = (r * cos_sim).sum(dim=1).mean()

    return weighted_cos.item()


def activation_cosine_correlation(W: torch.Tensor, X: torch.Tensor, a: torch.Tensor) -> float:
    """
    Correlation between activation magnitude and cosine similarity.

    For each component k, correlate activation a_k with cos(x, w_k) across samples.

    Args:
        W: Weight matrix (K, D)
        X: Input data (N, D)
        a: Activation matrix (N, K) - pre-nonlinearity

    Returns:
        Mean correlation across components

    Expected:
        - Cosine regime: +0.7 to +0.9 (activation tracks similarity)
        - Mahalanobis regime: -0.3 to +0.3 (activation tracks distance)
    """
    W_norm = F.normalize(W, dim=1)  # (K, D)
    X_norm = F.normalize(X, dim=1)  # (N, D)

    cos_sim = X_norm @ W_norm.T  # (N, K)

    # Compute correlation for each component
    K = W.shape[0]
    correlations = []

    for k in range(K):
        corr = corrcoef(a[:, k], cos_sim[:, k])
        correlations.append(corr)

    return np.mean(correlations)


def bias_data_relationship(W: torch.Tensor, b: torch.Tensor, X: torch.Tensor) -> Dict[str, float]:
    """
    Analyze bias position relative to data projection distribution.

    Args:
        W: Weight matrix (K, D)
        b: Bias vector (K,)
        X: Input data (N, D)

    Returns:
        Dictionary with:
        - frac_bias_below_mean: Fraction of components where bias < mean projection
        - mean_relative_position: Mean of (bias - proj_mean) / proj_std

    Expected:
        - Cosine regime: frac_bias_below_mean < 0.5 (threshold above zero)
        - Mahalanobis regime: frac_bias_below_mean > 0.5 (centered on data)
    """
    # Project data onto weight directions (without bias)
    z_no_bias = X @ W.T  # (N, K)

    proj_mean = z_no_bias.mean(dim=0)  # (K,)
    proj_std = z_no_bias.std(dim=0) + 1e-8  # (K,)

    # What fraction of biases are below the mean projection?
    frac_below = (b < proj_mean).float().mean().item()

    # Relative position: how many std devs is bias from mean?
    relative_pos = ((b - proj_mean) / proj_std).mean().item()

    return {
        "frac_bias_below_mean": frac_below,
        "mean_relative_position": relative_pos
    }


def variance_normalization_index(W: torch.Tensor, X: torch.Tensor) -> float:
    """
    Correlation between weight norm and inverse projection variance.

    In Mahalanobis geometry, weight norms encode precision (inverse variance).

    Args:
        W: Weight matrix (K, D)
        X: Input data (N, D)

    Returns:
        Correlation between ||w_k|| and 1/sqrt(var(x·w_k/||w_k||))

    Expected:
        - Cosine regime: ~0 (norms don't encode variance)
        - Mahalanobis regime: +0.3 to +0.7 (norms = precision)
    """
    w_norms = W.norm(dim=1)  # (K,)

    # Get unit directions
    W_dirs = F.normalize(W, dim=1)  # (K, D)

    # Project data onto each direction
    projections = X @ W_dirs.T  # (N, K)

    # Variance along each direction
    proj_var = projections.var(dim=0)  # (K,)

    # Inverse sqrt variance (precision)
    inv_sqrt_var = 1.0 / (proj_var.sqrt() + 1e-8)

    return corrcoef(w_norms, inv_sqrt_var)


def boundary_placement(W: torch.Tensor, b: torch.Tensor, X: torch.Tensor) -> float:
    """
    Where does the activation=0 boundary fall relative to data?

    Args:
        W: Weight matrix (K, D)
        b: Bias vector (K,)
        X: Input data (N, D)

    Returns:
        Mean fraction of data points with positive pre-activation

    Expected:
        - Cosine regime: < 0.5 (threshold excludes most data)
        - Mahalanobis regime: ~0.5 (boundary at data center)
    """
    z = X @ W.T + b  # (N, K)
    frac_positive = (z > 0).float().mean(dim=0)  # (K,)

    return frac_positive.mean().item()


def responsibility_concentration(r: torch.Tensor) -> Dict[str, float]:
    """
    Entropy and concentration of responsibility distribution.

    Args:
        r: Responsibility matrix (N, K) from softmax(-a)

    Returns:
        Dictionary with:
        - entropy: Mean entropy of responsibility per sample
        - top1_prob: Mean probability of most responsible component
        - effective_k: Mean effective number of components (exp(entropy))

    Expected:
        - Cosine regime: Higher entropy, lower top1 (more distributed)
        - Mahalanobis regime: Lower entropy, higher top1 (more concentrated)
    """
    # Clamp for numerical stability
    r_clamped = r.clamp(min=1e-10)

    # Per-sample entropy
    entropy = -(r_clamped * r_clamped.log()).sum(dim=1)  # (N,)
    mean_entropy = entropy.mean().item()

    # Top-1 probability
    top1 = r.max(dim=1).values.mean().item()

    # Effective number of components
    effective_k = entropy.exp().mean().item()

    return {
        "entropy": mean_entropy,
        "top1_prob": top1,
        "effective_k": effective_k
    }


def template_correlation(W: torch.Tensor, X: torch.Tensor, r: torch.Tensor) -> float:
    """
    Correlation between weights and responsibility-weighted data prototypes.

    For each component, compute the weighted mean of inputs (prototype),
    then correlate with the weight vector.

    Args:
        W: Weight matrix (K, D)
        X: Input data (N, D)
        r: Responsibility matrix (N, K)

    Returns:
        Mean correlation between w_k and prototype_k

    Expected:
        - Cosine regime: High (weights look like data prototypes)
        - Mahalanobis regime: Low (weights are boundary normals)
    """
    K = W.shape[0]
    correlations = []

    for k in range(K):
        # Responsibility-weighted prototype for component k
        weights_k = r[:, k]  # (N,)
        if weights_k.sum() < 1e-8:
            continue

        prototype = (weights_k.unsqueeze(1) * X).sum(dim=0) / weights_k.sum()  # (D,)

        # Correlation with weight vector
        corr = corrcoef(W[k], prototype)
        correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def center_surround_structure(W: torch.Tensor, image_size: int = 28) -> Dict[str, float]:
    """
    Analyze center-surround structure in image weights.

    For MNIST-like images, compute correlation between center and surround regions.
    Templates have positive correlation; boundary detectors have negative.

    Args:
        W: Weight matrix (K, D) where D = image_size^2
        image_size: Side length of square image

    Returns:
        Dictionary with:
        - center_surround_corr: Mean correlation between center and surround
        - center_mean: Mean of center region weights
        - surround_mean: Mean of surround region weights

    Expected:
        - Cosine regime: Positive correlation (template structure)
        - Mahalanobis regime: Negative correlation (edge/boundary structure)
    """
    K, D = W.shape

    if D != image_size * image_size:
        return {"center_surround_corr": 0.0, "center_mean": 0.0, "surround_mean": 0.0}

    # Reshape to images
    W_img = W.view(K, image_size, image_size)

    # Define center region (middle 40% of image)
    margin = int(image_size * 0.3)
    center_slice = slice(margin, image_size - margin)

    correlations = []
    center_means = []
    surround_means = []

    for k in range(K):
        img = W_img[k]

        # Extract center
        center = img[center_slice, center_slice].flatten()

        # Extract surround (all minus center)
        mask = torch.ones_like(img, dtype=torch.bool)
        mask[center_slice, center_slice] = False
        surround = img[mask]

        center_means.append(center.mean().item())
        surround_means.append(surround.mean().item())

        # Correlation between center and surround means across spatial positions
        # Simplified: just compare mean values
        # A template has center and surround with same sign
        # A boundary detector has opposite signs

    # Compute overall correlation
    center_mean = np.mean(center_means)
    surround_mean = np.mean(surround_means)

    # Sign agreement: do center and surround have same sign?
    sign_agreement = np.mean([
        1.0 if (c > 0) == (s > 0) else -1.0
        for c, s in zip(center_means, surround_means)
        if abs(c) > 1e-6 and abs(s) > 1e-6
    ]) if center_means else 0.0

    return {
        "center_surround_corr": sign_agreement,
        "center_mean": center_mean,
        "surround_mean": surround_mean
    }


def compute_all_geometric_metrics(
    W: torch.Tensor,
    b: torch.Tensor,
    X: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    image_size: int = 28
) -> Dict[str, float]:
    """
    Compute all geometric metrics for a trained model.

    Args:
        W: Encoder weight matrix (K, D)
        b: Encoder bias vector (K,)
        X: Input data batch (N, D)
        a: Pre-activation values (N, K) = X @ W.T + b
        r: Responsibilities (N, K) = softmax(-a)
        image_size: For center-surround analysis

    Returns:
        Dictionary of all metrics
    """
    metrics = {}

    # Metric 1: Weight-Input Alignment
    metrics["weight_input_alignment"] = weight_input_alignment(W, X, r)

    # Metric 2: Activation-Cosine Correlation
    metrics["activation_cosine_corr"] = activation_cosine_correlation(W, X, a)

    # Metric 3: Bias Data Relationship
    bias_metrics = bias_data_relationship(W, b, X)
    metrics["frac_bias_below_mean"] = bias_metrics["frac_bias_below_mean"]
    metrics["bias_relative_position"] = bias_metrics["mean_relative_position"]

    # Metric 4: Variance Normalization Index
    metrics["variance_norm_index"] = variance_normalization_index(W, X)

    # Metric 5: Boundary Placement
    metrics["frac_positive_preact"] = boundary_placement(W, b, X)

    # Metric 6: Responsibility Concentration
    resp_metrics = responsibility_concentration(r)
    metrics["responsibility_entropy"] = resp_metrics["entropy"]
    metrics["responsibility_top1"] = resp_metrics["top1_prob"]
    metrics["effective_k"] = resp_metrics["effective_k"]

    # Metric 7: Template Correlation
    metrics["template_correlation"] = template_correlation(W, X, r)

    # Metric 8: Center-Surround Structure
    cs_metrics = center_surround_structure(W, image_size)
    metrics["center_surround_corr"] = cs_metrics["center_surround_corr"]

    return metrics


def print_metric_comparison(results: Dict[str, Dict[str, float]], metric_names: Optional[list] = None):
    """
    Print a comparison table of metrics across conditions.

    Args:
        results: Dict mapping condition name to metrics dict
        metric_names: Optional list of metric names to include
    """
    if metric_names is None:
        # Get all metric names from first result
        first_key = list(results.keys())[0]
        metric_names = list(results[first_key].keys())

    # Header
    conditions = list(results.keys())
    header = f"{'Metric':<30} | " + " | ".join(f"{c:>15}" for c in conditions)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for metric in metric_names:
        row = f"{metric:<30} | "
        values = []
        for cond in conditions:
            val = results[cond].get(metric, float('nan'))
            values.append(f"{val:>15.4f}")
        row += " | ".join(values)
        print(row)

    print("=" * len(header) + "\n")


def get_expected_ranges() -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Return expected metric ranges for each regime.

    Returns:
        Dict mapping metric name to dict of regime -> (min, max) expected range
    """
    return {
        "weight_input_alignment": {
            "cosine": (0.3, 0.8),
            "mahalanobis": (-0.2, 0.2)
        },
        "activation_cosine_corr": {
            "cosine": (0.7, 0.9),
            "mahalanobis": (-0.3, 0.3)
        },
        "frac_bias_below_mean": {
            "cosine": (0.0, 0.5),
            "mahalanobis": (0.5, 1.0)
        },
        "variance_norm_index": {
            "cosine": (-0.2, 0.2),
            "mahalanobis": (0.3, 0.7)
        },
        "frac_positive_preact": {
            "cosine": (0.0, 0.5),
            "mahalanobis": (0.4, 0.6)
        },
        "responsibility_entropy": {
            "cosine": (2.0, 4.0),  # Higher
            "mahalanobis": (0.5, 2.0)  # Lower
        },
        "template_correlation": {
            "cosine": (0.5, 1.0),
            "mahalanobis": (-0.3, 0.5)
        },
        "center_surround_corr": {
            "cosine": (0.0, 1.0),
            "mahalanobis": (-1.0, 0.0)
        }
    }
