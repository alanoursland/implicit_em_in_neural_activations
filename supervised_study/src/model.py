"""Models for supervised implicit EM study."""

import torch
import torch.nn as nn
from typing import Dict, Any

from losses import variance_loss, correlation_loss


def neg_log_softmin(d: torch.Tensor) -> torch.Tensor:
    """Calibrate distances: y_j = d_j + log Z where Z = sum exp(-d_k).

    Args:
        d: distances (batch, K), non-negative

    Returns:
        y: calibrated distances (batch, K)
    """
    return d + torch.logsumexp(-d, dim=1, keepdim=True)


class ImplicitEMLayer(nn.Module):
    """Single computational unit implementing implicit EM theory.

    Three components, each independently configurable:
      1. Distance computation: Linear + Softplus -> non-negative distances
      2. Calibration: NegLogSoftmin -> probabilistically calibrated distances
      3. Volume control: variance + decorrelation regularization

    Forward pass returns calibrated distances y. Raw distances d are stored
    internally and used by regularization_loss().

    Args:
        input_dim: Input dimensionality
        hidden_dim: Number of components (mixture components / hidden units)
        use_neg_log_softmin: Apply NegLogSoftmin calibration in forward pass
        lambda_var: Weight for variance loss (anti-collapse)
        lambda_tc: Weight for correlation loss (anti-redundancy)
        variance_eps: Epsilon for variance loss numerical stability
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 16,
        use_neg_log_softmin: bool = True,
        lambda_var: float = 1.0,
        lambda_tc: float = 1.0,
        variance_eps: float = 1e-6,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.use_neg_log_softmin = use_neg_log_softmin

        # Volume control weights (stored as plain attributes, not parameters)
        self.lambda_var = lambda_var
        self.lambda_tc = lambda_tc
        self.variance_eps = variance_eps

        # Stored from most recent forward pass (retains computation graph)
        self._d: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Stores raw distances internally for regularization_loss().

        Args:
            x: Input (batch, input_dim)

        Returns:
            y: Output distances (batch, hidden_dim) — calibrated if NLS enabled
        """
        d = self.activation(self.linear(x))
        self._d = d
        if self.use_neg_log_softmin:
            y = neg_log_softmin(d)
        else:
            y = d
        return y

    def regularization_loss(self) -> Dict[str, Any]:
        """Compute volume control regularization on stored distances.

        Must be called after forward(). The stored distances retain their
        computation graph so gradients flow back through the layer.

        Returns:
            Dict with total, var, tc (scalars)
        """
        var = variance_loss(self._d, eps=self.variance_eps)
        tc = correlation_loss(self._d)
        total = self.lambda_var * var + self.lambda_tc * tc
        return {
            "total": total,
            "var": var.detach(),
            "tc": tc.detach(),
        }

    @property
    def distances(self) -> torch.Tensor:
        """Raw distances from the most recent forward pass."""
        return self._d

    @property
    def W(self) -> torch.Tensor:
        """Weight matrix (hidden_dim, input_dim)."""
        return self.linear.weight


class SupervisedModel(nn.Module):
    """Two-layer supervised model with ImplicitEM layer.

    Architecture:
        x -> ImplicitEMLayer -> Linear(K, num_classes) -> LayerNorm -> logits

    forward() returns only class logits. Regularization loss and raw distances
    are accessed via regularization_loss() and distances after calling forward().

    Args:
        input_dim: Input dimensionality (784 for flattened MNIST)
        hidden_dim: Number of hidden units / mixture components
        num_classes: Number of output classes
        use_neg_log_softmin: Whether ImplicitEMLayer applies NegLogSoftmin
        lambda_var: Weight for variance loss
        lambda_tc: Weight for correlation loss
        variance_eps: Epsilon for variance loss
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 16,
        num_classes: int = 10,
        use_neg_log_softmin: bool = False,
        lambda_var: float = 0.0,
        lambda_tc: float = 0.0,
        variance_eps: float = 1e-6,
    ):
        super().__init__()
        self.em_layer = ImplicitEMLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_neg_log_softmin=use_neg_log_softmin,
            lambda_var=lambda_var,
            lambda_tc=lambda_tc,
            variance_eps=variance_eps,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.layer_norm = nn.LayerNorm(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (batch, input_dim)

        Returns:
            h: Class logits (batch, num_classes)
        """
        y = self.em_layer(x)
        h = self.layer_norm(self.classifier(y))
        return h

    def regularization_loss(self) -> Dict[str, Any]:
        """Compute volume control regularization from the most recent forward pass."""
        return self.em_layer.regularization_loss()

    @property
    def distances(self) -> torch.Tensor:
        """Raw distances from the most recent forward pass."""
        return self.em_layer.distances

    @property
    def W1(self) -> torch.Tensor:
        """First layer weight matrix (hidden_dim, input_dim) for visualization."""
        return self.em_layer.W
