"""Model definitions for encoder study."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class Encoder(nn.Module):
    """Single layer encoder. No decoder.

    Args:
        input_dim: Size of input (e.g., 784 for MNIST)
        hidden_dim: Number of features/components
        activation: Activation function ('relu', 'identity', 'softplus')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_name = activation

        self.linear = nn.Linear(input_dim, hidden_dim)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "identity":
            self.activation = nn.Identity()
        elif activation == "softplus":
            self.activation = nn.Softplus()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            a: Activations after activation function (batch, hidden_dim)
            z: Pre-activations before activation function (batch, hidden_dim)
        """
        z = self.linear(x)
        a = self.activation(z)
        return a, z

    def get_weights(self) -> torch.Tensor:
        """Returns weight matrix W (hidden_dim x input_dim)."""
        return self.linear.weight.data

    @property
    def W(self) -> torch.Tensor:
        """Weight matrix (hidden_dim x input_dim)."""
        return self.linear.weight

    @property
    def b(self) -> torch.Tensor:
        """Bias vector (hidden_dim)."""
        return self.linear.bias


class StandardSAE(nn.Module):
    """Standard sparse autoencoder for comparison.

    Encoder: Linear + ReLU
    Decoder: Linear
    Loss: MSE reconstruction + L1 sparsity

    Args:
        input_dim: Size of input
        hidden_dim: Number of hidden units
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            recon: Reconstruction (batch, input_dim)
            a: Activations (batch, hidden_dim)
        """
        a = self.encoder(x)
        recon = self.decoder(a)
        return recon, a

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to activations.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            a: Activations (batch, hidden_dim)
        """
        return self.encoder(x)

    def get_encoder_weights(self) -> torch.Tensor:
        """Returns encoder weight matrix."""
        return self.encoder[0].weight.data

    def get_decoder_weights(self) -> torch.Tensor:
        """Returns decoder weight matrix."""
        return self.decoder.weight.data

    def parameter_count(self) -> int:
        """Returns total number of parameters."""
        return sum(p.numel() for p in self.parameters())
