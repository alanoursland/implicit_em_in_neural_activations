"""Encoder Study: Decoder-free sparse autoencoders from implicit EM."""

from .model import Encoder, StandardSAE
from .losses import lse_loss, variance_loss, correlation_loss, weight_redundancy_loss, combined_loss
from .metrics import (
    dead_units,
    redundancy_score,
    weight_redundancy,
    effective_rank,
    responsibility_entropy,
    usage_distribution,
    reconstruction_mse,
    sparsity_l0,
)
from .data import get_mnist, get_synthetic, get_llm_activations
from .training import train_epoch, train, evaluate

__all__ = [
    "Encoder",
    "StandardSAE",
    "lse_loss",
    "variance_loss",
    "correlation_loss",
    "weight_redundancy_loss",
    "combined_loss",
    "dead_units",
    "redundancy_score",
    "weight_redundancy",
    "effective_rank",
    "responsibility_entropy",
    "usage_distribution",
    "reconstruction_mse",
    "sparsity_l0",
    "get_mnist",
    "get_synthetic",
    "get_llm_activations",
    "train_epoch",
    "train",
    "evaluate",
]
