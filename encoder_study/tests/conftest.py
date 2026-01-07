"""Pytest configuration and fixtures."""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def device():
    """Return the available device (CPU for tests)."""
    return torch.device("cpu")


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def small_encoder():
    """Create a small encoder for testing."""
    from src.model import Encoder
    return Encoder(input_dim=100, hidden_dim=20, activation="relu")


@pytest.fixture
def small_sae():
    """Create a small SAE for testing."""
    from src.model import StandardSAE
    return StandardSAE(input_dim=100, hidden_dim=20)


@pytest.fixture
def synthetic_batch():
    """Create a synthetic batch of data."""
    torch.manual_seed(42)
    return torch.randn(32, 100)


@pytest.fixture
def synthetic_activations():
    """Create synthetic activations."""
    torch.manual_seed(42)
    return torch.randn(32, 20)


@pytest.fixture
def loss_config():
    """Return a default loss configuration."""
    return {
        "lambda_lse": 1.0,
        "lambda_var": 1.0,
        "lambda_tc": 1.0,
        "lambda_wr": 0.0,
        "variance_eps": 1e-6,
    }
