import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activations import (
    get_activation,
    Identity,
    ReLU,
    LeakyReLU,
    Softmax,
    Tanh,
    Softplus
)

def test_identity():
    act = Identity()
    x = torch.randn(10, 5)
    assert torch.allclose(act(x), x)

def test_relu():
    act = ReLU()
    x = torch.randn(10, 5)
    y = act(x)
    assert (y >= 0).all()

def test_softmax():
    act = Softmax()
    x = torch.randn(10, 5)
    y = act(x)
    assert torch.allclose(y.sum(dim=-1), torch.ones(10))
    assert (y >= 0).all()

def test_tanh():
    act = Tanh()
    x = torch.randn(10, 5)
    y = act(x)
    assert (y >= -1).all() and (y <= 1).all()

def test_softplus():
    act = Softplus()
    x = torch.randn(10, 5)
    y = act(x)
    assert (y >= 0).all()

def test_leaky_relu():
    act = LeakyReLU(negative_slope=0.1)
    x = torch.tensor([[-1.0, 0.0, 1.0]])
    y = act(x)
    assert y[0, 0] == -0.1
    assert y[0, 1] == 0.0
    assert y[0, 2] == 1.0

def test_get_activation():
    act = get_activation("relu")
    assert isinstance(act, ReLU)

    act = get_activation("tanh")
    assert isinstance(act, Tanh)

def test_get_activation_invalid():
    with pytest.raises(ValueError):
        get_activation("invalid_activation")
