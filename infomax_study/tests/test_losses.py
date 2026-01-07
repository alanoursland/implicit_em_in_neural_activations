import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.losses import batch_variance, batch_correlation, infomax_loss

def test_batch_variance_shape():
    a = torch.randn(32, 16)
    var = batch_variance(a)
    assert var.shape == (16,)

def test_batch_variance_positive():
    a = torch.randn(32, 16)
    var = batch_variance(a)
    assert (var > 0).all()

def test_batch_correlation_shape():
    a = torch.randn(32, 16)
    corr = batch_correlation(a)
    assert corr.shape == (16, 16)

def test_batch_correlation_diagonal():
    a = torch.randn(32, 16)
    corr = batch_correlation(a)
    diag = torch.diag(corr)
    assert torch.allclose(diag, torch.ones(16), atol=1e-5)

def test_infomax_loss_keys():
    a = torch.randn(32, 16)
    losses = infomax_loss(a)
    assert "total" in losses
    assert "entropy" in losses
    assert "tc" in losses

def test_infomax_loss_differentiable():
    a = torch.randn(32, 16, requires_grad=True)
    losses = infomax_loss(a)
    losses["total"].backward()
    assert a.grad is not None
