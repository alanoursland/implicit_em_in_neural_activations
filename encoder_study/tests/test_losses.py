"""Tests for losses.py"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.losses import (
    lse_loss,
    variance_loss,
    correlation_loss,
    weight_redundancy_loss,
    combined_loss,
    sae_loss,
)


class TestLSELoss:
    """Tests for lse_loss function."""

    def test_lse_loss_returns_tuple(self):
        """Test that lse_loss returns loss and responsibilities."""
        a = torch.randn(32, 64)
        loss, r = lse_loss(a)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(r, torch.Tensor)

    def test_lse_loss_scalar(self):
        """Test that loss is a scalar."""
        a = torch.randn(32, 64)
        loss, _ = lse_loss(a)

        assert loss.dim() == 0, "Loss should be scalar"

    def test_responsibilities_shape(self):
        """Test responsibilities have correct shape."""
        batch_size, hidden_dim = 32, 64
        a = torch.randn(batch_size, hidden_dim)
        _, r = lse_loss(a)

        assert r.shape == (batch_size, hidden_dim)

    def test_responsibilities_sum_to_one(self):
        """Test that responsibilities sum to 1 per sample."""
        a = torch.randn(32, 64)
        _, r = lse_loss(a)

        sums = r.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_responsibilities_nonnegative(self):
        """Test that responsibilities are non-negative."""
        a = torch.randn(32, 64)
        _, r = lse_loss(a)

        assert (r >= 0).all(), "Responsibilities should be non-negative"

    def test_gradient_equals_responsibility(self):
        """Test the key theorem: gradient of LSE loss equals responsibility."""
        torch.manual_seed(42)
        a = torch.randn(16, 32, requires_grad=True)

        loss, r = lse_loss(a)
        loss.backward()

        # Gradient should equal responsibility (scaled by 1/batch_size due to mean reduction)
        assert torch.allclose(a.grad * a.shape[0], r, atol=1e-5), \
            "Gradient should equal responsibility (key theorem)"

    def test_lse_loss_gradient_theorem_identity_activation(self):
        """Verify theorem with identity activation (z = a)."""
        torch.manual_seed(123)

        # Create random pre-activations
        z = torch.randn(64, 128, requires_grad=True)

        # With identity activation, a = z
        loss, r = lse_loss(z)
        loss.backward()

        # Verify gradient = responsibility
        correlation = torch.corrcoef(torch.stack([z.grad.flatten(), r.flatten()]))[0, 1]
        assert correlation > 0.9999, f"Correlation should be ~1, got {correlation}"


class TestVarianceLoss:
    """Tests for variance_loss function."""

    def test_variance_loss_scalar(self):
        """Test that variance loss is a scalar."""
        a = torch.randn(32, 64)
        loss = variance_loss(a)

        assert loss.dim() == 0, "Loss should be scalar"

    def test_variance_loss_low_variance(self):
        """Test that low variance gives high loss."""
        # Create data with very low variance
        a_low = torch.ones(32, 64) + 0.001 * torch.randn(32, 64)
        # Create data with high variance
        a_high = torch.randn(32, 64) * 10

        loss_low = variance_loss(a_low)
        loss_high = variance_loss(a_high)

        assert loss_low > loss_high, "Low variance should give higher loss"

    def test_variance_loss_dead_units(self):
        """Test variance loss penalizes dead (constant) units."""
        # Create data where some units are constant (dead)
        a = torch.randn(32, 64)
        a[:, 0] = 0.0  # First unit is constant (dead)

        loss_with_dead = variance_loss(a, eps=1e-8)

        # Loss should be high (negative log of near-zero variance)
        # Actually, with eps, it won't be infinite, but should be large
        a_no_dead = torch.randn(32, 64)
        loss_no_dead = variance_loss(a_no_dead)

        # With dead unit, loss should be higher
        assert loss_with_dead > loss_no_dead

    def test_variance_loss_gradients(self):
        """Test that gradients flow through variance loss."""
        a = torch.randn(32, 64, requires_grad=True)
        loss = variance_loss(a)
        loss.backward()

        assert a.grad is not None


class TestCorrelationLoss:
    """Tests for correlation_loss function."""

    def test_correlation_loss_scalar(self):
        """Test that correlation loss is a scalar."""
        a = torch.randn(32, 64)
        loss = correlation_loss(a)

        assert loss.dim() == 0, "Loss should be scalar"

    def test_correlation_loss_uncorrelated(self):
        """Test that uncorrelated data gives low loss."""
        # Create approximately uncorrelated data
        torch.manual_seed(42)
        a = torch.randn(1000, 10)  # Large batch for stable statistics

        loss = correlation_loss(a)

        # Should be relatively low for random data
        # Expected: ~0 for truly independent, but some correlation exists
        assert loss < 1.0, "Random data should have low correlation loss"

    def test_correlation_loss_highly_correlated(self):
        """Test that correlated data gives high loss."""
        # Create highly correlated data (all columns similar)
        base = torch.randn(100, 1)
        a = base.expand(100, 10) + 0.01 * torch.randn(100, 10)

        loss = correlation_loss(a)

        # Should be high due to correlation
        assert loss > 10.0, "Highly correlated data should have high loss"

    def test_correlation_loss_gradients(self):
        """Test that gradients flow through correlation loss."""
        a = torch.randn(32, 64, requires_grad=True)
        loss = correlation_loss(a)
        loss.backward()

        assert a.grad is not None

    def test_correlation_loss_nonnegative(self):
        """Test that correlation loss is non-negative."""
        a = torch.randn(32, 64)
        loss = correlation_loss(a)

        assert loss >= 0, "Correlation loss should be non-negative"


class TestWeightRedundancyLoss:
    """Tests for weight_redundancy_loss function."""

    def test_weight_redundancy_scalar(self):
        """Test that weight redundancy loss is a scalar."""
        W = torch.randn(64, 784)
        loss = weight_redundancy_loss(W)

        assert loss.dim() == 0, "Loss should be scalar"

    def test_weight_redundancy_orthogonal(self):
        """Test that orthogonal weights give low loss."""
        # Create orthogonal weight matrix
        W = torch.eye(64)[:, :64]  # 64x64 identity
        loss = weight_redundancy_loss(W)

        # Should be approximately 0 for orthogonal weights
        assert loss < 0.1, "Orthogonal weights should have low redundancy"

    def test_weight_redundancy_redundant(self):
        """Test that redundant (similar) weights give high loss."""
        # Create redundant weights (all rows similar)
        base = torch.randn(1, 100)
        W = base.expand(20, 100) + 0.01 * torch.randn(20, 100)

        loss = weight_redundancy_loss(W)

        # Should be high due to redundancy
        assert loss > 100, "Redundant weights should have high loss"

    def test_weight_redundancy_nonnegative(self):
        """Test that weight redundancy loss is non-negative."""
        W = torch.randn(64, 784)
        loss = weight_redundancy_loss(W)

        assert loss >= 0, "Weight redundancy loss should be non-negative"


class TestCombinedLoss:
    """Tests for combined_loss function."""

    def test_combined_loss_returns_dict(self):
        """Test that combined_loss returns a dictionary."""
        a = torch.randn(32, 64)
        W = torch.randn(64, 784)
        config = {"lambda_lse": 1.0, "lambda_var": 1.0, "lambda_tc": 1.0}

        result = combined_loss(a, W, config)

        assert isinstance(result, dict)
        assert "total" in result
        assert "lse" in result
        assert "var" in result
        assert "tc" in result
        assert "wr" in result
        assert "responsibilities" in result

    def test_combined_loss_respects_lambdas(self):
        """Test that lambda weights are applied correctly."""
        a = torch.randn(32, 64)
        W = torch.randn(64, 784)

        # Only LSE
        config_lse = {"lambda_lse": 1.0, "lambda_var": 0.0, "lambda_tc": 0.0, "lambda_wr": 0.0}
        result_lse = combined_loss(a, W, config_lse)

        # Only variance
        config_var = {"lambda_lse": 0.0, "lambda_var": 1.0, "lambda_tc": 0.0, "lambda_wr": 0.0}
        result_var = combined_loss(a, W, config_var)

        # Total should equal individual component when only one is active
        # (approximately, due to floating point)
        assert torch.allclose(result_lse["total"], result_lse["lse"], atol=1e-5)

    def test_combined_loss_gradients(self):
        """Test that gradients flow through combined loss."""
        a = torch.randn(32, 64, requires_grad=True)
        W = torch.randn(64, 784, requires_grad=True)
        config = {"lambda_lse": 1.0, "lambda_var": 1.0, "lambda_tc": 1.0, "lambda_wr": 1.0}

        result = combined_loss(a, W, config)
        result["total"].backward()

        assert a.grad is not None
        assert W.grad is not None


class TestSAELoss:
    """Tests for sae_loss function."""

    def test_sae_loss_returns_dict(self):
        """Test that sae_loss returns a dictionary."""
        x = torch.randn(32, 784)
        recon = torch.randn(32, 784)
        a = torch.randn(32, 64)

        result = sae_loss(x, recon, a)

        assert isinstance(result, dict)
        assert "total" in result
        assert "mse" in result
        assert "l1" in result

    def test_sae_loss_perfect_reconstruction(self):
        """Test that perfect reconstruction gives zero MSE."""
        x = torch.randn(32, 784)
        recon = x.clone()
        a = torch.randn(32, 64)

        result = sae_loss(x, recon, a)

        assert result["mse"] < 1e-10, "Perfect reconstruction should have zero MSE"

    def test_sae_loss_l1_weight(self):
        """Test that L1 weight affects total loss."""
        x = torch.randn(32, 784)
        recon = torch.randn(32, 784)
        a = torch.abs(torch.randn(32, 64))  # Non-negative activations

        result_low = sae_loss(x, recon, a, l1_weight=0.001)
        result_high = sae_loss(x, recon, a, l1_weight=1.0)

        assert result_high["total"] > result_low["total"]

    def test_sae_loss_gradients(self):
        """Test that gradients flow through SAE loss."""
        x = torch.randn(32, 784)
        recon = torch.randn(32, 784, requires_grad=True)
        a = torch.randn(32, 64, requires_grad=True)

        result = sae_loss(x, recon, a)
        result["total"].backward()

        assert recon.grad is not None
        assert a.grad is not None
