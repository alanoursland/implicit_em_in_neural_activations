"""Tests for metrics.py"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    dead_units,
    redundancy_score,
    weight_redundancy,
    effective_rank,
    responsibility_entropy,
    usage_distribution,
    reconstruction_mse,
    sparsity_l0,
    feature_similarity_matrix,
    activation_correlation_matrix,
    compute_all_metrics,
)


class TestDeadUnits:
    """Tests for dead_units function."""

    def test_dead_units_no_dead(self):
        """Test with no dead units."""
        a = torch.randn(100, 10)  # High variance in all units
        count = dead_units(a, threshold=0.01)
        assert count == 0

    def test_dead_units_all_dead(self):
        """Test with all dead units."""
        a = torch.zeros(100, 10)  # Zero variance in all units
        count = dead_units(a, threshold=0.01)
        assert count == 10

    def test_dead_units_some_dead(self):
        """Test with some dead units."""
        a = torch.randn(100, 10)
        a[:, :3] = 0.0  # First 3 units are dead
        count = dead_units(a, threshold=0.01)
        assert count == 3

    def test_dead_units_threshold(self):
        """Test that threshold parameter works."""
        a = torch.randn(100, 10) * 0.05  # Low variance
        a[:, 0] = 0.0  # One dead unit

        # With low threshold, only truly dead unit counts
        count_low = dead_units(a, threshold=0.0001)
        assert count_low == 1

        # With high threshold, more units count as dead
        count_high = dead_units(a, threshold=0.1)
        assert count_high >= 1


class TestRedundancyScore:
    """Tests for redundancy_score function."""

    def test_redundancy_score_nonnegative(self):
        """Test that redundancy score is non-negative."""
        a = torch.randn(100, 10)
        score = redundancy_score(a)
        assert score >= 0

    def test_redundancy_score_uncorrelated(self):
        """Test that uncorrelated data has low redundancy."""
        torch.manual_seed(42)
        a = torch.randn(1000, 10)
        score = redundancy_score(a)
        # Should be relatively low for random data
        assert score < 1.0

    def test_redundancy_score_correlated(self):
        """Test that correlated data has high redundancy."""
        base = torch.randn(100, 1)
        a = base.expand(100, 10) + 0.01 * torch.randn(100, 10)
        score = redundancy_score(a)
        assert score > 10.0


class TestWeightRedundancy:
    """Tests for weight_redundancy function."""

    def test_weight_redundancy_nonnegative(self):
        """Test that weight redundancy is non-negative."""
        W = torch.randn(64, 784)
        score = weight_redundancy(W)
        assert score >= 0

    def test_weight_redundancy_orthogonal(self):
        """Test that orthogonal weights have low redundancy."""
        W = torch.eye(20)
        score = weight_redundancy(W)
        assert score < 0.01

    def test_weight_redundancy_identical(self):
        """Test that identical weights have high redundancy."""
        base = torch.randn(1, 100)
        W = base.expand(10, 100).clone()
        score = weight_redundancy(W)
        # All rows identical means Gram matrix is all 1s
        # ||ones - I||^2 = n^2 - n for n×n matrix
        assert score > 50


class TestEffectiveRank:
    """Tests for effective_rank function."""

    def test_effective_rank_identity(self):
        """Test effective rank of identity matrix."""
        W = torch.eye(10)
        rank = effective_rank(W)
        # Identity has all singular values = 1, so rank should be n
        assert abs(rank - 10) < 0.1

    def test_effective_rank_rank_one(self):
        """Test effective rank of rank-1 matrix."""
        u = torch.randn(10, 1)
        v = torch.randn(1, 5)
        W = u @ v
        rank = effective_rank(W)
        assert rank < 2

    def test_effective_rank_bounds(self):
        """Test that effective rank is bounded."""
        W = torch.randn(20, 50)
        rank = effective_rank(W)
        assert 1 <= rank <= min(20, 50)


class TestResponsibilityEntropy:
    """Tests for responsibility_entropy function."""

    def test_entropy_uniform(self):
        """Test entropy of uniform distribution."""
        n = 10
        r = torch.ones(100, n) / n  # Uniform responsibilities
        entropy = responsibility_entropy(r)
        expected = np.log(n)  # Max entropy for n categories
        assert abs(entropy - expected) < 0.01

    def test_entropy_peaked(self):
        """Test entropy of peaked distribution."""
        r = torch.zeros(100, 10)
        r[:, 0] = 1.0  # All responsibility on first component
        entropy = responsibility_entropy(r)
        assert entropy < 0.01  # Should be near 0

    def test_entropy_nonnegative(self):
        """Test that entropy is non-negative."""
        r = torch.softmax(torch.randn(50, 20), dim=1)
        entropy = responsibility_entropy(r)
        assert entropy >= 0


class TestUsageDistribution:
    """Tests for usage_distribution function."""

    def test_usage_distribution_shape(self):
        """Test that usage distribution has correct shape."""
        r = torch.randn(100, 64).softmax(dim=1)
        usage = usage_distribution(r)
        assert usage.shape == (64,)

    def test_usage_distribution_sums_to_one(self):
        """Test that usage distribution sums to 1."""
        r = torch.randn(100, 10).softmax(dim=1)
        usage = usage_distribution(r)
        assert abs(usage.sum() - 1.0) < 1e-5

    def test_usage_distribution_uniform(self):
        """Test usage distribution with uniform responsibilities."""
        r = torch.ones(100, 10) / 10
        usage = usage_distribution(r)
        expected = np.ones(10) / 10
        assert np.allclose(usage, expected)


class TestReconstructionMSE:
    """Tests for reconstruction_mse function."""

    def test_reconstruction_mse_perfect(self):
        """Test MSE with perfect reconstruction."""
        W = torch.eye(10)  # Identity decoder
        x = torch.randn(32, 10)
        a = x  # Perfect encoding

        mse = reconstruction_mse(x, a, W)
        assert mse < 1e-10

    def test_reconstruction_mse_nonnegative(self):
        """Test that MSE is non-negative."""
        x = torch.randn(32, 784)
        a = torch.randn(32, 64)
        W = torch.randn(64, 784)

        mse = reconstruction_mse(x, a, W)
        assert mse >= 0

    def test_reconstruction_mse_zero_activations(self):
        """Test MSE with zero activations."""
        x = torch.randn(32, 100)
        a = torch.zeros(32, 20)
        W = torch.randn(20, 100)

        # Reconstruction is zero, so MSE = ||x||^2 / n
        mse = reconstruction_mse(x, a, W)
        expected = (x ** 2).mean().item()
        assert abs(mse - expected) < 1e-5


class TestSparsityL0:
    """Tests for sparsity_l0 function."""

    def test_sparsity_l0_all_active(self):
        """Test L0 with all units active."""
        a = torch.ones(32, 64)  # All units have value 1
        l0 = sparsity_l0(a, threshold=0.01)
        assert l0 == 64

    def test_sparsity_l0_all_inactive(self):
        """Test L0 with all units inactive."""
        a = torch.zeros(32, 64)
        l0 = sparsity_l0(a, threshold=0.01)
        assert l0 == 0

    def test_sparsity_l0_some_active(self):
        """Test L0 with some units active."""
        a = torch.zeros(32, 64)
        a[:, :10] = 1.0  # First 10 units active
        l0 = sparsity_l0(a, threshold=0.01)
        assert l0 == 10

    def test_sparsity_l0_threshold(self):
        """Test that threshold affects L0."""
        a = torch.ones(32, 64) * 0.05  # Small values

        l0_low = sparsity_l0(a, threshold=0.01)
        l0_high = sparsity_l0(a, threshold=0.1)

        assert l0_low > l0_high


class TestFeatureSimilarityMatrix:
    """Tests for feature_similarity_matrix function."""

    def test_similarity_matrix_shape(self):
        """Test that similarity matrix has correct shape."""
        W = torch.randn(64, 784)
        sim = feature_similarity_matrix(W)
        assert sim.shape == (64, 64)

    def test_similarity_matrix_diagonal(self):
        """Test that diagonal is all ones."""
        W = torch.randn(20, 100)
        sim = feature_similarity_matrix(W)
        assert np.allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_similarity_matrix_symmetric(self):
        """Test that similarity matrix is symmetric."""
        W = torch.randn(20, 100)
        sim = feature_similarity_matrix(W)
        assert np.allclose(sim, sim.T)

    def test_similarity_matrix_bounds(self):
        """Test that similarity values are in [-1, 1]."""
        W = torch.randn(20, 100)
        sim = feature_similarity_matrix(W)
        assert sim.min() >= -1.0 - 1e-5
        assert sim.max() <= 1.0 + 1e-5


class TestActivationCorrelationMatrix:
    """Tests for activation_correlation_matrix function."""

    def test_correlation_matrix_shape(self):
        """Test that correlation matrix has correct shape."""
        a = torch.randn(100, 64)
        corr = activation_correlation_matrix(a)
        assert corr.shape == (64, 64)

    def test_correlation_matrix_diagonal(self):
        """Test that diagonal is all ones."""
        a = torch.randn(100, 20)
        corr = activation_correlation_matrix(a)
        assert np.allclose(np.diag(corr), 1.0, atol=1e-5)

    def test_correlation_matrix_symmetric(self):
        """Test that correlation matrix is symmetric."""
        a = torch.randn(100, 20)
        corr = activation_correlation_matrix(a)
        assert np.allclose(corr, corr.T, atol=1e-5)


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_compute_all_metrics_returns_dict(self):
        """Test that compute_all_metrics returns a dictionary."""
        a = torch.randn(100, 64)
        W = torch.randn(64, 784)
        x = torch.randn(100, 784)
        r = torch.softmax(-a, dim=1)

        metrics = compute_all_metrics(a, W, x, r)

        assert isinstance(metrics, dict)
        assert "dead_units" in metrics
        assert "redundancy_score" in metrics
        assert "weight_redundancy" in metrics
        assert "effective_rank" in metrics
        assert "responsibility_entropy" in metrics
        assert "reconstruction_mse" in metrics
        assert "sparsity_l0" in metrics
