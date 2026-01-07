"""Tests for data.py"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_synthetic, get_synthetic_loader


class TestGetSynthetic:
    """Tests for get_synthetic function."""

    def test_synthetic_data_shape(self):
        """Test that synthetic data has correct shape."""
        n_samples = 100
        n_features = 50
        n_components = 5

        data, components = get_synthetic(
            n_samples=n_samples,
            n_features=n_features,
            n_components=n_components,
        )

        assert data.shape == (n_samples, n_features)
        assert components.shape == (n_components, n_features)

    def test_synthetic_components_normalized(self):
        """Test that synthetic components are normalized."""
        _, components = get_synthetic(
            n_samples=100,
            n_features=50,
            n_components=5,
        )

        norms = components.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_synthetic_reproducibility(self):
        """Test that seed makes data reproducible."""
        data1, comp1 = get_synthetic(n_samples=100, n_features=50, seed=42)
        data2, comp2 = get_synthetic(n_samples=100, n_features=50, seed=42)

        assert torch.allclose(data1, data2)
        assert torch.allclose(comp1, comp2)

    def test_synthetic_different_seeds(self):
        """Test that different seeds produce different data."""
        data1, _ = get_synthetic(n_samples=100, n_features=50, seed=42)
        data2, _ = get_synthetic(n_samples=100, n_features=50, seed=43)

        assert not torch.allclose(data1, data2)

    def test_synthetic_noise(self):
        """Test that noise parameter affects data."""
        data_low_noise, _ = get_synthetic(
            n_samples=100, n_features=50, noise_std=0.01, seed=42
        )
        data_high_noise, _ = get_synthetic(
            n_samples=100, n_features=50, noise_std=1.0, seed=42
        )

        # Different noise levels should produce different data
        assert not torch.allclose(data_low_noise, data_high_noise)


class TestGetSyntheticLoader:
    """Tests for get_synthetic_loader function."""

    def test_loader_returns_dataloader(self):
        """Test that function returns a DataLoader."""
        loader, components = get_synthetic_loader(
            n_samples=100,
            n_features=50,
            n_components=5,
            batch_size=32,
        )

        assert hasattr(loader, "__iter__")
        assert hasattr(loader, "__len__")

    def test_loader_batch_size(self):
        """Test that batches have correct size."""
        batch_size = 32
        loader, _ = get_synthetic_loader(
            n_samples=100,
            n_features=50,
            batch_size=batch_size,
        )

        for batch in loader:
            # Last batch might be smaller
            assert batch[0].shape[0] <= batch_size
            break

    def test_loader_feature_dim(self):
        """Test that batches have correct feature dimension."""
        n_features = 50
        loader, _ = get_synthetic_loader(
            n_samples=100,
            n_features=n_features,
            batch_size=32,
        )

        for batch in loader:
            assert batch[0].shape[1] == n_features
            break

    def test_loader_components_shape(self):
        """Test that returned components have correct shape."""
        n_components = 5
        n_features = 50
        _, components = get_synthetic_loader(
            n_samples=100,
            n_features=n_features,
            n_components=n_components,
        )

        assert components.shape == (n_components, n_features)


# Check if MNIST data directory exists
MNIST_DATA_DIR = Path("E:/ml_datasets")
MNIST_AVAILABLE = MNIST_DATA_DIR.exists() and (MNIST_DATA_DIR / "MNIST").exists()


class TestMNIST:
    """Tests for MNIST loading.

    Tests run only if MNIST data exists at E:/ml_datasets/MNIST/.
    """

    @pytest.mark.skipif(not MNIST_AVAILABLE, reason="MNIST data not available at E:/ml_datasets")
    def test_mnist_loaders(self):
        """Test that MNIST loaders work."""
        from src.data import get_mnist

        train_loader, test_loader = get_mnist(batch_size=64)

        # Check we can iterate
        for batch in train_loader:
            x, y = batch
            assert x.shape == (64, 784)
            assert y.shape == (64,)
            break

        # Check test loader
        for batch in test_loader:
            x, y = batch
            assert x.shape[1] == 784
            break

    @pytest.mark.skipif(not MNIST_AVAILABLE, reason="MNIST data not available at E:/ml_datasets")
    def test_mnist_flatten(self):
        """Test MNIST flatten parameter."""
        from src.data import get_mnist

        train_loader, _ = get_mnist(batch_size=32, flatten=True)

        for batch in train_loader:
            x, _ = batch
            assert x.shape[1] == 784
            break

    @pytest.mark.skipif(not MNIST_AVAILABLE, reason="MNIST data not available at E:/ml_datasets")
    def test_mnist_unflatten(self):
        """Test MNIST without flattening."""
        from src.data import get_mnist

        train_loader, _ = get_mnist(batch_size=32, flatten=False)

        for batch in train_loader:
            x, _ = batch
            # Should be (batch, 1, 28, 28)
            assert x.shape == (32, 1, 28, 28)
            break

    @pytest.mark.skipif(not MNIST_AVAILABLE, reason="MNIST data not available at E:/ml_datasets")
    def test_mnist_custom_data_dir(self):
        """Test MNIST with custom data directory."""
        from src.data import get_mnist

        train_loader, test_loader = get_mnist(
            batch_size=32,
            data_dir="E:/ml_datasets"
        )

        # Should work without error
        for batch in train_loader:
            assert batch[0].shape[0] <= 32
            break

    @pytest.mark.skipif(not MNIST_AVAILABLE, reason="MNIST data not available at E:/ml_datasets")
    def test_mnist_train_test_sizes(self):
        """Test that train/test loaders have correct sizes."""
        from src.data import get_mnist

        train_loader, test_loader = get_mnist(batch_size=128)

        # MNIST has 60000 train, 10000 test samples
        train_samples = sum(batch[0].shape[0] for batch in train_loader)
        test_samples = sum(batch[0].shape[0] for batch in test_loader)

        assert train_samples == 60000
        assert test_samples == 10000

    @pytest.mark.skipif(not MNIST_AVAILABLE, reason="MNIST data not available at E:/ml_datasets")
    def test_mnist_data_range(self):
        """Test that MNIST data is normalized to [0, 1]."""
        from src.data import get_mnist

        train_loader, _ = get_mnist(batch_size=128)

        for batch in train_loader:
            x, _ = batch
            assert x.min() >= 0.0
            assert x.max() <= 1.0
            break
