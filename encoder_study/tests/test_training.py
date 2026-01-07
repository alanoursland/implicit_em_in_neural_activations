"""Tests for training.py"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import Encoder, StandardSAE
from src.data import get_synthetic_loader
from src.training import train_epoch, evaluate, train, set_seed


class TestSetSeed:
    """Tests for set_seed function."""

    def test_set_seed_reproducibility(self):
        """Test that set_seed makes random operations reproducible."""
        set_seed(42)
        a = torch.randn(10)

        set_seed(42)
        b = torch.randn(10)

        assert torch.allclose(a, b)

    def test_different_seeds(self):
        """Test that different seeds give different results."""
        set_seed(42)
        a = torch.randn(10)

        set_seed(43)
        b = torch.randn(10)

        assert not torch.allclose(a, b)


class TestTrainEpoch:
    """Tests for train_epoch function."""

    def test_train_epoch_encoder(self):
        """Test training epoch with Encoder."""
        set_seed(42)
        model = Encoder(input_dim=50, hidden_dim=10, activation="relu")
        loader, _ = get_synthetic_loader(n_samples=64, n_features=50, batch_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_config = {"lambda_lse": 1.0, "lambda_var": 1.0, "lambda_tc": 1.0}
        device = torch.device("cpu")

        metrics = train_epoch(model, loader, optimizer, loss_config, device)

        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_train_epoch_sae(self):
        """Test training epoch with StandardSAE."""
        set_seed(42)
        model = StandardSAE(input_dim=50, hidden_dim=10)
        loader, _ = get_synthetic_loader(n_samples=64, n_features=50, batch_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_config = {"l1_weight": 0.01}
        device = torch.device("cpu")

        metrics = train_epoch(model, loader, optimizer, loss_config, device)

        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_train_epoch_decreases_loss(self):
        """Test that training decreases loss over epochs."""
        set_seed(42)
        model = Encoder(input_dim=50, hidden_dim=10, activation="relu")
        loader, _ = get_synthetic_loader(n_samples=128, n_features=50, batch_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_config = {"lambda_lse": 1.0, "lambda_var": 1.0, "lambda_tc": 0.1}
        device = torch.device("cpu")

        losses = []
        for _ in range(5):
            metrics = train_epoch(model, loader, optimizer, loss_config, device)
            losses.append(metrics["loss"])

        # Loss should generally decrease (allow some fluctuation)
        assert losses[-1] < losses[0], "Loss should decrease during training"


class TestEvaluate:
    """Tests for evaluate function."""

    def test_evaluate_encoder(self):
        """Test evaluation with Encoder."""
        set_seed(42)
        model = Encoder(input_dim=50, hidden_dim=10, activation="relu")
        loader, _ = get_synthetic_loader(n_samples=64, n_features=50, batch_size=32)
        loss_config = {"lambda_lse": 1.0, "lambda_var": 1.0, "lambda_tc": 1.0}
        device = torch.device("cpu")

        model = model.to(device)
        metrics = evaluate(model, loader, loss_config, device)

        assert "loss" in metrics
        assert "dead_units" in metrics
        assert "redundancy_score" in metrics
        assert "reconstruction_mse" in metrics
        assert "sparsity_l0" in metrics

    def test_evaluate_sae(self):
        """Test evaluation with StandardSAE."""
        set_seed(42)
        model = StandardSAE(input_dim=50, hidden_dim=10)
        loader, _ = get_synthetic_loader(n_samples=64, n_features=50, batch_size=32)
        loss_config = {"l1_weight": 0.01}
        device = torch.device("cpu")

        model = model.to(device)
        metrics = evaluate(model, loader, loss_config, device)

        assert "loss" in metrics
        assert "reconstruction_mse" in metrics

    def test_evaluate_no_gradients(self):
        """Test that evaluation doesn't accumulate gradients."""
        set_seed(42)
        model = Encoder(input_dim=50, hidden_dim=10, activation="relu")
        loader, _ = get_synthetic_loader(n_samples=64, n_features=50, batch_size=32)
        loss_config = {"lambda_lse": 1.0, "lambda_var": 1.0, "lambda_tc": 1.0}
        device = torch.device("cpu")

        model = model.to(device)
        evaluate(model, loader, loss_config, device)

        # Model should not have gradients after evaluation
        for param in model.parameters():
            assert param.grad is None or (param.grad == 0).all()


class TestTrain:
    """Tests for train function."""

    def test_train_returns_history(self):
        """Test that train returns history dict."""
        set_seed(42)
        model = Encoder(input_dim=50, hidden_dim=10, activation="relu")
        train_loader, _ = get_synthetic_loader(n_samples=64, n_features=50, batch_size=32)
        test_loader, _ = get_synthetic_loader(n_samples=32, n_features=50, batch_size=32, seed=43)

        config = {
            "training": {"epochs": 2, "lr": 0.01, "optimizer": "adam"},
            "loss": {"lambda_lse": 1.0, "lambda_var": 1.0, "lambda_tc": 1.0},
        }

        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            verbose=False,
        )

        assert "train_loss" in history
        assert "test_loss" in history
        assert len(history["train_loss"]) == 2

    def test_train_loss_decreases(self):
        """Test that training loss decreases."""
        set_seed(42)
        model = Encoder(input_dim=50, hidden_dim=10, activation="relu")
        train_loader, _ = get_synthetic_loader(n_samples=256, n_features=50, batch_size=32)
        test_loader, _ = get_synthetic_loader(n_samples=64, n_features=50, batch_size=32, seed=43)

        config = {
            "training": {"epochs": 10, "lr": 0.01, "optimizer": "adam"},
            "loss": {"lambda_lse": 1.0, "lambda_var": 1.0, "lambda_tc": 0.1},
        }

        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            verbose=False,
        )

        # Loss should decrease
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_train_sgd_optimizer(self):
        """Test training with SGD optimizer."""
        set_seed(42)
        model = Encoder(input_dim=50, hidden_dim=10, activation="relu")
        train_loader, _ = get_synthetic_loader(n_samples=64, n_features=50, batch_size=32)
        test_loader, _ = get_synthetic_loader(n_samples=32, n_features=50, batch_size=32, seed=43)

        config = {
            "training": {"epochs": 2, "lr": 0.01, "optimizer": "sgd"},
            "loss": {"lambda_lse": 1.0, "lambda_var": 1.0, "lambda_tc": 1.0},
        }

        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            verbose=False,
        )

        assert len(history["train_loss"]) == 2

    def test_train_sae(self):
        """Test training StandardSAE."""
        set_seed(42)
        model = StandardSAE(input_dim=50, hidden_dim=10)
        train_loader, _ = get_synthetic_loader(n_samples=64, n_features=50, batch_size=32)
        test_loader, _ = get_synthetic_loader(n_samples=32, n_features=50, batch_size=32, seed=43)

        config = {
            "training": {"epochs": 3, "lr": 0.01, "optimizer": "adam"},
            "loss": {"l1_weight": 0.01},
        }

        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            verbose=False,
        )

        assert len(history["train_loss"]) == 3
        assert history["test_reconstruction_mse"][-1] > 0


class TestIntegration:
    """Integration tests for training pipeline."""

    def test_encoder_full_pipeline(self):
        """Test full training pipeline with Encoder."""
        set_seed(42)

        # Create model
        model = Encoder(input_dim=50, hidden_dim=20, activation="relu")

        # Create data
        train_loader, _ = get_synthetic_loader(
            n_samples=200, n_features=50, n_components=5, batch_size=32
        )
        test_loader, _ = get_synthetic_loader(
            n_samples=50, n_features=50, n_components=5, batch_size=32, seed=43
        )

        # Config
        config = {
            "training": {"epochs": 5, "lr": 0.001, "optimizer": "adam"},
            "loss": {
                "lambda_lse": 1.0,
                "lambda_var": 1.0,
                "lambda_tc": 0.5,
                "lambda_wr": 0.0,
            },
        }

        # Train
        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            verbose=False,
        )

        # Verify training happened
        assert len(history["train_loss"]) == 5
        assert all(isinstance(v, float) or isinstance(v, int) for v in history["test_dead_units"])

    def test_sae_full_pipeline(self):
        """Test full training pipeline with StandardSAE."""
        set_seed(42)

        model = StandardSAE(input_dim=50, hidden_dim=20)

        train_loader, _ = get_synthetic_loader(
            n_samples=200, n_features=50, batch_size=32
        )
        test_loader, _ = get_synthetic_loader(
            n_samples=50, n_features=50, batch_size=32, seed=43
        )

        config = {
            "training": {"epochs": 5, "lr": 0.001, "optimizer": "adam"},
            "loss": {"l1_weight": 0.01},
        }

        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            verbose=False,
        )

        assert len(history["train_loss"]) == 5
        # SAE should achieve reasonable reconstruction
        assert history["test_reconstruction_mse"][-1] < history["test_reconstruction_mse"][0]
