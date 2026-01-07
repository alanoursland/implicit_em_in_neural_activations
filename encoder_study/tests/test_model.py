"""Tests for model.py"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import Encoder, StandardSAE


class TestEncoder:
    """Tests for Encoder class."""

    def test_encoder_init(self):
        """Test encoder initialization."""
        model = Encoder(input_dim=784, hidden_dim=64, activation="relu")
        assert model.input_dim == 784
        assert model.hidden_dim == 64
        assert model.activation_name == "relu"

    def test_encoder_init_different_activations(self):
        """Test encoder with different activation functions."""
        for act in ["relu", "identity", "softplus"]:
            model = Encoder(input_dim=100, hidden_dim=10, activation=act)
            assert model.activation_name == act

    def test_encoder_invalid_activation(self):
        """Test encoder raises error for invalid activation."""
        with pytest.raises(ValueError, match="Unknown activation"):
            Encoder(input_dim=100, hidden_dim=10, activation="invalid")

    def test_encoder_forward_shape(self):
        """Test encoder forward pass output shapes."""
        model = Encoder(input_dim=784, hidden_dim=64, activation="relu")
        x = torch.randn(32, 784)

        a, z = model(x)

        assert a.shape == (32, 64)
        assert z.shape == (32, 64)

    def test_encoder_forward_relu_nonnegativity(self):
        """Test that ReLU activation produces non-negative outputs."""
        model = Encoder(input_dim=100, hidden_dim=20, activation="relu")
        x = torch.randn(16, 100)

        a, z = model(x)

        assert (a >= 0).all(), "ReLU outputs should be non-negative"

    def test_encoder_forward_identity(self):
        """Test that identity activation returns pre-activations."""
        model = Encoder(input_dim=100, hidden_dim=20, activation="identity")
        x = torch.randn(16, 100)

        a, z = model(x)

        assert torch.allclose(a, z), "Identity activation should return z unchanged"

    def test_encoder_get_weights(self):
        """Test get_weights returns correct shape."""
        model = Encoder(input_dim=784, hidden_dim=64, activation="relu")
        W = model.get_weights()

        assert W.shape == (64, 784)

    def test_encoder_weight_property(self):
        """Test W property returns weight matrix."""
        model = Encoder(input_dim=100, hidden_dim=10, activation="relu")
        assert model.W.shape == (10, 100)

    def test_encoder_bias_property(self):
        """Test b property returns bias vector."""
        model = Encoder(input_dim=100, hidden_dim=10, activation="relu")
        assert model.b.shape == (10,)

    def test_encoder_gradients(self):
        """Test that gradients flow through encoder."""
        model = Encoder(input_dim=100, hidden_dim=10, activation="relu")
        x = torch.randn(8, 100)

        a, z = model(x)
        loss = a.sum()
        loss.backward()

        assert model.linear.weight.grad is not None
        assert model.linear.bias.grad is not None


class TestStandardSAE:
    """Tests for StandardSAE class."""

    def test_sae_init(self):
        """Test SAE initialization."""
        model = StandardSAE(input_dim=784, hidden_dim=64)
        assert model.input_dim == 784
        assert model.hidden_dim == 64

    def test_sae_forward_shape(self):
        """Test SAE forward pass output shapes."""
        model = StandardSAE(input_dim=784, hidden_dim=64)
        x = torch.randn(32, 784)

        recon, a = model(x)

        assert recon.shape == (32, 784)
        assert a.shape == (32, 64)

    def test_sae_encode_shape(self):
        """Test SAE encode method output shape."""
        model = StandardSAE(input_dim=784, hidden_dim=64)
        x = torch.randn(32, 784)

        a = model.encode(x)

        assert a.shape == (32, 64)

    def test_sae_encode_nonnegativity(self):
        """Test that SAE encoder produces non-negative outputs (ReLU)."""
        model = StandardSAE(input_dim=100, hidden_dim=20)
        x = torch.randn(16, 100)

        a = model.encode(x)

        assert (a >= 0).all(), "ReLU outputs should be non-negative"

    def test_sae_encoder_weights(self):
        """Test get_encoder_weights returns correct shape."""
        model = StandardSAE(input_dim=784, hidden_dim=64)
        W = model.get_encoder_weights()

        assert W.shape == (64, 784)

    def test_sae_decoder_weights(self):
        """Test get_decoder_weights returns correct shape."""
        model = StandardSAE(input_dim=784, hidden_dim=64)
        W = model.get_decoder_weights()

        assert W.shape == (784, 64)

    def test_sae_parameter_count(self):
        """Test parameter_count method."""
        model = StandardSAE(input_dim=784, hidden_dim=64)
        count = model.parameter_count()

        # Encoder: 784*64 + 64 = 50240
        # Decoder: 64*784 + 784 = 50960
        # Total: 101200
        expected = 784 * 64 + 64 + 64 * 784 + 784
        assert count == expected

    def test_sae_gradients(self):
        """Test that gradients flow through SAE."""
        model = StandardSAE(input_dim=100, hidden_dim=10)
        x = torch.randn(8, 100)

        recon, a = model(x)
        loss = ((x - recon) ** 2).mean()
        loss.backward()

        assert model.encoder[0].weight.grad is not None
        assert model.decoder.weight.grad is not None


class TestModelComparison:
    """Tests comparing Encoder and StandardSAE."""

    def test_encoder_fewer_params_than_sae(self):
        """Test that Encoder has fewer parameters than SAE."""
        encoder = Encoder(input_dim=784, hidden_dim=64, activation="relu")
        sae = StandardSAE(input_dim=784, hidden_dim=64)

        encoder_params = sum(p.numel() for p in encoder.parameters())
        sae_params = sae.parameter_count()

        assert encoder_params < sae_params, "Encoder should have fewer params"
        # Encoder: 784*64 + 64 = 50240
        # SAE: 101200
        assert encoder_params == 784 * 64 + 64
