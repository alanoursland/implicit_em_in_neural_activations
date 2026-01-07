import torch
import torch.nn as nn
import torch.nn.functional as F

class Activation(nn.Module):
    """Base class for activations."""
    pass

class Identity(Activation):
    def forward(self, z):
        return z

class ReLU(Activation):
    def forward(self, z):
        return F.relu(z)

class LeakyReLU(Activation):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, z):
        return F.leaky_relu(z, self.negative_slope)

class Softmax(Activation):
    def forward(self, z):
        return F.softmax(z, dim=-1)

class Tanh(Activation):
    def forward(self, z):
        return torch.tanh(z)

class Softplus(Activation):
    def forward(self, z):
        return F.softplus(z)

def get_activation(name: str) -> Activation:
    """Factory function."""
    activations = {
        "identity": Identity,
        "relu": ReLU,
        "leaky_relu": LeakyReLU,
        "softmax": Softmax,
        "tanh": Tanh,
        "softplus": Softplus,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]()