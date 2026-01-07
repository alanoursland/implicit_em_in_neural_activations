import torch
import torch.nn as nn
from .activations import get_activation

class SingleLayer(nn.Module):
    """
    Single layer model: z = Wx + b, a = activation(z)
    """
    def __init__(self, input_dim: int, hidden_dim: int, activation: str):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = get_activation(activation)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        z = self.linear(x)
        a = self.activation(z)
        return a, z

    @property
    def weight_matrix(self):
        return self.linear.weight.detach()
