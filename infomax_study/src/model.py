import torch
import torch.nn as nn
from .activations import get_activation

class SingleLayer(nn.Module):
    """
    Single layer model: z = Wx + b, h = activation(z), a = softmax(h)

    Architecture:
      Input → Linear → Activation → Softmax → Output

    The softmax output 'a' represents responsibilities/posterior probabilities,
    creating EM-like behavior where each sample has a soft assignment across K units.
    """
    def __init__(self, input_dim: int, hidden_dim: int, activation: str):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = get_activation(activation)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        z = self.linear(x)
        h = self.activation(z)
        a = self.softmax(h)
        return a, z

    @property
    def weight_matrix(self):
        return self.linear.weight.detach()
