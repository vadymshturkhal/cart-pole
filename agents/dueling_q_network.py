import torch.nn as nn
from .q_network import QNetwork


class DuelingQNetwork(QNetwork):
    """
    Dueling architecture built on top of QNetwork.
    Retains configurable hidden structure from QNetwork but
    replaces the final output layer with separate Value and Advantage streams.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)

        # Identify and remove the final Linear layer (QNetwork always ends with one)
        last_linear = None
        for m in reversed(self.net):
            if isinstance(m, nn.Linear):
                last_linear = m
                break

        if last_linear is None:
            raise RuntimeError("QNetwork has no Linear layer to replace in DuelingQNetwork")

        last_dim = last_linear.in_features

        # Build feature extractor: all layers except the last Linear
        layers = []
        for layer in self.net:
            layers.append(layer)
            if layer is last_linear:
                break
        # Remove the last linear layer (keep activations/dropouts before it)
        while layers and isinstance(layers[-1], nn.Linear):
            layers.pop()
        self.feature = nn.Sequential(*layers)

        # Define dueling heads
        self.value = nn.Linear(last_dim, 1)
        self.advantage = nn.Linear(last_dim, action_dim)

        # Reinitialize new heads only
        self._init_dueling_weights()

    def _init_dueling_weights(self):
        for layer in [self.value, self.advantage]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.feature(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - a.mean(dim=1, keepdim=True))
