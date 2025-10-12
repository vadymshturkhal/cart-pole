import torch.nn as nn
import config


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}

def get_activation(activation_function: str) -> nn.Module:
    if activation_function not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {activation_function}, supported {list(_ACTIVATIONS)}")
    
    return _ACTIVATIONS[activation_function]()


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Shared feature layers
        layers = []
        input_dim = state_dim
        for hidden_dim in config.HIDDEN_LAYERS:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_activation(config.HIDDEN_ACTIVATION))

            if config.DROPOUT > 0:
                layers.append(nn.Dropout(p=config.DROPOUT))
            
            input_dim = hidden_dim

        self.feature = nn.Sequential(*layers)

        # Dueling streams
        self.value = nn.Linear(input_dim, 1)
        self.advantage = nn.Linear(input_dim, action_dim)

        # Initialize weights after model is built
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Perfect for ReLU-family activations
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.feature(x)
        value = self.value(x)
        advantage = self.advantage(x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q
