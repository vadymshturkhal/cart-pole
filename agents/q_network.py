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


class QNetwork(nn.Module):
    """
    Feedforward neural network (FNN) for approximating Q-values in discrete-action RL.
    This network maps environment states to estimated Q-values for each possible action.
    Architecture (hidden layers, activations, dropout) can be set from GUI.
    """

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Hidden layers from config
        for hidden_dim in config.HIDDEN_LAYERS:
            layers.append(nn.Linear(input_dim, hidden_dim))
            activation = get_activation(config.HIDDEN_ACTIVATION)
            print(activation)
            
            # Fresh activation for each layer
            layers.append(activation)

            #  Dropout after activation
            if config.DROPOUT > 0:
                layers.append(nn.Dropout(p=config.DROPOUT))

            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.net = nn.Sequential(*layers)

        # Initialize weights after model is built
        self._init_weights()

    def _init_weights(self):
        """Activation-aware initialization (He for ReLU, Xavier for tanh)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if config.ACTIVATION.lower() == "relu":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif config.ACTIVATION.lower() == "tanh":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    # Fallback for other activations
                    nn.init.uniform_(m.weight, -0.01, 0.01)

                #  Biases to 0
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
