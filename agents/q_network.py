import torch.nn as nn
import config


def get_activation(name):
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {name}")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        activation = get_activation(config.ACTIVATION)
        
        # Hidden layers from config
        for hidden_dim in config.HIDDEN_LAYERS:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            if config.DROPOUT > 0:   # ✅ Add dropout if enabled
                layers.append(nn.Dropout(p=config.DROPOUT))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
