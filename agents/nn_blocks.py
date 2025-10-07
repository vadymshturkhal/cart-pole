import torch
import torch.nn as nn


def make_mlp(in_dim, hidden_layers, activation="relu", dropout=0.0, out_dim=None, out_act=None):
    acts = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    Act = acts.get(activation, nn.ReLU)

    layers = []
    last = in_dim
    for h in hidden_layers:
        layers += [nn.Linear(last, h), Act()]
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        last = h
    if out_dim is not None:
        layers.append(nn.Linear(last, out_dim))
        if out_act is not None:
            layers.append(out_act)
    return nn.Sequential(*layers)


class ActorMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, activation="relu", dropout=0.0, out_tanh=True):
        super().__init__()
        self.net = make_mlp(
            in_dim=state_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            out_dim=action_dim,
            out_act=nn.Tanh() if out_tanh else None,
        )

    def forward(self, x):
        return self.net(x)  # typically in [-1,1] if out_tanh=True


class CriticMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, activation="relu", dropout=0.0):
        super().__init__()
        self.state = make_mlp(
            in_dim=state_dim,
            hidden_layers=hidden_layers[:-1] if len(hidden_layers) > 1 else [],
            activation=activation,
            dropout=dropout,
        )
        last_state = state_dim if len(hidden_layers) <= 1 else hidden_layers[-2]
        # concat [state_features, action] -> final hidden -> 1
        self.head = make_mlp(
            in_dim=last_state + action_dim,
            hidden_layers=[hidden_layers[-1]] if len(hidden_layers) > 0 else [128],
            activation=activation,
            dropout=dropout,
            out_dim=1,
            out_act=None,
        )

    def forward(self, s, a):
        if len(list(self.state.children())) == 0:
            sf = s
        else:
            sf = self.state(s)
        x = torch.cat([sf, a], dim=-1)
        return self.head(x)
