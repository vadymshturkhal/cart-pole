import torch.optim as optim


OPTIMIZERS = {
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
    "sgd": optim.SGD,
}

def build_optimizer(optimizer_name: str, model_params, lr: float):
    OptimizerClass = OPTIMIZERS.get(optimizer_name)

    if OptimizerClass is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return OptimizerClass(model_params, lr=lr)
