import config


AGENT_SPECS = {
    "nstep_dqn": {
        "gamma":        ("float", 0.0, 0.9999, 0.001, config.GAMMA),
        "lr":           ("float", 1e-6, 1.0, 1e-6,  config.LR),
        "buffer_size":  ("int",   1000, 10_000_000, 1000, config.BUFFER_SIZE),
        "batch_size":   ("int",   8,    65536,      1,    config.BATCH_SIZE),
        "n_step":       ("int",   1,    10,         1,    config.N_STEP),
        "eps_start":    ("float", 0.0,  1.0,        0.01, config.EPSILON_START),
        "eps_end":      ("float", 0.0,  1.0,        0.01, config.EPSILON_END),
        "eps_decay":    ("int",   1,    10_000_000, 1,    config.EPSILON_DECAY),
    },

    "nstep_ddqn": {
        "gamma":        ("float", 0.0,  0.9999,     0.001, config.GAMMA),
        "lr":           ("float", 1e-6, 1.0,        1e-6,  config.LR),
        "buffer_size":  ("int",   1000, 10_000_000, 1000,  config.BUFFER_SIZE),
        "batch_size":   ("int",   8,    65536,      1,     config.BATCH_SIZE),
        "n_step":       ("int",   1,    10,         1,     config.N_STEP),
        "eps_start":    ("float", 0.0,  1.0,        0.01,  config.EPSILON_START),
        "eps_end":      ("float", 0.0,  1.0,        0.01,  config.EPSILON_END),
        "eps_decay":    ("int",   1,    10_000_000, 1,     config.EPSILON_DECAY),
    },

    # Actor–Critic w/ eligibilities (your ase_ace)
    "ase_ace": {
        "gamma":        ("float", 0.0,  0.9999, 0.001, 0.99),
        "lam_v":        ("float", 0.0,  1.0,    0.01,  0.8),
        "lam_pi":       ("float", 0.0,  1.0,    0.01,  0.8),
        "alpha_v":      ("float", 1e-5, 1.0,    1e-3,  0.2),
        "alpha_pi":     ("float", 1e-5, 1.0,    1e-3,  0.05),
        "max_steps":    ("int",   1,    100_000, 1,    2000),
        "sparse_reward":("bool",),
    },
}

def default_hparams(agent_name: str) -> dict:
    spec = AGENT_SPECS[agent_name]
    out = {}
    for k, desc in spec.items():
        if desc[0] == "bool":
            out[k] = False
        else:
            out[k] = desc[-1]  # default
    return out
