import torch, json, os


# ===== Default values =====
DEFAULTS = {
    # System
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # Environment
    "DEFAULT_ENVIRONMENT": "CartPole-v1",
    "AVAILABLE_ENVIRONMENTS": ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"],
    "SEED": 42,

    # Rendering
    "RESOLUTION": [1280, 720],

    # File management
    "REWARDS_FILE": "rewards.csv",
    "TRAINED_CONSOLE_MODEL_FILENAME": "trained_console_qnet.pth",
    "TRAINED_MODELS_FOLDER": "trained_models",
    "N_STEP_DQN": "nstep_dqn_qnet.pth",

    # Neural network architecture
    "HIDDEN_LAYERS": [256, 128],
    "ACTIVATION": "relu",     # "relu" or "tanh"
    "DROPOUT": 0.2,

    # Training
    "EPISODES": 500,
    "TARGET_UPDATE_FREQ": 20,

    # Agent
    "GAMMA": 0.99,
    "LR": 1e-3,
    "BUFFER_SIZE": 10000,
    "BATCH_SIZE": 64,
    "N_STEP": 3,
    "SUTTON_BARTO_REWARD": False,

    # Exploration
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.05,
    "EPSILON_DECAY": 10000,

    # Logs
    "LOG_AFTER_EPISODES": 20,
    "LOG_DIR": "runs/cartpole_nstep",
    "TARGET_UPDATE": 10,
}

# ===== User config handling =====
USER_CONFIG_FILE = "user_config.json"

def load_user_config():
    """Load defaults, then override with user config if available."""
    config = DEFAULTS.copy()
    if os.path.exists(USER_CONFIG_FILE):
        try:
            with open(USER_CONFIG_FILE, "r") as f:
                user_cfg = json.load(f)
                config.update(user_cfg)
        except Exception as e:
            print("⚠ Failed to load user config:", e)
    return config

def save_user_config(updates: dict):
    """Save updates to user config file (merge with existing)."""
    data = {}
    if os.path.exists(USER_CONFIG_FILE):
        try:
            with open(USER_CONFIG_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.update(updates)
    with open(USER_CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)

def restore_defaults():
    """Delete user config and reload defaults only."""
    if os.path.exists(USER_CONFIG_FILE):
        try:
            os.remove(USER_CONFIG_FILE)
            print("✅ User config removed, defaults will be used next run.")
        except Exception as e:
            print(f"⚠ Could not remove user config: {e}")

    # Reset module globals immediately
    globals().update(DEFAULTS)
    # Special case: DEVICE needs to be a torch.device
    globals()["DEVICE"] = torch.device(DEFAULTS["DEVICE"])


# ===== Merge defaults with overrides =====
_OVERRIDES = load_user_config()
CONFIG = DEFAULTS.copy()
CONFIG.update(_OVERRIDES)

# ===== Export as module variables =====
globals().update(CONFIG)
DEVICE = torch.device(CONFIG["DEVICE"])
