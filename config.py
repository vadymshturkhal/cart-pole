import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_ENVIRONMENT = "CartPole-v1"
AVAILABLE_ENVIRONMENTS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
SEED = 42

# Rendering
WIDTH = 800
HEIGHT = 600

# File management
REWARDS_FILE = "rewards.csv"
TRAINED_CONSOLE_MODEL_FILENAME = "trained_console_qnet.pth"
TRAINED_MODELS_FOLDER = "trained_models"
N_STEP_DQN = "nstep_dqn_qnet.pth"

# Neural network architecture
HIDDEN_LAYERS = [256, 128]     # list of hidden layer sizes
ACTIVATION = "relu"       # activation function ("relu" or "tanh")
DROPOUT = 0.2             # dropout probability (0.0 = disable)

# Training
EPISODES = 500
TARGET_UPDATE_FREQ = 20

# Agent
GAMMA = 0.99
LR = 1e-3
BUFFER_SIZE = 10000
BATCH_SIZE = 64
N_STEP = 3
SUTTON_BARTO_REWARD = False

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 10000

# Logs
LOG_AFTER_EPISODES = 20
LOG_DIR = "runs/cartpole_nstep"

TARGET_UPDATE = 10   # update target net every 10 episodes
