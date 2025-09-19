import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV_NAME = "CartPole-v1"
SEED = 42

# File management
REWARDS_FILE = "rewards.csv"
MODEL_FILE = "trained_qnet.pth"

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

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 10000
