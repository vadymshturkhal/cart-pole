from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from agents.base_agent import BaseAgent
from .q_network import QNetwork
from memory.replay_buffer import NStepReplayBuffer
import config
from datetime import datetime
from utils.optim_factory import build_optimizer


class NStepDeepQLearningAgent(BaseAgent):
    DEFAULT_PARAMS = {
        "gamma": config.GAMMA,
        "buffer_size": config.BUFFER_SIZE,
        "batch_size": config.BATCH_SIZE,
        "n_step": config.N_STEP,
        "eps_start": config.EPSILON_START,
        "eps_end": config.EPSILON_END,
        "eps_decay": config.EPSILON_DECAY,
        "eps_fixed": config.EPSILON_FIXED,
        "epsilon_schedule": "linear",
        "target_update": config.TARGET_UPDATE,
    }
        
    def __init__(self, state_dim, action_dim, **kwargs):
        """
        N-step DQN agent.

        """
        
        self.name = "nstep_dqn"
        self.rng = np.random.default_rng(config.SEED)
        hyperparams = self.DEFAULT_PARAMS.copy()
        hyperparams.update(kwargs)
        self.hyperparams = hyperparams
        self.checkpoint = {}
        self.n_step = hyperparams["n_step"]

        # Hyperparameters
        self.gamma = hyperparams["gamma"]
        self.batch_size = hyperparams["batch_size"]
        self.action_dim = action_dim
        self.steps_done = 0
        self.episodes = 0

        # Exploration
        self.eps_start = hyperparams["eps_start"]
        self.eps_end = hyperparams["eps_end"]
        self.eps_decay = hyperparams["eps_decay"]
        self.eps_fixed = hyperparams["eps_fixed"]

        self.current_epsilon = self.eps_start

        # Replay buffer (use n-step > 1)
        self.memory = NStepReplayBuffer(hyperparams["buffer_size"], hyperparams["n_step"], self.gamma)

        # Networks
        self.q_net = QNetwork(state_dim, action_dim).to(config.DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(config.DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = build_optimizer(config.OPTIMIZER, self.q_net.parameters(), lr=config.LR)

        #Loss
        self._losses = [0]

    def select_action(self, state, greedy: bool = False):
        """
        Select an action from the state.

        Args:
            state (ndarray): environment state
            greedy (bool): 
                - True  → always choose best action (evaluation/testing)
                - False → epsilon-greedy (training)
        """

        if greedy:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)
                q_values = self.q_net(state)
                return q_values.argmax().item()

        schedule = self.hyperparams.get("epsilon_schedule", "linear")

        if schedule == "fixed":
            eps = self.eps_fixed
        elif schedule == "linear":
            eps = self.eps_start - (self.eps_start - self.eps_end) * (self.episodes / self.total_episodes)
        elif schedule == "exponential":
            eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-5 * (self.episodes / self.total_episodes))
        elif schedule == "manual":
            eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1.0 * self.steps_done / self.eps_decay)
        else:
            eps = self.eps_end
        
        # For Logger
        self.current_epsilon = eps
        self.steps_done += 1

        if self.rng.random() < eps:
            return self.rng.integers(0, self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)
                q_values = self.target_net(state)
                return q_values.argmax().item()

    def update_step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.update()

    def update(self):
        if self.batch_size == 0:
            return

        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        states = torch.from_numpy(batch.state).to(config.DEVICE)
        actions = torch.from_numpy(batch.action).unsqueeze(1).to(config.DEVICE)
        rewards = torch.from_numpy(batch.reward).to(config.DEVICE)
        next_states = torch.from_numpy(batch.next_state).to(config.DEVICE)
        dones = torch.from_numpy(batch.done).to(config.DEVICE)

        q_values = self.q_net(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]

        # Compute target with n-step returns already handled by replay buffer
        expected_q = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values
        
        # Loss
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(q_values, expected_q.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        self._losses.append(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    @classmethod
    def get_default_hyperparams(cls) -> dict:
        return cls.DEFAULT_PARAMS.copy()

    def get_hyperparams(self) -> dict:
        return self.hyperparams.copy()
    
    def get_checkpoint(self):
        return deepcopy(self.checkpoint)

    def update_checkpoint(self, extra):
        self.checkpoint.update(extra)

    def save(self, path: str, extra: dict = None):
        # Add latest epsilon
        self.hyperparams["eps_latest"] = self.current_epsilon

        self.checkpoint = {
            "agent_name": self.name,
            "model_state": self.target_net.state_dict(),
            "hyperparams": self.hyperparams,
            "nn_config": {  # Save full NN architecture & optimizer info
                "hidden_layers": config.HIDDEN_LAYERS,
                "activation": config.HIDDEN_ACTIVATION,
                "dropout": config.DROPOUT,
                "lr": config.LR,
                "optimizer": config.OPTIMIZER,
                "device": str(config.DEVICE),
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if extra:
            self.checkpoint.update(extra)

        torch.save(self.checkpoint, path)

    def load(self, path: str, hyperparams: dict = None, apply_nn_config: bool = True):
        """
        Load model weights, optimizer, and NN configuration from a checkpoint file.
        """
        checkpoint = torch.load(path, map_location=config.DEVICE)

        # === Restore agent hyperparameters ===
        loaded_hyperparams = self.DEFAULT_PARAMS.copy()
        loaded_hyperparams.update(checkpoint.get("hyperparams", {}))

        if hyperparams:
            loaded_hyperparams.update(hyperparams)

        self.hyperparams = loaded_hyperparams

        self.eps_start = loaded_hyperparams["eps_start"]
        self.eps_end = loaded_hyperparams["eps_end"]
        self.eps_decay = loaded_hyperparams["eps_decay"]

        # === Restore NN configuration (if present) ===
        nn_cfg = checkpoint.get("nn_config", {})
        hidden_layers = nn_cfg.get("hidden_layers", config.HIDDEN_LAYERS)
        activation = nn_cfg.get("activation", config.HIDDEN_ACTIVATION)
        dropout = nn_cfg.get("dropout", config.DROPOUT)
        lr = nn_cfg.get("lr", config.LR)
        optimizer_name = nn_cfg.get("optimizer", config.OPTIMIZER)

        # Conditional config override
        if apply_nn_config:
            # Update global config (runtime consistency)
            config.HIDDEN_LAYERS = hidden_layers
            config.HIDDEN_ACTIVATION = activation
            config.DROPOUT = dropout
            config.LR = lr
            config.OPTIMIZER = optimizer_name

        # === Recreate networks ===
        self.q_net = QNetwork(self.q_net.net[0].in_features, self.action_dim).to(config.DEVICE)
        self.target_net = QNetwork(self.q_net.net[0].in_features, self.action_dim).to(config.DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # === Restore model weights ===
        if "model_state" in checkpoint:
            self.q_net.load_state_dict(checkpoint["model_state"])
            self.target_net.load_state_dict(self.q_net.state_dict())

        # === Recreate optimizer with correct type and LR ===
        self.optimizer = build_optimizer(optimizer_name, self.q_net.parameters(), lr=lr)
        self.q_net.eval()

    @property
    def losses(self):
        return self._losses.copy()
    
    @property
    def loss(self):
        return self._losses.copy()[-1]
    
    def _clear_losses(self):
        self._losses.clear()
        self._losses.append(0)

    def add_episode(self):
        """Increment episode count for scheduling (epsilon decay, etc.)."""
        self.episodes += 1
        self._clear_losses()

    def set_total_episodes(self, episodes):
        self.total_episodes = episodes
