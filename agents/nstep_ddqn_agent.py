import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from agents.base_agent import BaseAgent
from .q_network import QNetwork
from memory.replay_buffer import NStepReplayBuffer
import config
from datetime import datetime


class NStepDoubleDeepQLearningAgent(BaseAgent):
    DEFAULT_PARAMS = {
        "gamma": config.GAMMA,
        "lr": config.LR,
        "buffer_size": config.BUFFER_SIZE,
        "batch_size": config.BATCH_SIZE,
        "n_step": config.N_STEP,
        "eps_start": config.EPSILON_START,
        "eps_end": config.EPSILON_END,
        "eps_decay": config.EPSILON_DECAY,
    }
        
    def __init__(self, state_dim, action_dim, **kwargs):
        """
        N-step DDQN agent.

        Args:
            state_dim (int): dimension of state space
            action_dim (int): dimension of action space
            gamma (float): discount factor
            lr (float): learning rate
            buffer_size (int): replay buffer capacity
            batch_size (int): minibatch size
            n_step (int): N-step horizon
            eps_start (float): starting epsilon for exploration
            eps_end (float): final epsilon
            eps_decay (int): decay factor for epsilon
        """

        hyperparams = self.DEFAULT_PARAMS.copy()
        hyperparams.update(kwargs)
        self.hyperparams = hyperparams
        self.checkpoint = {}

        # Hyperparameters
        self.gamma = hyperparams["gamma"]
        self.batch_size = hyperparams["batch_size"]
        self.action_dim = action_dim
        self.steps_done = 0

        # Exploration
        self.eps_start = hyperparams["eps_start"]
        self.eps_end = hyperparams["eps_end"]
        self.eps_decay = hyperparams["eps_decay"]
        
        # Replay buffer (use n-step > 1)
        self.memory = NStepReplayBuffer(hyperparams["buffer_size"], hyperparams["n_step"], self.gamma)

        # Networks
        self.q_net = QNetwork(state_dim, action_dim).to(config.DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(config.DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=hyperparams["lr"])

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

        # Epsilon-greedy for training
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
                np.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() < eps:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)
                q_values = self.q_net(state)
                return q_values.argmax().item()

    def update_step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.update()

    def update(self):
        """Sample batch and update Q-network (Double DQN + N-step)"""
        
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        states = torch.from_numpy(batch.state).to(config.DEVICE)
        actions = torch.from_numpy(batch.action).unsqueeze(1).to(config.DEVICE)
        rewards = torch.from_numpy(batch.reward).to(config.DEVICE)
        next_states = torch.from_numpy(batch.next_state).to(config.DEVICE)
        dones = torch.from_numpy(batch.done).to(config.DEVICE)

        # Current Q estimates
        q_values = self.q_net(states).gather(1, actions).squeeze(1)

        # --- Double DQN target ---
        # 1. Select next action using online net
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)

        # 2. Evaluate action using target net
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        # 3. Compute target with n-step returns already handled by replay buffer
        expected_q = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(q_values, expected_q.detach())
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        """Update target network weights."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    @classmethod
    def get_default_hyperparams(cls) -> dict:
        return cls.DEFAULT_PARAMS.copy()
    
    def get_hyperparams(self) -> dict:
        return self.hyperparams.copy()
    
    def get_checkpoint(self):
        return self.checkpoint
    
    def save(self, path: str, extra: dict = None):
        self.checkpoint = {
            "agent_name": "nstep_ddqn",
            "model_state": self.q_net.state_dict(),
            "hyperparams": self.hyperparams,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if extra:
            self.checkpoint.update(extra)

        torch.save(self.checkpoint, path)

    def load(self, path: str):
        """
        Load model weights and hyperparameters from a checkpoint file.

        Args:
            path (str): path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=config.DEVICE)

        # Restore hyperparams (merge defaults + checkpoint)
        hyperparams = self.DEFAULT_PARAMS.copy()
        hyperparams.update(checkpoint.get("hyperparams", {}))
        self.hyperparams = hyperparams

        # Restore exploration settings
        self.eps_start = hyperparams["eps_start"]
        self.eps_end = hyperparams["eps_end"]
        self.eps_decay = hyperparams["eps_decay"]

        # Load model weights
        if "model_state" in checkpoint:
            self.q_net.load_state_dict(checkpoint["model_state"])
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Re-create optimizer with correct learning rate
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=hyperparams["lr"])
        self.q_net.eval()
