# FIXME
# EXPERIMENTAL

from copy import deepcopy
import torch
import numpy as np
import random
from agents.base_agent import BaseAgent
from .dueling_q_network import DuelingQNetwork
from memory.prioritized_replay_buffer import PrioritizedReplayBuffer
import config
from datetime import datetime
from utils.optim_factory import build_optimizer


class RainbowAgent(BaseAgent):
    DEFAULT_PARAMS = {
        "gamma": config.GAMMA,
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
            buffer_size (int): replay buffer capacity
            batch_size (int): minibatch size
            n_step (int): N-step horizon
            eps_start (float): starting epsilon for exploration
            eps_end (float): final epsilon
            eps_decay (int): decay factor for epsilon
        """

        self.name = "rainbow"
        self.state_dim = state_dim
        self.action_dim = action_dim

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
        self.memory = PrioritizedReplayBuffer(hyperparams["buffer_size"], hyperparams["n_step"], self.gamma)

        # Networks
        self.q_net = DuelingQNetwork(state_dim, action_dim).to(config.DEVICE)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(config.DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = build_optimizer(config.OPTIMIZER, self.q_net.parameters(), lr=config.LR)

        # Loss
        self.losses_ = []

        self.total_steps = 0

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
        self.total_steps += 1
        self.memory.push(state, action, reward, next_state, done)
        self.update()

    def update(self):
        """Sample batch and update Q-network (Double DQN + N-step)"""
        
        if len(self.memory) < self.batch_size:
            return

        batch, indices, weights = self.memory.sample(self.batch_size, self.total_steps)

        states = torch.from_numpy(batch.state).to(config.DEVICE)
        actions = torch.from_numpy(batch.action).unsqueeze(1).to(config.DEVICE)
        rewards = torch.from_numpy(batch.reward).to(config.DEVICE)
        next_states = torch.from_numpy(batch.next_state).to(config.DEVICE)
        dones = torch.from_numpy(batch.done).to(config.DEVICE)

        # Current Q estimates
        q_values = self.q_net(states).gather(1, actions).squeeze(1)

        # 1. Select next action using online net
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)

        # 2. Evaluate action using target net
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        # 3. Compute target with n-step returns already handled by replay buffer
        expected_q = rewards + (1 - dones) * self.gamma * next_q_values

        # Priority loss
        self.optimizer.zero_grad()
        td_error = q_values - expected_q.detach()
        td_errors = td_error.detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        loss = (torch.FloatTensor(weights).to(config.DEVICE) * td_error.pow(2)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        self.losses_.append(loss.item())

    def update_target(self):
        """Update target network weights."""
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
        self.checkpoint = {
            "agent_name": "rainbow",
            "model_state": self.q_net.state_dict(),
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

    def _infer_arch_from_keys(self, model_state: dict) -> str:
        keys = list(model_state.keys())
        if any(k.startswith(("value.", "advantage.", "feature.")) for k in keys):
            return "DuelingQNetwork"
        if any(k.startswith("net.") for k in keys):
            return "QNetwork"
        return "QNetwork"  # safe default

    def load(self, path: str, hyperparams: dict = None):
        checkpoint = torch.load(path, map_location=config.DEVICE)

        # 1) Decide architecture
        arch_name = checkpoint.get("architecture")
        if arch_name is None:
            arch_name = self._infer_arch_from_keys(checkpoint["model_state"])
        print(f"🧠 Loading RainbowAgent with architecture: {arch_name}")

        # 2) Decide dims
        state_dim = checkpoint.get("state_dim", getattr(self, "state_dim", None))
        action_dim = checkpoint.get("action_dim", getattr(self, "action_dim", None))
        if state_dim is None or action_dim is None:
            raise ValueError("Checkpoint missing state_dim/action_dim and agent has none set.")

        # 3) Build correct net class
        if arch_name == "DuelingQNetwork":
            from .dueling_q_network import DuelingQNetwork as NetClass
        else:
            from .q_network import QNetwork as NetClass

        self.q_net = NetClass(state_dim, action_dim).to(config.DEVICE)
        self.target_net = NetClass(state_dim, action_dim).to(config.DEVICE)

        # 4) Load weights (non-strict so minor diffs don’t crash)
        incompatible = self.q_net.load_state_dict(checkpoint["model_state"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"⚠️ Partial load. missing={len(incompatible.missing_keys)} "
                f"unexpected={len(incompatible.unexpected_keys)}")

        self.target_net.load_state_dict(self.q_net.state_dict())

        # 5) Optimizer (rebuild if shape mismatch)
        try:
            if "optimizer_state" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as e:
            print(f"⚠️ Optimizer state incompatible, resetting optimizer. ({e})")
            self.optimizer = build_optimizer(config.OPTIMIZER, self.q_net.parameters(), lr=config.LR)

        # 6) Restore misc
        self.hyperparams = checkpoint.get("hyperparams", self.hyperparams)

        if hyperparams:
            self.hyperparams.update(hyperparams)

        self.steps_done = checkpoint.get("steps_done", 0)
        self.total_steps = checkpoint.get("total_steps", 0)

        self.q_net.eval()
        print("✅ Loaded successfully.")


    @property
    def losses(self):
        return self.losses_.copy()
    
    def clear_losses(self):
        self.losses_.clear()
