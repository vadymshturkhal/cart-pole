from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from agents.base_agent import BaseAgent
from agents.nn_blocks import ActorMLP, CriticMLP
from memory.replay_buffer import NStepReplayBuffer  # we’ll use n_step=1 (plain replay)
import config


class OUNoise:
    """Classic OU noise for exploration in continuous control."""
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.x = np.ones(self.action_dim, dtype=np.float32) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.action_dim).astype(np.float32)
        self.x = self.x + dx
        return self.x


class DDPGAgent(BaseAgent):
    """
    Deterministic Policy Gradient for continuous actions.
    Conforms to BaseAgent.
    """
    DEFAULT_PARAMS = {
        # Core
        "gamma": config.GAMMA,
        "tau": 0.005,             # soft target update rate
        "lr_actor": 1e-3,
        "lr_critic": 1e-3,
        "batch_size": config.BATCH_SIZE,
        "buffer_size": 100_000,
        "learning_starts": 5_000, # steps before updates
        "train_freq": 1,          # env steps per gradient step
        "clip_grad": 1.0,

        # Network
        "hidden_layers": config.HIDDEN_LAYERS,
        "activation": config.ACTIVATION,  # "relu" or "tanh"
        "dropout": config.DROPOUT,

        # Exploration
        "use_ou": True,
        "ou_theta": 0.15,
        "ou_sigma": 0.2,
        "action_noise_std": 0.1,  # used if not OU

        # Action scaling (IMPORTANT for Pendulum: max_action=2.0)
        "max_action": 1.0,
        "min_action": -1.0,

        # N-step behavior for the shared buffer
        "n_step": 1,  # DDPG uses 1-step TD target
    }

    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim)

        # Hyperparams
        hp = self.DEFAULT_PARAMS.copy()
        hp.update(kwargs)
        self.hyperparams = hp
        self.checkpoint = {}

        self.device = config.DEVICE

        # Replay buffer (n_step=1 behaves like standard replay)
        self.memory = NStepReplayBuffer(hp["buffer_size"], n_step=hp["n_step"], gamma=hp["gamma"])

        # Networks
        self.actor = ActorMLP(
            state_dim, action_dim,
            hidden_layers=hp["hidden_layers"],
            activation=hp["activation"],
            dropout=hp["dropout"],
            out_tanh=True,                    # actor outputs in [-1,1]
        ).to(self.device)

        self.actor_target = ActorMLP(
            state_dim, action_dim,
            hidden_layers=hp["hidden_layers"],
            activation=hp["activation"],
            dropout=hp["dropout"],
            out_tanh=True,
        ).to(self.device)

        self.critic = CriticMLP(
            state_dim, action_dim,
            hidden_layers=hp["hidden_layers"],
            activation=hp["activation"],
            dropout=hp["dropout"],
        ).to(self.device)

        self.critic_target = CriticMLP(
            state_dim, action_dim,
            hidden_layers=hp["hidden_layers"],
            activation=hp["activation"],
            dropout=hp["dropout"],
        ).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=hp["lr_actor"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=hp["lr_critic"])

        # Exploration
        self.total_steps = 0
        self.max_action = float(hp["max_action"])
        self.min_action = float(hp["min_action"])
        if hp["use_ou"]:
            self.ou = OUNoise(action_dim, theta=hp["ou_theta"], sigma=hp["ou_sigma"])
        else:
            self.ou = None
            self.gauss_std = float(hp["action_noise_std"])

    # ===== BaseAgent API =====
    @classmethod
    def get_default_hyperparams(cls):
        return cls.DEFAULT_PARAMS.copy()

    def get_hyperparams(self):
        return self.hyperparams.copy()

    def get_checkpoint(self):
        return self.checkpoint

    def select_action(self, state, greedy: bool = False):
        """
        Returns a NumPy action of shape (action_dim,).
        Training: add exploration noise, then scale to [min_action, max_action].
        """
        self.actor.eval()
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.actor(s).cpu().numpy()[0]  # in [-1,1] because of tanh head

        if not greedy:
            if self.ou is not None:
                a = a + self.ou.sample()
            else:
                a = a + np.random.normal(0.0, self.gauss_std, size=a.shape).astype(np.float32)

        # scale to env bounds
        a_scaled = self._scale_action_from_unit(a)
        return a_scaled.astype(np.float32)

    def update_step(self, state, action, reward, next_state, done):
        """
        Store transition and, if ready, perform gradient updates.
        Expect `action` as np.array shape (action_dim,) in env’s scale.
        """
        # store unscaled? we store scaled env action—consistent on both critic and target
        self.memory.push(state, action, reward, next_state, done)

        # Learn only after warmup
        self.total_steps += 1
        if self.total_steps < self.hyperparams["learning_starts"]:
            return

        # Train at specified frequency
        if (self.total_steps % self.hyperparams["train_freq"]) != 0:
            return

        if len(self.memory) < self.hyperparams["batch_size"]:
            return

        self._update()

    def save(self, path: str, extra: dict | None = None):
        self.checkpoint = {
            "agent_name": "ddpg",
            "hyperparams": self.hyperparams,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            self.checkpoint.update(extra)
        torch.save(self.checkpoint, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        hp = self.DEFAULT_PARAMS.copy()
        hp.update(ckpt.get("hyperparams", {}))
        self.hyperparams = hp

        # Recreate networks (in case shapes changed)
        self.__init__(self.state_dim, self.action_dim, **hp)  # re-init with hp
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor.eval(); self.actor_target.eval()
        return self

    # ===== internals =====
    def _scale_action_from_unit(self, a_unit):
        # a_unit ~ [-1,1]; scale to [min_action, max_action]
        lo, hi = self.min_action, self.max_action
        return lo + (0.5 * (a_unit + 1.0) * (hi - lo))

    def _scale_action_to_unit(self, a_scaled):
        # env action -> [-1,1] (only needed if you stored unscaled; we store scaled already)
        lo, hi = self.min_action, self.max_action
        return np.clip(2.0 * (a_scaled - lo) / (hi - lo) - 1.0, -1.0, 1.0)

    def _update(self):
        hp = self.hyperparams
        batch = self.memory.sample(hp["batch_size"])

        states      = torch.from_numpy(batch.state).float().to(self.device)
        actions     = torch.from_numpy(batch.action).float().to(self.device)
        rewards     = torch.from_numpy(batch.reward).unsqueeze(1).float().to(self.device)
        next_states = torch.from_numpy(batch.next_state).float().to(self.device)
        dones       = torch.from_numpy(batch.done).unsqueeze(1).float().to(self.device)

        # Target actions and Q-values
        with torch.no_grad():
            next_actions = self.actor_target(next_states)  # in [-1,1]; scaling is implicit in critic if needed
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + (1.0 - dones) * hp["gamma"] * target_q

        # --- Critic update ---
        self.critic_opt.zero_grad(set_to_none=True)
        q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q, y)
        critic_loss.backward()
        if hp["clip_grad"] is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), hp["clip_grad"])
        self.critic_opt.step()

        # --- Actor update ---
        self.actor_opt.zero_grad(set_to_none=True)
        # maximize Q(s, actor(s))  <=>  minimize -Q(...)
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        actor_loss.backward()
        if hp["clip_grad"] is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), hp["clip_grad"])
        self.actor_opt.step()

        # --- Soft update targets ---
        self._soft_update(self.actor_target, self.actor, hp["tau"])
        self._soft_update(self.critic_target, self.critic, hp["tau"])

    @staticmethod
    def _soft_update(target_net, online_net, tau):
        with torch.no_grad():
            for tp, p in zip(target_net.parameters(), online_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

    def update_target(self):
        """Optional hard update, if you prefer."""
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
