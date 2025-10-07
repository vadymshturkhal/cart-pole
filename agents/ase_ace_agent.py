from __future__ import annotations
from copy import deepcopy
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import torch
from agents.base_agent import BaseAgent
from datetime import datetime


@dataclass
class BoxDiscretizer:
    # bounds roughly aligned with CartPole termination; velocity bounds are conventional
    x_bounds: Tuple[float, float] = (-2.4, 2.4)
    xdot_bounds: Tuple[float, float] = (-3.0, 3.0)
    th_bounds: Tuple[float, float] = (-(12 * math.pi / 180), (12 * math.pi / 180))  # ±12°
    thdot_bounds: Tuple[float, float] = (-3.5, 3.5)

    x_bins: int = 3
    xdot_bins: int = 3
    th_bins: int = 6
    thdot_bins: int = 3

    def __post_init__(self):
        self.dim = self.x_bins * self.xdot_bins * self.th_bins * self.thdot_bins  # 162
        def edges(lo, hi, k): return np.linspace(lo, hi, k + 1)
        self._x_edges = edges(*self.x_bounds, self.x_bins)
        self._xdot_edges = edges(*self.xdot_bounds, self.xdot_bins)
        self._th_edges = edges(*self.th_bounds, self.th_bins)
        self._thdot_edges = edges(*self.thdot_bounds, self.thdot_bins)

    def _bin_index(self, val: float, edges: np.ndarray) -> int:
        idx = int(np.digitize([val], edges[1:-1])[0])
        return max(0, min(idx, len(edges) - 2))

    def encode_index(self, s: np.ndarray) -> int:
        bx  = self._bin_index(float(s[0]), self._x_edges)
        bxd = self._bin_index(float(s[1]), self._xdot_edges)
        bth = self._bin_index(float(s[2]), self._th_edges)
        bthd= self._bin_index(float(s[3]), self._thdot_edges)
        return int((((bx * self.xdot_bins) + bxd) * self.th_bins + bth) * self.thdot_bins + bthd)

    def encode_one_hot(self, s: np.ndarray) -> np.ndarray:
        idx = self.encode_index(s)
        phi = np.zeros(self.dim, dtype=np.float32)
        phi[idx] = 1.0
        return phi


@dataclass
class ACConfig:
    gamma: float = 0.99
    lam_v: float = 0.8
    lam_pi: float = 0.8
    alpha_v: float = 0.2   # critic step size (per-feature scale for sparse features)
    alpha_pi: float = 0.05 # actor step size
    max_steps: int = 2000  # safety cap per episode
    sparse_reward: bool = False


class ActorCriticASEACE:
    def __init__(self, feat_dim: int, cfg: ACConfig):
        self.cfg = cfg
        self.A = 2  # Left, Right
        self.w = np.zeros(feat_dim, dtype=np.float32)                # critic weights
        self.theta = np.zeros((self.A, feat_dim), dtype=np.float32)  # actor preferences
        self.e_w = np.zeros_like(self.w)
        self.e_theta = np.zeros_like(self.theta)

    def value(self, phi: np.ndarray) -> float:
        return float(self.w @ phi)

    def logits(self, phi: np.ndarray) -> np.ndarray:
        return self.theta @ phi  # shape (2,)

    def policy(self, phi: np.ndarray) -> np.ndarray:
        z = self.logits(phi) - np.max(self.logits(phi))
        expz = np.exp(z)
        return expz / np.sum(expz)

    def sample_action(self, phi: np.ndarray, rng: np.random.Generator) -> int:
        p = self.policy(phi)
        return int(rng.choice(self.A, p=p))

    def best_action(self, phi: np.ndarray) -> int:
        return int(np.argmax(self.logits(phi)))

    def step(self, phi_t: np.ndarray, a_t: int, r_tp1: float, phi_tp1: np.ndarray, done: bool):
        cfg = self.cfg
        v_t = self.value(phi_t)
        v_tp1 = 0.0 if done else self.value(phi_tp1)
        delta = r_tp1 + cfg.gamma * v_tp1 - v_t

        # Critic traces
        self.e_w = cfg.gamma * cfg.lam_v * self.e_w + phi_t
        self.w += cfg.alpha_v * delta * self.e_w

        # Actor traces (compatible with softmax/REINFORCE with baseline)
        pi_t = self.policy(phi_t)
        grad_logp = -pi_t[:, None] * phi_t[None, :]
        grad_logp[a_t] += phi_t
        self.e_theta = cfg.gamma * cfg.lam_pi * self.e_theta + grad_logp
        self.theta += cfg.alpha_pi * delta * self.e_theta

        return delta


class ASEACEAgent(BaseAgent):
    """
        NumPy-based Actor–Critic with eligibility traces, using a 162-dim one-hot
        discretization of CartPole state. 
        Conforms to BaseAgent for GUI integration.
        Agent is from Barto, Sutton, and Anderson (1983) for solving pole-balancing problem.
    """

    DEFAULT_PARAMS = {
        "gamma": 0.99,
        "lam_v": 0.8,
        "lam_pi": 0.8,
        "alpha_v": 0.2,
        "alpha_pi": 0.05,
        "max_steps": 2000,
        "sparse_reward": False,
        "seed": 0,
        # no epsilon here (stochastic policy); we’ll add eval_deterministic flag
        "eval_deterministic": False,
    }

    def __init__(self, state_dim=None, action_dim=None, **kwargs):
        super().__init__(state_dim, action_dim)
        hyperparams = self.DEFAULT_PARAMS.copy()
        hyperparams.update(kwargs)
        self.hyperparams = hyperparams
        self.checkpoint = {}

        self.cfg = ACConfig(
            gamma=hyperparams["gamma"],
            lam_v=hyperparams["lam_v"],
            lam_pi=hyperparams["lam_pi"],
            alpha_v=hyperparams["alpha_v"],
            alpha_pi=hyperparams["alpha_pi"],
            max_steps=hyperparams["max_steps"],
            sparse_reward=hyperparams["sparse_reward"],
        )
        self.disc = BoxDiscretizer()
        self.core = ActorCriticASEACE(self.disc.dim, self.cfg)
        self.rng = np.random.default_rng(hyperparams["seed"])

    # ---- BaseAgent API ----
    @classmethod
    def get_default_hyperparams(cls) -> Dict[str, Any]:
        return cls.DEFAULT_PARAMS.copy()

    def get_hyperparams(self) -> Dict[str, Any]:
        return self.hyperparams.copy()

    def set_hyperparams(self, updates: Dict[str, Any]) -> None:
        # Only allow "safe" updates at runtime
        for k, v in updates.items():
            if k not in self.hyperparams:
                raise ValueError(f"Unknown parameter: {k}")
            if k in ["gamma", "lam_v", "lam_pi", "alpha_v", "alpha_pi", "max_steps", "sparse_reward"]:
                # Requires re-creating cfg / traces for full correctness. Keep it simple:
                self.hyperparams[k] = v
            elif k in ["seed", "eval_deterministic"]:
                self.hyperparams[k] = v
            else:
                self.hyperparams[k] = v
        # If core-level params changed, re-init minimal parts:
        self.cfg = ACConfig(
            gamma=self.hyperparams["gamma"],
            lam_v=self.hyperparams["lam_v"],
            lam_pi=self.hyperparams["lam_pi"],
            alpha_v=self.hyperparams["alpha_v"],
            alpha_pi=self.hyperparams["alpha_pi"],
            max_steps=self.hyperparams["max_steps"],
            sparse_reward=self.hyperparams["sparse_reward"],
        )

    def select_action(self, state, greedy=False) -> int:
        phi = self.disc.encode_one_hot(np.asarray(state, dtype=np.float32))
        if self.hyperparams.get("eval_deterministic", True):
            return self.core.best_action(phi)
        return self.core.sample_action(phi, self.rng)

    def update_step(self, state, action, reward, next_state, done):
        phi_t = self.disc.encode_one_hot(np.asarray(state, dtype=np.float32))
        phi_tp1 = np.zeros_like(phi_t) if done else self.disc.encode_one_hot(np.asarray(next_state, dtype=np.float32))
        return self.core.step(phi_t, action, reward, phi_tp1, done)

    def get_checkpoint(self):
        return deepcopy(self.checkpoint)
    
    def update_target(self):
        pass

    def save(self, path: str, extra: dict | None = None) -> None:
        self.checkpoint = {
            "agent_name": "ase_ace",
            "hyperparams": self.hyperparams,
            # numpy arrays serialize fine with torch.save
            "w": self.core.w,
            "theta": self.core.theta,
            "e_w": self.core.e_w,
            "e_theta": self.core.e_theta,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if extra:
            self.checkpoint.update(extra)

        torch.save(self.checkpoint, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")  # this agent is CPU/NumPy
        hyperparams = self.DEFAULT_PARAMS.copy()
        hyperparams.update(checkpoint.get("hyperparams", {}))
        self.hyperparams = hyperparams
        # rebuild config & core (dim must match discretizer)
        self.cfg = ACConfig(
            gamma=hyperparams["gamma"],
            lam_v=hyperparams["lam_v"],
            lam_pi=hyperparams["lam_pi"],
            alpha_v=hyperparams["alpha_v"],
            alpha_pi=hyperparams["alpha_pi"],
            max_steps=hyperparams["max_steps"],
            sparse_reward=hyperparams["sparse_reward"],
        )
        self.core = ActorCriticASEACE(self.disc.dim, self.cfg)

        # For testing
        self.hyperparams.update({"eval_deterministic": True})

        for name in ["w", "theta", "e_w", "e_theta"]:
            if name in checkpoint:
                getattr(self.core, name)[:] = checkpoint[name]
        return self
