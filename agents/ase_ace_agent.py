from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np


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
        # Precompute edges for uniform binning
        def edges(lo, hi, k):
            return np.linspace(lo, hi, k + 1)
        self._x_edges = edges(*self.x_bounds, self.x_bins)
        self._xdot_edges = edges(*self.xdot_bounds, self.xdot_bins)
        self._th_edges = edges(*self.th_bounds, self.th_bins)
        self._thdot_edges = edges(*self.thdot_bounds, self.thdot_bins)

    def _bin_index(self, val: float, edges: np.ndarray) -> int:
        # Place values below/above range into edge bins
        idx = int(np.digitize([val], edges[1:-1])[0])
        return max(0, min(idx, len(edges) - 2))

    def encode_index(self, s: np.ndarray) -> int:
        # s = [x, xdot, theta, thetadot]
        bx = self._bin_index(float(s[0]), self._x_edges)
        bxd = self._bin_index(float(s[1]), self._xdot_edges)
        bth = self._bin_index(float(s[2]), self._th_edges)
        bthd = self._bin_index(float(s[3]), self._thdot_edges)
        # map 4D indices to flat index
        idx = (((bx * self.xdot_bins) + bxd) * self.th_bins + bth) * self.thdot_bins + bthd
        return int(idx)

    def encode_one_hot(self, s: np.ndarray) -> np.ndarray:
        idx = self.encode_index(s)
        phi = np.zeros(self.dim, dtype=np.float32)
        phi[idx] = 1.0
        return phi

# -------------------------------------------------------------
# Actor–Critic with eligibility traces (softmax actor, linear critic)
# -------------------------------------------------------------

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
    

class ASEACEAgent:
    """Thin wrapper exposing a simple Agent API with `select_action` and `update`.

    - Encodes raw state with the 162-BOXES discretizer
    - Uses the Actor–Critic core with eligibilities under the hood
    """
    def __init__(self, cfg: ACConfig, seed: int = 0):
        self.cfg = cfg
        self.disc = BoxDiscretizer()
        self.core = ActorCriticASEACE(self.disc.dim, cfg)
        self.rng = np.random.default_rng(seed)

    # -------- public API --------
    def select_action(self, state: np.ndarray) -> int:
        """Return an action (0=Left, 1=Right) given the raw environment state."""
        phi = self.disc.encode_one_hot(np.asarray(state, dtype=np.float32))
        return self.core.sample_action(phi, self.rng)

    def update_step(self, state, action, reward, next_state, done):
        # For actor–critic: directly update online, no replay buffer
        self.update(state, action, reward, next_state, done)
        
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        """Single-step actor–critic update. Returns TD error δ.
        Accepts raw states; handles feature encoding internally.
        """
        phi_t = self.disc.encode_one_hot(np.asarray(state, dtype=np.float32))
        if done:
            phi_tp1 = np.zeros_like(phi_t)
        else:
            phi_tp1 = self.disc.encode_one_hot(np.asarray(next_state, dtype=np.float32))
        return self.core.step(phi_t, action, reward, phi_tp1, done)

    def reset_traces(self):
        self.core.e_w.fill(0.0)
        self.core.e_theta.fill(0.0)

    # Optional helpers
    def value(self, state: np.ndarray) -> float:
        phi = self.disc.encode_one_hot(np.asarray(state, dtype=np.float32))
        return self.core.value(phi)
    
    def update_target(self):
        """Update target network weights."""
        pass
