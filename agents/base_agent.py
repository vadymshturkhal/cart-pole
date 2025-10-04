from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
        Abstract base class for all RL agents.
        Defines the required interface for training, action selection, and saving.
    """

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def select_action(self, state, greedy: bool = False):
        """
        Choose an action given the current state.
        - If greedy=True, always choose the best action (no exploration).
        """
        pass

    @abstractmethod
    def update_step(self, state, action, reward, next_state, done):
        """
        Perform one learning step from the transition.
        - Replay-based agents: push to buffer, sample, update.
        - Online agents: update directly from the transition.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model parameters + hyperparameters to a file."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model parameters + hyperparameters from a file."""
        pass

    @classmethod
    @abstractmethod
    def get_default_hyperparams(cls):
        """Get default parameters specific for the agent"""
        pass

    @abstractmethod
    def get_hyperparams(self):
        """Get parameters specific for the agent in particular training section"""
        pass

    @abstractmethod
    def get_checkpoint(self):
        """Get the original checkpoint created right after training."""
        pass
