import numpy as np
from memory.replay_buffer import NStepReplayBuffer, Transition


class PrioritizedReplayBuffer(NStepReplayBuffer):
    def __init__(self, capacity, n_step=3, gamma=0.99,
                 alpha=0.6, beta_start=0.4, beta_frames=100000):
        super().__init__(capacity, n_step, gamma)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.pos = 0
        self.priorities = np.ones((capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def sample(self, batch_size, step):
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        # Compute probabilities
        priorities = self.priorities[:len(self.buffer)]
        
        # Ensure nonzero and finite
        if priorities.max() == 0 or np.isnan(priorities).any():
            priorities += 1e-5

        probs = priorities ** self.alpha
        probs_sum = probs.sum()

        if probs_sum == 0 or np.isnan(probs_sum):
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        batch = Transition(*zip(*samples))

        # Importance-sampling weights
        total = len(self.buffer)
        beta = min(1.0, self.beta_start + step * (1.0 - self.beta_start) / self.beta_frames)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        # Convert to numpy arrays
        states = np.array(batch.state, dtype=np.float32)
        actions = np.array(batch.action)
        rewards = np.array(batch.reward, dtype=np.float32)
        next_states = np.array(batch.next_state, dtype=np.float32)
        dones = np.array(batch.done, dtype=np.float32)

        return (Transition(states, actions, rewards, next_states, dones),
                indices, weights)
    
    def update_priorities(self, indices, td_errors, eps=1e-5):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + eps

def push(self, *args):
    super().push(*args)
    self.priorities[self.pos - 1] = self.max_priority
