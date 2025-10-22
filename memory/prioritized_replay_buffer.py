import numpy as np
from memory.replay_buffer import NStepReplayBuffer, Transition


class PrioritizedReplayBuffer(NStepReplayBuffer):
    def __init__(self, capacity, n_step=3, gamma=0.99,
                 alpha=0.6, beta_start=0.4, beta_frames=100000):
        super().__init__(capacity, n_step, gamma)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.capacity = capacity
        self.priorities = np.ones((capacity,), dtype=np.float32)
        self.max_priority = 1.0
        self.pos = 0  # explicit write pointer

    def push(self, *args):
        super().push(*args)
        # update position pointer
        if len(self.buffer) == self.capacity:
            idx = self.pos
        else:
            idx = len(self.buffer) - 1
        self.priorities[idx] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, step):
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        # valid priorities slice
        valid_priorities = self.priorities[:len(self.buffer)]
        valid_priorities = np.where(valid_priorities <= 0, 1e-5, valid_priorities)

        probs = valid_priorities ** self.alpha
        probs /= probs.sum() if probs.sum() > 0 else len(valid_priorities)

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        batch = Transition(*zip(*samples))

        total = len(self.buffer)
        beta = min(1.0, self.beta_start + step * (1.0 - self.beta_start) / self.beta_frames)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max() if weights.max() > 0 else 1.0

        states = np.array(batch.state, dtype=np.float32)
        actions = np.array(batch.action)
        rewards = np.array(batch.reward, dtype=np.float32)
        next_states = np.array(batch.next_state, dtype=np.float32)
        dones = np.array(batch.done, dtype=np.float32)

        return (Transition(states, actions, rewards, next_states, dones),
                indices, weights)

    def update_priorities(self, indices, td_errors, eps=1e-5):
        td_errors = np.abs(td_errors) + eps
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(err)
        self.max_priority = max(self.max_priority, td_errors.max())

    def __len__(self):
        return len(self.buffer)
