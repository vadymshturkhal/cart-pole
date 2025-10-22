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
        self.pos = 0
        self.frame = 0  # used if step is not provided

    def push(self, *args):
        super().push(*args)
        idx = self.pos if len(self.buffer) == self.capacity else len(self.buffer) - 1
        self.priorities[idx] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, step=None):
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        valid_n = len(self.buffer)
        prios = self.priorities[:valid_n]
        prios = np.where(np.isfinite(prios) & (prios > 0), prios, 1e-5)

        probs = prios ** self.alpha
        psum = probs.sum()
        probs = probs / psum if psum > 0 else np.full(valid_n, 1.0 / valid_n, dtype=np.float32)

        indices = np.random.choice(valid_n, batch_size, p=probs)
        batch = Transition(*zip(*[self.buffer[i] for i in indices]))

        # anneal beta
        if step is None:
            self.frame += 1
            step = self.frame
        beta = min(1.0, self.beta_start + step * (1.0 - self.beta_start) / self.beta_frames)

        weights = (valid_n * probs[indices]) ** (-beta)
        wmax = weights.max()
        weights = weights / wmax if wmax > 0 else weights

        states = np.array(batch.state, dtype=np.float32)
        actions = np.array(batch.action)
        rewards = np.array(batch.reward, dtype=np.float32)
        next_states = np.array(batch.next_state, dtype=np.float32)
        dones = np.array(batch.done, dtype=np.float32)

        # return Transition(states, actions, rewards, next_states, dones), indices, weights.astype(np.float32)

        # Removed indices, weights.astype(np.float32) for consistency with NStepReplayBuffer
        return Transition(states, actions, rewards, next_states, dones)

    def update_priorities(self, indices, td_errors, eps=1e-5):
        td = np.abs(td_errors) + eps
        td = np.where(np.isfinite(td), td, eps)
        for i, e in zip(indices, td):
            self.priorities[i] = float(e)
        self.max_priority = max(self.max_priority, float(td.max()))

    def __len__(self):
        return len(self.buffer)
