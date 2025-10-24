from collections import deque, namedtuple
import numpy as np
import config


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class NStepReplayBuffer:
    def __init__(self, capacity, n_step=3, gamma=0.99):
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma
        self.rng = np.random.default_rng(config.SEED)
    
    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        
        reward, next_state, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[2], transition[3], transition[4]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        return reward, next_state, done
    
    def sample(self, batch_size):
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[idx] for idx in indices]
        batch = Transition(*zip(*transitions))

        # Numpy arrays
        states = np.array(batch.state, dtype=np.float32)
        actions = np.array(batch.action)
        rewards = np.array(batch.reward, dtype=np.float32)
        next_states = np.array(batch.next_state, dtype=np.float32)
        dones = np.array(batch.done, dtype=np.float32)

        return Transition(states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.buffer)
