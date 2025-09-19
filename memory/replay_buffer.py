from collections import deque, namedtuple
import random


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class NStepReplayBuffer:
    def __init__(self, capacity, n_step=3, gamma=0.99):
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma
    
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
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return batch
    
    def __len__(self):
        return len(self.buffer)
