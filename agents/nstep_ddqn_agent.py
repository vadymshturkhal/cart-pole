import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from .q_network import QNetwork
from memory.replay_buffer import NStepReplayBuffer
import config


class NStepDoubleDeepQLearningAgent:
    def __init__(self, state_dim, action_dim):
        # Networks
        self.q_net = QNetwork(state_dim, action_dim).to(config.DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(config.DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.LR)

        # Replay buffer (use n-step > 1)
        self.memory = NStepReplayBuffer(config.BUFFER_SIZE, n_step=config.N_STEP, gamma=config.GAMMA)

        # Hyperparameters
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self.action_dim = action_dim
        self.steps_done = 0

    def select_action(self, state, greedy: bool = False):
        """Choose an action (epsilon-greedy for training, greedy-only for eval)."""
        if greedy:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)
                q_values = self.q_net(state)
                return q_values.argmax().item()

        # Epsilon-greedy for training
        eps = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * \
              np.exp(-1. * self.steps_done / config.EPSILON_DECAY)
        self.steps_done += 1

        if random.random() < eps:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)
                q_values = self.q_net(state)
                return q_values.argmax().item()

    def update(self):
        """Sample batch and update Q-network (Double DQN + N-step)."""
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(batch.state).to(config.DEVICE)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(config.DEVICE)
        rewards = torch.FloatTensor(batch.reward).to(config.DEVICE)
        next_states = torch.FloatTensor(batch.next_state).to(config.DEVICE)
        dones = torch.FloatTensor(batch.done).to(config.DEVICE)

        # Current Q estimates
        q_values = self.q_net(states).gather(1, actions).squeeze(1)

        # --- Double DQN target ---
        # 1. Select next action using online net
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)

        # 2. Evaluate action using target net
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        # 3. Compute target with n-step returns already handled by replay buffer
        expected_q = rewards + (1 - dones) * (self.gamma ** config.N_STEP) * next_q_values

        # Loss
        loss = nn.MSELoss()(q_values, expected_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        """Update target network weights."""
        self.target_net.load_state_dict(self.q_net.state_dict())
