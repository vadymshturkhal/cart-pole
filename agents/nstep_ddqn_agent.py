from .nstep_dqn_agent import NStepDeepQLearningAgent
import torch
import torch.nn as nn
import config


class NStepDoubleDeepQLearningAgent(NStepDeepQLearningAgent):
    """N-step Double DQN Agent inheriting from N-step DQN."""

    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.name = "nstep_ddqn"

    def update(self):
        """Sample batch and update Q-network (Double DQN + N-step)"""
        
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        states = torch.from_numpy(batch.state).to(config.DEVICE)
        actions = torch.from_numpy(batch.action).unsqueeze(1).to(config.DEVICE)
        rewards = torch.from_numpy(batch.reward).to(config.DEVICE)
        next_states = torch.from_numpy(batch.next_state).to(config.DEVICE)
        dones = torch.from_numpy(batch.done).to(config.DEVICE)

        # Current Q estimates
        q_values = self.q_net(states).gather(1, actions).squeeze(1)

        # --- Double DQN target ---
        # 1. Select next action using online net
        next_actions = self.target_net(next_states).argmax(1, keepdim=True)

        # 2. Evaluate action using target net
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        # 3. Compute target with n-step returns already handled by replay buffer
        expected_q = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values

        # Loss
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(q_values, expected_q.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        self._losses.append(loss.item())
