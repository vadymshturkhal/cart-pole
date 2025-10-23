from .nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from memory.prioritized_replay_buffer import PrioritizedReplayBuffer
import torch
import torch.nn as nn
import numpy as np
import config


class RainbowAgent(NStepDoubleDeepQLearningAgent):
    "Rainbow Agent"

    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.name = "rainbow"

        hyperparams = self.DEFAULT_PARAMS.copy()
        self.memory = PrioritizedReplayBuffer(hyperparams["buffer_size"], hyperparams["n_step"], self.gamma)


    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch, indices, weights = self.memory.sample(self.batch_size)

        states = torch.from_numpy(batch.state).to(config.DEVICE)
        actions = torch.from_numpy(batch.action).unsqueeze(1).to(config.DEVICE)
        rewards = torch.from_numpy(batch.reward).to(config.DEVICE)
        next_states = torch.from_numpy(batch.next_state).to(config.DEVICE)
        dones = torch.from_numpy(batch.done).to(config.DEVICE)
        weights = torch.from_numpy(weights).to(config.DEVICE)

        q_values = self.q_net(states).gather(1, actions).squeeze(1)

        # Double DQN target
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        expected_q = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = q_values - expected_q.detach()

        # Weighted MSE
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        self._losses.append(loss.item())