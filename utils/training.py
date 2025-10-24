import config
import numpy as np


def train(env, agent, episodes=config.DEFAULT_EPISODES,
          progress_cb=None, stop_flag=lambda: False,
          render=False):
    
    rewards = []
    losses = []

    for ep in range(episodes):
        state, _ = env.reset(seed=config.SEED)
        done = False
        episode_total_reward = 0
        mean_loss = 0
        i = 1  # Mean loss counter

        if stop_flag():
            break

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update_step(state, action, reward, next_state, done)
            state = next_state
            episode_total_reward += reward

            mean_loss = mean_loss + (1/i) * (agent.loss - mean_loss)

            if render:
                env.render()
            i += 1

        rewards.append(episode_total_reward)
        losses.append(mean_loss)

        # ✅ periodically update target net
        target_update = agent.hyperparams.get("target_update", config.TARGET_UPDATE)
        if (ep + 1) % target_update == 0:
            agent.update_target()

        # ✅ callback for Qt menu
        if progress_cb:
            progress_cb(ep, episodes, episode_total_reward, rewards, losses, agent.current_epsilon)

        agent.add_episode()

    return rewards


def train_episode(env, agent, stop_flag=lambda: False, render=False):
    """Run one training episode and return the total reward."""
    state, _ = env.reset(seed=config.SEED)
    done = False
    episode_total_reward = 0

    while not done:
        if stop_flag():
            break

        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.memory.push(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        episode_total_reward += reward

    if render and not stop_flag():
        env.render()

    return episode_total_reward
