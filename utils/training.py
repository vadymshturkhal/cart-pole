import config
import pandas as pd


def train(env, agent, episodes=config.EPISODES, rewards_file=config.REWARDS_FILE, progress_cb=None, stop_flag=lambda: False):
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset(seed=config.SEED)
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            agent.update()  # still guarded by batch_size in agent
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        # ✅ periodically update target net
        if (ep + 1) % config.TARGET_UPDATE == 0:
            agent.update_target()

        # ✅ callback for pygame
        if progress_cb:
            progress_cb(ep, episodes, total_reward, rewards)

        if ep % config.LOG_AFTER_EPISODES == 0:
            print(f"Episode {ep}, Reward: {total_reward}")

    # ✅ Save rewards to CSV (path in config)
    df = pd.DataFrame({"episode": range(len(rewards)), "reward": rewards})
    df.to_csv(config.REWARDS_FILE, index=False)
    print(f"✅ Rewards saved to {config.REWARDS_FILE}")

    return rewards

def train_episode(env, agent):
    """Run one training episode and return the total reward."""
    state, _ = env.reset(seed=config.SEED)
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.memory.push(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        total_reward += reward

    return total_reward
