import config


def train(env, agent, episodes=config.DEFAULT_EPISODES,
          rewards_file=config.REWARDS_FILE,
          progress_cb=None, stop_flag=lambda: False,
          render=False):
    
    rewards = []
    losses = []

    for ep in range(episodes):
        state, _ = env.reset(seed=config.SEED)
        done = False
        total_reward = 0
        total_loss = 0

        if stop_flag():
            break

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update_step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            total_loss += agent.loss

            if render:
                env.render()

        rewards.append(total_reward)
        losses.append(total_loss)

        # ✅ periodically update target net
        if (ep + 1) % config.TARGET_UPDATE == 0:
            agent.update_target()

        # ✅ callback for Qt / pygame menus
        if progress_cb:
            progress_cb(ep, episodes, total_reward, rewards, losses, agent.current_epsilon)

        agent.add_episode()

    return rewards


def train_episode(env, agent, stop_flag=lambda: False, render=False):
    """Run one training episode and return the total reward."""
    state, _ = env.reset(seed=config.SEED)
    done = False
    total_reward = 0

    while not done:
        if stop_flag():
            break

        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.memory.push(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        total_reward += reward

    if render and not stop_flag():
        env.render()

    return total_reward
