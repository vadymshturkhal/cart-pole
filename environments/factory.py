import gymnasium as gym
import config


def create_environment(env_name=config.DEFAULT_ENVIRONMENT, render='off'):
    """Generic environment factory, no special per-env kwargs."""

    if render == "human":
        env = gym.make(env_name, render_mode="human")
    elif render in ["gif", "mp4"]:
        env = gym.make(env_name, render_mode="rgb_array")
    else:  # off
        env = gym.make(env_name)

    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n

    return env, state_dim, action_dim
