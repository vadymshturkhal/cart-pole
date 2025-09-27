import gymnasium as gym


def create_environment(env_name, render='off', sutton_barto_reward=False):
    if render == "human":
        env = gym.make(env_name, render_mode="human", sutton_barto_reward=sutton_barto_reward)
    elif render in ["gif", "mp4"]:
        env = gym.make(env_name, render_mode="rgb_array", sutton_barto_reward=sutton_barto_reward)
    else:  # off
        env = gym.make(env_name, sutton_barto_reward=sutton_barto_reward)

    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n

    return env, state_dim, action_dim
