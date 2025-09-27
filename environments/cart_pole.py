import gymnasium as gym
import config


def create_cart_pole_environment(render, sutton_barto_reward=False):
        # set correct render mode
        if render == "human":
            env = gym.make(config.CART_POLE_ENVIRONMENT, render_mode="human", sutton_barto_reward=sutton_barto_reward)
        elif render in ["gif", "mp4"]:
            env = gym.make(config.CART_POLE_ENVIRONMENT, render_mode="rgb_array", sutton_barto_reward=sutton_barto_reward)
        else:  # off
            env = gym.make(config.CART_POLE_ENVIRONMENT, sutton_barto_reward=sutton_barto_reward)

        state_dim, action_dim = env.observation_space.shape[0], env.action_space.n

        return env, state_dim, action_dim
