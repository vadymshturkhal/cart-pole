import torch, gymnasium as gym, os
import config
from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from utils.training import train
from utils.plotting import plot_rewards
from ui.menu import menu_loop
from ui.progress import progress_callback
from ui.test_menu import test_menu_loop
from utils.rendering import render_agent

def make_agent(agent_name, state_dim, action_dim):
    if agent_name == "nstep_dqn":
        return NStepDeepQLearningAgent(state_dim, action_dim)
    if agent_name == "nstep_ddqn":
        return NStepDoubleDeepQLearningAgent(state_dim, action_dim)
    raise ValueError(f"Unknown agent: {agent_name}")

def main():
    agent_name, render_mode = menu_loop()
    env = gym.make(config.ENV_NAME)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    agent = make_agent(agent_name, state_dim, action_dim)
    rewards = train(env, agent, episodes=config.EPISODES, progress_cb=progress_callback)
    os.makedirs(config.TRAINED_MODELS_FOLDER, exist_ok=True)
    model_path = f"{config.TRAINED_MODELS_FOLDER}/{agent_name}_qnet.pth"
    torch.save(agent.q_net.state_dict(), model_path)

    # After training:
    model_file, render_mode, episodes = test_menu_loop()
    model_path = os.path.join(config.TRAINED_MODELS_FOLDER, model_file)

    env = gym.make(config.ENV_NAME, render_mode="rgb_array" if render_mode in ["gif", "mp4"] else "human")
    agent = make_agent(agent_name, env.observation_space.shape[0], env.action_space.n)
    agent.q_net.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    agent.q_net.eval()

    render_agent(env, agent, mode=render_mode, episodes=episodes)
    env.close()


if __name__ == "__main__":
    main()
