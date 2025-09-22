import torch, gymnasium as gym, os
import config
from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from utils.training import train
from utils.plotting import plot_rewards
from ui.menu import menu_loop
from ui.progress import progress_callback


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
    env.close()


if __name__ == "__main__":
    main()
