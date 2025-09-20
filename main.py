import torch
import gymnasium as gym
import config
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from utils.training import train
from utils.plotting import plot_rewards


def main():
    print("🚀 Starting training...")
    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = NStepDoubleDeepQLearningAgent(state_dim, action_dim)
    train(env, agent, episodes=config.EPISODES)
    
    # ✅ Save the trained model to config path
    torch.save(agent.q_net.state_dict(), config.MODEL_FILE)
    print(f"✅ Training complete. Model saved to {config.MODEL_FILE}")
    
    # ✅ Plot from CSV
    plot_rewards(from_file=True)
    
    env.close()


if __name__ == "__main__":
    main()
