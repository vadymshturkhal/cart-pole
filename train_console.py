import torch
import gymnasium as gym
import config
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from utils.training import train


def main():
    print("🚀 Starting training...")
    env = gym.make(config.DEFAULT_ENVIRONMENT)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = NStepDeepQLearningAgent(state_dim, action_dim)
    train(env, agent, episodes=config.EPISODES)
    
    # ✅ Save the trained model to config path
    model_path = f"{config.TRAINED_MODELS_FOLDER}/{config.TRAINED_CONSOLE_MODEL_FILENAME}"
    torch.save(agent.q_net.state_dict(), model_path)
    print(f"✅ Training complete. Model saved to {model_path}")
    
    # ✅ Plot from CSV
    # plot_rewards(from_file=True)
    
    env.close()


if __name__ == "__main__":
    main()
