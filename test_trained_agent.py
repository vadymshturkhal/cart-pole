import gymnasium as gym
import config
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from agents.nstep_dqn_agent import NStepDeepQLearningAgent
import torch
import os


def render_trained_agent(episodes=3):
    model_path = f"{config.TRAINED_MODELS_FOLDER}/{config.TRAINED_CONSOLE_MODEL_FILENAME}"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Trained model not found: {model_path}. Run main.py first.")
    
    env = gym.make(config.DEFAULT_ENVIRONMENT, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = NStepDeepQLearningAgent(state_dim, action_dim)
    agent.q_net.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    agent.q_net.eval()
    
    all_rewards = []
    for ep in range(episodes):
        state, _ = env.reset(seed=config.SEED)
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()
        
        all_rewards.append(total_reward)
        print(f"🎬 Rendered Episode {ep}, Reward: {total_reward}")
    
    env.close()
    print(f"📊 Average reward over {episodes} episodes: {sum(all_rewards)/len(all_rewards):.2f}")


if __name__ == "__main__":
    # ✅ Plot training performance first
    # plot_rewards(from_file=True)
    
    # ✅ Render trained agent
    render_trained_agent(episodes=20)
