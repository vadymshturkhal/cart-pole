import gymnasium as gym
import config
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
import torch
from utils.plotting import plot_rewards
import os


def render_trained_agent(episodes=3):
    if not os.path.exists(config.MODEL_FILE):
        raise FileNotFoundError(f"❌ Trained model not found: {config.MODEL_FILE}. Run main.py first.")
    
    env = gym.make(config.ENV_NAME, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = NStepDoubleDeepQLearningAgent(state_dim, action_dim)
    agent.q_net.load_state_dict(torch.load(config.MODEL_FILE, map_location=config.DEVICE))
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
