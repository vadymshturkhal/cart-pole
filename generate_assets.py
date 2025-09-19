import os
import imageio
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import torch

import config
from agents.nstep_agent import NStepQLearningAgent


def generate_rewards_plot():
    """Generate rewards plot from rewards.csv and save as PNG."""
    if not os.path.exists(config.REWARDS_FILE):
        print(f"❌ Rewards file not found: {config.REWARDS_FILE}. Run training first.")
        return
    
    df = pd.read_csv(config.REWARDS_FILE)
    rewards = df["reward"].values

    plt.figure(figsize=(8, 5))
    plt.plot(rewards, label="Episode reward", alpha=0.7)
    plt.plot(pd.Series(rewards).rolling(20).mean(), label="Moving avg (20 ep)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/rewards_plot.png")
    plt.close()
    print("✅ Saved training curve to docs/rewards_plot.png")


def generate_cartpole_gif(episodes=1, max_frames=500):
    """Render trained agent and save a GIF."""
    if not os.path.exists(config.MODEL_FILE):
        print(f"❌ Model file not found: {config.MODEL_FILE}. Run training first.")
        return

    env = gym.make(config.ENV_NAME, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = NStepQLearningAgent(state_dim, action_dim)
    agent.q_net.load_state_dict(torch.load(config.MODEL_FILE, map_location=config.DEVICE))
    agent.q_net.eval()

    frames = []
    state, _ = env.reset(seed=config.SEED)
    done = False
    step = 0

    while not done and step < max_frames:
        frame = env.render()
        frames.append(frame)

        action = agent.select_action(state, greedy=True)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

    env.close()

    os.makedirs("docs", exist_ok=True)
    gif_path = "docs/cartpole.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"✅ Saved rendering demo to {gif_path}")


if __name__ == "__main__":
    generate_rewards_plot()
    generate_cartpole_gif()
