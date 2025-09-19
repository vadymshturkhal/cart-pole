import pandas as pd
import matplotlib.pyplot as plt
import config

def plot_rewards(from_file=True, rewards=None):
    if from_file:
        df = pd.read_csv(config.REWARDS_FILE)
        rewards = df["reward"].values
    elif rewards is None:
        raise ValueError("Either set from_file=True or pass rewards array.")
    
    plt.figure(figsize=(8, 5))
    plt.plot(rewards, label="Episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
