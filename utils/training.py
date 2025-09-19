import config
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


def train(env, agent, episodes=config.EPISODES, target_update=config.TARGET_UPDATE_FREQ, log_dir="runs/cartpole_nstep"):
    writer = SummaryWriter(log_dir=log_dir)
    rewards_history = []
    
    for ep in range(episodes):
        state, _ = env.reset(seed=config.SEED)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            total_reward += reward
        
        if ep % target_update == 0:
            agent.update_target()
        
        rewards_history.append(total_reward)
        writer.add_scalar("Reward", total_reward, ep)
        
        if ep % 20 == 0:
            print(f"Episode {ep}, Reward: {total_reward}")
    
    writer.close()
    
    # ✅ Save rewards to CSV (path in config)
    df = pd.DataFrame({"episode": range(len(rewards_history)), "reward": rewards_history})
    df.to_csv(config.REWARDS_FILE, index=False)
    print(f"✅ Rewards saved to {config.REWARDS_FILE}")
    
    return rewards_history


