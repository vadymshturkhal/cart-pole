import os, imageio, torch, time
import config
import pygame


def render_agent(env, agent, mode="human", episodes=3, out_path="docs/cartpole.gif"):
    rewards = []
    frames = []
    quit_flag = False

    for ep in range(episodes):
        if quit_flag:
            break

        state, _ = env.reset(seed=config.SEED)
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if mode in ["human", "rgb_array"]:
                frame = env.render()
                if mode == "rgb_array":
                    frames.append(frame)

            if mode == "human":
                time.sleep(1/60)  # slow down for visibility (~60 FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    quit_flag = True

            if quit_flag:
                break

        print(f"🎬 Episode {ep+1}, Reward: {total_reward}")
        rewards.append(total_reward)

    if mode == "rgb_array":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if out_path.endswith(".gif"):
            imageio.mimsave(out_path, frames, fps=30)
        elif out_path.endswith(".mp4"):
            imageio.mimsave(out_path, frames, fps=30, format="mp4")

    avg_reward = sum(rewards) / len(rewards)
    print(f"✅ Tested {episodes} episodes — Avg reward: {avg_reward:.1f}")
    return avg_reward
