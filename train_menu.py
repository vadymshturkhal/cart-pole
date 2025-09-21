import pygame
import sys
import torch
import gymnasium as gym
import config
import os
import imageio

from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from utils.training import train_episode, train
from utils.plotting import plot_rewards


SCROLL_OFFSET = 0
dragging = False
drag_offset_y = 0


def progress_callback(ep, episodes, ep_reward, rewards):
    screen.fill(WHITE)

    # === Main progress ===
    title = font.render(f"Training {ep+1}/{episodes}", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2 - 100, 30))

    avg = sum(rewards[-20:]) / min(len(rewards), 20)
    avg_txt = small_font.render(f"Avg (last 20): {avg:.1f}", True, BLACK)
    screen.blit(avg_txt, (WIDTH // 2 - avg_txt.get_width() // 2 - 100, 70))

    bar_w = int((ep+1) / episodes * 400)
    pygame.draw.rect(screen, BLUE, (50, 120, bar_w, 30))
    pygame.draw.rect(screen, BLACK, (50, 120, 400, 30), 2)

    # === Sidebar with episode list ===
    sidebar_x = 480
    sidebar_y = 30
    sidebar_width = 110
    sidebar_height = 340
    visible_rows = 14

    total = len(rewards)
    max_offset = max(0, total - visible_rows)
    global SCROLL_OFFSET
    SCROLL_OFFSET = min(SCROLL_OFFSET, max_offset)

    # Draw sidebar background
    pygame.draw.rect(screen, GRAY, (sidebar_x, sidebar_y, sidebar_width, sidebar_height))

    # Up button
    up_rect = pygame.Rect(sidebar_x, sidebar_y, sidebar_width, 20)
    pygame.draw.rect(screen, BLUE if SCROLL_OFFSET > 0 else GRAY, up_rect)
    up_txt = small_font.render("▲", True, WHITE if SCROLL_OFFSET > 0 else BLACK)
    screen.blit(up_txt, (sidebar_x + sidebar_width // 2 - up_txt.get_width() // 2, sidebar_y))

    # Down button
    down_rect = pygame.Rect(sidebar_x, sidebar_y + sidebar_height - 20, sidebar_width, 20)
    pygame.draw.rect(screen, BLUE if SCROLL_OFFSET < max_offset else GRAY, down_rect)
    down_txt = small_font.render("▼", True, WHITE if SCROLL_OFFSET < max_offset else BLACK)
    screen.blit(down_txt, (sidebar_x + sidebar_width // 2 - down_txt.get_width() // 2,
                           sidebar_y + sidebar_height - 20))

    # Episode list area
    start_idx = max(0, total - visible_rows - SCROLL_OFFSET)
    end_idx = min(total, start_idx + visible_rows)

    for i, idx in enumerate(range(start_idx, end_idx)):
        reward_val = rewards[idx]
        color = BLUE if idx == total - 1 else BLACK
        txt = small_font.render(f"{idx+1}: {reward_val:.0f}", True, color)
        screen.blit(txt, (sidebar_x + 5, sidebar_y + 25 + i*20))

    # === Scrollbar indicator ===
    if total > visible_rows:
        indicator_height = max(20, int((visible_rows / total) * (sidebar_height - 40)))
        indicator_y = sidebar_y + 20 + int((start_idx / total) * (sidebar_height - 40))
        scrollbar_rect = pygame.Rect(sidebar_x + sidebar_width - 8, indicator_y, 6, indicator_height)
        pygame.draw.rect(screen, BLACK, scrollbar_rect)
    else:
        scrollbar_rect = None

    pygame.display.flip()

    # === Input handling ===
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                SCROLL_OFFSET = min(SCROLL_OFFSET + 1, max_offset)
            elif event.key == pygame.K_DOWN:
                SCROLL_OFFSET = max(SCROLL_OFFSET - 1, 0)
        elif event.type == pygame.MOUSEWHEEL:
            if event.y > 0:  # scroll up
                SCROLL_OFFSET = min(SCROLL_OFFSET + 1, max_offset)
            elif event.y < 0:  # scroll down
                SCROLL_OFFSET = max(SCROLL_OFFSET - 1, 0)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if up_rect.collidepoint(x, y) and SCROLL_OFFSET > 0:
                SCROLL_OFFSET = min(SCROLL_OFFSET + 1, max_offset)
            elif down_rect.collidepoint(x, y) and SCROLL_OFFSET < max_offset:
                SCROLL_OFFSET = max(SCROLL_OFFSET - 1, 0)


# ============== AGENT FACTORY ==============
def make_agent(agent_name, state_dim, action_dim):
    if agent_name == "nstep_dqn":
        return NStepDeepQLearningAgent(state_dim, action_dim)
    if agent_name == "nstep_ddqn":
        return NStepDoubleDeepQLearningAgent(state_dim, action_dim)
    raise ValueError(f"Unknown agent: {agent_name}")


# ============== PYGAME MENU ==============
pygame.init()
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CartPole RL Launcher")

font = pygame.font.SysFont("arial", 28)
small_font = pygame.font.SysFont("arial", 20)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (70, 130, 180)
GREEN = (34, 139, 34)
RED = (178, 34, 34)
GRAY = (200, 200, 200)

agents = ["nstep_dqn", "nstep_ddqn"]
render_modes = ["off", "human", "gif", "mp4"]

selected_agent = 0
selected_render = 0


def draw_menu():
    screen.fill(WHITE)

    title = font.render("CartPole RL Launcher", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 30))

    # Agent selection
    agent_label = small_font.render("Choose Agent:", True, BLACK)
    screen.blit(agent_label, (50, 100))
    for i, agent in enumerate(agents):
        color = BLUE if i == selected_agent else GRAY
        pygame.draw.rect(screen, color, (200 + i * 150, 90, 140, 40))
        txt = small_font.render(agent, True, BLACK)
        screen.blit(txt, (200 + i * 150 + 20, 100))

    # Render selection
    render_label = small_font.render("Rendering Mode:", True, BLACK)
    screen.blit(render_label, (50, 170))
    for i, mode in enumerate(render_modes):
        color = GREEN if i == selected_render else GRAY
        pygame.draw.rect(screen, color, (200 + i * 90, 160, 80, 40))
        txt = small_font.render(mode, True, BLACK)
        screen.blit(txt, (200 + i * 90 + 10, 170))

    # Start button
    pygame.draw.rect(screen, RED, (WIDTH // 2 - 100, 250, 200, 60))
    start_txt = font.render("Start Training", True, WHITE)
    screen.blit(start_txt, (WIDTH // 2 - start_txt.get_width() // 2, 265))

    pygame.display.flip()


def menu_loop():
    global selected_agent, selected_render
    while True:
        draw_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                # Agent buttons
                for i in range(len(agents)):
                    if 200 + i * 150 <= x <= 200 + i * 150 + 140 and 90 <= y <= 130:
                        selected_agent = i
                # Render buttons
                for i in range(len(render_modes)):
                    if 200 + i * 90 <= x <= 200 + i * 90 + 80 and 160 <= y <= 200:
                        selected_render = i
                # Start button
                if WIDTH // 2 - 100 <= x <= WIDTH // 2 + 100 and 250 <= y <= 310:
                    return agents[selected_agent], render_modes[selected_render]


# ============== RENDER AGENT (unchanged) ==============
def render_agent(env, agent, mode="human", episodes=3, out_path="docs/cartpole.gif"):
    rewards = []
    frames = []

    for ep in range(episodes):
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

        rewards.append(total_reward)

    if mode == "rgb_array":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        imageio.mimsave(out_path, frames, fps=30)

    return sum(rewards) / len(rewards)


# ============== MAIN ==============
def main():
    agent_name, render_mode = menu_loop()

    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = make_agent(agent_name, state_dim, action_dim)

    rewards = train(env, agent, episodes=config.EPISODES, progress_cb=progress_callback)

    # Save model
    os.makedirs(config.TRAINED_MODELS_FOLDER, exist_ok=True)
    model_path = f"{config.TRAINED_MODELS_FOLDER}/{agent_name}_qnet.pth"
    torch.save(agent.q_net.state_dict(), model_path)

    # Plot rewards
    # plot_rewards(from_file=False, rewards=rewards)

    # Rendering after training
    if render_mode != "off":
        render_env = gym.make(
            config.ENV_NAME,
            render_mode="rgb_array" if render_mode in ["gif", "mp4"] else "human"
        )
        agent.q_net.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        agent.q_net.eval()

        avg_reward = 0
        if render_mode == "human":
            avg_reward = render_agent(render_env, agent, mode="human", episodes=5)
        elif render_mode == "gif":
            avg_reward = render_agent(render_env, agent, mode="rgb_array", episodes=1, out_path="docs/cartpole.gif")
        elif render_mode == "mp4":
            avg_reward = render_agent(render_env, agent, mode="rgb_array", episodes=1, out_path="docs/cartpole.mp4")

        render_env.close()

        # Show final message
        screen.fill(WHITE)
        msg = font.render(f"Training Done! Avg Reward: {avg_reward:.1f}", True, BLACK)
        screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 20))
        pygame.display.flip()
        pygame.time.wait(3000)

    env.close()

    # === Post-training loop ===
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                # Keep sidebar interactive even after training
                progress_callback(len(rewards)-1, config.EPISODES, rewards[-1], rewards)

        clock.tick(30)  # refresh ~30 FPS

    pygame.quit()


if __name__ == "__main__":
    main()
