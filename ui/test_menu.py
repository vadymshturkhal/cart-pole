import pygame, os, sys
from .colors import *

def list_models(folder="trained_models"):
    return [f for f in os.listdir(folder) if f.endswith(".pth")]

def draw_test_menu(models, selected_idx, selected_render, episodes):
    screen.fill(WHITE)
    title = font.render("Test Saved Model", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 30))

    # Model selection
    model_label = small_font.render("Choose Model:", True, BLACK)
    screen.blit(model_label, (50, 100))
    for i, m in enumerate(models):
        color = BLUE if i == selected_idx else GRAY
        pygame.draw.rect(screen, color, (200, 90 + i*40, 250, 30))
        txt = small_font.render(m, True, BLACK)
        screen.blit(txt, (205, 95 + i*40))

    # Render mode selection
    render_modes = ["human", "gif", "mp4"]
    render_label = small_font.render("Rendering Mode:", True, BLACK)
    screen.blit(render_label, (50, 250))
    for i, mode in enumerate(render_modes):
        color = GREEN if i == selected_render else GRAY
        pygame.draw.rect(screen, color, (200 + i*90, 240, 80, 40))
        txt = small_font.render(mode, True, BLACK)
        screen.blit(txt, (200 + i*90 + 10, 250))

    # Episodes selection
    ep_label = small_font.render(f"Episodes: {episodes}", True, BLACK)
    screen.blit(ep_label, (50, 310))
    pygame.draw.rect(screen, BLUE, (200, 300, 40, 30))   # - button
    screen.blit(small_font.render("-", True, WHITE), (212, 305))
    pygame.draw.rect(screen, BLUE, (250, 300, 40, 30))   # + button
    screen.blit(small_font.render("+", True, WHITE), (262, 305))

    # Start button
    pygame.draw.rect(screen, RED, (WIDTH // 2 - 100, 350, 200, 40))
    start_txt = font.render("Run Test", True, WHITE)
    screen.blit(start_txt, (WIDTH // 2 - start_txt.get_width() // 2, 355))

    pygame.display.flip()

def test_menu_loop():
    models = list_models()
    if not models:
        print("❌ No models found in trained_models/")
        pygame.quit(); sys.exit()

    selected_idx = 0
    selected_render = 0
    render_modes = ["human", "gif", "mp4"]
    episodes = 5

    while True:
        draw_test_menu(models, selected_idx, selected_render, episodes)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                # Model select
                for i in range(len(models)):
                    if 200 <= x <= 450 and 90 + i*40 <= y <= 120 + i*40:
                        selected_idx = i
                # Render mode select
                for i in range(len(render_modes)):
                    if 200 + i*90 <= x <= 280 + i*90 and 240 <= y <= 280:
                        selected_render = i
                # Episodes select
                if 200 <= x <= 240 and 300 <= y <= 330:
                    episodes = max(1, episodes - 1)
                elif 250 <= x <= 290 and 300 <= y <= 330:
                    episodes += 1
                # Run button
                if WIDTH // 2 - 100 <= x <= WIDTH // 2 + 100 and 350 <= y <= 390:
                    return models[selected_idx], render_modes[selected_render], episodes
