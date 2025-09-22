import pygame, sys
from .colors import screen, WHITE, BLACK, BLUE, GREEN, GRAY, RED, font, small_font, WIDTH, HEIGHT

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
                pygame.quit(); 
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
