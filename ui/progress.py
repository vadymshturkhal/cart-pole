import pygame, sys
from .colors import *

SCROLL_OFFSET = 0
dragging = False
drag_offset_y = 0

def progress_callback(ep, episodes, ep_reward, rewards):
    screen.fill(WHITE)

    # === Main progress ===
    title = font.render(f"Training {ep+1}/{episodes}", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2 - 100, 30))

     # === Averages ===
    avg20 = sum(rewards[-20:]) / min(len(rewards), 20)
    avg20_txt = small_font.render(f"Avg (last 20): {avg20:.1f}", True, BLACK)
    screen.blit(avg20_txt, (WIDTH // 2 - avg20_txt.get_width() // 2 - 100, 70))

    global_avg = sum(rewards) / len(rewards) if rewards else 0
    global_avg_txt = small_font.render(f"Global avg: {global_avg:.1f}", True, BLACK)
    screen.blit(global_avg_txt, (WIDTH // 2 - global_avg_txt.get_width() // 2 - 100, 95))

    # === Progress bar ===
    bar_w = int((ep+1) / episodes * 400)
    pygame.draw.rect(screen, BLUE, (50, 120, bar_w, 30))
    pygame.draw.rect(screen, BLACK, (50, 120, 400, 30), 2)

    # === Training curve under progress bar ===
    if len(rewards) > 1:
        curve_x, curve_y = 50, 170
        curve_w, curve_h = 400, 100
        pygame.draw.rect(screen, WHITE, (curve_x, curve_y, curve_w, curve_h))
        pygame.draw.rect(screen, BLACK, (curve_x, curve_y, curve_w, curve_h), 1)

        # Axis labels
        x_label = small_font.render("Episodes", True, BLACK)
        screen.blit(x_label, (curve_x + curve_w // 2 - x_label.get_width() // 2, curve_y + curve_h + 5))

        y_label = small_font.render("Reward (norm.)", True, BLACK)
        # Rotate text for vertical axis
        y_surf = pygame.transform.rotate(y_label, 90)
        screen.blit(y_surf, (curve_x - 40, curve_y + curve_h // 2 - y_surf.get_height() // 2))

        max_r = max(rewards)
        min_r = min(rewards)
        span = max(1, max_r - min_r)

        # Normalize rewards to [0,1] for plotting
        points = []
        for i, r in enumerate(rewards):
            x = curve_x + int(i / max(1, episodes-1) * curve_w)
            y = curve_y + curve_h - int((r - min_r) / span * curve_h)
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(screen, BLUE, False, points, 2)

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
