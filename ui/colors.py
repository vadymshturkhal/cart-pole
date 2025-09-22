import pygame

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (70, 130, 180)
GREEN = (34, 139, 34)
RED = (178, 34, 34)
GRAY = (200, 200, 200)

WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CartPole RL Launcher")

font = pygame.font.SysFont("arial", 28)
small_font = pygame.font.SysFont("arial", 20)
