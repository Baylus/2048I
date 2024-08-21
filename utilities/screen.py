import pygame

from config.settings import WINDOW_HEIGHT, WINDOW_WIDTH
from utilities.singleton import Singleton

pygame.init()

class SharedScreen(metaclass=Singleton):
    def __init__(self):
        self.screen: pygame.Surface = None
    
    def init_screen(self):
        if not self.screen:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("2048I")

    def get_screen(self):
        return self.screen
    
    def draw_text(self, text, x, y, font_size=20, color=(255, 255, 255)):
        font = pygame.font.SysFont(None, font_size)
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
