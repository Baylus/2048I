from enum import IntEnum, auto
import pygame
import sys


from action import Action
from game import Board, GameDone

from config.settings import *

if WINDOW_HEIGHT < BOARD_HEIGHT or WINDOW_WIDTH < BOARD_WIDTH:
    raise ValueError("Board can't be bigger than the window")

if WINDOW_HEIGHT != WINDOW_WIDTH:
    print("WARNING: Just so you know your window ratio isnt 1:1")
if BOARD_HEIGHT != BOARD_WIDTH:
    raise ValueError("ERROR: Your board ratio isnt 1:1, which is definitely disgusting")

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("2048I")

board = Board()

# BG = pygame.image.load("assets/stage.png")

def draw():
    # Fill background
    screen.fill(BACKGROUND_COLOR)

    board.draw(screen)

    pygame.display.update()

def main():
    # Initial housekeeping

    clock = pygame.time.Clock()

    # Main game loop
    running = True
    while running:
        clock.tick(TPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action = get_action(keys)
        # board.step(action)
        if action:
            board.act(action)

        draw()
        if (board.is_done()):
            raise GameDone

    pygame.quit()
    sys.exit()

last_action = None

def get_action(inputs) -> Action:
    """Determine action to do

    Args:
        inputs (_type_): keys being pressed
    """
    global last_action

    action = None
    if inputs[pygame.K_w]: # W
        action = Action.UP
    if inputs[pygame.K_s]: # S
        action = Action.DOWN
    if inputs[pygame.K_a]: # A
        action = Action.LEFT
    if inputs[pygame.K_d]: # D
        action = Action.RIGHT
    
    if action:
        print(f"This is our action {action}")
    # This makes it so that you can't just hold down a key and it will continue to spam the inputs
    # Only useful for user interactions, as the network will be able to keep up. Shocked it runs so fast tho
    if action and action != last_action:
        # These are different, and we found an action, so we can actually use it
        last_action = action
        return action

    last_action = action
    return None


if __name__ == "__main__":
    main()