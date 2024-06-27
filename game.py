import numpy as np
import pygame as pg
import random

from action import Action
from config.settings import *

pg.font.init()
FONT = pg.font.Font(None, FONT_SIZE)

class GameDone(Exception):
    pass

# Define the game environment
class Board:
    def __init__(
            self,
            height = BOARD_HEIGHT,
            width = BOARD_WIDTH,
        ):
        self.height = height
        self.width = width
        # These help me remember when I am looking at rows or columns
        self.rows = 4
        self.cols = 4

        # Def top corner
        self.x = (WINDOW_WIDTH - BOARD_WIDTH) / 2
        # print(f"{self.x}")
        self.y = (WINDOW_HEIGHT - BOARD_HEIGHT) / 2
        # print(f"{self.y}")

        # Tile size (I am forcing 1:1 ratio for board, so they are equally wide as tall)
        self.tile_size = (BOARD_HEIGHT - (TILE_GAP * (self.rows + 1))) / self.rows

        # Initialize the game state
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        # Initialize other game parameters
        self.score = 0

        # Two at the start
        self.add_tile()
        self.add_tile()

    def reset(self):
        # Reset the game state
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        # Reset other game parameters
        self.score = 0
        self.add_tile()
        self.add_tile()

    def add_tile(self):
        """Adds a random 2 or 4 tile to an empty spot on the board
        """
        empty_tiles = [(row, col) for row in range(self.rows) for col in range(self.cols) if self.grid[row][col] == 0]
        
        row, col = random.choice(empty_tiles)
        # Decide the value of the new tile (90% chance for 2, 10% chance for 4)
        new_tile_value = 2 if random.random() < 0.9 else 4

        self.grid[row][col] = new_tile_value

        if self.is_done():
            # We might add an exception here, which is why I am adding this check, but this is the first place
            # that we are able to check for a done board state, since it can only happen from filling the last spot
            # without the 
            raise GameDone
            return

        pass


    def act(self, action):
        # print(self.grid)
        # print("########################################################################")
        new_grid = None
        if action == Action.UP:
            new_grid = self.move_up(self.grid)
        elif action == Action.RIGHT:
            new_grid = self.move_right(self.grid)
        elif action == Action.DOWN:
            new_grid = self.move_down(self.grid)
        elif action == Action.LEFT:
            new_grid = self.move_left(self.grid)
        
        # print(new_grid)
        if np.all(self.grid == new_grid):
            # We didnt move anything at all, so this wasn't a valid action
            # print("This grid didnt change at all, so it doesnt count")
            return

        self.grid = new_grid
        # Now we have to add a tile to replace
        self.add_tile()
    
    def slide_and_merge_row_left(self, row):
        # Slide non-zero values to the left
        new_row = [i for i in row if i != 0]
        # Merge tiles
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                # We merged, so we adjust our score 
                self.score += new_row[i]
                new_row[i + 1] = 0
        # Slide again after merging
        new_row = [i for i in new_row if i != 0]
        # Pad with zeros to the right to maintain the length
        new_row += [0] * (len(row) - len(new_row))
        return new_row

    def move_left(self, state):
        new_state = np.zeros((self.rows, self.cols), dtype=int)
        for i in range(self.rows):
            new_state[i] = self.slide_and_merge_row_left(state[i])
        return new_state

    def move_right(self, state):
        new_state = np.zeros((self.rows, self.cols), dtype=int)
        for i in range(4):
            new_state[i, :] = self.slide_and_merge_row_left(state[i, :][::-1])[::-1]
        return new_state

    def move_up(self, state):
        new_state = np.zeros((self.rows, self.cols), dtype=int)
        for i in range(self.cols):
            new_state[:, i] = self.slide_and_merge_row_left(state[:, i])
        return new_state

    def move_down(self, state):
        new_state = np.zeros((self.rows, self.cols), dtype=int)
        for i in range(self.cols):
            new_state[:, i] = self.slide_and_merge_row_left(state[:, i][::-1])[::-1]
        return new_state

    def get_state(self):
        return self.grid.flatten()

    def set_state(self, new_state):
        self.grid = np.reshape(new_state, (self.rows, self.cols))

    def is_done(self):
        # Check if the game is over

        # Check rows first
        for i in range(self.rows):
            for j in range(self.cols - 1):
                # - 1, dont check off the edge
                if self.grid[i][j] == self.grid[i][j + 1]:
                    # Found match to the right, so we can combine them
                    return False
        
        # Now columns
        for i in range(self.rows - 1):
            # - 1, dont fall off bottom
            for j in range(self.cols):
                if self.grid[i][j] == self.grid[i + 1][j]:
                    # Found match below, so we can combine them
                    return False
        
        # print(f"It is done. Heres the grid\n{self.grid}")
        return True

    def draw(self, surface):
        color = BOARD_COLOR
        rect = pg.Rect(self.x, self.y, self.width, self.height)
        pg.draw.rect(surface, color, rect, border_radius = 15)

        text_surface = FONT.render(f"Score: {str(self.score)}", True, FONT_COLOR)
        text_rect = text_surface.get_rect(center=rect.center)
        text_rect.y = 10
        surface.blit(text_surface, text_rect)
        
        for row in range(self.rows):
            for col in range(self.cols):
                # print("Trying to draw tile")
                value = self.grid[row][col]
                color = TILE_COLOR.get(value, TILE_COLOR[16384])
                x = self.x + (col * self.tile_size) + ((col + 1) * TILE_GAP)
                y = self.y + (row * self.tile_size) + ((row + 1) * TILE_GAP)
                rect = pg.Rect(x, y, self.tile_size, self.tile_size)
                pg.draw.rect(surface, color, rect, border_radius = 15)
                
                if value > 0:
                    text_surface = FONT.render(str(value), True, FONT_COLOR)
                
                    text_rect = text_surface.get_rect(center=rect.center)
                    text_rect.y += 5
                    surface.blit(text_surface, text_rect)
        
        pg.display.update()
        

