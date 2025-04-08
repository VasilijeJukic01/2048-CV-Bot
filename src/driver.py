#!/usr/bin/env python

import numpy as np
import time
import random
from PIL import ImageGrab
from pynput.keyboard import Controller, Listener, Key
from copy import deepcopy
from typing import List, Tuple
from move_recorder import MoveRecorder
from grid_locator import GridFinder

# CONFIG
GRID_X, GRID_Y, GRID_WIDTH, GRID_HEIGHT = 708, 310, 487, 490
CELL_SIZE = GRID_WIDTH // 4
# Delay between consecutive moves
BASE_MOVE_DELAY = 0.1
# Max number of consecutive unchanged grid states before attempting to break
MAX_STUCK_COUNT = 5
# Animation stability check
ANIMATION_CHECK_INTERVAL = 0.05
ANIMATION_STABILITY_THRESHOLD = 1

# Color Map
COLOR_MAP = {
    (238, 228, 218): 2,
    (237, 224, 200): 4,
    (242, 177, 121): 8,
    (245, 149, 99): 16,
    (246, 124, 95): 32,
    (246, 94, 59): 64,
    (237, 207, 114): 128,
    (237, 204, 97): 256,
    (237, 200, 80): 512,
    (237, 197, 63): 1024,
    (237, 194, 46): 2048,
    (205, 193, 180): 0
}


def on_press(key):
    """Stop the bot when the 'ESC' key is pressed."""
    try:
        if key == Key.esc:
            listener.stop()
    except AttributeError:
        pass


listener = Listener(on_press=on_press)
keyboard_controller = Controller()


class Bot2048:
    """A bot that plays the 2048 game using computer vision and the minimax algorithm.

        It automates gameplay by capturing the game grid from the screen, analyzing its state,
        deciding optimal moves using a heuristic-based minimax algorithm, and executing those moves
        via keyboard inputs.

        Attributes:
            grid (List[List[int]]): A 4x4 list representing the current state of the game grid.
            moves (List[str]): List of possible move directions: 'up', 'down', 'left', 'right'.
            move_keys (dict): Mapping of move directions to keyboard keys.
            last_move (str or None): The most recent move executed, or None if no move has been made.
            stuck_count (int): Number of consecutive moves that resulted in no grid change.
            previous_grid (List[List[int]] or None): The grid state before the last move.
            move_history (List[str]): A list of recent moves, limited to the last 10.
            target_corner (Tuple[int, int]): The grid position (row, col) to prioritize for high-value tiles.
            move_recorder (MoveRecorder): An instance to record gameplay frames to a video file.
            detected_colors (set): A set of RGB color tuples detected from the grid (FOR DEBUG).
    """

    def __init__(self):
        self.grid = [[0] * 4 for _ in range(4)]
        self.moves = ['up', 'down', 'left', 'right']
        self.move_keys = {'up': 'w', 'down': 's', 'left': 'a', 'right': 'd'}
        self.last_move = None
        self.stuck_count = 0
        self.previous_grid = None
        self.move_history = []
        self.target_corner = (0, 0)
        self.move_recorder = MoveRecorder("output.avi")
        self.detected_colors = set()

    def capture_screen(self) -> np.ndarray:
        """Capture a screenshot of the 2048 game grid area.

            Returns:
                np.ndarray: A NumPy array that represents the screenshot in RGB.
        """
        screenshot = ImageGrab.grab(bbox=(GRID_X, GRID_Y, GRID_X + GRID_WIDTH, GRID_Y + GRID_HEIGHT))
        return np.array(screenshot)

    def get_cell_color(self, image: np.ndarray, x: int, y: int) -> Tuple:
        """Determine the color of a pixel above the center of a specified cell in the grid image.

            Args:
                image (np.ndarray): The screenshot of the game grid.
                x (int): The x-coordinate of the cell's top-left corner in pixels.
                y (int): The y-coordinate of the cell's top-left corner in pixels.

            Returns:
                Tuple[int, int, int]: The RGB color of the pixel as a tuple of integers.
        """
        center_x, center_y = x + CELL_SIZE // 2, y + CELL_SIZE // 2
        pixel_x, pixel_y = center_x, max(0, center_y - 30)
        color = tuple(image[pixel_y, pixel_x])
        # self.detected_colors.add(color)
        return color

    def is_grid_stable(self) -> bool:
        """Check if the game grid is stable (no animations occurring).

            Stability is determined by comparing hashes of consecutive screenshots. If the hash
            remains unchanged for a specified number of checks, the grid is considered stable.

            Returns:
                bool: True if the grid is stable, False otherwise.
        """
        stable_count = 0
        last_hash = None
        for _ in range(ANIMATION_STABILITY_THRESHOLD + 1):
            screen = self.capture_screen()
            current_hash = hash(screen.tobytes())
            if last_hash is not None and current_hash == last_hash:
                stable_count += 1
            last_hash = current_hash
            if stable_count >= ANIMATION_STABILITY_THRESHOLD:
                return True
            time.sleep(ANIMATION_CHECK_INTERVAL)
        return False

    def read_grid(self, image: np.ndarray) -> List[List[int]]:
        """Analyzes the color of each cell and maps it to a tile value using COLOR_MAP.

            Args:
                image (np.ndarray): The screenshot of the game grid.

            Returns:
                List[List[int]]: A 4x4 grid where each element is a tile value.
        """
        grid = [[0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                color = self.get_cell_color(image, j * CELL_SIZE, i * CELL_SIZE)
                min_dist = float('inf')
                value = 0
                for mapped_color, tile_value in COLOR_MAP.items():
                    # Euclidean distance between colors
                    dist = sum((a - b) ** 2 for a, b in zip(color, mapped_color))
                    if dist < min_dist and dist < 50:
                        min_dist = dist
                        value = tile_value
                grid[i][j] = value
        return grid

    def move(self, grid: List[List[int]], direction: str) -> Tuple[List[List[int]], bool, int]:
        """Simulate a move in the specified direction on the grid.

            Args:
                grid (List[List[int]]): The current 4x4 game grid.
                direction (str): The direction to move ('up', 'down', 'left', 'right').

            Returns:
                Tuple[List[List[int]], bool, int]:
                    - The new grid state after the move.
                    - A boolean indicating if any tiles moved or merged.
                    - The score gained from merging tiles.
        """
        new_grid = [row[:] for row in grid]
        moved = False
        score = 0
        if direction == 'up':
            for j in range(4):
                tiles = [new_grid[i][j] for i in range(4)]
                new_tiles, m, s = self.merge_tiles(tiles)
                moved |= m
                score += s
                for i in range(4):
                    new_grid[i][j] = new_tiles[i]
        elif direction == 'down':
            for j in range(4):
                tiles = [new_grid[i][j] for i in range(4)][::-1]
                new_tiles, m, s = self.merge_tiles(tiles)
                moved |= m
                score += s
                for i in range(4):
                    new_grid[3 - i][j] = new_tiles[i]
        elif direction == 'left':
            for i in range(4):
                tiles = new_grid[i]
                new_tiles, m, s = self.merge_tiles(tiles)
                moved |= m
                score += s
                new_grid[i] = new_tiles
        elif direction == 'right':
            for i in range(4):
                tiles = new_grid[i][::-1]
                new_tiles, m, s = self.merge_tiles(tiles)
                moved |= m
                score += s
                new_grid[i] = new_tiles[::-1]
        return new_grid, moved, score

    def merge_tiles(self, tiles: List[int]) -> Tuple[List[int], bool, int]:
        """Merge a list of tiles.

            Tiles are shifted towards the start and adjacent equal tiles are merged.

            Args:
                tiles (List[int]): A list of 4 tile values (row or column).

            Returns:
                Tuple[List[int], bool, int]:
                    - The resulting list of tiles after merging.
                    - A boolean indicating if any movement or merging occurred.
                    - The score gained from merges.
        """
        result = [0] * 4
        pos = 0
        moved = False
        score = 0
        nonzero = [x for x in tiles if x != 0]
        i = 0
        while i < len(nonzero):
            if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
                merged = nonzero[i] * 2
                result[pos] = merged
                score += merged
                moved = True
                i += 2
            else:
                result[pos] = nonzero[i]
                i += 1
            pos += 1
        return result, moved or tiles != result, score

    def evaluate(self, grid: List[List[int]]) -> float:
        """Evaluate the desirability of a grid state using a heuristic score.

            The score considers factors such as empty cells, maximum tile value, corner placement,
            monotonicity, merge potential, distance from the target corner, trapped tiles, and a
            winning bonus.

            Args:
                grid (List[List[int]]): The 4x4 game grid to evaluate.

            Returns:
                float: A heuristic score representing the grid's quality -> higher == better.
        """
        empty = sum(row.count(0) for row in grid)
        max_tile = max(max(row) for row in grid)

        # Corner focus - encourage keeping the max tile in a specific corner
        ti, tj = self.target_corner
        if grid[ti][tj] == max_tile:
            # Max tile is in corner -> BONUS
            corner_score = max_tile * 10
        else:
            corner_score = -max_tile * 5

        # Monotonicity - reward rows/columns sorted in descending order (easier to merge)
        mono_score = 0
        for i in range(4):
            row = [x for x in grid[i] if x]
            col = [grid[j][i] for j in range(4) if grid[j][i]]
            if row == sorted(row, reverse=True):
                mono_score += sum(row) * (4 - i)
            if col == sorted(col, reverse=True):
                mono_score += sum(col) * (4 - i)

        # Merge potential - reward adjacent equal tiles
        merge_potential = 0
        for i in range(4):
            for j in range(4):
                if grid[i][j]:
                    if i < 3 and grid[i][j] == grid[i + 1][j]:
                        merge_potential += grid[i][j] * 5
                    if j < 3 and grid[i][j] == grid[i][j + 1]:
                        merge_potential += grid[i][j] * 5

        # Distance penalty - penalize high tiles far from the target corner
        distance_penalty = 0
        for i in range(4):
            for j in range(4):
                if grid[i][j] > 0:
                    # Manhattan distance to corner
                    distance = abs(i - ti) + abs(j - tj)
                    distance_penalty += grid[i][j] * distance

        # Trap penalty - penalize tiles surrounded by much smaller tiles
        trap_penalty = 0
        for i in range(4):
            for j in range(4):
                if grid[i][j] > 0:
                    neighbors = []
                    if i > 0 and grid[i - 1][j] > 0:
                        neighbors.append(grid[i - 1][j])
                    if i < 3 and grid[i + 1][j] > 0:
                        neighbors.append(grid[i + 1][j])
                    if j > 0 and grid[i][j - 1] > 0:
                        neighbors.append(grid[i][j - 1])
                    if j < 3 and grid[i][j + 1] > 0:
                        neighbors.append(grid[i][j + 1])
                    for n in neighbors:
                        if n < grid[i][j] / 4:
                            trap_penalty += grid[i][j]

        # Winning focus
        win_bonus = 10000 if max_tile >= 2048 else max_tile * 10

        return (empty * 500 +
                max_tile * 150 +
                corner_score * 100 +
                mono_score * 50 +
                merge_potential * 75 +
                win_bonus - 10 * distance_penalty - 5 * trap_penalty)

    def get_empty_cells(self, grid: List[List[int]]) -> List[Tuple[int, int]]:
        """Identify all empty cells in the grid.

            Args:
                grid (List[List[int]]): The 4x4 game grid.

            Returns:
                List[Tuple[int, int]]: A list of (row, col) coordinates of empty cells.
        """
        return [(i, j) for i in range(4) for j in range(4) if grid[i][j] == 0]

    def minimax(self, grid: List[List[int]], depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, str]:
        """Implement the minimax algorithm with alpha-beta pruning to evaluate moves.

            This method simulates a game tree:
                - Maximizing Player (Bot)* Tries all possible moves (up, down, left, right) to maximize the score.
                - Minimizing Player (Environment): Adds a 2 or 4 tile randomly, aiming to minimize the bot’s advantage.
            Alpha-beta pruning skips branches that won’t affect the outcome, saving time.

            Args:
                grid (List[List[int]]): The current 4x4 game grid.
                depth (int): The remaining depth to search.
                alpha (float): The best score the maximizing player can guarantee.
                beta (float): The best score the minimizing player can guarantee.
                maximizing (bool): True if maximizing (player's turn), False if minimizing (random tile).

            Returns:
                Tuple[float, str]:
                    - The evaluated score of the best move or state.
                    - The best move direction, or an empty string if none is chosen.
        """
        # Stop recursion if depth is 0 or no moves are possible
        if depth == 0 or not any(self.move(grid, m)[1] for m in self.moves):
            return self.evaluate(grid), ""

        best_move = ""
        # Bot’s turn: MAXIMIZE
        if maximizing:
            max_eval = float('-inf')
            # Sort moves by score to prioritize better moves
            moves = sorted(self.moves, key=lambda m: -self.move(grid, m)[2])
            for move in moves:
                new_grid, valid, _ = self.move(grid, move)
                # Move changes the grid
                if valid:
                    # Recurse to environment’s turn - minimizing
                    eval, _ = self.minimax(new_grid, depth - 1, alpha, beta, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    # Pruning - environment won’t let this happen
                    if beta <= alpha:
                        break
            # Best score found, or current grid’s score if no valid moves
            return max_eval if max_eval != float('-inf') else self.evaluate(grid), best_move
        # Environment’s turn: MINIMIZE
        else:
            empty = self.get_empty_cells(grid)
            if not empty:
                return self.evaluate(grid), ""
            avg_eval = 0
            count = 0
            for i, j in empty:
                # 90% chance of 2, 10% of 4
                for value, prob in [(2, 0.9), (4, 0.1)]:
                    new_grid = [row[:] for row in grid]
                    new_grid[i][j] = value
                    # Recurse to bot’s turn - maximizing
                    eval, _ = self.minimax(new_grid, depth - 1, alpha, beta, True)
                    avg_eval += eval * prob
                    count += prob
                    beta = min(beta, eval)
                    # Pruning - bot won’t let this happen
                    if beta <= alpha:
                        break
            # Average score over all possibilities
            return avg_eval / count if count > 0 else self.evaluate(grid), ""

    def choose_best_move(self) -> str:
        """Select the best move based on the current grid state.

            Uses iterative deepening with minimax to find the optimal move within a time limit.
            Includes logic to break stagnation if the grid remains unchanged for too long.

            Returns:
                str: The direction of the best move ('up', 'down', 'left', 'right').
        """
        empty_cells = len(self.get_empty_cells(self.grid))
        # max_tile = max(max(row) for row in self.grid)

        # Dynamic depth based on number of empty cells
        base_depth = 3 if empty_cells > 8 else 5
        max_depth = base_depth + 1

        # Break stagnation if stuck for too long
        if self.stuck_count >= MAX_STUCK_COUNT:
            valid_moves = []
            for m in self.moves:
                new_grid, moved, score = self.move(self.grid, m)
                if moved:
                    empty_after = sum(row.count(0) for row in new_grid)
                    merge_score = sum(
                        new_grid[i][j] for i in range(4) for j in range(4) if new_grid[i][j] > self.grid[i][j])
                    valid_moves.append((m, empty_after * 100 + merge_score * 10 + score * 15))
            if valid_moves:
                self.stuck_count = 0
                return max(valid_moves, key=lambda x: x[1])[0]

        # Iterative deepening
        start_time = time.time()
        best_move = ""
        best_score = float('-inf')
        for d in range(1, max_depth + 1):
            # Time limit per decision
            if time.time() - start_time > 0.1:
                break
            score, move = self.minimax(self.grid, d, float('-inf'), float('inf'), True)
            if move and score > best_score:
                best_score = score
                best_move = move

        if best_move:
            # Track stagnation
            if best_move == self.last_move:
                self.stuck_count += 1
            else:
                self.stuck_count = 0
            self.move_history.append(best_move)
            if len(self.move_history) > 10:
                self.move_history.pop(0)
            return best_move

        # Fallback to a random valid move
        valid_moves = [m for m in self.moves if self.move(self.grid, m)[1]]
        return random.choice(valid_moves) if valid_moves else random.choice(self.moves)

    def run(self):
        """Run the 2048 bot in a continuous loop until stopped.

            Stops when the 'ESC' key is pressed or a 2048 tile is reached.
        """
        global listener
        print("Starting bot... Press 'ESC' to quit")
        self.move_recorder.start_recording()
        last_move_time = time.time()

        listener.start()
        try:
            while listener.running:
                grid_finder = GridFinder()
                try:
                    global GRID_X, GRID_Y, GRID_WIDTH, GRID_HEIGHT, CELL_SIZE
                    GRID_X, GRID_Y, GRID_WIDTH, GRID_HEIGHT = grid_finder.detect_grid()
                    CELL_SIZE = GRID_WIDTH // 4
                    print(f"Detected grid at: ({GRID_X}, {GRID_Y}) with size ({GRID_WIDTH}, {GRID_HEIGHT})")
                except ValueError:
                    print("Could not detect the 2048 game grid. Waiting to detect grid...")
                    time.sleep(1)
                    continue

                while listener.running:
                    if not self.is_grid_stable():
                        continue

                    screen = self.capture_screen()
                    self.previous_grid = deepcopy(self.grid)
                    self.grid = self.read_grid(screen)

                    if self.previous_grid == self.grid and self.last_move:
                        self.stuck_count += 1

                    move = self.choose_best_move()
                    self.last_move = move
                    keyboard_controller.press(self.move_keys[move])
                    keyboard_controller.release(self.move_keys[move])
                    self.move_recorder.record_frame(screen, move)

                    max_tile = max(max(row) for row in self.grid)
                    print(f"Move: {move} (Stuck: {self.stuck_count}, Max Tile: {max_tile})")
                    for row in self.grid:
                        print(row)
                    print()

                    elapsed = time.time() - last_move_time
                    sleep_time = max(0, BASE_MOVE_DELAY - elapsed)
                    time.sleep(sleep_time)
                    last_move_time = time.time()

                    if max_tile >= 2048:
                        print("Reached 2048! You win!")
                        break

                    try:
                        GRID_X, GRID_Y, GRID_WIDTH, GRID_HEIGHT = grid_finder.detect_grid()
                    except ValueError:
                        print("Lost grid focus. Stopping the bot.")
                        break
        finally:
            listener.stop()
            self.move_recorder.stop_recording()
            # self.show_detected_colors()

    # def show_detected_colors(self):
    #     color_list_bgr = [(color[2], color[1], color[0]) for color in self.detected_colors]
    #     color_image = np.zeros((100, 100 * len(color_list_bgr), 3), dtype=np.uint8)
    #     for i, color in enumerate(color_list_bgr):
    #         color_image[:, i * 100:(i + 1) * 100] = color
    #         print(f"Detected color: {color}")
    #     cv2.imshow("Detected Colors", color_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    bot = Bot2048()
    bot.run()
