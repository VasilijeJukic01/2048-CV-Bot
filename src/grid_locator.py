import cv2
import numpy as np
from PIL import ImageGrab
from typing import Tuple


class GridFinder:
    """A class to automatically detect the 2048 game grid on the screen.

    Attributes:
        screen (np.ndarray): The full screenshot of the screen.
        background_color (Tuple[int, int, int]): Typical background color of the 2048 grid (RGB).
        min_grid_size (int): Minimum expected size of the grid in pixels.
        max_grid_size (int): Maximum expected size of the grid in pixels.
    """

    def __init__(self):
        self.screen = None
        self.background_color = (187, 173, 160)
        self.min_grid_size = 300
        self.max_grid_size = 600

    def capture_full_screen(self) -> np.ndarray:
        """Capture a screenshot of the entire screen.

        Returns:
            np.ndarray: A NumPy array representing the screenshot in RGB.
        """
        screenshot = ImageGrab.grab().convert("RGB")
        return np.array(screenshot)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image to enhance grid detection.

        Converts to grayscale, applies Gaussian blur, and performs edge detection.

        Args:
            image (np.ndarray): The input RGB image.

        Returns:
            np.ndarray: The preprocessed image (edges).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def find_grid_contours(self, edges: np.ndarray) -> list:
        """Find contours in the edge-detected image that could represent the grid.

        Args:
            edges (np.ndarray): The edge-detected image.

        Returns:
            list: A list of contours sorted by area (largest first).
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours

    def is_valid_grid(self, w: int, h: int) -> bool:
        """Check if the detected rectangle is a valid 2048 grid based on size and aspect ratio.

        Args:
            w (int): Width of the rectangle.
            h (int): Height of the rectangle.

        Returns:
            bool: True if the rectangle is a valid grid, False otherwise.
        """
        # Check size constraints
        if not (self.min_grid_size <= w <= self.max_grid_size and self.min_grid_size <= h <= self.max_grid_size):
            return False
        # Check aspect ratio (2048 grid is nearly square)
        aspect_ratio = w / h
        if not (0.9 <= aspect_ratio <= 1.1):
            return False
        return True

    def verify_grid_content(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """Verify that the detected region contains 2048 grid-like features (e.g., background color).

        Args:
            image (np.ndarray): The full screen image.
            x (int): X-coordinate of the top-left corner.
            y (int): Y-coordinate of the top-left corner.
            w (int): Width of the rectangle.
            h (int): Height of the rectangle.

        Returns:
            bool: True if the region resembles a 2048 grid, False otherwise.
        """
        region = image[y:y+h, x:x+w]
        # Calculate average color in the region
        avg_color = np.mean(region, axis=(0, 1)).astype(int)
        # Euclidean distance to expected background color
        color_dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(avg_color, self.background_color)))
        # Allow some tolerance in color matching
        return color_dist < 50

    def detect_grid(self) -> Tuple[int, int, int, int]:
        """Detect the 2048 game grid on the screen and return its coordinates and size.

        Returns:
            Tuple[int, int, int, int]: (GRID_X, GRID_Y, GRID_WIDTH, GRID_HEIGHT)
        Raises:
            ValueError: If no valid grid is detected.
        """
        self.screen = self.capture_full_screen()
        edges = self.preprocess_image(self.screen)
        contours = self.find_grid_contours(edges)

        for contour in contours[:10]:
            x, y, w, h = cv2.boundingRect(contour)
            if self.is_valid_grid(w, h) and self.verify_grid_content(self.screen, x, y, w, h):
                # Adjust the grid position and size
                x += 8
                y += 8
                w -= 16
                h -= 16
                return x, y, w, h

        raise ValueError("Could not detect the 2048 game grid on the screen.")

    def visualize_detection(self, x: int, y: int, w: int, h: int):
        """Visualize the detected grid on the screen for debugging purposes.

        Args:
            x (int): X-coordinate of the top-left corner.
            y (int): Y-coordinate of the top-left corner.
            w (int): Width of the rectangle.
            h (int): Height of the rectangle.
        """
        debug_image = self.screen.copy()
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Detected Grid", cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_screenshot_with_grid(self, x: int, y: int, w: int, h: int, cell_size: int):
        """Save a screenshot with the grid's hitbox and cell size.

        Args:
            x (int): X-coordinate of the top-left corner.
            y (int): Y-coordinate of the top-left corner.
            w (int): Width of the rectangle.
            h (int): Height of the rectangle.
            cell_size (int): Size of each cell in the grid.
        """
        debug_image = self.screen.copy()
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for i in range(1, 4):
            cv2.line(debug_image, (x + i * cell_size, y), (x + i * cell_size, y + h), (0, 255, 0), 1)
            cv2.line(debug_image, (x, y + i * cell_size), (x + w, y + i * cell_size), (0, 255, 0), 1)
        cv2.imwrite("grid_screenshot.png", cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
        print("Screenshot saved as grid_screenshot.png")
