import unittest
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
from src.visualizations.path_visualization import PathVisualizer

class TestPathVisualizer(unittest.TestCase):

    def setUp(self):
        """Set up the necessary components before each test."""
        self.image_processor = MagicMock()
        self.path_visualizer = PathVisualizer(self.image_processor)

    def test_visualize_path_with_valid_data(self):
        """Test the visualization of a path with valid nodes."""
        # Create a mock image with a specific size
        width, height = 100, 100
        mock_image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        self.image_processor.image = mock_image
        self.image_processor.start_point = (10, 10)
        self.image_processor.end_point = (90, 90)

        path = [(10, 10), (20, 20), (30, 30), (90, 90)]
        visited_nodes = [(10, 10), (20, 20), (30, 30)]

        self.path_visualizer.visualize_path(path, visited_nodes)

        # Verify that the display image is updated
        self.image_processor.canvas.create_image.assert_called_once()

        # Check the pixels for visited nodes
        pixels = mock_image.load()
        for node in visited_nodes:
            x, y = node
            self.assertEqual(pixels[x, y], (255 - 255, 255 - 0, 255 - 0, 255))  # Check for inverted color

        # Check start and end points
        start_color = pixels[10, 10]
        end_color = pixels[90, 90]
        self.assertEqual(start_color, (0, 255, 0, 255))  # Start point should be green
        self.assertEqual(end_color, (255, 0, 0, 255))  # End point should be red

    def test_invert_color(self):
        """Test the color inversion logic."""
        color = (100, 150, 200, 255)
        inverted = self.path_visualizer.invert_color(color)
        self.assertEqual(inverted, (155, 105, 55, 255))  # Check inverted color

    def test_draw_point(self):
        """Test drawing a point on the image."""
        width, height = 10, 10
        mock_image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        self.image_processor.image = mock_image
        pixels = mock_image.load()

        self.path_visualizer.draw_point(pixels, 5, 5, (255, 0, 0, 255))

        # Verify that the point was drawn
        for dx in range(-6, 7):
            for dy in range(-6, 7):
                if abs(dx) <= 6 and abs(dy) <= 6:
                    x, y = 5 + dx, 5 + dy
                    if 0 <= x < width and 0 <= y < height:
                        self.assertEqual(pixels[x, y], (255, 0, 0, 255))  # Check for red color

    def test_draw_line_placeholder(self):
        """Test placeholder for the line drawing function."""
        # Here we would typically validate the line drawing logic
        # As the draw_line method is not implemented yet, we can only check that it exists.
        self.assertTrue(hasattr(self.path_visualizer, 'draw_line'))

if __name__ == '__main__':
    unittest.main()
