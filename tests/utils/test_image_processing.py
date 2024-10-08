import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
from src.utils.image_processing import ImageProcessor

class TestImageProcessor(unittest.TestCase):

    def setUp(self):
        """Set up the necessary components before each test."""
        self.canvas = MagicMock()
        self.image_processor = ImageProcessor(self.canvas)

    @patch('src.image_processing.Image.open')
    def test_load_image_success(self, mock_open):
        """Test loading a valid image."""
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image.resize.return_value = mock_image
        mock_open.return_value = mock_image

        self.image_processor.load_image('path/to/image.png')

        mock_open.assert_called_once_with('path/to/image.png')
        mock_image.convert.assert_called_once_with("RGBA")
        mock_image.resize.assert_called_once_with((800, 600), Image.ANTIALIAS)
        self.canvas.create_image.assert_called_once_with(0, 0, anchor='nw', image=self.image_processor.display_image)

    @patch('src.image_processing.Image.open')
    def test_load_image_failure(self, mock_open):
        """Test loading an invalid image."""
        mock_open.side_effect = IOError("File not found")
        with self.assertLogs(level='INFO') as log:
            self.image_processor.load_image('invalid/path/image.png')
            self.assertIn("Error loading image: File not found", log.output[0])

    def test_create_image_graph_empty_image(self):
        """Test create_image_graph when no image is loaded."""
        graph = self.image_processor.create_image_graph()
        self.assertIsNone(graph)

    @patch('src.image_processing.ImageProcessor.calculate_pixel_cost')
    def test_create_image_graph_success(self, mock_calculate_pixel_cost):
        """Test creating a graph from a loaded image."""
        # Create a dummy image
        test_image = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]], 
                                [[0, 0, 255, 255], [255, 255, 0, 255]]], dtype=np.uint8)
        self.image_processor.image = Image.fromarray(test_image)

        mock_calculate_pixel_cost.return_value = 1.0

        graph = self.image_processor.create_image_graph()
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.nodes), 4)  # 4 pixels in the 2x2 image
        self.assertEqual(len(graph.edges), 4)  # 4 edges in a 2x2 grid

    def test_calculate_pixel_cost(self):
        """Test the calculation of pixel cost."""
        color1 = (255, 0, 0, 255)  # Red
        color2 = (0, 255, 0, 255)  # Green
        cost = self.image_processor.calculate_pixel_cost(color1, color2)

        expected_cost = np.sqrt((255 - 0) ** 2 + (0 - 255) ** 2 + (0 - 0) ** 2 + (255 - 255) ** 2)
        self.assertAlmostEqual(cost, expected_cost)

    def test_euclidean_distance_color(self):
        """Test the Euclidean distance calculation between two colors."""
        color1 = (255, 0, 0, 255)
        color2 = (0, 255, 0, 255)
        distance = self.image_processor.euclidean_distance_color(color1, color2)

        expected_distance = np.sqrt((255 - 0) ** 2 + (0 - 255) ** 2 + (0 - 0) ** 2 + (255 - 255) ** 2)
        self.assertAlmostEqual(distance, expected_distance)

if __name__ == '__main__':
    unittest.main()
