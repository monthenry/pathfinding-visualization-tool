import unittest
from unittest.mock import MagicMock, patch
import tkinter as tk
from src.main import PathFindingApp

class TestPathFindingApp(unittest.TestCase):

    def setUp(self):
        """Set up the tkinter root window and the PathFindingApp instance."""
        self.root = tk.Tk()
        self.app = PathFindingApp(self.root)

    def tearDown(self):
        """Destroy the tkinter root window after each test."""
        self.root.destroy()

    @patch('utils.image_processing.ImageProcessor.load_image')
    def test_load_image(self, mock_load_image):
        """Test loading an image."""
        # Simulate file selection by setting the mock
        mock_load_image.return_value = None
        with patch('tkinter.filedialog.askopenfilename', return_value='../assets/images/dalle-gallet.png'):
            self.app.load_image()

        # Verify that load_image method was called
        mock_load_image.assert_called_once_with('../assets/images/dalle-gallet.png')

    def test_canvas_click_start_point(self):
        """Test clicking on the canvas to set the start point."""
        self.app.on_canvas_click(MockEvent(100, 150))
        self.assertEqual(self.app.start_point, (100, 150))

        # Check if the green circle is created on the canvas
        self.assertEqual(len(self.app.canvas.find_all()), 1)  # Only one shape (start point) should be created

    def test_canvas_click_end_point(self):
        """Test clicking on the canvas to set the end point."""
        self.app.on_canvas_click(MockEvent(100, 150))  # Set start point
        self.app.on_canvas_click(MockEvent(200, 250))  # Set end point
        self.assertEqual(self.app.end_point, (200, 250))

        # Check if the red circle is created on the canvas
        self.assertEqual(len(self.app.canvas.find_all()), 2)  # Two shapes (start and end points) should be created

    @patch('src.main.Dijkstra')
    def test_run_algorithm(self, mock_dijkstra):
        """Test running an algorithm."""
        self.app.image_processor.image = MagicMock()  # Mock image processor image
        self.app.start_point = (100, 150)
        self.app.end_point = (200, 250)

        self.app.run_algorithm(mock_dijkstra)

        # Verify that the algorithm was instantiated and run
        mock_dijkstra.assert_called_once_with(self.app.image_processor, self.app.start_point, self.app.end_point, self.app.path_visualizer)
        mock_dijkstra.return_value.run.assert_called_once()

    def test_run_algorithm_without_image(self):
        """Test running an algorithm without an image loaded."""
        self.app.start_point = (100, 150)
        self.app.end_point = (200, 250)

        self.app.run_algorithm(MagicMock())

        # Algorithm should not run as no image is loaded
        self.assertFalse(self.app.algorithm_in_progress)

    def test_run_algorithm_without_start_point(self):
        """Test running an algorithm without a start point."""
        self.app.image_processor.image = MagicMock()  # Mock image processor image
        self.app.end_point = (200, 250)

        self.app.run_algorithm(MagicMock())

        # Algorithm should not run as no start point is set
        self.assertFalse(self.app.algorithm_in_progress)

    def test_run_algorithm_without_end_point(self):
        """Test running an algorithm without an end point."""
        self.app.image_processor.image = MagicMock()  # Mock image processor image
        self.app.start_point = (100, 150)

        self.app.run_algorithm(MagicMock())

        # Algorithm should not run as no end point is set
        self.assertFalse(self.app.algorithm_in_progress)

    def test_canvas_click_while_algorithm_in_progress(self):
        """Test clicking on the canvas while an algorithm is in progress."""
        self.app.algorithm_in_progress = True
        self.app.on_canvas_click(MockEvent(100, 150))
        self.assertIsNone(self.app.start_point)  # Start point should not be set

    def test_algorithm_in_progress_toggle(self):
        """Test toggling the algorithm in progress state."""
        self.app.algorithm_in_progress = True
        self.app.run_algorithm(MagicMock())
        self.assertFalse(self.app.algorithm_in_progress)  # Should reset to False after running

class MockEvent:
    """Mock class to simulate tkinter event."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

if __name__ == '__main__':
    unittest.main()
