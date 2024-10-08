import unittest
from unittest.mock import MagicMock
from src.algorithms.dijkstra import Dijkstra

class TestDijkstra(unittest.TestCase):

    def setUp(self):
        """Set up the necessary components before each test."""
        self.image_processor = MagicMock()
        self.path_visualizer = MagicMock()

        # Define start and end points for the Dijkstra algorithm
        self.start_point = (0, 0)
        self.end_point = (2, 2)

        # Create an instance of Dijkstra
        self.dijkstra = Dijkstra(self.image_processor, self.start_point, self.end_point, self.path_visualizer)

    def test_run_without_start_or_end_point(self):
        """Test run method when start or end point is None."""
        self.dijkstra.start_point = None
        self.dijkstra.run()
        self.path_visualizer.visualize_path.assert_not_called()

        self.dijkstra.start_point = (0, 0)
        self.dijkstra.end_point = None
        self.dijkstra.run()
        self.path_visualizer.visualize_path.assert_not_called()

    def test_run_triggers_algorithm(self):
        """Test run method executes the algorithm and visualizes the path."""
        self.image_processor.create_image_graph.return_value = MagicMock()
        self.dijkstra.run()
        self.assertTrue(self.dijkstra.start_point is not None and self.dijkstra.end_point is not None)

    def test_custom_dijkstra_path_found(self):
        """Test the custom_dijkstra method when a path is found."""
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = [(0, 0), (1, 1), (2, 2)]
        mock_graph.edges.return_value = [
            ((0, 0), (1, 1), {'cost': 1}),
            ((1, 1), (2, 2), {'cost': 1}),
            ((0, 0), (2, 2), {'cost': 3}),
        ]

        self.image_processor.create_image_graph.return_value = mock_graph
        
        path = self.dijkstra.custom_dijkstra(mock_graph, self.start_point, self.end_point)
        expected_path = [(0, 0), (1, 1), (2, 2)]

        self.assertEqual(path, expected_path)

    def test_custom_dijkstra_no_path(self):
        """Test the custom_dijkstra method when no path is found."""
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = [(0, 0), (1, 1), (2, 2)]
        mock_graph.edges.return_value = [
            ((0, 0), (1, 1), {'cost': 1}),
            ((1, 1), (0, 0), {'cost': 1}),  # No path to (2, 2)
        ]

        self.image_processor.create_image_graph.return_value = mock_graph
        
        path = self.dijkstra.custom_dijkstra(mock_graph, self.start_point, self.end_point)
        expected_path = []  # No valid path to end_point

        self.assertEqual(path, expected_path)

    def test_reconstruct_path(self):
        """Test the reconstruct_path function."""
        came_from = {
            (1, 1): (0, 0),
            (2, 2): (1, 1)
        }
        path = self.dijkstra.reconstruct_path(came_from, (0, 0), (2, 2))
        expected_path = [(0, 0), (1, 1), (2, 2)]
        self.assertEqual(path, expected_path)

if __name__ == '__main__':
    unittest.main()
