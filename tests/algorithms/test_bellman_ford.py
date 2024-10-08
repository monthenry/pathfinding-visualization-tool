import unittest
from unittest.mock import MagicMock
from src.algorithms.bellman_ford import BellmanFord

class TestBellmanFord(unittest.TestCase):

    def setUp(self):
        """Set up the necessary components before each test."""
        self.image_processor = MagicMock()
        self.path_visualizer = MagicMock()

        # Define start and end points for the Bellman-Ford algorithm
        self.start_point = (0, 0)
        self.end_point = (2, 2)

        # Create an instance of BellmanFord
        self.bellman_ford = BellmanFord(self.image_processor, self.start_point, self.end_point, self.path_visualizer)

    def test_run_without_start_or_end_point(self):
        """Test run method when start or end point is None."""
        self.bellman_ford.start_point = None
        self.bellman_ford.run()
        self.path_visualizer.visualize_path.assert_not_called()

        self.bellman_ford.start_point = (0, 0)
        self.bellman_ford.end_point = None
        self.bellman_ford.run()
        self.path_visualizer.visualize_path.assert_not_called()

    def test_run_triggers_algorithm(self):
        """Test run method executes the algorithm and visualizes the path."""
        self.image_processor.create_image_graph.return_value = MagicMock()
        self.bellman_ford.run()
        self.assertTrue(self.bellman_ford.start_point is not None and self.bellman_ford.end_point is not None)

    def test_custom_bellman_ford_path_found(self):
        """Test the custom_bellman_ford method when a path is found."""
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = [(0, 0), (1, 1), (2, 2)]
        mock_graph.edges.return_value = [
            ((0, 0), (1, 1), {'cost': 1}),
            ((1, 1), (2, 2), {'cost': 1}),
            ((0, 0), (2, 2), {'cost': 3}),
        ]

        self.image_processor.create_image_graph.return_value = mock_graph
        
        path = self.bellman_ford.custom_bellman_ford(mock_graph, self.start_point)
        expected_path = [(0, 0), (1, 1), (2, 2)]

        self.assertEqual(path, expected_path)

    def test_custom_bellman_ford_no_path(self):
        """Test the custom_bellman_ford method when no path is found."""
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = [(0, 0), (1, 1), (2, 2)]
        mock_graph.edges.return_value = [
            ((0, 0), (1, 1), {'cost': 1}),
            ((1, 1), (0, 0), {'cost': 1}),  # No path to (2, 2)
        ]

        self.image_processor.create_image_graph.return_value = mock_graph
        
        path = self.bellman_ford.custom_bellman_ford(mock_graph, self.start_point)
        expected_path = []  # No valid path to end_point

        self.assertEqual(path, expected_path)

    def test_reconstruct_path(self):
        """Test the reconstruct_path function."""
        predecessors = {
            (1, 1): (0, 0),
            (2, 2): (1, 1)
        }
        path = self.bellman_ford.reconstruct_path(predecessors, (0, 0))
        expected_path = [(0, 0), (1, 1), (2, 2)]
        self.assertEqual(path, expected_path)

if __name__ == '__main__':
    unittest.main()
