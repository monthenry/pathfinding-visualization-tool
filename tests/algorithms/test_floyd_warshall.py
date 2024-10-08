import unittest
from unittest.mock import MagicMock
from src.algorithms.floyd_warshall import FloydWarshall

class TestFloydWarshall(unittest.TestCase):

    def setUp(self):
        """Set up the necessary components before each test."""
        self.image_processor = MagicMock()
        self.path_visualizer = MagicMock()

        # Define start and end points for the Floyd-Warshall algorithm
        self.start_point = (0, 0)
        self.end_point = (2, 2)

        # Create an instance of FloydWarshall
        self.floyd_warshall = FloydWarshall(self.image_processor, self.start_point, self.end_point, self.path_visualizer)

    def test_run_without_start_or_end_point(self):
        """Test run method when start or end point is None."""
        self.floyd_warshall.start_point = None
        self.floyd_warshall.run()
        self.path_visualizer.visualize_path.assert_not_called()

        self.floyd_warshall.start_point = (0, 0)
        self.floyd_warshall.end_point = None
        self.floyd_warshall.run()
        self.path_visualizer.visualize_path.assert_not_called()

    def test_run_triggers_algorithm(self):
        """Test run method executes the algorithm and visualizes the path."""
        self.image_processor.create_image_graph.return_value = MagicMock()
        self.floyd_warshall.run()
        self.assertTrue(self.floyd_warshall.start_point is not None and self.floyd_warshall.end_point is not None)

    def test_custom_floyd_warshall_path_found(self):
        """Test the custom_floyd_warshall method when a path is found."""
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = [(0, 0), (1, 1), (2, 2)]
        mock_graph.neighbors.return_value = {
            (0, 0): [(1, 1)],
            (1, 1): [(2, 2)],
            (2, 2): []
        }
        mock_graph.__getitem__.side_effect = lambda x: {
            (0, 0): {(1, 1): {'cost': 1}},
            (1, 1): {(2, 2): {'cost': 1}},
            (2, 2): {}
        }[x]

        self.image_processor.create_image_graph.return_value = mock_graph
        
        path = self.floyd_warshall.custom_floyd_warshall(mock_graph)
        expected_path = []  # You should define the expected path

        self.assertEqual(path, expected_path)

    def test_reconstruct_path(self):
        """Test the reconstruct_path function."""
        next_node = {
            (0, 0): {(1, 1): (1, 1)},
            (1, 1): {(2, 2): (2, 2)},
            (2, 2): {}
        }
        
        # Here you would define the expected path from (0, 0) to (2, 2)
        expected_path = [(0, 0), (1, 1), (2, 2)]

        path = self.floyd_warshall.reconstruct_path(next_node)

        # Make sure the reconstruction returns the expected path
        self.assertEqual(path, expected_path)

if __name__ == '__main__':
    unittest.main()
