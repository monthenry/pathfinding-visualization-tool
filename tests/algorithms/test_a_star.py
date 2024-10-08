import unittest
from unittest.mock import MagicMock
from src.algorithms.a_star import AStar

class TestAStar(unittest.TestCase):

    def setUp(self):
        """Set up the necessary components before each test."""
        self.image_processor = MagicMock()
        self.path_visualizer = MagicMock()

        # Define start and end points for the A* algorithm
        self.start_point = (0, 0)
        self.end_point = (2, 2)

        # Create an instance of AStar
        self.a_star = AStar(self.image_processor, self.start_point, self.end_point, self.path_visualizer)

    def test_run_without_start_or_end_point(self):
        """Test run method when start or end point is None."""
        self.a_star.start_point = None
        self.a_star.run()
        self.assertFalse(self.a_star.algorithm_in_progress)

        self.a_star.start_point = (0, 0)
        self.a_star.end_point = None
        self.a_star.run()
        self.assertFalse(self.a_star.algorithm_in_progress)

    def test_run_triggers_execution(self):
        """Test run method triggers the execute method."""
        self.image_processor.create_image_graph.return_value = MagicMock()
        self.a_star.run()
        self.assertTrue(self.a_star.algorithm_in_progress)

    def test_custom_a_star_path_found(self):
        """Test the custom_a_star method when a path is found."""
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = [(0, 0), (1, 1), (2, 2)]
        mock_graph.neighbors.side_effect = [
            [(1, 1)],   # Neighbors of (0, 0)
            [(0, 0), (2, 2)],  # Neighbors of (1, 1)
            []  # Neighbors of (2, 2)
        ]
        mock_graph.get_edge_data.side_effect = [
            {'cost': 1},  # Cost to move from (0, 0) to (1, 1)
            {'cost': 1},  # Cost to move from (1, 1) to (2, 2)
        ]
        
        self.image_processor.create_image_graph.return_value = mock_graph
        
        path = self.a_star.custom_a_star(mock_graph, self.start_point, self.end_point)
        expected_path = [(0, 0), (1, 1), (2, 2)]

        self.assertEqual(path, expected_path)

    def test_custom_a_star_no_path(self):
        """Test the custom_a_star method when no path is found."""
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = [(0, 0), (1, 1), (2, 2)]
        mock_graph.neighbors.side_effect = [
            [(1, 1)],  # Neighbors of (0, 0)
            [(0, 0)],  # Neighbors of (1, 1) (isolated, no path to (2, 2))
            []         # Neighbors of (2, 2)
        ]
        
        self.image_processor.create_image_graph.return_value = mock_graph
        
        path = self.a_star.custom_a_star(mock_graph, self.start_point, self.end_point)
        expected_path = []

        self.assertEqual(path, expected_path)

    def test_heuristic(self):
        """Test the heuristic function."""
        heuristic_value = self.a_star.heuristic((1, 1), (2, 2))
        self.assertEqual(heuristic_value, 2)

    def test_reconstruct_path(self):
        """Test the reconstruct_path function."""
        came_from = {
            (1, 1): (0, 0),
            (2, 2): (1, 1)
        }
        path = self.a_star.reconstruct_path(came_from, (0, 0), (2, 2))
        expected_path = [(0, 0), (1, 1), (2, 2)]
        self.assertEqual(path, expected_path)

if __name__ == '__main__':
    unittest.main()
