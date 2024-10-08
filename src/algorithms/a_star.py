import threading
from queue import PriorityQueue

class AStar:
    def __init__(self, image_processor, start_point, end_point, path_visualizer):
        self.image_processor = image_processor
        self.start_point = start_point
        self.end_point = end_point
        self.path_visualizer = path_visualizer
        self.algorithm_in_progress = False

    def run(self):
        if self.start_point is None or self.end_point is None:
            return

        self.algorithm_in_progress = True
        threading.Thread(target=self.execute).start()

    def execute(self):
        graph = self.image_processor.create_image_graph()
        visited_nodes = self.custom_a_star(graph, self.start_point, self.end_point)
        self.path_visualizer.visualize_path(visited_nodes)

        self.algorithm_in_progress = False

    def custom_a_star(self, graph, start, goal):
        open_set = PriorityQueue()
        open_set.put(start)
        came_from = {start: None}
        g_score = {node: float('inf') for node in graph.nodes()}
        g_score[start] = 0
        f_score = {node: float('inf') for node in graph.nodes()}
        f_score[start] = self.heuristic(start, goal)

        while not open_set.empty():
            current = open_set.get()

            if current == goal:
                return self.reconstruct_path(came_from, start, goal)

            for neighbor in graph.neighbors(current):
                edge_data = graph.get_edge_data(current, neighbor)
                tentative_g_score = g_score[current] + edge_data['cost']

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)

                    if neighbor not in open_set.queue:
                        open_set.put(neighbor)

        return []

    def heuristic(self, node, goal):
        # Implement a heuristic function for A* (e.g., Manhattan distance)
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def reconstruct_path(self, came_from, start, goal):
        current = goal
        path = []

        while current is not None:
            path.append(current)
            current = came_from[current]

        path.reverse()
        return path
