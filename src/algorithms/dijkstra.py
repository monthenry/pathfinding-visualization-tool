import threading
import networkx as nx
from queue import PriorityQueue

class Dijkstra:
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
        visited_nodes = self.custom_dijkstra(graph, self.start_point, self.end_point)
        self.path_visualizer.visualize_path(visited_nodes)

        self.algorithm_in_progress = False

    def custom_dijkstra(self, graph, start, goal):
        visited = set()
        pq = PriorityQueue()
        pq.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while not pq.empty():
            current_cost, current_node = pq.get()

            if current_node == goal:
                break

            visited.add(current_node)

            for neighbor in graph.neighbors(current_node):
                edge_data = graph.get_edge_data(current_node, neighbor)
                new_cost = cost_so_far[current_node] + edge_data['cost']

                if neighbor not in visited and (neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]):
                    cost_so_far[neighbor] = new_cost
                    pq.put((new_cost, neighbor))
                    came_from[neighbor] = current_node

        return self.reconstruct_path(came_from, start, goal)

    def reconstruct_path(self, came_from, start, goal):
        current = goal
        path = []

        while current is not None:
            path.append(current)
            current = came_from[current]

        path.reverse()
        return path
