class BellmanFord:
    def __init__(self, image_processor, start_point, end_point, path_visualizer):
        self.image_processor = image_processor
        self.start_point = start_point
        self.end_point = end_point
        self.path_visualizer = path_visualizer

    def run(self):
        if self.start_point is None or self.end_point is None:
            return

        graph = self.image_processor.create_image_graph()
        visited_nodes = self.custom_bellman_ford(graph, self.start_point)
        self.path_visualizer.visualize_path(visited_nodes)

    def custom_bellman_ford(self, graph, start):
        distances = {node: float('inf') for node in graph.nodes()}
        distances[start] = 0
        predecessors = {node: None for node in graph.nodes()}

        for _ in range(len(graph.nodes()) - 1):
            for u, v, edge_data in graph.edges(data=True):
                weight = edge_data['cost']
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u

        return self.reconstruct_path(predecessors, start)

    def reconstruct_path(self, predecessors, start):
        path = []
        current = self.end_point

        while current is not None:
            path.append(current)
            current = predecessors[current]

        path.reverse()
        return path
