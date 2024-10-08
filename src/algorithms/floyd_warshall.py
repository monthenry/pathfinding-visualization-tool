class FloydWarshall:
    def __init__(self, image_processor, start_point, end_point, path_visualizer):
        self.image_processor = image_processor
        self.start_point = start_point
        self.end_point = end_point
        self.path_visualizer = path_visualizer

    def run(self):
        if self.start_point is None or self.end_point is None:
            return

        graph = self.image_processor.create_image_graph()
        visited_nodes = self.custom_floyd_warshall(graph)
        self.path_visualizer.visualize_path(visited_nodes)

    def custom_floyd_warshall(self, graph):
        distances = {}
        next_node = {}

        for node in graph.nodes():
            distances[node] = {}
            for neighbor in graph.neighbors(node):
                distances[node][neighbor] = graph[node][neighbor]['cost']
                next_node[node] = {neighbor: neighbor}

        for k in graph.nodes():
            for i in graph.nodes():
                for j in graph.nodes():
                    if (i in distances) and (k in distances[i]) and (j in distances[k]):
                        if (j not in distances[i]) or (distances[i][k] + distances[k][j] < distances[i][j]):
                            distances[i][j] = distances[i][k] + distances[k][j]
                            next_node[i][j] = next_node[i][k]

        path = self.reconstruct_path(next_node)
        return path

    def reconstruct_path(self, next_node):
        path = []

        # Implement path reconstruction logic based on next_node
        # Add logic to build the path from start to end

        return path
