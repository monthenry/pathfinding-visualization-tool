import tkinter as tk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import networkx as nx
import matplotlib.pyplot as plt
import threading
import numpy as np
from queue import PriorityQueue

class PathFindingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Path Finding App")

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.image = None
        self.start_point = None
        self.end_point = None
        self.algorithm_in_progress = False

        # Create a single frame for all buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        # Create and pack algorithm buttons to the left in the frame
        self.algorithm_buttons = []
        self.create_algorithm_buttons()

        # Create and pack the load image button to the right in the same frame
        self.load_image_button = tk.Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_image_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def create_algorithm_buttons(self):
        algorithms = ["Dijkstra", "A*", "Floyd-Warshall", "Bellman-Ford"]
        for algorithm in algorithms:
            button = tk.Button(self.button_frame, text=algorithm, command=lambda algo=algorithm: self.run_algorithm(algo))
            button.pack(side=tk.LEFT, padx=10, pady=10)
            self.algorithm_buttons.append(button)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.image = Image.open(file_path)
                # Ensure the image is in RGBA mode (with alpha channel)
                self.image = self.image.convert("RGBA")
                self.image = self.image.resize((800, 600), Image.ANTIALIAS)
                self.display_image = ImageTk.PhotoImage(self.image)  # Create ImageTk.PhotoImage
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
            except Exception as e:
                print("Error loading image:", str(e))

    def on_canvas_click(self, event):
        if self.algorithm_in_progress:
            return

        x, y = event.x, event.y
        diameter = 10

        if self.start_point is None:
            self.start_point = (x, y)
            self.canvas.create_oval(x - diameter, y - diameter, x + diameter, y + diameter, fill="green")
        elif self.end_point is None:
            self.end_point = (x, y)
            self.canvas.create_oval(x - diameter, y - diameter, x + diameter, y + diameter, fill="red")

    def run_algorithm(self, algorithm):
        if self.image is None or self.start_point is None or self.end_point is None:
            return

        if self.algorithm_in_progress:
            self.algorithm_in_progress = False
            return

        self.algorithm_in_progress = True

        if algorithm == "Dijkstra":
            threading.Thread(target=self.run_dijkstra).start()
        elif algorithm == "A*":
            threading.Thread(target=self.run_a_star).start()
        elif algorithm == "Floyd-Warshall":
            threading.Thread(target=self.run_floyd_warshall).start()
        elif algorithm == "Bellman-Ford":
            threading.Thread(target=self.run_bellman_ford).start()

    def run_dijkstra(self):
        if self.start_point is None or self.end_point is None:
            self.algorithm_in_progress = False
            return

        graph = self.create_image_graph()
        self.custom_dijkstra(graph, self.start_point, self.end_point)

        self.algorithm_in_progress = False

    def custom_dijkstra(self, graph, start, goal):
        visited = set()
        queue = PriorityQueue()
        queue.put((0, start))
        visited_nodes = []

        while not queue.empty():
            (dist, current_node) = queue.get()
            if current_node == goal:
                break

            if current_node in visited:
                continue

            visited.add(current_node)
            visited_nodes.append(current_node)  # Store the visited node

            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    queue.put((dist + graph[current_node][neighbor]["cost"], neighbor))

        # After exploration, visualize the shortest path
        try:
            path = nx.shortest_path(graph, source=start, target=goal, weight='cost')
            self.visualize_path(path, visited_nodes)
        except nx.NetworkXNoPath:
            pass

    def run_a_star(self):
        if self.start_point is None or self.end_point is None:
            self.algorithm_in_progress = False
            return

        graph = self.create_image_graph()
        self.custom_a_star(graph, self.start_point, self.end_point)

        self.algorithm_in_progress = False

    def custom_a_star(self, graph, start, goal):
        visited = set()
        queue = PriorityQueue()
        queue.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        visited_nodes = []

        while not queue.empty():
            _, current = queue.get()

            if current == goal:
                break

            if current not in visited:
                visited.add(current)
                visited_nodes.append(current)  # Store the visited node

            for next_node in graph.neighbors(current):
                new_cost = cost_so_far[current] + graph[current][next_node]['cost']
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.euclidean_distance_points(next_node, goal)
                    queue.put((priority, next_node))
                    came_from[next_node] = current

        # Reconstruct and visualize the path
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        self.visualize_path(path, visited_nodes)

    def run_floyd_warshall(self):
        if self.start_point is None or self.end_point is None:
            self.algorithm_in_progress = False
            return

        # Create a graph from the image pixels
        graph = self.create_image_graph()

        print("start")

        # Apply Floyd-Warshall algorithm
        try:
            dist, next_node = self.floyd_warshall(graph)
            path = self.reconstruct_path_floyd_warshall(self.start_point, self.end_point, next_node)
        except Exception as e:
            print("Error in Floyd-Warshall algorithm:", str(e))
            path = None

        print("end")

        if path:
            self.visualize_path(path)

        self.algorithm_in_progress = False

    def floyd_warshall(self, graph):
        # Number of vertices in the graph
        num_vertices = len(graph)

        # Initialize distance and next matrices
        dist = {v: dict.fromkeys(graph, float('inf')) for v in graph}
        next_node = {v: dict.fromkeys(graph, None) for v in graph}

        # Set distance to 0 for self loops and initialize next matrix
        for v in graph:
            dist[v][v] = 0
            for neighbor in graph[v]:
                dist[v][neighbor] = graph[v][neighbor]['cost']
                next_node[v][neighbor] = neighbor

        # Floyd-Warshall main loop
        for k in graph:
            for i in graph:
                for j in graph:
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]

        return dist, next_node

    def reconstruct_path(self, start, end, next_node):
        # Reconstructs the path from start to end using the next matrix
        if next_node[start][end] is None:
            return []

        path = [start]
        while start != end:
            start = next_node[start][end]
            path.append(start)

        return path


    def run_bellman_ford(self):
        if self.start_point is None or self.end_point is None:
            self.algorithm_in_progress = False
            return

        graph = self.create_image_graph()

        try:
            # Using NetworkX's single source shortest path with Bellman-Ford method
            # This function returns a tuple (lengths, paths) where paths is a dictionary of shortest paths
            lengths, paths = nx.single_source_bellman_ford(graph, self.start_point, weight='cost')
            path = paths[self.end_point]
        except (nx.NetworkXNoPath, KeyError):
            path = None

        if path:
            self.visualize_path(path)

        self.algorithm_in_progress = False


    def create_image_graph(self):
        if self.image is None:
            return None

        img_array = np.array(self.image)

        if img_array.ndim == 2:  # Convert grayscale to RGB if needed
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] not in {3, 4}:
            raise ValueError("Image must have 3 or 4 channels (RGB or RGBA format)")

        if img_array.size == 0:
            raise ValueError("Empty image")

        G = nx.Graph()

        if img_array.ndim == 3:
            height, width, _ = img_array.shape
        else:
            raise ValueError("Unsupported image format")

        for y in range(height):
            for x in range(width):
                G.add_node((x, y))

                if x < width - 1:
                    right_pixel_cost = self.calculate_pixel_cost(img_array[y, x], img_array[y, x + 1])
                    G.add_edge((x, y), (x + 1, y), cost=right_pixel_cost)

                if y < height - 1:
                    bottom_pixel_cost = self.calculate_pixel_cost(img_array[y, x], img_array[y + 1, x])
                    G.add_edge((x, y), (x, y + 1), cost=bottom_pixel_cost)

        return G

    def calculate_pixel_cost(self, color1, color2):
        # Calculate the cost between two nodes based on Euclidean distance between colors
        color_distance = self.euclidean_distance_color(color1, color2)
        # color_distance = self.manhattan_distance_color(color1, color2)
        # color_distance = self.chebyshev_distance_color(color1, color2)
        # color_distance = self.threshold_distance_color(color1, color2)

        return color_distance
    
    def euclidean_distance_points(self, point1, point2):
        # Calculate Euclidean distance between two points
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def euclidean_distance_color(self, color1, color2):
        # Normalize the color channel values to [0, 255] and cast to integers
        r1, g1, b1, a1 = map(int, color1)
        r2, g2, b2, a2 = map(int, color2)

        # Calculate the color difference for each channel (R, G, B, and A)
        delta_r = r1 - r2
        delta_g = g1 - g2
        delta_b = b1 - b2
        delta_a = a1 - a2

        # Calculate the Euclidean distance between the colors
        distance = np.sqrt(delta_r ** 2 + delta_g ** 2 + delta_b ** 2 + delta_a ** 2)

        return distance
    
    def manhattan_distance_color(self, color1, color2):
        # Calculate the Manhattan distance between two RGB colors
        r1, g1, b1, a1 = map(int, color1)
        r2, g2, b2, a2 = map(int, color2)

        delta_r = abs(r1 - r2)
        delta_g = abs(g1 - g2)
        delta_b = abs(b1 - b2)
        delta_a = abs(a1 - a2)

        distance = delta_r + delta_g + delta_b + delta_a

        return distance

    def chebyshev_distance_color(self, color1, color2):
        # Calculate the Chebyshev distance between two RGB colors
        r1, g1, b1, a1 = map(int, color1)
        r2, g2, b2, a2 = map(int, color2)

        delta_r = abs(r1 - r2)
        delta_g = abs(g1 - g2)
        delta_b = abs(b1 - b2)
        delta_a = abs(a1 - a2)

        distance = max(delta_r, delta_g, delta_b, delta_a)

        return distance

    def threshold_distance_color(self, color1, color2, threshold=20):
        # Calculate a threshold-based distance between two RGB colors
        r1, g1, b1, a1 = map(int, color1)
        r2, g2, b2, a2 = map(int, color2)

        delta_r = abs(r1 - r2)
        delta_g = abs(g1 - g2)
        delta_b = abs(b1 - b2)
        delta_a = abs(a1 - a2)

        # Check if any color channel difference exceeds the threshold
        if max(delta_r, delta_g, delta_b, delta_a) > threshold:
            return float('inf')  # Return infinity as the distance if threshold is exceeded
        else:
            return 0  # Return zero as the distance if within the threshold

    def visualize_path(self, path, visited_nodes):
        # Create a copy of the original image to modify
        img = self.image.copy()
        pixels = img.load()

        # Invert the color of the visited nodes
        for node in visited_nodes:
            x, y = node
            if x < img.width and y < img.height:
                inverted_color = self.invert_color(pixels[x, y])
                pixels[x, y] = inverted_color

        # Draw the path on the image
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            self.draw_line(pixels, x1, y1, x2, y2, (0, 0, 255, 255))  # Blue color for the path

        # Redraw the start and end points
        start_x, start_y = self.start_point
        end_x, end_y = self.end_point
        self.draw_point(pixels, start_x, start_y, (0, 255, 0, 255))  # Green color for the start point
        self.draw_point(pixels, end_x, end_y, (255, 0, 0, 255))  # Red color for the end point

        # Update the canvas with the modified image
        self.display_image = ImageTk.PhotoImage(img)  # Convert to a format suitable for Tkinter
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)

    def invert_color(self, color):
        """Invert a given color."""
        r, g, b, a = color
        return (255 - r, 255 - g, 255 - b, a)

    def draw_point(self, pixels, x, y, color, size=6):
        """Draw a point on the image."""
        img_width, img_height = self.image.size
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if 0 <= x + i < img_width and 0 <= y + j < img_height:
                    pixels[x + i, y + j] = color

    def draw_line(self, pixels, x1, y1, x2, y2, color, thickness=3):
        """Draw a thicker line on the image using Bresenham's line algorithm."""
        def plot_line_low(x0, y0, x1, y1):
            dx = x1 - x0
            dy = y1 - y0
            yi = 1
            if dy < 0:
                yi = -1
                dy = -dy
            D = 2*dy - dx
            y = y0

            for x in range(x0, x1 + 1):
                for t in range(-thickness // 2, thickness // 2 + 1):
                    pixels[x, y + t] = color
                if D > 0:
                    y += yi
                    D -= 2*dx
                D += 2*dy

        def plot_line_high(x0, y0, x1, y1):
            dx = x1 - x0
            dy = y1 - y0
            xi = 1
            if dx < 0:
                xi = -1
                dx = -dx
            D = 2*dx - dy
            x = x0

            for y in range(y0, y1 + 1):
                for t in range(-thickness // 2, thickness // 2 + 1):
                    pixels[x + t, y] = color
                if D > 0:
                    x += xi
                    D -= 2*dy
                D += 2*dx

        if abs(y2 - y1) < abs(x2 - x1):
            if x1 > x2:
                plot_line_low(x2, y2, x1, y1)
            else:
                plot_line_low(x1, y1, x2, y2)
        else:
            if y1 > y2:
                plot_line_high(x2, y2, x1, y1)
            else:
                plot_line_high(x1, y1, x2, y2)



if __name__ == "__main__":
    root = tk.Tk()
    app = PathFindingApp(root)
    root.mainloop()
