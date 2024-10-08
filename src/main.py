import tkinter as tk
from tkinter import filedialog
from utils.image_processing import ImageProcessor
from visualizations.path_visualization import PathVisualizer
from algorithms.dijkstra import Dijkstra
from algorithms.a_star import AStar
from algorithms.floyd_warshall import FloydWarshall
from algorithms.bellman_ford import BellmanFord

class PathFindingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Path Finding App")

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.image_processor = ImageProcessor(self.canvas)
        self.path_visualizer = PathVisualizer(self.image_processor)

        self.start_point = None
        self.end_point = None
        self.algorithm_in_progress = False

        # Create a single frame for all buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        # Create and pack algorithm buttons to the left in the frame
        self.create_algorithm_buttons()

        # Create and pack the load image button to the right in the same frame
        self.load_image_button = tk.Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_image_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def create_algorithm_buttons(self):
        algorithms = [("Dijkstra", Dijkstra), ("A*", AStar), ("Floyd-Warshall", FloydWarshall), ("Bellman-Ford", BellmanFord)]
        for name, algorithm in algorithms:
            button = tk.Button(self.button_frame, text=name, command=lambda algo=algorithm: self.run_algorithm(algo))
            button.pack(side=tk.LEFT, padx=10, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_processor.load_image(file_path)

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
        if self.image_processor.image is None or self.start_point is None or self.end_point is None:
            return

        if self.algorithm_in_progress:
            self.algorithm_in_progress = False
            return

        self.algorithm_in_progress = True
        algorithm_instance = algorithm(self.image_processor, self.start_point, self.end_point, self.path_visualizer)
        algorithm_instance.run()

if __name__ == "__main__":
    root = tk.Tk()
    app = PathFindingApp(root)
    root.mainloop()