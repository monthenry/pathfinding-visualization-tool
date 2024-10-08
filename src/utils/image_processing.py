import numpy as np
from PIL import Image, ImageTk
import networkx as nx

class ImageProcessor:
    def __init__(self, canvas):
        self.canvas = canvas
        self.image = None
        self.display_image = None

    def load_image(self, file_path):
        try:
            self.image = Image.open(file_path)
            self.image = self.image.convert("RGBA")
            self.image = self.image.resize((800, 600), Image.ANTIALIAS)
            self.display_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor='nw', image=self.display_image)
        except Exception as e:
            print("Error loading image:", str(e))

    def create_image_graph(self):
        if self.image is None:
            return None

        img_array = np.array(self.image)

        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] not in {3, 4}:
            raise ValueError("Image must have 3 or 4 channels (RGB or RGBA format)")

        if img_array.size == 0:
            raise ValueError("Empty image")

        G = nx.Graph()
        height, width, _ = img_array.shape

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
        return self.euclidean_distance_color(color1, color2)

    def euclidean_distance_color(self, color1, color2):
        r1, g1, b1, a1 = map(int, color1)
        r2, g2, b2, a2 = map(int, color2)

        distance = np.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2 + (a1 - a2) ** 2)
        return distance
