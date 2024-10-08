from PIL import ImageTk

class PathVisualizer:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def visualize_path(self, path, visited_nodes):
        img = self.image_processor.image.copy()
        pixels = img.load()

        for node in visited_nodes:
            x, y = node
            if x < img.width and y < img.height:
                inverted_color = self.invert_color(pixels[x, y])
                pixels[x, y] = inverted_color

        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            self.draw_line(pixels, x1, y1, x2, y2, (0, 0, 255, 255))

        start_x, start_y = self.image_processor.start_point
        end_x, end_y = self.image_processor.end_point
        self.draw_point(pixels, start_x, start_y, (0, 255, 0, 255))
        self.draw_point(pixels, end_x, end_y, (255, 0, 0, 255))

        self.image_processor.display_image = ImageTk.PhotoImage(img)
        self.image_processor.canvas.create_image(0, 0, anchor='nw', image=self.image_processor.display_image)

    def invert_color(self, color):
        r, g, b, a = color
        return (255 - r, 255 - g, 255 - b, a)

    def draw_point(self, pixels, x, y, color, size=6):
        img_width, img_height = self.image_processor.image.size
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if 0 <= x + i < img_width and 0 <= y + j < img_height:
                    pixels[x + i, y + j] = color

    def draw_line(self, pixels, x1, y1, x2, y2, color, thickness=3):
        # Implement line drawing (Bresenham's line algorithm or similar)
        pass  # Placeholder for the line drawing implementation
