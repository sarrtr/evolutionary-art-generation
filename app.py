import pygame
import numpy as np
from PIL import Image, ImageDraw
from deap import base, creator, tools, algorithms
from genetic_algorithm import (
    create_individual,
    mutate,
    cxTwoPoint,
    cluster_individuals,
    SHAPE_TYPES,
    COLOR_RANGE
)
import math

# Constants
POPULATION_SIZE = 50
USER_SELECTION_SIZE = 6 # Images per screen
# Adjust grid size dynamically (smallest integer >= sqrt(POPULATION_SIZE))
GRID_SIZE = math.ceil(USER_SELECTION_SIZE ** 0.5)
WIDTH, HEIGHT = 1000, 600
SHAPES_PER_IMAGE = 8
MUTATION_RATE = 0.5
SPECIES_COUNT = 4
PADDING = 10

class EvolutionApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.generation = 0
        self.current_batch = 0
        self.batch_size = USER_SELECTION_SIZE  # Images per screen
        self.current_batch = 0
        self.total_batches = (POPULATION_SIZE + self.batch_size - 1) // self.batch_size
        self.population = []
        self.selected_indices = []
        self.save_selected = []
        self.selection_mode = 'generation'
        self.running = True
        self.init_deap()
        self.init_ui()

    def init_deap(self):
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", create_individual, SHAPES_PER_IMAGE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", cxTwoPoint)
        self.toolbox.register("mutate", mutate, mutation_rate=MUTATION_RATE)
        self.population = self.toolbox.population(n=POPULATION_SIZE)

    def init_ui(self):
        self.font = pygame.font.SysFont("Arial", 20)
        self.buttons = {
            'save': pygame.Rect(10, HEIGHT - 50, 100, 40),
            'reset': pygame.Rect(120, HEIGHT - 50, 100, 40),
            'choose next': pygame.Rect(WIDTH // 2 - 70, HEIGHT - 50, 140, 40)
        }

    def genome_to_image(self, genome, img_size):
        """Convert genome to a PIL Image with scaling."""
        img = Image.new("RGB", img_size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        for shape in genome:
            color = tuple(shape['color'])
            scale_x = lambda x: int(x * img_size[0])
            scale_y = lambda y: int(y * img_size[1])
            
            if shape['type'] == 'circle':
                x = scale_x(shape['x'])
                y = scale_y(shape['y'])
                radius = scale_x(shape['radius'])
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
                
            elif shape['type'] == 'rect':
                x = scale_x(shape['x'])
                y = scale_y(shape['y'])
                w = scale_x(shape['w'])
                h = scale_y(shape['h'])
                draw.rectangle([x, y, x + w, y + h], fill=color)
                
            elif shape['type'] == 'line':
                x1 = scale_x(shape['x1'])
                y1 = scale_y(shape['y1'])
                x2 = scale_x(shape['x2'])
                y2 = scale_y(shape['y2'])
                draw.line([x1, y1, x2, y2], fill=color, width=shape['width'])
            
            elif shape['type'] == 'triangle':
                points = [
                    (scale_x(shape['x1']), scale_y(shape['y1'])),
                    (scale_x(shape['x2']), scale_y(shape['y2'])),
                    (scale_x(shape['x3']), scale_y(shape['y3']))
                ]
                draw.polygon(points, fill=color)
            
            elif shape['type'] == 'polygon':
                points = [(scale_x(x), scale_y(y)) for x, y in shape['points']]
                draw.polygon(points, fill=color)
            
            elif shape['type'] == 'ellipse':
                x1 = scale_x(shape['x1'])
                y1 = scale_y(shape['y1'])
                x2 = scale_x(shape['x2'])
                y2 = scale_y(shape['y2'])
                draw.ellipse([x1, y1, x2, y2], fill=color)
        
        return img

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                if self.buttons['save'].collidepoint(x, y):
                    self.handle_save_button()
                elif self.buttons['reset'].collidepoint(x, y):
                    self.init_deap()
                    self.generation = 0
                elif self.buttons['choose next'].collidepoint(x, y):
                    self.handle_image_click(x, y, choose=False)
                else:
                    self.handle_image_click(x, y, choose=True)

    def handle_save_button(self):
        if self.selection_mode == 'generation':
            self.selection_mode = 'save'
        else:
            for i, idx in enumerate(self.save_selected):
                img = self.genome_to_image(self.population[idx], (1024, 768))
                img.save(f"image_{i + 1}.png")
            self.selection_mode = 'generation'
            self.save_selected = []

    def handle_image_click(self, x, y, choose):
        if choose:
            col = x // (WIDTH // GRID_SIZE)
            row = y // (HEIGHT // GRID_SIZE)
            idx = row * GRID_SIZE + col
        else:
            idx = 0
        
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.population))

        if start_idx + idx < end_idx:
            if self.selection_mode == 'generation':
                # Allow only one selection per batch
                if start_idx + idx not in self.selected_indices:
                    if choose:
                        self.selected_indices.append(start_idx + idx)
                    self.current_batch += 1  # Move to the next batch
                    if self.current_batch >= self.total_batches:
                        self.create_next_generation()  # Process the next generation
            elif self.selection_mode == 'save':
                if idx in self.save_selected:
                    self.save_selected.remove(idx)
                else:
                    self.save_selected.append(idx)

    def calculate_diversity(self, individual):
        """
        Calculate the diversity of an individual based on the variety of shapes and their properties.
        Returns a diversity score (higher is more diverse).
        """
        shape_counts = {shape_type: 0 for shape_type in SHAPE_TYPES}
        color_values = []
        size_values = []

        for shape in individual:
            # Count shape types
            shape_counts[shape['type']] += 1

            # Collect color values (average of RGB)
            color_values.append(np.mean(shape['color']))

            # Collect size-related values
            if shape['type'] == 'circle':
                size_values.append(shape['radius'])
            elif shape['type'] == 'rect':
                size_values.append(shape['w'] * shape['h'])  # Area of the rectangle
            elif shape['type'] == 'line':
                length = np.sqrt((shape['x2'] - shape['x1'])**2 + (shape['y2'] - shape['y1'])**2)
                size_values.append(length)
            elif shape['type'] == 'triangle':
                # Approximate area of the triangle using vertices
                x1, y1 = shape['x1'], shape['y1']
                x2, y2 = shape['x2'], shape['y2']
                x3, y3 = shape['x3'], shape['y3']
                area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2
                size_values.append(area)
            elif shape['type'] == 'polygon':
                # Approximate diversity by the number of points
                size_values.append(len(shape['points']))
            elif shape['type'] == 'ellipse':
                width = abs(shape['x2'] - shape['x1'])
                height = abs(shape['y2'] - shape['y1'])
                size_values.append(width * height)  # Approximate area of the ellipse

        # Calculate diversity metrics
        shape_diversity = len([count for count in shape_counts.values() if count > 0]) / len(SHAPE_TYPES)
        color_std = np.std(color_values) if color_values else 0
        size_std = np.std(size_values) if size_values else 0

        # Combine metrics into a single diversity score
        diversity_score = shape_diversity + 0.5 * color_std + 0.5 * size_std
        return diversity_score

    def create_next_generation(self):
        # Assign fitness based on selections.
        # for ind in self.population:
        #     ind.fitness.values = (0.0,)
        # for idx in self.selected_indices:
        #     self.population[idx].fitness.values = (1.0,)

        # Assign fitness based on selections with a more complex function.
        selection_counts = {}
        for idx in set(self.selected_indices):
            shape_types = [item['type'] for item in self.population[idx]]
            unique_shapes = len(set(shape_types))
            selection_counts[idx] = unique_shapes * 0.4
            shape_colors = [item['color'] for item in self.population[idx]]
            unique_colors = len(set(shape_colors))
            selection_counts[idx] += unique_colors * 0.6

        max_count = max(selection_counts.values()) if selection_counts else 1

        for idx, count in selection_counts.items():
            # Fitness is a weighted combination of selection count and diversity factor.
            diversity_score = self.calculate_diversity(self.population[idx])
            diversity_factor = 1.0 / (1.0 + np.std(diversity_score))
            normalized_count = count / max_count
            self.population[idx].fitness.values = (0.7 * normalized_count + 0.3 * diversity_factor,)
            print(self.population[idx].fitness.values)

        # for idx, individual in enumerate(self.population):
        #     if idx not in selection_counts:
        #         diversity_score = self.calculate_diversity(individual)
        #         self.population[idx].fitness.values = (diversity_score*0.1,)

        # Speciation using clustering.
        clusters = cluster_individuals(self.population, SPECIES_COUNT)

        # Select survivors from each species.
        survivors = []
        for cluster_id in range(SPECIES_COUNT):
            cluster = [ind for ind, c in zip(self.population, clusters) if c == cluster_id]
            if cluster:
                survivors.append(max(cluster, key=lambda ind: ind.fitness.values))
        
        # Generate offspring to refill the population.
        offspring = algorithms.varOr(
            survivors, 
            self.toolbox, 
            len(self.population) - len(survivors),
            cxpb=0.5, 
            mutpb=0.2
        )
        
        # Inject new random individuals to boost diversity (e.g., 2 new individuals).
        random_individuals = self.toolbox.population(n=10)
        
        # Combine survivors, offspring, and random individuals.
        new_population = survivors + offspring + random_individuals
        self.population = new_population[:POPULATION_SIZE]
        
        self.generation += 1
        self.current_batch = 0  # Reset to first batch
        self.selected_indices = []
    
    def draw_gradient_background(self):
        """Draw a gradient background."""
        colors = [(82, 1, 32), (8, 64, 62), (112, 101, 19), (181, 113, 20), (150, 43, 9)]
        num_colors = len(colors)
        segment_height = HEIGHT // (num_colors - 1)

        for i in range(num_colors - 1):
            start_color = colors[i]
            end_color = colors[i + 1]
            for y in range(segment_height):
                ratio = y / segment_height
                color = tuple(
                    int(start_color[j] * (1 - ratio) + end_color[j] * ratio) for j in range(3)
                )
                pygame.draw.line(
                    self.screen, color, (0, i * segment_height + y), (WIDTH, i * segment_height + y)
                )

    def draw_rounded_button(self, rect, text, is_hovered):
        """Draw a rounded button with hover effect."""
        color = (180, 180, 180) if is_hovered else (200, 200, 200)
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        btn_text = self.font.render(text.capitalize(), True, (0, 0, 0))
        self.screen.blit(btn_text, (rect.x + rect.width // 2 - btn_text.get_width() // 2, rect.y + rect.height // 2 - btn_text.get_height() // 2))

    def draw_ui(self):
        """Draw the UI elements including the generation counter and buttons."""
        # Draw generation counter
        gen_text = self.font.render(f"Generation: {self.generation}", True, (255, 255, 255))
        self.screen.blit(gen_text, (WIDTH // 2 - gen_text.get_width() // 2, 10))

        # Draw buttons with hover effects
        mouse_pos = pygame.mouse.get_pos()
        for name, rect in self.buttons.items():
            is_hovered = rect.collidepoint(mouse_pos)
            self.draw_rounded_button(rect, name, is_hovered)

    def run(self):
        while self.running:
            MARGIN_TOP = 50
            self.draw_gradient_background() 
            self.handle_events()

            start_idx = self.current_batch * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.population))
            batch = self.population[start_idx:end_idx]


            cell_width = WIDTH // GRID_SIZE
            cell_height = HEIGHT // GRID_SIZE
            image_size = (cell_width - 2 * PADDING, cell_height - 2 * PADDING)

            for i, ind in enumerate(batch):
                row = i // GRID_SIZE
                col = i % GRID_SIZE
                x = col * cell_width + PADDING
                y = row * cell_height + PADDING + MARGIN_TOP
                
                img = self.genome_to_image(ind, image_size)
                mode = img.mode
                data = img.tobytes()
                py_image = pygame.image.fromstring(data, img.size, mode)
                
                if start_idx + i in self.selected_indices or start_idx + i in self.save_selected:
                    border_color = (0, 255, 0) if self.selection_mode == 'generation' else (255, 0, 0)
                    pygame.draw.rect(self.screen, border_color, 
                                     (x - PADDING, y - PADDING, image_size[0] + 2 * PADDING, image_size[1] + 2 * PADDING), 3)
                
                self.screen.blit(py_image, (x, y))
            
            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(30)

if __name__ == "__main__":
    app = EvolutionApp()
    app.run()
