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
POPULATION_SIZE = 16
# Adjust grid size dynamically (smallest integer >= sqrt(POPULATION_SIZE))
GRID_SIZE = math.ceil(POPULATION_SIZE ** 0.5)
WIDTH, HEIGHT = 1200, 800
SHAPES_PER_IMAGE = 8
MUTATION_RATE = 0.4
SPECIES_COUNT = 4
PADDING = 5

class EvolutionApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.generation = 0
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
            'reset': pygame.Rect(120, HEIGHT - 50, 100, 40)
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
        
        return img

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if self.buttons['save'].collidepoint(x, y):
                    self.handle_save_button()
                elif self.buttons['reset'].collidepoint(x, y):
                    self.init_deap()
                    self.generation = 0
                else:
                    self.handle_image_click(x, y)

    def handle_save_button(self):
        if self.selection_mode == 'generation':
            self.selection_mode = 'save'
        else:
            for i, idx in enumerate(self.save_selected):
                img = self.genome_to_image(self.population[idx], (1024, 768))
                img.save(f"image_{i + 1}.png")
            self.selection_mode = 'generation'
            self.save_selected = []

    def handle_image_click(self, x, y):
        col = x // (WIDTH // GRID_SIZE)
        row = y // (HEIGHT // GRID_SIZE)
        idx = row * GRID_SIZE + col
        if idx < len(self.population):
            if self.selection_mode == 'generation':
                # Allow up to 2 selections instead of 3.
                if idx not in self.selected_indices and len(self.selected_indices) < 2:
                    self.selected_indices.append(idx)
                    if len(self.selected_indices) == 2:
                        self.create_next_generation()
            elif self.selection_mode == 'save':
                if idx in self.save_selected:
                    self.save_selected.remove(idx)
                else:
                    self.save_selected.append(idx)

    def create_next_generation(self):
        # Assign fitness based on selections.
        for ind in self.population:
            ind.fitness.values = (0.0,)
        for idx in self.selected_indices:
            self.population[idx].fitness.values = (1.0,)

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
        random_individuals = self.toolbox.population(n=2)
        
        # Combine survivors, offspring, and random individuals.
        new_population = survivors + offspring + random_individuals
        self.population = new_population[:POPULATION_SIZE]
        
        self.generation += 1
        self.selected_indices = []

    def draw_ui(self):
        """Draw the UI elements including the generation counter and buttons."""
        text = self.font.render(f"Gen: {self.generation} | Mode: {self.selection_mode}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        
        for name, rect in self.buttons.items():
            color = (150, 150, 150) if name == 'save' and self.selection_mode == 'save' else (200, 200, 200)
            pygame.draw.rect(self.screen, color, rect)
            btn_text = self.font.render(name.capitalize(), True, (0, 0, 0))
            self.screen.blit(btn_text, (rect.x + 10, rect.y + 10))

    def run(self):
        while self.running:
            DARK_BLUE = (0, 0, 139)
            self.screen.fill(DARK_BLUE)
            self.handle_events()

            cell_width = WIDTH // GRID_SIZE
            cell_height = HEIGHT // GRID_SIZE
            image_size = (cell_width - 2 * PADDING, cell_height - 2 * PADDING)

            for i, ind in enumerate(self.population):
                row = i // GRID_SIZE
                col = i % GRID_SIZE
                x = col * cell_width + PADDING
                y = row * cell_height + PADDING
                
                img = self.genome_to_image(ind, image_size)
                mode = img.mode
                data = img.tobytes()
                py_image = pygame.image.fromstring(data, img.size, mode)
                
                if i in self.selected_indices or i in self.save_selected:
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
