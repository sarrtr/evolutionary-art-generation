import pygame
import numpy as np
from PIL import Image, ImageDraw
from deap import base, creator, tools, algorithms
from genetic_algorithm import (
    create_individual,
    evaluate_fitness,
    mutate,
    cxTwoPoint,
    cluster_individuals,
    SHAPE_TYPES,
    COLOR_RANGE
)
import math
import os
import glob

# Constants
POPULATION_SIZE = 34
USER_SELECTION_SIZE = 6 # Images per screen
# Adjust grid size dynamically (smallest integer >= sqrt(POPULATION_SIZE))
GRID_SIZE = math.ceil(USER_SELECTION_SIZE ** 0.5)
WIDTH, HEIGHT = 1000, 600
SHAPES_PER_IMAGE = 8
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
NEW_SHAPE_CHANCE = 0.2
SPECIES_COUNT = 4
PADDING = 10

class EvolutionApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.batch_size = USER_SELECTION_SIZE  # Images per screen
        self.total_batches = (POPULATION_SIZE + self.batch_size - 1) // self.batch_size
        self.init_deap()
        self.init_ui()

    def init_deap(self):
        self.clear_history_folder()
        self.generation = 0
        self.current_batch = 0
        self.selected_indices = []
        self.selection_mode = 'generation'
        self.running = True
        self.selected_history = []
        self.selected_history_all = []
        self.stopped = False
        self.best_individuals = []
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", create_individual, SHAPES_PER_IMAGE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate_fitness, selected_individuals=[])
        self.toolbox.register("mate", cxTwoPoint)
        self.toolbox.register("mutate", mutate, mutation_rate=MUTATION_RATE, new_shape_chance=NEW_SHAPE_CHANCE)
        self.population = self.toolbox.population(n=POPULATION_SIZE)
        
    def init_ui(self):
        self.font = pygame.font.SysFont("Arial", 20)
        self.buttons = {
            'reset': pygame.Rect(120, HEIGHT - 50, 100, 40),
            'choose next': pygame.Rect(WIDTH // 2 - 70, HEIGHT - 50, 140, 40),
            'stop': pygame.Rect(WIDTH // 4 * 3 - 50, HEIGHT - 50, 100, 40),
            'save best': pygame.Rect(WIDTH // 4 * 3 + 100, HEIGHT - 50, 100, 40)
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
                if self.buttons['reset'].collidepoint(x, y):
                    self.init_deap()
                elif self.buttons['choose next'].collidepoint(x, y):
                    self.handle_image_click(x, y, choose=False)
                elif self.buttons['stop'].collidepoint(x, y):
                    self.save_selected_history()
                    self.stop_evolution()
                elif self.buttons['save best'].collidepoint(x, y):
                    self.save_best_individuals()
                else:
                    self.handle_image_click(x, y, choose=True)

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
                        self.save_image(self.generation, start_idx + idx, self.population[start_idx + idx])
                    self.current_batch += 1  # Move to the next batch
                    if self.current_batch >= self.total_batches:
                        self.create_next_generation()  # Process the next generation

    def create_next_generation(self):
        selected_individuals = [self.population[idx] for idx in self.selected_indices]
        self.selected_history.append(selected_individuals)

        selected_individuals = [ind for generation in self.selected_history[-3:] for ind in generation]

        # Assign fitness based on user-selected individuals
        for ind in self.population:
            ind.fitness.values = evaluate_fitness(ind, selected_individuals)

        # Speciation using clustering
        clusters = cluster_individuals(self.population, SPECIES_COUNT)

        # Select survivors from each species
        survivors = []
        for cluster_id in range(SPECIES_COUNT):
            cluster = [ind for ind, c in zip(self.population, clusters) if c == cluster_id]
            if cluster:
                survivors.append(max(cluster, key=lambda ind: ind.fitness.values))

        # Generate offspring to refill the population
        offspring = algorithms.varOr(
            survivors,
            self.toolbox,
            len(self.population) - len(survivors),
            cxpb=CROSSOVER_RATE,
            mutpb=MUTATION_RATE
        )

        # Inject new random individuals to boost diversity
        random_individuals = self.toolbox.population(n=10)

        # Combine survivors, offspring, random individuals, and selected history
        new_population = survivors + offspring + random_individuals + selected_individuals
        self.population = new_population[:POPULATION_SIZE]  # Limit to population size

        # Update generation and reset batch
        self.generation += 1
        self.current_batch = 0
        self.selected_indices = []

    def stop_evolution(self):
        """Stop the evolution process and select the best 6 individuals."""
        self.stopped = True

        for ind in self.population:
            if not ind.fitness.valid:
                ind.fitness.values = self.toolbox.evaluate(ind)
        # Sort the population by fitness in descending order
        self.population.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        # Select the top 6 individuals
        self.best_individuals = self.population[:6]
        print("Evolution stopped. Best 6 individuals selected.")

    def save_best_individuals(self):
        """Save the best 6 individuals as images."""
        for i, ind in enumerate(self.best_individuals):
            img = self.genome_to_image(ind, (1024, 768))
            img.save(f"result/best_ind_{i}.png")
        print("Best 6 individuals saved.")

    # def save_selected_history(self):
    #     for gen_idx, generation in enumerate(self.selected_history):
    #         for ind_idx, individual in enumerate(generation):
    #             img = self.genome_to_image(individual, (1024, 768))
    #             img.save(f"history/gen{gen_idx + 1}_ind{ind_idx + 1}.png")

    def save_image(self, gen, idx, individual):
        img = self.genome_to_image(individual, (1024, 768))
        img.save(f"history/gen{gen}_ind{idx}.png")

    def clear_history_folder(self):
        """Clear all files from the history folder."""
        history_folder = "history"
        if not os.path.exists(history_folder):
            os.makedirs(history_folder)  # Create the folder if it doesn't exist
        else:
            # Remove all files in the folder
            files = glob.glob(os.path.join(history_folder, "*"))
            for file in files:
                os.remove(file)

    def draw_pulsating_background(self, time):
        """Draw a pulsating gradient background."""
        colors = [
            (82, 1, 32), (8, 64, 62), (112, 101, 19), (181, 113, 20), (150, 43, 9)
        ]
        num_colors = len(colors)
        segment_height = HEIGHT // (num_colors - 1)

        for i in range(num_colors - 1):
            start_color = colors[i]
            end_color = colors[i + 1]
            for y in range(segment_height):
                ratio = (y / segment_height) * (0.5 + 0.5 * math.sin(time / 500))
                color = tuple(
                    int(start_color[j] * (1 - ratio) + end_color[j] * ratio) for j in range(3)
                )
                pygame.draw.line(
                    self.screen, color, (0, i * segment_height + y), (WIDTH, i * segment_height + y)
                )

    def draw_rounded_button(self, rect, text, is_hovered):
        """Draw a rounded button with hover scaling effect."""
        scale_factor = 1.1 if is_hovered else 1.0
        scaled_rect = pygame.Rect(
            rect.x - (rect.width * (scale_factor - 1)) // 2,
            rect.y - (rect.height * (scale_factor - 1)) // 2,
            int(rect.width * scale_factor),
            int(rect.height * scale_factor)
        )

        # Button colors
        color = (255, 165, 0) if is_hovered else (200, 200, 200)
        pygame.draw.rect(self.screen, color, scaled_rect, border_radius=10)

        # Draw button text
        btn_text = self.font.render(text.capitalize(), True, (0, 0, 0))
        self.screen.blit(btn_text, (scaled_rect.x + scaled_rect.width // 2 - btn_text.get_width() // 2, scaled_rect.y + scaled_rect.height // 2 - btn_text.get_height() // 2))

    def draw_progress_bar(self):
        """Draw a progress bar for the current generation."""
        bar_width = WIDTH - 40
        bar_height = 20
        progress = self.current_batch / self.total_batches if self.total_batches > 0 else 0

        # Draw background bar
        pygame.draw.rect(self.screen, (200, 200, 200), (20, HEIGHT - 80, bar_width, bar_height), border_radius=5)

        # Draw progress
        pygame.draw.rect(self.screen, (255, 165, 0), (20, HEIGHT - 80, int(bar_width * progress), bar_height), border_radius=5)

        # Draw text
        progress_text = self.font.render(f"Progress: {int(progress * 100)}%", True, (255, 255, 255))
        self.screen.blit(progress_text, (WIDTH // 2 - progress_text.get_width() // 2, HEIGHT - 110))

    def draw_ui(self):
        """Draw the UI elements including the generation counter, buttons, and progress bar."""
        # Draw generation counter
        gen_text = self.font.render(f"Generation: {self.generation}", True, (255, 255, 255))
        self.screen.blit(gen_text, (WIDTH // 2 - gen_text.get_width() // 2, 10))

        # Draw buttons with hover effects
        mouse_pos = pygame.mouse.get_pos()
        for name, rect in self.buttons.items():
            is_hovered = rect.collidepoint(mouse_pos)
            self.draw_rounded_button(rect, name, is_hovered)

        # Draw progress bar
        self.draw_progress_bar()

    def run(self):
        while self.running:
            MARGIN_TOP = 50
            self.draw_pulsating_background(pygame.time.get_ticks())
            self.draw_progress_bar()
            self.handle_events()

            if self.stopped:
                # Display the best 6 individuals
                cell_width = WIDTH // GRID_SIZE
                cell_height = HEIGHT // GRID_SIZE
                image_size = (cell_width - 2 * PADDING, cell_height - 2 * PADDING)

                for i, ind in enumerate(self.best_individuals):
                    row = i // GRID_SIZE
                    col = i % GRID_SIZE
                    x = col * cell_width + PADDING
                    y = row * cell_height + PADDING + MARGIN_TOP

                    img = self.genome_to_image(ind, image_size)
                    mode = img.mode
                    data = img.tobytes()
                    py_image = pygame.image.fromstring(data, img.size, mode)

                    self.screen.blit(py_image, (x, y))

                self.draw_ui()
                pygame.display.flip()
                self.clock.tick(30)
                continue

            # Normal evolution display
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

                if start_idx + i in self.selected_indices:
                    border_color = (0, 255, 0)  # Green border for selected individuals
                    pygame.draw.rect(self.screen, border_color, 
                                     (x - PADDING, y - PADDING, image_size[0] + 2 * PADDING, image_size[1] + 2 * PADDING), 3)

                self.screen.blit(py_image, (x, y))

            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(30)

if __name__ == "__main__":
    app = EvolutionApp()
    app.run()
