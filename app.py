import pygame
from PIL import Image, ImageDraw
from deap import base, tools, algorithms
from genetic_algorithm import (
    create_individual,
    evaluate_fitness,
    mutate,
    cxTwoPoint,
    cluster_individuals
)
import math
import glob
import os
import multiprocessing

# Initialize multiprocessing settings to optimize performance and stability
# Using minimum between physical cores and 4 to prevent overloading system
cpu_count = min(multiprocessing.cpu_count(), 4)  # Max of 4 cores even if more available
os.environ['LOKY_MAX_CPU_COUNT'] = str(cpu_count)

# Evolutionary algorithm parameters
POPULATION_SIZE = 34                # Total number of individuals in each generation
USER_SELECTION_SIZE = 6             # Number of images displayed per selection screen
GRID_SIZE = math.ceil(USER_SELECTION_SIZE ** 0.5)  # Dynamic grid size for display
WIDTH, HEIGHT = 1000, 600           # Main window dimensions
SHAPES_PER_IMAGE = 8                # Maximum shapes per generated image
CROSSOVER_RATE = 0.8                # Probability of crossover between individuals
MUTATION_RATE = 0.2                 # Probability of mutation
NEW_SHAPE_CHANCE = 0.2              # Chance to add a new shape during mutation
SPECIES_COUNT = 4                   # Number of species for speciation/clustering
PADDING = 10                        # UI padding between elements

class EvolutionApp:
    def __init__(self):
        """Initialize the evolution application with pygame, DEAP toolbox, and UI components."""
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.batch_size = USER_SELECTION_SIZE  # Images per screen
        self.total_batches = (POPULATION_SIZE + self.batch_size - 1) // self.batch_size
        self.init_deap()  # Initialize genetic algorithm components
        self.init_ui()    # Initialize user interface elements

    def init_deap(self):
        """Initialize DEAP genetic algorithm components and evolutionary state."""
        self.clear_history_folder()  # Clear previous run's images
        self.generation = 0         # Current generation counter
        self.current_batch = 0       # Current batch of displayed individuals
        self.selected_indices = []   # Indices of user-selected individuals
        self.selection_mode = 'generation'  # Current selection mode
        self.running = True          # Main loop control flag
        self.selected_history = []   # History of selected individuals per generation
        self.selected_history_all = []  # Complete selection history
        self.stopped = False         # Flag for stopping evolution
        self.best_individuals = []   # Collection of best performing individuals
        
        # DEAP toolbox configuration
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", create_individual, SHAPES_PER_IMAGE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate_fitness, selected_individuals=[])
        self.toolbox.register("mate", cxTwoPoint)
        self.toolbox.register("mutate", mutate, mutation_rate=MUTATION_RATE, new_shape_chance=NEW_SHAPE_CHANCE)
        self.population = self.toolbox.population(n=POPULATION_SIZE)  # Initial population
        
    def init_ui(self):
        """Initialize user interface elements including buttons and fonts."""
        self.font = pygame.font.SysFont("Arial", 20)
        # Define interactive buttons with their positions and dimensions
        self.buttons = {
            'reset': pygame.Rect(120, HEIGHT - 50, 100, 40),
            'choose next': pygame.Rect(WIDTH // 2 - 70, HEIGHT - 50, 140, 40),
            'stop evolution': pygame.Rect(WIDTH // 4 * 3 - 50, HEIGHT - 50, 100, 40)
        }

    def genome_to_image(self, genome, img_size):
        """
        Convert a genetic representation (genome) into a visual image.
        
        Args:
            genome: The genetic representation of an individual
            img_size: Tuple (width, height) for output image dimensions
            
        Returns:
            PIL Image object representing the individual
        """
        img = Image.new("RGB", img_size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Process each shape in the genome
        for shape in genome:
            color = tuple(shape['color'])
            # Scaling functions for normalized coordinates to pixel coordinates
            scale_x = lambda x: int(x * img_size[0])
            scale_y = lambda y: int(y * img_size[1])
            
            # Handle different shape types with appropriate drawing commands
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
        """Process all pygame events including user input and system events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                # Handle button clicks
                if self.buttons['reset'].collidepoint(x, y):
                    self.init_deap()  # Reset the entire application
                elif self.buttons['choose next'].collidepoint(x, y):
                    self.handle_image_click(x, y, choose=False)  # Advance without selection
                elif self.buttons['stop evolution'].collidepoint(x, y):
                    self.save_selected_history()  # Save progress before stopping
                    self.stop_evolution()       # End the evolutionary process
                else:
                    self.handle_image_click(x, y, choose=True)  # Handle image selection

    def handle_image_click(self, x, y, choose):
        """
        Process user clicks on images for selection and navigation.
        
        Args:
            x, y: Mouse click coordinates
            choose: Boolean indicating whether this is a selection click
        """
        if choose:
            # Calculate which image was clicked based on grid position
            col = x // (WIDTH // GRID_SIZE)
            row = y // (HEIGHT // GRID_SIZE)
            idx = row * GRID_SIZE + col
        else:
            idx = 0  # Default to first image for non-selection clicks
        
        # Calculate valid index range for current batch
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.population))

        if start_idx + idx < end_idx:
            if self.selection_mode == 'generation':
                # Only allow selection if not already selected
                if start_idx + idx not in self.selected_indices:
                    if choose:
                        self.selected_indices.append(start_idx + idx)
                        self.save_image(self.generation, start_idx + idx, self.population[start_idx + idx])
                    # Advance to next batch or generation
                    self.current_batch += 1
                    if self.current_batch >= self.total_batches:
                        self.create_next_generation()  # Proceed to next generation if all batches processed

    def create_next_generation(self):
        """
        Create a new generation of individuals using genetic operators:
        - Selection based on user preferences
        - Crossover between individuals
        - Mutation of individuals
        - Speciation to maintain diversity
        """
        # Get selected individuals from current generation
        selected_individuals = [self.population[idx] for idx in self.selected_indices]
        self.selected_history.append(selected_individuals)

        # Consider selections from last 3 generations for fitness evaluation
        selected_individuals = [ind for generation in self.selected_history[-3:] for ind in generation]

        # Evaluate fitness of all individuals relative to user selections
        for ind in self.population:
            ind.fitness.values = evaluate_fitness(ind, selected_individuals)

        # Cluster population into species to maintain diversity
        clusters = cluster_individuals(self.population, SPECIES_COUNT)

        # Select top individuals from each species
        survivors = []
        for cluster_id in range(SPECIES_COUNT):
            cluster = [ind for ind, c in zip(self.population, clusters) if c == cluster_id]
            if cluster:
                survivors.append(max(cluster, key=lambda ind: ind.fitness.values))

        # Generate offspring through crossover and mutation
        offspring = algorithms.varOr(
            survivors,
            self.toolbox,
            len(self.population) - len(survivors),
            cxpb=CROSSOVER_RATE,
            mutpb=MUTATION_RATE
        )

        # Inject new random individuals to maintain genetic diversity
        random_individuals = self.toolbox.population(n=10)

        # Combine all components to form new population
        new_population = survivors + offspring + random_individuals + selected_individuals
        self.population = new_population[:POPULATION_SIZE]  # Trim to population size

        # Update evolutionary state
        self.generation += 1
        self.current_batch = 0
        self.selected_indices = []

    def stop_evolution(self):
        """Cleanly stop the evolutionary process and prepare final output."""
        self.stopped = True
        self.running = False
        print("Evolution stopped.")

    def save_selected_history(self):
        """Save all selected individuals from all generations to image files."""
        for gen_idx, generation in enumerate(self.selected_history):
            for ind_idx, individual in enumerate(generation):
                img = self.genome_to_image(individual, (1024, 768))
                img.save(f"history/gen{gen_idx + 1}_ind{ind_idx + 1}.png")

    def save_image(self, gen, idx, individual):
        """Save an individual's image to file with generation and index information."""
        img = self.genome_to_image(individual, (1024, 768))
        img.save(f"history/gen{gen}_ind{idx}.png")

    def clear_history_folder(self):
        """Ensure clean history directory by removing previous run's files."""
        history_folder = "history"
        if not os.path.exists(history_folder):
            os.makedirs(history_folder)  # Create folder if missing
        else:
            # Remove all existing files in history folder
            files = glob.glob(os.path.join(history_folder, "*"))
            for file in files:
                os.remove(file)

    def draw_pulsating_background(self, time):
        """
        Draw an animated gradient background that changes over time.
        
        Args:
            time: Current time value for animation progression
        """
        colors = [
            (82, 1, 32), (8, 64, 62), (112, 101, 19), (181, 113, 20), (150, 43, 9)
        ]
        num_colors = len(colors)
        segment_height = HEIGHT // (num_colors - 1)

        # Draw gradient segments with pulsating effect
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
        """
        Draw a button with rounded corners and hover animation.
        
        Args:
            rect: Button position and dimensions
            text: Button label text
            is_hovered: Boolean indicating hover state
        """
        # Apply scaling effect when hovered
        scale_factor = 1.1 if is_hovered else 1.0
        scaled_rect = pygame.Rect(
            rect.x - (rect.width * (scale_factor - 1)) // 2,
            rect.y - (rect.height * (scale_factor - 1)) // 2,
            int(rect.width * scale_factor),
            int(rect.height * scale_factor)
        )

        # Button appearance changes based on hover state
        color = (255, 165, 0) if is_hovered else (200, 200, 200)
        pygame.draw.rect(self.screen, color, scaled_rect, border_radius=10)

        # Center text in button
        btn_text = self.font.render(text.capitalize(), True, (0, 0, 0))
        self.screen.blit(btn_text, (scaled_rect.x + scaled_rect.width // 2 - btn_text.get_width() // 2, scaled_rect.y + scaled_rect.height // 2 - btn_text.get_height() // 2))

    def draw_progress_bar(self):
        """Draw a progress bar showing current batch completion status."""
        bar_width = WIDTH - 40
        bar_height = 20
        progress = self.current_batch / self.total_batches if self.total_batches > 0 else 0

        # Background bar
        pygame.draw.rect(self.screen, (200, 200, 200), (20, HEIGHT - 80, bar_width, bar_height), border_radius=5)

        # Progress indicator
        pygame.draw.rect(self.screen, (255, 165, 0), (20, HEIGHT - 80, int(bar_width * progress), bar_height), border_radius=5)

        # Progress text
        progress_text = self.font.render(f"Progress: {int(progress * 100)}%", True, (255, 255, 255))
        self.screen.blit(progress_text, (WIDTH // 2 - progress_text.get_width() // 2, HEIGHT - 110))

    def draw_ui(self):
        """Draw all user interface elements including buttons, text, and progress indicators."""
        # Generation counter
        gen_text = self.font.render(f"Generation: {self.generation}", True, (255, 255, 255))
        self.screen.blit(gen_text, (WIDTH // 2 - gen_text.get_width() // 2, 10))

        # Interactive buttons with hover detection
        mouse_pos = pygame.mouse.get_pos()
        for name, rect in self.buttons.items():
            is_hovered = rect.collidepoint(mouse_pos)
            self.draw_rounded_button(rect, name, is_hovered)

        # Progress visualization
        self.draw_progress_bar()

    def run(self):
        """Main application loop handling display, user input, and evolution progress."""
        try:
            while self.running:
                MARGIN_TOP = 50
                self.draw_pulsating_background(pygame.time.get_ticks())
                self.draw_progress_bar()
                self.handle_events()

                if self.stopped:
                    # Final display of best individuals when evolution is stopped
                    cell_width = WIDTH // GRID_SIZE
                    cell_height = HEIGHT // GRID_SIZE
                    image_size = (cell_width - 2 * PADDING, cell_height - 2 * PADDING)

                    # Display each best individual in a grid
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

                # Normal evolution display showing current batch
                start_idx = self.current_batch * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.population))
                batch = self.population[start_idx:end_idx]

                cell_width = WIDTH // GRID_SIZE
                cell_height = HEIGHT // GRID_SIZE
                image_size = (cell_width - 2 * PADDING, cell_height - 2 * PADDING)

                # Display each individual in the current batch
                for i, ind in enumerate(batch):
                    row = i // GRID_SIZE
                    col = i % GRID_SIZE
                    x = col * cell_width + PADDING
                    y = row * cell_height + PADDING + MARGIN_TOP

                    img = self.genome_to_image(ind, image_size)
                    mode = img.mode
                    data = img.tobytes()
                    py_image = pygame.image.fromstring(data, img.size, mode)

                    # Highlight selected individuals with green border
                    if start_idx + i in self.selected_indices:
                        border_color = (0, 255, 0)  # Green border for selected individuals
                        pygame.draw.rect(self.screen, border_color, 
                                        (x - PADDING, y - PADDING, image_size[0] + 2 * PADDING, image_size[1] + 2 * PADDING), 3)

                    self.screen.blit(py_image, (x, y))

                self.draw_ui()
                pygame.display.flip()
                self.clock.tick(30)

        finally:
            pygame.quit()  # Ensure proper cleanup of pygame resources
            print("Pygame cleaned up")

if __name__ == "__main__":
    app = EvolutionApp()
    app.run()