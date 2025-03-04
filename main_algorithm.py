import pygame
import random
import numpy as np
from PIL import Image, ImageDraw
from deap import base, creator, tools, algorithms
from sklearn.cluster import KMeans  # For speciation

# Initialize Pygame
pygame.init()

# ======================
#  CONSTANTS & SETTINGS
# ======================
WIDTH, HEIGHT = 1200, 800
GRID_SIZE = 4
POPULATION_SIZE = GRID_SIZE ** 2
SHAPES_PER_IMAGE = 8
MAX_SHAPE_SIZE = 100
SHAPE_TYPES = ['circle', 'rect', 'line']
COLOR_RANGE = (0, 255)
MUTATION_RATE = 0.4
SPECIES_COUNT = 4

# DEAP Configuration
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# ======================
#  GENOME REPRESENTATION
# ======================
def create_shape():
    """Create a random shape with parameters"""
    shape_type = random.choice(SHAPE_TYPES)
    color = (random.randint(*COLOR_RANGE), 
            random.randint(*COLOR_RANGE),
            random.randint(*COLOR_RANGE))
    
    if shape_type == 'circle':
        return {
            'type': 'circle',
            'x': random.uniform(0, 1),
            'y': random.uniform(0, 1),
            'radius': random.uniform(0.05, 0.2),
            'color': color
        }
    elif shape_type == 'rect':
        return {
            'type': 'rect',
            'x': random.uniform(0, 1),
            'y': random.uniform(0, 1),
            'w': random.uniform(0.05, 0.3),
            'h': random.uniform(0.05, 0.3),
            'color': color
        }
    elif shape_type == 'line':
        return {
            'type': 'line',
            'x1': random.uniform(0, 1),
            'y1': random.uniform(0, 1),
            'x2': random.uniform(0, 1),
            'y2': random.uniform(0, 1),
            'width': random.randint(1, 5),
            'color': color
        }

def create_individual():
    return creator.Individual([create_shape() for _ in range(SHAPES_PER_IMAGE)])

# ======================
#  GENETIC OPERATIONS
# ======================
def mutate(individual):
    """DEAP-compatible mutation operator"""
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            shape = individual[i].copy()
            
            # Select parameter to mutate
            mutation_type = random.choice(['position', 'size', 'color'])
            
            if mutation_type == 'position':
                if shape['type'] in ['circle', 'rect']:
                    shape['x'] = np.clip(shape['x'] + random.uniform(-0.1, 0.1), 0, 1)
                    shape['y'] = np.clip(shape['y'] + random.uniform(-0.1, 0.1), 0, 1)
                elif shape['type'] == 'line':
                    shape['x1'] = np.clip(shape['x1'] + random.uniform(-0.1, 0.1), 0, 1)
                    shape['y1'] = np.clip(shape['y1'] + random.uniform(-0.1, 0.1), 0, 1)
                    shape['x2'] = np.clip(shape['x2'] + random.uniform(-0.1, 0.1), 0, 1)
                    shape['y2'] = np.clip(shape['y2'] + random.uniform(-0.1, 0.1), 0, 1)
                    
            elif mutation_type == 'size':
                if shape['type'] == 'circle':
                    shape['radius'] = np.clip(shape['radius'] + random.uniform(-0.05, 0.05), 0.01, 0.3)
                elif shape['type'] == 'rect':
                    shape['w'] = np.clip(shape['w'] + random.uniform(-0.05, 0.05), 0.01, 0.4)
                    shape['h'] = np.clip(shape['h'] + random.uniform(-0.05, 0.05), 0.01, 0.4)
                elif shape['type'] == 'line':
                    shape['width'] = np.clip(shape['width'] + random.randint(-2, 2), 1, 8)
                    
            elif mutation_type == 'color':
                shape['color'] = (
                    np.clip(shape['color'][0] + random.randint(-50, 50), 0, 255),
                    np.clip(shape['color'][1] + random.randint(-50, 50), 0, 255),
                    np.clip(shape['color'][2] + random.randint(-50, 50), 0, 255)
                )
            
            individual[i] = shape
    return individual,

def cxTwoPoint(ind1, ind2):
    """DEAP two-point crossover"""
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    
    return ind1, ind2

# ======================
#  SPECIATION & SELECTION
# ======================
def cluster_individuals(population, n_clusters):
    """Cluster population using K-means based on shape features"""
    features = []
    for ind in population:
        # Create feature vector: average color and position
        avg_color = np.mean([np.mean(s['color']) for s in ind])
        avg_pos = np.mean([s.get('x', 0.5) for s in ind] + [s.get('y', 0.5) for s in ind])
        features.append([avg_color, avg_pos])
    
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(features)
    return clusters

# ======================
#  RENDERING & UI
# ======================
def genome_to_image(genome, img_size=(300, 200)):
    """Convert genome to PIL Image using Pillow"""
    img = Image.new("RGB", img_size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    for shape in genome:
        color = tuple(shape['color'])
        
        # Scale coordinates to image size
        scale = lambda x: int(x * img_size[0] if x <= 1 else x)
        
        if shape['type'] == 'circle':
            x = scale(shape['x'] * img_size[0])
            y = scale(shape['y'] * img_size[1])
            radius = scale(shape['radius'] * img_size[0])
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
            
        elif shape['type'] == 'rect':
            x = scale(shape['x'] * img_size[0])
            y = scale(shape['y'] * img_size[1])
            w = scale(shape['w'] * img_size[0])
            h = scale(shape['h'] * img_size[1])
            draw.rectangle([x, y, x+w, y+h], fill=color)
            
        elif shape['type'] == 'line':
            x1 = scale(shape['x1'] * img_size[0])
            y1 = scale(shape['y1'] * img_size[1])
            x2 = scale(shape['x2'] * img_size[0])
            y2 = scale(shape['y2'] * img_size[1])
            draw.line([x1, y1, x2, y2], fill=color, width=shape['width'])
    
    return img

# ======================
#  MAIN APPLICATION
# ======================
class EvolutionApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.generation = 0
        self.population = []
        self.selected = None
        self.running = True
        self.init_deap()
        self.init_ui()
        
    def init_deap(self):
        """Initialize DEAP toolbox"""
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", cxTwoPoint)
        self.toolbox.register("mutate", mutate)
        self.population = self.toolbox.population(n=POPULATION_SIZE)
        
    def init_ui(self):
        """Initialize UI elements"""
        self.font = pygame.font.SysFont("Arial", 20)
        self.buttons = {
            'save': pygame.Rect(10, HEIGHT-50, 100, 40),
            'reset': pygame.Rect(120, HEIGHT-50, 100, 40)
        }
        
    def draw_ui(self):
        """Draw UI elements"""
        # Generation counter
        text = self.font.render(f"Generation: {self.generation}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        
        # Buttons
        for name, rect in self.buttons.items():
            pygame.draw.rect(self.screen, (200, 200, 200), rect)
            text = self.font.render(name.capitalize(), True, (0, 0, 0))
            self.screen.blit(text, (rect.x+10, rect.y+10))
            
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                
                # Check buttons
                if self.buttons['save'].collidepoint(x, y):
                    self.save_image()
                elif self.buttons['reset'].collidepoint(x, y):
                    self.init_deap()
                    self.generation = 0
                else:
                    # Select individual
                    col = x // (WIDTH//GRID_SIZE)
                    row = y // (HEIGHT//GRID_SIZE)
                    idx = row * GRID_SIZE + col
                    if idx < len(self.population):
                        self.selected = self.population[idx]
                        self.create_next_generation()
                        
    def create_next_generation(self):
        """Evolve new generation using DEAP"""
        # Cluster population into species
        clusters = cluster_individuals(self.population, SPECIES_COUNT)
        
        # Select survivors from each species
        survivors = []
        for cluster_id in range(SPECIES_COUNT):
            cluster = [ind for ind, c in zip(self.population, clusters) if c == cluster_id]
            if cluster:
                survivors.append(max(cluster, key=lambda ind: ind.fitness.values))
        
        # Create offspring using DEAP algorithms
        offspring = algorithms.varOr(survivors, self.toolbox, lambda x: len(self.population) - len(survivors),cxpb=0.5, mutpb=0.2)
        
        self.population = survivors + offspring
        self.generation += 1
        
    def save_image(self):
        """Save current best image using Pillow"""
        best = max(self.population, key=lambda ind: ind.fitness.values)
        img = genome_to_image(best, (1024, 768))
        img.save(f"generation_{self.generation}.png")
        
    def run(self):
        """Main loop"""
        while self.running:
            self.screen.fill((255, 255, 255))
            self.handle_events()
            
            # Draw population
            for i, ind in enumerate(self.population):
                row = i // GRID_SIZE
                col = i % GRID_SIZE
                x = col * (WIDTH//GRID_SIZE)
                y = row * (HEIGHT//GRID_SIZE)
                
                # Convert to Pygame surface
                img = genome_to_image(ind, (WIDTH//GRID_SIZE, HEIGHT//GRID_SIZE))
                mode = img.mode
                size = img.size
                data = img.tobytes()
                py_image = pygame.image.fromstring(data, size, mode)
                self.screen.blit(py_image, (x, y))
                
            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(30)

if __name__ == "__main__":
    app = EvolutionApp()
    app.run()