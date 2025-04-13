import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.cluster import KMeans

# DEAP Configuration: Only create types if they don't already exist.
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # we want to maximize fitness
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

# shapes and color
SHAPE_TYPES = ['circle', 'rect', 'line', 'triangle', 'polygon', 'ellipse']
COLOR_RANGE = (0, 255)

def create_shape():
    """Create a random shape with parameters."""
    shape_type = random.choice(SHAPE_TYPES)
    color = (
        random.randint(*COLOR_RANGE),
        random.randint(*COLOR_RANGE),
        random.randint(*COLOR_RANGE)
    )

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
    elif shape_type == 'triangle':
        return {
            'type': 'triangle',
            'x1': random.uniform(0, 1),
            'y1': random.uniform(0, 1),
            'x2': random.uniform(0, 1),
            'y2': random.uniform(0, 1),
            'x3': random.uniform(0, 1),
            'y3': random.uniform(0, 1),
            'color': color
        }
    elif shape_type == 'polygon':
        num_points = random.randint(3, 6)  # Random number of points for the polygon
        points = [(random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)) for _ in range(num_points)]
        return {
            'type': 'polygon',
            'points': points,
            'color': color
        }
    elif shape_type == 'ellipse':
        x1, x2 = sorted([random.uniform(0, 1), random.uniform(0, 1)])
        y1, y2 = sorted([random.uniform(0, 1), random.uniform(0, 1)])
        return {
            'type': 'ellipse',
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'color': color
        }

def create_individual(shapes_per_image):
    """Create an individual (a single image) with random shapes."""
    return creator.Individual([create_shape() for _ in range(random.randint(2, shapes_per_image))])

def mutate(individual, mutation_rate, new_shape_chance):
    """More diverse mutation with larger occasional jumps"""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            shape = individual[i].copy()
            
            # 10% chance for a completely new shape
            if random.random() < new_shape_chance:
                individual[i] = create_shape()
                continue
                
            mutation_type = random.choice(['position', 'size', 'color', 'all'])
            
            # Larger mutations occasionally
            mutation_scale = mutation_rate + 0.3 if random.random() < 0.1 else mutation_rate
            
            if mutation_type in ['position', 'all']:
                if shape['type'] in ['circle', 'rect']:
                    shape['x'] = np.clip(shape['x'] + random.uniform(-mutation_scale, mutation_scale), 0, 1)
                    shape['y'] = np.clip(shape['y'] + random.uniform(-mutation_scale, mutation_scale), 0, 1)
                elif shape['type'] == 'line':
                    shape['x1'] = np.clip(shape['x1'] + random.uniform(-mutation_scale, mutation_scale), 0, 1)
                    shape['y1'] = np.clip(shape['y1'] + random.uniform(-mutation_scale, mutation_scale), 0, 1)
                    shape['x2'] = np.clip(shape['x2'] + random.uniform(-mutation_scale, mutation_scale), 0, 1)
                    shape['y2'] = np.clip(shape['y2'] + random.uniform(-mutation_scale, mutation_scale), 0, 1)
                    
            if mutation_type in ['size', 'all']:
                if shape['type'] == 'circle':
                    shape['radius'] = np.clip(shape['radius'] + random.uniform(-mutation_scale, mutation_scale), 0.01, 0.4)
                elif shape['type'] == 'rect':
                    shape['w'] = np.clip(shape['w'] + random.uniform(-mutation_scale, mutation_scale), 0.01, 0.5)
                    shape['h'] = np.clip(shape['h'] + random.uniform(-mutation_scale, mutation_scale), 0.01, 0.5)
                elif shape['type'] == 'line':
                    shape['width'] = np.clip(shape['width'] + random.randint(-4, 4), 1, 10)
                    
            if mutation_type in ['color', 'all']:
                color_shift = random.randint(-80, 80) if random.random() < 0.3 else random.randint(-30, 30)
                shape['color'] = (
                    np.clip(shape['color'][0] + color_shift, 0, 255),
                    np.clip(shape['color'][1] + color_shift, 0, 255),
                    np.clip(shape['color'][2] + color_shift, 0, 255)
                )
            
            individual[i] = shape
    return individual,

def cxTwoPoint(ind1, ind2):
    """Two-point crossover between two individuals."""
    size = len(ind1)
    # Choose two valid crossover points (ensuring they are not the first or last index)
    cxpoint1 = random.randint(1, size - 1)
    cxpoint2 = random.randint(1, size - 1)
    
    if cxpoint2 < cxpoint1:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Swap the slices between the two individuals
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    
    return ind1, ind2

def cluster_individuals(population, n_clusters):
    """Cluster individuals into species using K-means."""
    features = []
    for ind in population:
        # Compute average color and average position for each individual.
        avg_color = np.mean([np.mean(s['color']) for s in ind])
        avg_pos = np.mean([s.get('x', 0.5) for s in ind] + [s.get('y', 0.5) for s in ind])
        features.append([avg_color, avg_pos])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return kmeans.fit_predict(features)


def get_size(shape):
    """Helper function to calculate size-related properties of a shape."""
    if shape['type'] == 'circle':
        return shape['radius']
    elif shape['type'] == 'rect':
        return shape['w'] * shape['h']
    elif shape['type'] == 'line':
        return np.sqrt((shape['x2'] - shape['x1'])**2 + (shape['y2'] - shape['y1'])**2)
    elif shape['type'] == 'triangle':
        x1, y1 = shape['x1'], shape['y1']
        x2, y2 = shape['x2'], shape['y2']
        x3, y3 = shape['x3'], shape['y3']
        return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2
    elif shape['type'] == 'polygon':
        return len(shape['points'])
    elif shape['type'] == 'ellipse':
        width = abs(shape['x2'] - shape['x1'])
        height = abs(shape['y2'] - shape['y1'])
        return width * height
    return 0

def evaluate_fitness(individual, selected_individuals):
    """
    Custom fitness evaluation function.
    Combines user selection priority with diversity in shapes, sizes, and colors.
    """
    # Initialize fitness components
    similarity_score = 0
    diversity_score = 0

    # Calculate similarity to user-selected individuals
    for selected in selected_individuals:
        shape_similarity = sum(1 for s1, s2 in zip(individual, selected) if s1['type'] == s2['type'])
        color_similarity = sum(
            1 - (abs(np.mean(s1['color']) - np.mean(s2['color'])) / 255)
            for s1, s2 in zip(individual, selected)
        )
        size_similarity = sum(
            1 - abs(get_size(s1) - get_size(s2)) / max(get_size(s1), get_size(s2), 1)
            for s1, s2 in zip(individual, selected)
        )
        similarity_score += shape_similarity + color_similarity + size_similarity

    # Normalize similarity score
    similarity_score /= len(selected_individuals) if selected_individuals else 1

    # Calculate diversity in the individual
    shape_counts = {shape_type: 0 for shape_type in SHAPE_TYPES}
    color_values = []
    size_values = []

    for shape in individual:
        shape_counts[shape['type']] += 1
        color_values.append(np.mean(shape['color']))
        size_values.append(get_size(shape))

    shape_diversity = len([count for count in shape_counts.values() if count > 0]) / len(SHAPE_TYPES)
    color_std = np.std(color_values) if color_values else 0
    size_std = np.std(size_values) if size_values else 0
    diversity_score = shape_diversity + 0.5 * color_std + 0.5 * size_std

    # Combine similarity and diversity into a single fitness score
    fitness_score = 0.7 * similarity_score + 0.3 * diversity_score
    return fitness_score,