import random
import numpy as np
from deap import base, creator
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Union

# Define shape and color parameters
SHAPE_TYPES = ['circle', 'rect', 'line', 'triangle', 'polygon', 'ellipse']
COLOR_RANGE = (0, 255)

# Setup DEAP types (only if not previously defined)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

Shape = Dict[str, Union[str, float, int, Tuple[int, int, int], List[Tuple[float, float]]]]

def create_shape() -> Shape:
    """Generate a random shape with properties based on its type."""
    shape_type = random.choice(SHAPE_TYPES)
    color = tuple(random.randint(*COLOR_RANGE) for _ in range(3))

    if shape_type == 'circle':
        return {'type': 'circle', 'x': random.random(), 'y': random.random(),
                'radius': random.uniform(0.05, 0.2), 'color': color}
    elif shape_type == 'rect':
        return {'type': 'rect', 'x': random.random(), 'y': random.random(),
                'w': random.uniform(0.05, 0.3), 'h': random.uniform(0.05, 0.3), 'color': color}
    elif shape_type == 'line':
        return {'type': 'line', 'x1': random.random(), 'y1': random.random(),
                'x2': random.random(), 'y2': random.random(),
                'width': random.randint(1, 5), 'color': color}
    elif shape_type == 'triangle':
        return {'type': 'triangle',
                'x1': random.random(), 'y1': random.random(),
                'x2': random.random(), 'y2': random.random(),
                'x3': random.random(), 'y3': random.random(), 'color': color}
    elif shape_type == 'polygon':
        points = [(random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)) for _ in range(random.randint(3, 6))]
        return {'type': 'polygon', 'points': points, 'color': color}
    elif shape_type == 'ellipse':
        x1, x2 = sorted([random.random(), random.random()])
        y1, y2 = sorted([random.random(), random.random()])
        return {'type': 'ellipse', 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'color': color}

def create_individual(shapes_per_image: int) -> creator.Individual:
    """Create an individual composed of randomly generated shapes."""
    return creator.Individual([create_shape() for _ in range(random.randint(2, shapes_per_image))])

def mutate(ind: creator.Individual, mutation_rate: float, new_shape_chance: float):
    """Mutate an individual's shapes with possible position, size, or color changes."""
    for i in range(len(ind)):
        if random.random() < mutation_rate:
            shape = ind[i].copy()
            if random.random() < new_shape_chance:
                ind[i] = create_shape()
                continue

            mutation_type = random.choice(['position', 'size', 'color', 'all'])
            mutation_scale = mutation_rate + 0.3 if random.random() < 0.1 else mutation_rate

            if mutation_type in ['position', 'all']:
                if shape['type'] in ['circle', 'rect']:
                    shape['x'] = np.clip(shape['x'] + random.uniform(-mutation_scale, mutation_scale), 0, 1)
                    shape['y'] = np.clip(shape['y'] + random.uniform(-mutation_scale, mutation_scale), 0, 1)
                elif shape['type'] == 'line':
                    for key in ['x1', 'y1', 'x2', 'y2']:
                        shape[key] = np.clip(shape[key] + random.uniform(-mutation_scale, mutation_scale), 0, 1)

            if mutation_type in ['size', 'all']:
                if shape['type'] == 'circle':
                    shape['radius'] = np.clip(shape['radius'] + random.uniform(-mutation_scale, mutation_scale), 0.01, 0.4)
                elif shape['type'] == 'rect':
                    shape['w'] = np.clip(shape['w'] + random.uniform(-mutation_scale, mutation_scale), 0.01, 0.5)
                    shape['h'] = np.clip(shape['h'] + random.uniform(-mutation_scale, mutation_scale), 0.01, 0.5)
                elif shape['type'] == 'line':
                    shape['width'] = np.clip(shape['width'] + random.randint(-4, 4), 1, 10)

            if mutation_type in ['color', 'all']:
                shift = random.randint(-80, 80) if random.random() < 0.3 else random.randint(-30, 30)
                shape['color'] = tuple(np.clip(c + shift, 0, 255) for c in shape['color'])

            ind[i] = shape
    return ind,

def cxTwoPoint(ind1: creator.Individual, ind2: creator.Individual) -> Tuple[creator.Individual, creator.Individual]:
    """Apply two-point crossover between individuals."""
    size = len(ind1)
    cx1, cx2 = sorted([random.randint(1, size - 1) for _ in range(2)])
    ind1[cx1:cx2], ind2[cx1:cx2] = ind2[cx1:cx2], ind1[cx1:cx2]
    return ind1, ind2

def cluster_individuals(population: List[creator.Individual], n_clusters: int) -> List[int]:
    """Cluster population into species using KMeans based on shape color and position."""
    features = []
    for ind in population:
        avg_color = np.mean([np.mean(shape['color']) for shape in ind])
        avg_pos = np.mean([shape.get('x', 0.5) for shape in ind] + [shape.get('y', 0.5) for shape in ind])
        features.append([avg_color, avg_pos])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return kmeans.fit_predict(features)

def get_size(shape: Shape) -> float:
    """Estimate size of a shape using geometric or heuristic calculation."""
    if shape['type'] == 'circle':
        return shape['radius']
    elif shape['type'] == 'rect':
        return shape['w'] * shape['h']
    elif shape['type'] == 'line':
        return np.hypot(shape['x2'] - shape['x1'], shape['y2'] - shape['y1'])
    elif shape['type'] == 'triangle':
        x1, y1, x2, y2, x3, y3 = shape['x1'], shape['y1'], shape['x2'], shape['y2'], shape['x3'], shape['y3']
        return abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2
    elif shape['type'] == 'polygon':
        return len(shape['points'])
    elif shape['type'] == 'ellipse':
        return abs((shape['x2'] - shape['x1']) * (shape['y2'] - shape['y1']))
    return 0.0

def evaluate_fitness(ind: creator.Individual, selected: List[creator.Individual]) -> Tuple[float]:
    """Evaluate individual based on similarity to user-selected samples and internal diversity."""
    sim_score = 0
    for sel in selected:
        shape_sim = sum(1 for s1, s2 in zip(ind, sel) if s1['type'] == s2['type'])
        color_sim = sum(1 - abs(np.mean(s1['color']) - np.mean(s2['color'])) / 255 for s1, s2 in zip(ind, sel))
        size_sim = sum(1 - abs(get_size(s1) - get_size(s2)) / max(get_size(s1), get_size(s2), 1)
                       for s1, s2 in zip(ind, sel))
        sim_score += shape_sim + color_sim + size_sim
    sim_score /= max(len(selected), 1)

    shape_counts = {stype: 0 for stype in SHAPE_TYPES}
    color_vals = []
    size_vals = []

    for shape in ind:
        shape_counts[shape['type']] += 1
        color_vals.append(np.mean(shape['color']))
        size_vals.append(get_size(shape))

    shape_div = sum(1 for c in shape_counts.values() if c > 0) / len(SHAPE_TYPES)
    color_std = np.std(color_vals) if color_vals else 0
    size_std = np.std(size_vals) if size_vals else 0
    diversity = shape_div + 0.5 * color_std + 0.5 * size_std

    return 0.7 * sim_score + 0.3 * diversity,