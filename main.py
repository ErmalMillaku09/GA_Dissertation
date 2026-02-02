import random

import numpy as np

# Problem Parameters
DIMENSION = 5
BOUNDS = [(-5.0, 5.0)] * DIMENSION

def objective_function(x):

    """
    Shifted Sphere Function
    Global minimum at x_i = 1
    """
    return sum((xi - 1.0)**2 for xi in x)

def fitness_function(x):
    return 1.0/(1.0 + objective_function(x))

POP_SIZE = 20

def initialize_population():
    population = []
    for _ in range(POP_SIZE):
        individual = [
            random.uniform(low, high) for low, high in BOUNDS
        ]
        population.append(individual)
    return population


def evaluate_population(population):
    return [fitness_function(ind) for ind in population]


def linear_fitness_scaling(fitnesses, c = 2.0):
        f_avg = np.mean(fitnesses)
        f_max = np.max(fitnesses)

        if f_max == f_avg:
                return fitnesses.copy()
        a = (c - 1.0) * f_avg/(f_max - f_avg)
        b = f_avg * (1.0 - a)

        scaled_fitness = [a * f + b for f in fitnesses]
        return scaled_fitness
#In fitness-proportionate selection, preserving the correspondence between
# individuals and their fitness values is essential; any reordering must be applied consistently to both

def roulette_wheel_selection(population, scaled_fitness):
    total_fitness = sum(scaled_fitness)

    # Normalizing to probabilities
    probabilities = [f / total_fitness for f in scaled_fitness]

    # Cumulative Distribution
    cumulative_prob = []
    cumulative_sum = 0.0
    for p in probabilities:
        cumulative_sum += p
        cumulative_prob.append(cumulative_sum)
    # Spinning the wheel

    r = random.random()
    for i, threshold in enumerate(cumulative_prob):
        if r <= threshold:
            return population[i]

        # Fallback
    return population[-1]

CROSSOVER_RATE = 0.9

def arithmetic_crossover(parent1, parent2):
    """
    Real-value arithmetic crossover
    Returns two offspring.
    """
    if random.random() > CROSSOVER_RATE:
    #  No Crossover off spring are copies of parents
        return parent1.copy(), parent2.copy()

    alpha = random.random()
    child1 = [
        alpha * x + (1-alpha) * y
        for x, y in zip(parent1, parent2)
    ]

    child2 = [
        (1-alpha) * x + alpha * y
        for x, y in zip(parent1, parent2)
    ]

    return child1, child2

p1 = [1, 2, 3, 4, 5]
p2 = [5, 4, 9, 2, 1]

for _ in range(5):
    c1, c2 = arithmetic_crossover(p1, p2)
    print("Child 1:", c1)
    print("Child 2:", c2)
    print()


population = initialize_population()
fitnesses = evaluate_population(population)
scaled = linear_fitness_scaling(fitnesses)


selection_counts = [0] * POP_SIZE
TRIALS = 100000

for i in range(TRIALS):
    selected = roulette_wheel_selection(population, scaled)
    idx = population.index(selected)
    selection_counts[idx] += 1


indexed = list(enumerate(scaled))

# Sort by fitness descending
indexed.sort(key=lambda x: x[1], reverse=True)

for idx, fit in indexed:
    print(f"Individual {idx}: fitness = {fit:.4f}, selected = {selection_counts[idx]}")
