import random

import numpy as np

# Problem Parameters
DIMENSION = 5 # Each chromosome will encode 5 parameters telling GA that its a vector of 5 dimensions( 5  Genes)
BOUNDS = [(-5.0, 5.0)] * DIMENSION

# def objective_function(x):
#
#     """
#     Shifted Sphere Function
#     Global minimum at x_i = 1
#     """
#     return sum((xi - 1.0)**2 for xi in x)
def objective_function(x):
    """
    Rastrigin Function
    Global minimum at x_i = 0
    Many local minima — tricky landscape
    """
    A = 120
    n = len(x)
    return A * n + sum((xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x)

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

        if f_max == f_avg: # if both equal avoid division by 0
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

MUTATION_RATE = 0.1 # PROBABILITY PER GENE
MUTATION_STD = 0.1  # GAUSSIAN NOISE

def mutate(individual):

    """
    Gaussian mutation with boundary handling
    """
    for i, (low, high) in enumerate(BOUNDS):
      if random.random() < MUTATION_RATE:
          individual[i] += random.gauss(0, MUTATION_STD)


          if individual[i] < low:   # Enforcing Bounds
                individual[i] = low
          elif individual[i] > high:
                individual[i] = high

    return individual


selection_counts = [0] * POP_SIZE
TRIALS = 10000

def evolve_one_generation(population):
    """
    Creates next gen using:
     - roulette
     - arithmetic crossover
     - gaussian mutation with boundary handling
    """
    # Evaluate + scale fitness
    fitnesses = evaluate_population(population)
    scaled = linear_fitness_scaling(fitnesses)

    new_population = []

    # Generate offspring till population size is reached
    while len(new_population) < POP_SIZE:

        # PARENT SELECTION
        parent1 = roulette_wheel_selection(population, scaled)
        parent2 = roulette_wheel_selection(population, scaled)

        # Crossover
        child1, child2 = arithmetic_crossover(parent1, parent2)

        # Mutation
        child1 = mutate(child1)
        child2 = mutate(child2)

        # Add next gen

        new_population.append(child1)

        if len(new_population) < POP_SIZE:
            new_population.append(child2)

    return new_population



GENERATIONS = 100

population = initialize_population()

for gen in range(GENERATIONS):
    population = evolve_one_generation(population)

    # Monitor progress
    fitnesses = evaluate_population(population)
    best = np.max(fitnesses)
    avg =  np.mean(fitnesses)

    print(f"Gen {gen}: best={best:3f}, avg={avg:3f}")

#
# for i in range(TRIALS):
#     selected = roulette_wheel_selection(population, scaled)
#     idx = population.index(selected)
#     selection_counts[idx] += 1
#
#
# indexed = list(enumerate(scaled))
#
# # Sort by fitness descending
# indexed.sort(key=lambda x: x[1], reverse=True)
#
# for idx, fit in indexed:
#     print(f"Individual {idx}: fitness = {fit:.4f}, selected = {selection_counts[idx]}")
