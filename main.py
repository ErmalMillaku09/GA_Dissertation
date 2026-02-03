import random
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# -------------------- PARAMETERS -------------------------
# =========================================================

DIMENSION = 5 # Each chromosome will encode 5 parameters telling GA that its a vector of 5 dimensions( 5  Genes)
BOUNDS = [(-5.0, 5.0)] * DIMENSION

POP_SIZE = 20
GENERATIONS = 250

CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
MUTATION_STD = 0.1

USE_ELITISM = True

# =========================================================
# ---------------- OBJECTIVE FUNCTIONS --------------------
# =========================================================

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

# =========================================================
# ---------------- POPULATION SETUP -----------------------
# =========================================================

def initialize_population():
    return [
        [random.uniform(low, high) for low, high in BOUNDS]
        for _ in range(POP_SIZE)
    ]
    # for _ in range(POP_SIZE):
    #     individual = [
    #         random.uniform(low, high) for low, high in BOUNDS
    #     ]
    #     population.append(individual)
    # return population


def evaluate_population(population):
    return [fitness_function(ind) for ind in population]

# =========================================================
# ---------------- SELECTION ------------------------------
# ======================================================

def linear_fitness_scaling(fitnesses, c = 2.0):
        f_avg = np.mean(fitnesses)
        f_max = np.max(fitnesses)

        if f_max == f_avg: # if both equal avoid division by 0
                return fitnesses.copy()

        a = (c - 1.0) * f_avg/(f_max - f_avg)
        b = f_avg * (1.0 - a)

        return [a * f + b for f in fitnesses] # Return scaled fitness

#In fitness-proportionate selection, preserving the correspondence between
# individuals and their fitness values is essential; any reordering must be applied consistently to both

def roulette_wheel_selection(population, scaled_fitness):
    total = sum(scaled_fitness) # total fitness

    # Normalizing probabilities
    probs = [f / total for f in scaled_fitness] # probabilities

    # Cumulative Distribution

    r = random.random()
    cumulative = 0

    for ind, p in zip(population, probs):
        cumulative += p
        if r <= cumulative:
            return ind
        # Fallback
    return population[-1]

# =========================================================
# ---------------- GENETIC OPERATORS ----------------------
# =========================================================

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


def mutate(individual):
    """
    Gaussian mutation with boundary handling
    """
    for i, (low, high) in enumerate(BOUNDS):
        if random.random() < MUTATION_RATE:
            individual[i] += random.gauss(0, MUTATION_STD)
            individual[i] = np.clip(individual[i], low, high)

    return individual

# =========================================================
# ---------------- EVOLUTION STEP -------------------------
# =========================================================

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

    # ----- ELITISM -----
    if USE_ELITISM:
        best_idx = np.argmax(fitnesses)
        new_population.append(population[best_idx].copy())

    # Generate offspring till population size is reached
    while len(new_population) < POP_SIZE:

        # PARENT SELECTION
        p1 = roulette_wheel_selection(population, scaled)
        p2 = roulette_wheel_selection(population, scaled)

        # Crossover
        c1, c2 = arithmetic_crossover(p1, p2)

        # Add next gen
        new_population.append(mutate(c1))

        if len(new_population) < POP_SIZE:
            new_population.append(mutate(c2))

    return new_population


# =========================================================
# ---------------- RUN ONE GA -----------------------------
# =========================================================


def run_ga():

    population = initialize_population()

    best_history = []
    avg_history = []

    for gen in range(GENERATIONS):

        population = evolve_one_generation(population)

        fitnesses = evaluate_population(population)

        best = np.max(fitnesses)
        avg = np.mean(fitnesses)

        best_history.append(best)
        avg_history.append(avg)

        # ----- keep  generation print -----
        print(f"Gen {gen}: best={best:.4f}, avg={avg:.4f}")

    return best_history, avg_history

# =========================================================
# ---------- RANDOM SEARCHING TO GET BASELINE -------------
# =========================================================

def run_random_search():
    """
    Pure random sampling baseline
    Same evaluation budget as GA
    """

    best_history = []

    best_so_far = float("inf")

    for gen in range(GENERATIONS):

        population = initialize_population()

        objs = [objective_function(ind) for ind in population]

        best_so_far = min(best_so_far, min(objs))

        best_history.append(best_so_far)

    return best_history


def run_random_search_fitness():
    """
    Pure random sampling baseline
    Returns BEST FITNESS per generation
    (for fair comparison with GA fitness plots)
    """

    best_history = []

    best_so_far = 0.0  # fitness is maximized

    for gen in range(GENERATIONS):

        population = initialize_population()

        fits = [fitness_function(ind) for ind in population]

        best_so_far = max(best_so_far, max(fits))

        best_history.append(best_so_far)

    return best_history
# =========================================================
# ---------------- RUN MANY GA ----------------------------
# =========================================================

def run_experiment(runs = 100, seed=None):
    """
    Run GA multiple times and compute statistics
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    all_best = []
    for r in range(runs):
        print(f"\n========== Run {r+1}/{runs} ==========")
        best_history, _ = run_ga()
        all_best.append(best_history)
    all_best = np.array(all_best)

    mean_curve = np.mean(all_best, axis=0)
    std_curve = np.std(all_best, axis=0)

    plot_statistics(all_best, mean_curve, std_curve)

    return all_best

def run_experiment_with_baseline(runs=100):

    ga_runs = []
    rand_runs = []

    for r in range(runs):
        print(f"Run {r+1}/{runs}")

        best_ga, _ = run_ga()
        best_rand = run_random_search()

        ga_runs.append(best_ga)
        rand_runs.append(best_rand)

    ga_runs = np.array(ga_runs)
    rand_runs = np.array(rand_runs)

    plot_comparison(ga_runs, rand_runs)


def run_experiment_with_fitness_baseline(runs=100):

    ga_runs = []
    rand_runs = []

    for r in range(runs):
        print(f"Run {r+1}/{runs}")

        best_ga, _ = run_ga()                 # fitness
        best_rand = run_random_search_fitness()  # fitness

        ga_runs.append(best_ga)
        rand_runs.append(best_rand)

    ga_runs = np.array(ga_runs)
    rand_runs = np.array(rand_runs)

    plot_fitness_comparison(ga_runs, rand_runs)
# =========================================================
# ---------------- PLOTTING -------------------------------
# =========================================================
def plot_history(best_history, avg_history):

    plt.figure(figsize=(8, 5))

    plt.plot(best_history, label="Best fitness")
    plt.plot(avg_history, label="Average fitness")

    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("GA Convergence")

    plt.yscale("log")  # ← your log-scale line kept

    plt.legend()
    plt.grid()
    plt.show()

def plot_statistics(all_runs, mean_curve, std_curve):
    gens = np.arange(len(mean_curve))
    plt.figure(figsize=(9, 6))

    # faint individual runs (spread visualization)
    for run in all_runs:
        plt.plot(run, alpha=0.05)

    # mean curve
    plt.plot(gens, mean_curve, linewidth=3, label="Mean best")

    # std band
    plt.fill_between(
        gens,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.3,
        label="±1 std"
    )

    plt.xlabel("Generations")
    plt.ylabel("Best Fitness")
    plt.title(f"GA Convergence Statistics ({len(all_runs)}) runs")

    plt.yscale("log")

    plt.legend()
    plt.grid()

    # Remove comment in case we need images
    # plt.savefig("ga_statistics.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ---------- Boxplot of final performance ----------
    plt.figure(figsize=(6, 4))

    final_vals = all_runs[:, -1]
    plt.boxplot(final_vals)

    plt.ylabel("Final best fitness")
    plt.title("Final Performance Distribution")

    plt.yscale("log")

    # Remove comment in case we need images
  #  plt.savefig("ga_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_comparison(ga_runs, rand_runs):

    gens = np.arange(GENERATIONS)

    ga_mean = np.mean(ga_runs, axis=0)
    rand_mean = np.mean(rand_runs, axis=0)

    plt.figure(figsize=(9,6))

    plt.plot(ga_mean, label="GA")
    plt.plot(rand_mean, label="Random search")

    plt.xlabel("Generation")
    plt.ylabel("Objective value")
    plt.title("GA vs Random baseline")

    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

def plot_fitness_comparison(ga_runs, rand_runs):

    gens = np.arange(GENERATIONS)

    ga_mean = np.mean(ga_runs, axis=0)
    rand_mean = np.mean(rand_runs, axis=0)

    plt.figure(figsize=(9,6))

    plt.plot(ga_mean, label="GA (fitness)")
    plt.plot(rand_mean, label="Random (fitness)")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA vs Random Baseline (Fitness)")

    plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.savefig("ga_vs_random_fitness.png", dpi=300, bbox_inches="tight")
    plt.show()
# =========================================================
# ---------------- MAIN ----------------------------------
# =========================================================

if __name__ == "__main__":

    # ----- keep your child demonstration prints -----
    p1 = [1, 2, 3, 4, 5]
    p2 = [5, 4, 9, 2, 1]

    for _ in range(5):
        c1, c2 = arithmetic_crossover(p1, p2)
        print("Child 1:", c1)
        print("Child 2:", c2)
        print()

        # ---------- SINGLE RUN ----------
    best_hist, avg_hist = run_ga()
    plot_history(best_hist, avg_hist)

    # ---------- MULTI-RUN STRESS TEST ----------
    # run_experiment(runs=200, seed=42)
    run_experiment_with_fitness_baseline(runs=200)