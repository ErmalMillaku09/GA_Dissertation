import numpy as np
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =========================================================
# -------------------- PARAMETERS -------------------------
# =========================================================


@dataclass
class GAConfig:
    DIMENSION: int = 5
    BOUNDS: tuple = (-5.0, 5.0)

    POP_SIZE: int = 50
    GENERATIONS: int = 250

    CROSSOVER_RATE: float = 0.9
    MUTATION_RATE: float = 0.15
    MUTATION_STD: float = 0.1

    USE_ELITISM: bool = True

    # new
    SELECTION_METHOD: str = "ranking"  # roulette | tournament | ranking
    TOURNAMENT_K: int = 3

    OBJECTIVE_NAME: str = "sphere"  # "sphere" "rastrigin" "ackley" "rosenbrock"


cfg = GAConfig()


# =========================================================
# ---------------- OBJECTIVE FUNCTIONS --------------------
# =========================================================

class Benchmarks:
    """
    Benchmark suite for GA testing
    """

    @staticmethod
    def sphere(X):
        """Shifted Sphere | global minimum at 1"""
        return np.sum((X - 1.0) ** 2, axis=1)

    @staticmethod
    def rastrigin(X):
        """Rastrigin | many local minima"""
        A = 10
        n = X.shape[1]
        return A * n + np.sum(X ** 2 - A * np.cos(2 * np.pi * X), axis=1)

    @staticmethod
    def ackley(X):
        """Ackley function"""
        n = X.shape[1]
        s1 = np.sum(X ** 2, axis=1)
        s2 = np.sum(np.cos(2 * np.pi * X), axis=1)
        return (
                -20 * np.exp(-0.2 * np.sqrt(s1 / n))
                - np.exp(s2 / n)
                + 20 + np.e
        )

    @staticmethod
    def rosenbrock(X):
        """Rosenbrock valley"""
        return np.sum(
            100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2,
            axis=1
        )


def get_objective(name):
    return getattr(Benchmarks, name)


OBJECTIVE = get_objective(cfg.OBJECTIVE_NAME)


def fitness_from_objective(obj):
    return 1.0 / (1.0 + obj)


# =========================================================
# ---------------- POPULATION SETUP -----------------------
# =========================================================

def initialize_population():
    low, high = cfg.BOUNDS
    return np.random.uniform(low, high, (cfg.POP_SIZE, cfg.DIMENSION))


def evaluate_population(pop):
    obj = OBJECTIVE(pop)
    fit = fitness_from_objective(obj)
    return obj, fit


# =========================================================
# ---------------- SELECTION ------------------------------
# =========================================================

def roulette_selection(pop, fitness):
    idx = random.choices(range(len(pop)), weights=fitness, k=1)[0]
    return pop[idx]


def tournament_selection(pop, fitness):
    idxs = np.random.choice(len(pop), cfg.TOURNAMENT_K, replace=False)
    best = idxs[np.argmax(fitness[idxs])]
    return pop[best]


def ranking_selection(pop, fitness):
    ranks = np.argsort(np.argsort(fitness))
    probs = ranks + 1
    idx = random.choices(range(len(pop)), weights=probs, k=1)[0]
    return pop[idx]


def select(pop, fitness):
    if cfg.SELECTION_METHOD == "roulette":
        return roulette_selection(pop, fitness)
    if cfg.SELECTION_METHOD == "tournament":
        return tournament_selection(pop, fitness)
    if cfg.SELECTION_METHOD == "ranking":
        return ranking_selection(pop, fitness)
    raise ValueError("Unknown selection method")


# =========================================================
# ---------------- GENETIC OPERATORS ----------------------
# =========================================================

def arithmetic_crossover(p1, p2):
    if random.random() > cfg.CROSSOVER_RATE:
        return p1.copy(), p2.copy()

    alpha = random.random()
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = (1 - alpha) * p1 + alpha * p2
    return c1, c2


def mutate(child):
    mask = np.random.rand(cfg.DIMENSION) < cfg.MUTATION_RATE
    noise = np.random.normal(0, cfg.MUTATION_STD, cfg.DIMENSION)
    child = child + mask * noise

    low, high = cfg.BOUNDS
    return np.clip(child, low, high)


# =========================================================
# ---------------- EVOLUTION STEP -------------------------
# =========================================================

def evolve_one_generation(pop):
    obj, fitness = evaluate_population(pop)
    new_pop = []

    # ----- ELITISM -----
    if cfg.USE_ELITISM:
        elite = pop[np.argmax(fitness)]
        new_pop.append(elite.copy())

    while len(new_pop) < cfg.POP_SIZE:

        p1 = select(pop, fitness)
        p2 = select(pop, fitness)

        c1, c2 = arithmetic_crossover(p1, p2)

        new_pop.append(mutate(c1))

        if len(new_pop) < cfg.POP_SIZE:
            new_pop.append(mutate(c2))

    return np.array(new_pop)


# =========================================================
# ---------------- RUN ONE GA -----------------------------
# =========================================================

def run_ga(verbose=False):
    pop = initialize_population()

    best_fit_hist = []
    avg_fit_hist = []

    best_obj_hist = []
    avg_obj_hist = []

    for gen in range(cfg.GENERATIONS):

        pop = evolve_one_generation(pop)

        obj, fit = evaluate_population(pop)

        # ----- fitness -----
        best_fit_hist.append(np.max(fit))
        avg_fit_hist.append(np.mean(fit))

        # ----- objective (minimization) -----
        best_obj_hist.append(np.min(obj))
        avg_obj_hist.append(np.mean(obj))

        if verbose:
            print(
                f"Gen {gen}: "
                f"best_obj={best_obj_hist[-1]:.6f} "
                f"best_fit={best_fit_hist[-1]:.6f}"
            )

    return (
        np.array(best_fit_hist),
        np.array(avg_fit_hist),
        np.array(best_obj_hist),
        np.array(avg_obj_hist),
    )


def run_random_search():
    low, high = cfg.BOUNDS

    best_so_far = np.inf
    history = []

    for _ in range(cfg.GENERATIONS):
        samples = np.random.uniform(low, high,
                                    (cfg.POP_SIZE, cfg.DIMENSION))

        objs = OBJECTIVE(samples)
        best_so_far = min(best_so_far, np.min(objs))

        history.append(1 / (1 + best_so_far))  # convert to fitness

    return np.array(history)


# =========================================================
# ---------------- BENCHMARK EXPERIMENT -------------------
# =========================================================

def run_experiment(runs=50):
    all_fit = []
    all_obj = []

    for r in range(runs):
        random.seed(r)
        np.random.seed(r)

        best_fit, _, best_obj, _ = run_ga()

        all_fit.append(best_fit)
        all_obj.append(best_obj)

    all_fit = np.array(all_fit)
    all_obj = np.array(all_obj)

    # fitness plots
    plot_statistics(
        all_fit,
        np.mean(all_fit, axis=0),
        np.std(all_fit, axis=0)
    )

    # NEW objective plots
    plot_objective_statistics(all_obj)


def compare_ga_vs_random(runs=50):
    """
    Compare Genetic Algorithm vs Random Search performance over multiple runs.

    Parameters:
    -----------
    runs : int
        Number of experimental runs to perform (default: 50)
    """
    ga_runs = []
    rand_runs = []

    for r in range(runs):
        random.seed(r)
        np.random.seed(r)

        best_ga, _, _, _ = run_ga()
        best_rand = run_random_search()

        ga_runs.append(best_ga)
        rand_runs.append(best_rand)

    ga_runs = np.array(ga_runs)
    rand_runs = np.array(rand_runs)

    plot_fitness_comparison(ga_runs, rand_runs)


# Add an alias for backward compatibility
run_experiment_with_baseline = compare_ga_vs_random


# =========================================================
# ---------------- PLOTTING -------------------------------
# =========================================================

def plot_statistics(all_runs, mean, std):
    gens = np.arange(len(mean))

    # ---------- Mean + std band ----------
    plt.figure(figsize=(9, 6))

    for r in all_runs:
        plt.plot(r, alpha=0.05)

    plt.plot(mean, linewidth=3, label="Mean best")
    plt.fill_between(gens, mean - std, mean + std, alpha=0.3, label="±1 std")

    plt.xlabel("Generations")
    plt.ylabel("Best fitness")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------- Boxplot of final performance ----------
    plt.figure(figsize=(6, 4))

    final_vals = all_runs[:, -1]
    plt.boxplot(final_vals)

    plt.ylabel("Final best fitness")
    plt.title("Final Performance Distribution")

    plt.yscale("log")
    plt.grid()
    plt.show()


def plot_history(best_history, avg_history):
    plt.figure(figsize=(8, 5))

    plt.plot(best_history, label="Best fitness")
    plt.plot(avg_history, label="Average fitness")

    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("GA Convergence (single run)")

    plt.legend()
    plt.grid()
    plt.show()


def plot_fitness_comparison(ga_runs, rand_runs):
    gens = np.arange(cfg.GENERATIONS)

    ga_mean = np.mean(ga_runs, axis=0)
    rand_mean = np.mean(rand_runs, axis=0)

    plt.figure(figsize=(9, 6))

    plt.plot(ga_mean, label="GA")
    plt.plot(rand_mean, label="Random search")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA vs Random Baseline")

    plt.legend()
    plt.grid()
    plt.show()


def plot_objective_history(best_obj, avg_obj):
    plt.figure(figsize=(8, 5))

    plt.plot(best_obj, label="Best objective")
    plt.plot(avg_obj, label="Average objective")

    plt.xlabel("Generations")
    plt.ylabel("Objective value")
    plt.title("GA Convergence (objective)")

    plt.yscale("log")  # objective benefits from log scale

    plt.legend()
    plt.grid()
    plt.show()


def plot_objective_statistics(all_runs):
    mean = np.mean(all_runs, axis=0)
    std = np.std(all_runs, axis=0)
    gens = np.arange(len(mean))

    # ----- convergence band -----
    plt.figure(figsize=(9, 6))

    for r in all_runs:
        plt.plot(r, alpha=0.05)

    plt.plot(mean, linewidth=3, label="Mean best objective")
    plt.fill_between(gens, mean - std, mean + std, alpha=0.3, label="±1 std")

    plt.xlabel("Generations")
    plt.ylabel("Objective value")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

    # ----- boxplot -----
    plt.figure(figsize=(6, 4))

    final_vals = all_runs[:, -1]
    plt.boxplot(final_vals)

    plt.ylabel("Final best objective")
    plt.title("Final Objective Distribution")
    plt.yscale("log")
    plt.grid()
    plt.show()


# =========================================================
# ---------------- MAIN ----------------------------------
# =========================================================

if __name__ == "__main__":
    run_experiment(50)
