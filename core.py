import numpy as np
import random
from config import GAConfig
from benchmarks import get_objective, fitness_from_objective
from operators import select, arithmetic_crossover, mutate

# =========================================================
# ---------------- POPULATION SETUP -----------------------
# =========================================================

def initialize_population(cfg):
    low, high = cfg.BOUNDS
    return np.random.uniform(low, high, (cfg.POP_SIZE, cfg.DIMENSION))


def evaluate_population(pop, cfg):
    # Get objective function dynamically based on current config
    objective_func = get_objective(cfg.OBJECTIVE_NAME)
    obj = objective_func(pop)
    fit = fitness_from_objective(obj)
    return obj, fit


# =========================================================
# ---------------- EVOLUTION STEP -------------------------
# =========================================================

def evolve_one_generation(pop, cfg):
    obj, fitness = evaluate_population(pop, cfg)
    new_pop = []

    # ----- ELITISM -----
    if cfg.USE_ELITISM:
        elite = pop[np.argmax(fitness)]
        new_pop.append(elite.copy())

    while len(new_pop) < cfg.POP_SIZE:

        p1 = select(pop, fitness, cfg)
        p2 = select(pop, fitness, cfg)

        c1, c2 = arithmetic_crossover(p1, p2, cfg)

        new_pop.append(mutate(c1, cfg))

        if len(new_pop) < cfg.POP_SIZE:
            new_pop.append(mutate(c2, cfg))

    return np.array(new_pop)


# =========================================================
# ---------------- RUN ONE GA -----------------------------
# =========================================================

def run_ga(cfg, verbose=False):
    pop = initialize_population(cfg)

    best_fit_hist = []
    avg_fit_hist = []

    best_obj_hist = []
    avg_obj_hist = []

    for gen in range(cfg.GENERATIONS):

        pop = evolve_one_generation(pop, cfg)

        obj, fit = evaluate_population(pop, cfg)

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


def run_random_search(cfg):
    low, high = cfg.BOUNDS
    # Get objective function dynamically
    objective_func = get_objective(cfg.OBJECTIVE_NAME)

    best_so_far = np.inf
    history = []

    for _ in range(cfg.GENERATIONS):
        samples = np.random.uniform(low, high,
                                    (cfg.POP_SIZE, cfg.DIMENSION))

        objs = objective_func(samples)
        best_so_far = min(best_so_far, np.min(objs))

        history.append(1 / (1 + best_so_far))  # convert to fitness

    return np.array(history)