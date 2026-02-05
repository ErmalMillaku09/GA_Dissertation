import random
import numpy as np

# =========================================================
# ---------------- SELECTION ------------------------------
# =========================================================

def roulette_selection(pop, fitness, cfg):
    """Roulette wheel selection"""
    idx = random.choices(range(len(pop)), weights=fitness, k=1)[0]
    return pop[idx]


def tournament_selection(pop, fitness, cfg):
    """Tournament selection with size cfg.TOURNAMENT_K"""
    idxs = np.random.choice(len(pop), cfg.TOURNAMENT_K, replace=False)
    best = idxs[np.argmax(fitness[idxs])]
    return pop[best]


def ranking_selection(pop, fitness, cfg):
    """Ranking selection"""
    ranks = np.argsort(np.argsort(fitness))
    probs = ranks + 1
    idx = random.choices(range(len(pop)), weights=probs, k=1)[0]
    return pop[idx]


def select(pop, fitness, cfg):
    """Main selection dispatcher"""
    if cfg.SELECTION_METHOD == "roulette":
        return roulette_selection(pop, fitness, cfg)
    if cfg.SELECTION_METHOD == "tournament":
        return tournament_selection(pop, fitness, cfg)
    if cfg.SELECTION_METHOD == "ranking":
        return ranking_selection(pop, fitness, cfg)
    raise ValueError(f"Unknown selection method: {cfg.SELECTION_METHOD}")


# =========================================================
# ---------------- GENETIC OPERATORS ----------------------
# =========================================================

def arithmetic_crossover(p1, p2, cfg):
    """Arithmetic crossover with probability cfg.CROSSOVER_RATE"""
    if random.random() > cfg.CROSSOVER_RATE:
        return p1.copy(), p2.copy()

    alpha = random.random()
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = (1 - alpha) * p1 + alpha * p2
    return c1, c2


def mutate(child, cfg):
    """Gaussian mutation with probability cfg.MUTATION_RATE"""
    mask = np.random.rand(cfg.DIMENSION) < cfg.MUTATION_RATE
    noise = np.random.normal(0, cfg.MUTATION_STD, cfg.DIMENSION)
    child = child + mask * noise

    low, high = cfg.BOUNDS
    return np.clip(child, low, high)