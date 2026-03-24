# config.py
from dataclasses import dataclass


# =========================================================
# -------------------- PARAMETERS -------------------------
# =========================================================

@dataclass
class GAConfig:
    # GA parameters
    DIMENSION: int = 5
    BOUNDS: tuple = (-5.0, 5.0)

    POP_SIZE: int = 50
    GENERATIONS: int = 250

    CROSSOVER_RATE: float = 0.9
    MUTATION_RATE: float = 0.15
    MUTATION_STD: float = 0.1

    USE_ELITISM: bool = True

    # Selection parameters
    SELECTION_METHOD: str = "ranking"  # roulette | tournament | ranking
    TOURNAMENT_K: int = 3

    # Objective
    OBJECTIVE_NAME: str = "ackley"  # "sphere" "rastrigin" "ackley" "rosenbrock"

    # GD parameters (add these with proper defaults)
    GD_ALPHA: float = 0.001
    RUNS: int = 50
    NFE: int = 10000

    # Algorithm choice
    ALGORITHM: str = "GD"  # "COMPARE" "GD"


# Create a default instance
cfg = GAConfig()