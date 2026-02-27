from dataclasses import dataclass

# =========================================================
# -------------------- PARAMETERS -------------------------
# =========================================================

@dataclass
class GAConfig:
    DIMENSION: int = 5
    BOUNDS: tuple = (-5.0, 5)

    POP_SIZE: int = 50
    GENERATIONS: int = 250

    CROSSOVER_RATE: float = 0.9
    MUTATION_RATE: float = 0.15
    MUTATION_STD: float = 0.1

    USE_ELITISM: bool = True

    # new
    SELECTION_METHOD: str = "ranking"  # roulette | tournament | ranking
    TOURNAMENT_K: int = 3

    OBJECTIVE_NAME: str = "ackley"  # "sphere" "rastrigin" "ackley" "rosenbrock"
    GD_ALPHA = 0.001
    RUNS = 50
    NFE = 10000

    ALGORITHM = "GD" # "COMPARE" "GD"
cfg = GAConfig()


