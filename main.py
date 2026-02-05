from experiments import *
from config import GAConfig

cfg = GAConfig()

if __name__ == "__main__":
    # run_experiment(50)
    full_ga_study(
        objectives=["sphere", "rastrigin", "ackley", "rosenbrock"],
        selections=["roulette", "tournament", "ranking"],
        mutations=[0.05, 0.1, 0.15, 0.2],
        runs=25
    )