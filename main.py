
from experiments import *
from config import GAConfig
from plots import *
cfg = GAConfig()

if __name__ == "__main__":
    #run_experiment(cfg, 50)
    full_ga_study(
        objectives=["sphere", "rastrigin", "ackley", "rosenbrock"],
        selections=["roulette", "tournament", "ranking"],
        mutations=[0.05, 0.1, 0.15, 0.2],
        runs=25
    )

    #### 2

    # cfg = GAConfig(OBJECTIVE_NAME="rastrigin", GENERATIONS=100)
    # print(cfg)
    # best_fit, avg_fit, best_obj, avg_obj = run_ga(cfg, verbose=True)
    # plot_history(best_fit, avg_fit, cfg)  # Fitness convergence
    # plot_objective_history(best_obj, avg_obj, cfg)  # Objective convergence

    # 2. Statistical reliability
    #run_experiment(cfg, runs=30)  # Shows convergence and distribution