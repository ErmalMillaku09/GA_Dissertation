
from experiments import *
from config import GAConfig
from plots import *
cfg = GAConfig()

if __name__ == "__main__":

    gd_single_run(cfg)

    gd_statistics(cfg)

    alpha_results = gd_alpha_sweep(cfg)
    plot_alpha_sweep(alpha_results)

    ga_vs_gd_comparison(cfg)

    ga_vs_gd_statistics(cfg)
    #### 2

    # cfg = GAConfig(OBJECTIVE_NAME="rastrigin", GENERATIONS=100)
    # print(cfg)
    # best_fit, avg_fit, best_obj, avg_obj = run_ga(cfg, verbose=True)
    # plot_history(best_fit, avg_fit, cfg)  # Fitness convergence
    # plot_objective_history(best_obj, avg_obj, cfg)  # Objective convergence

    # 2. Statistical reliability
    #run_experiment(cfg, runs=30)  # Shows convergence and distribution