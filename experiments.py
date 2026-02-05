import numpy as np
import random
import matplotlib.pyplot as plt
from config import GAConfig
from core import run_ga, run_random_search
from plots import plot_statistics, plot_objective_statistics, plot_fitness_comparison
from benchmarks import get_objective


def run_experiment(cfg=None, runs=50):
    """
    Run multiple GA experiments and plot statistics.

    Parameters:
    -----------
    cfg : GAConfig or None
        Configuration to use. If None, uses default GAConfig()
    runs : int
        Number of experimental runs
    """
    if cfg is None:
        cfg = GAConfig()

    all_fit = []
    all_obj = []

    for r in range(runs):
        random.seed(r)
        np.random.seed(r)

        best_fit, _, best_obj, _ = run_ga(cfg)

        all_fit.append(best_fit)
        all_obj.append(best_obj)

    all_fit = np.array(all_fit)
    all_obj = np.array(all_obj)

    # fitness plots
    plot_statistics(
        all_fit,
        np.mean(all_fit, axis=0),
        np.std(all_fit, axis=0),
        cfg
    )

    # NEW objective plots
    plot_objective_statistics(all_obj, cfg)


def compare_ga_vs_random(cfg=None, runs=50):
    """
    Compare Genetic Algorithm vs Random Search performance over multiple runs.

    Parameters:
    -----------
    cfg : GAConfig or None
        Configuration to use. If None, uses default GAConfig()
    runs : int
        Number of experimental runs to perform (default: 50)
    """
    if cfg is None:
        cfg = GAConfig()

    ga_runs = []
    rand_runs = []

    for r in range(runs):
        random.seed(r)
        np.random.seed(r)

        best_ga, _, _, _ = run_ga(cfg)
        best_rand = run_random_search(cfg)

        ga_runs.append(best_ga)
        rand_runs.append(best_rand)

    ga_runs = np.array(ga_runs)
    rand_runs = np.array(rand_runs)

    plot_fitness_comparison(ga_runs, rand_runs, cfg)


# Add an alias for backward compatibility
run_experiment_with_baseline = compare_ga_vs_random


def run_parameter_sweep(param_name, values, base_cfg=None, runs=20):
    """
    Generic experiment runner.

    Example:
        run_parameter_sweep("SELECTION_METHOD", ["roulette","tournament","ranking"])
        run_parameter_sweep("MUTATION_RATE", [0.05, 0.1, 0.2])
        run_parameter_sweep("POP_SIZE", [30, 50, 100])

    Parameters:
    -----------
    param_name : str
        Name of parameter to sweep
    values : list
        List of values to test
    base_cfg : GAConfig or None
        Base configuration to use. If None, uses default GAConfig()
    runs : int
        Number of runs per parameter value
    """
    if base_cfg is None:
        base_cfg = GAConfig()

    results = {}
    gens = np.arange(base_cfg.GENERATIONS)

    plt.figure(figsize=(9, 6))

    print("\n=== PARAMETER SWEEP RESULTS ===")
    print(f"Parameter: {param_name}")
    print("-" * 50)
    print(f"{'Value':<15} {'Mean final':<15} {'Std':<15}")

    for val in values:
        # Create a copy of the config with the modified parameter
        cfg = GAConfig(
            DIMENSION=base_cfg.DIMENSION,
            BOUNDS=base_cfg.BOUNDS,
            POP_SIZE=base_cfg.POP_SIZE,
            GENERATIONS=base_cfg.GENERATIONS,
            CROSSOVER_RATE=base_cfg.CROSSOVER_RATE,
            MUTATION_RATE=base_cfg.MUTATION_RATE,
            MUTATION_STD=base_cfg.MUTATION_STD,
            USE_ELITISM=base_cfg.USE_ELITISM,
            SELECTION_METHOD=base_cfg.SELECTION_METHOD,
            TOURNAMENT_K=base_cfg.TOURNAMENT_K,
            OBJECTIVE_NAME=base_cfg.OBJECTIVE_NAME
        )

        # Set the parameter value
        setattr(cfg, param_name, val)

        runs_data = []

        for r in range(runs):
            random.seed(r)
            np.random.seed(r)

            best_fit, _, _, _ = run_ga(cfg)
            runs_data.append(best_fit)

        runs_data = np.array(runs_data)

        mean_curve = runs_data.mean(axis=0)
        final_vals = runs_data[:, -1]

        mean_final = final_vals.mean()
        std_final = final_vals.std()

        results[val] = mean_curve

        # print table row
        print(f"{str(val):<15} {mean_final:<15.6f} {std_final:<15.6f}")

        # plot curve
        plt.plot(mean_curve, linewidth=2, label=str(val))

    # ----- plot formatting -----
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.yscale("log")

    plt.title(
        f"Parameter Sweep: {param_name}\n"
        f"Objective: {base_cfg.OBJECTIVE_NAME}"
    )

    plt.legend(title=param_name)
    plt.tight_layout()
    plt.show()

    return results


def compare_objectives(objectives, base_cfg=None, runs=20):
    """
    Compare GA performance across multiple benchmark functions.

    Example:
        compare_objectives(["sphere", "rastrigin", "ackley", "rosenbrock"])

    Parameters:
    -----------
    objectives : list
        List of objective function names to compare
    base_cfg : GAConfig or None
        Base configuration to use. If None, uses default GAConfig()
    runs : int
        Number of runs per objective
    """
    if base_cfg is None:
        base_cfg = GAConfig()

    plt.figure(figsize=(9, 6))

    print("\n=== OBJECTIVE COMPARISON ===")
    print("-" * 50)
    print(f"{'Objective':<15} {'Mean final':<15} {'Std':<15}")

    for name in objectives:
        # Create a new config with the specified objective
        cfg = GAConfig(
            DIMENSION=base_cfg.DIMENSION,
            BOUNDS=base_cfg.BOUNDS,
            POP_SIZE=base_cfg.POP_SIZE,
            GENERATIONS=base_cfg.GENERATIONS,
            CROSSOVER_RATE=base_cfg.CROSSOVER_RATE,
            MUTATION_RATE=base_cfg.MUTATION_RATE,
            MUTATION_STD=base_cfg.MUTATION_STD,
            USE_ELITISM=base_cfg.USE_ELITISM,
            SELECTION_METHOD=base_cfg.SELECTION_METHOD,
            TOURNAMENT_K=base_cfg.TOURNAMENT_K,
            OBJECTIVE_NAME=name
        )

        runs_data = []

        for r in range(runs):
            random.seed(r)
            np.random.seed(r)

            best_fit, _, _, _ = run_ga(cfg)
            runs_data.append(best_fit)

        runs_data = np.array(runs_data)

        mean_curve = runs_data.mean(axis=0)
        final_vals = runs_data[:, -1]

        mean_final = final_vals.mean()
        std_final = final_vals.std()

        print(f"{name:<15} {mean_final:<15.6f} {std_final:<15.6f}")

        plt.plot(mean_curve, linewidth=2.5, label=name)

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.yscale("log")

    plt.title("GA Performance Across Benchmark Functions")

    plt.legend(title="Objective")
    plt.tight_layout()
    plt.show()


def full_ga_study(
        objectives,
        selections,
        mutations,
        base_cfg=None,
        runs=20
):
    """
    Full factorial GA study:
        objective × selection × mutation

    Parameters:
    -----------
    objectives : list
        List of objective function names
    selections : list
        List of selection methods
    mutations : list
        List of mutation rates
    base_cfg : GAConfig or None
        Base configuration to use. If None, uses default GAConfig()
    runs : int
        Number of runs per combination
    """
    if base_cfg is None:
        base_cfg = GAConfig()

    print("\n====== FULL GA STUDY ======\n")

    summary = []

    for obj_name in objectives:
        print(f"\n### Objective: {obj_name} ###")

        best_score = -np.inf
        best_combo = None

        for sel in selections:
            for mut in mutations:
                # Create config for this combination
                cfg = GAConfig(
                    DIMENSION=base_cfg.DIMENSION,
                    BOUNDS=base_cfg.BOUNDS,
                    POP_SIZE=base_cfg.POP_SIZE,
                    GENERATIONS=base_cfg.GENERATIONS,
                    CROSSOVER_RATE=base_cfg.CROSSOVER_RATE,
                    MUTATION_RATE=mut,
                    MUTATION_STD=base_cfg.MUTATION_STD,
                    USE_ELITISM=base_cfg.USE_ELITISM,
                    SELECTION_METHOD=sel,
                    TOURNAMENT_K=base_cfg.TOURNAMENT_K,
                    OBJECTIVE_NAME=obj_name
                )

                finals = []

                for r in range(runs):
                    random.seed(r)
                    np.random.seed(r)

                    best_fit, _, _, _ = run_ga(cfg)
                    finals.append(best_fit[-1])

                mean_final = np.mean(finals)

                print(
                    f"{sel:<10} mut={mut:<5} → {mean_final:.5f}"
                )

                if mean_final > best_score:
                    best_score = mean_final
                    best_combo = (sel, mut)

        summary.append((obj_name, *best_combo, best_score))

    # ----- print summary -----
    print("\n====== BEST COMBINATIONS ======")
    print(f"{'Objective':<12} {'Selection':<12} {'Mutation':<10} {'Score'}")

    for row in summary:
        print(f"{row[0]:<12} {row[1]:<12} {row[2]:<10} {row[3]:.5f}")

    return summary