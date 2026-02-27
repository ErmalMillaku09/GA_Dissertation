import numpy as np
import random
import matplotlib.pyplot as plt
from config import GAConfig
from core import run_ga, run_random_search
from plots import plot_statistics, plot_objective_statistics, plot_fitness_comparison
from benchmarks import get_objective
import time
from gradient_descent import gradient_descent

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
    """Full factorial GA study with logging."""

    if base_cfg is None:
        base_cfg = GAConfig()

    print("\n====== FULL GA STUDY ======\n")
    print(f"Objectives: {len(objectives)}")
    print(f"Selections: {len(selections)}")
    print(f"Mutations: {len(mutations)}")
    print(f"Runs per combo: {runs}")
    print(f"Total combos: {len(objectives) * len(selections) * len(mutations)}")
    print(f"Total GA runs: {len(objectives) * len(selections) * len(mutations) * runs}")
    print("-" * 50)

    total_evaluations = 0
    start_time = time.time()

    summary = []

    for obj_idx, obj_name in enumerate(objectives):
        print(f"\n### Objective {obj_idx + 1}/{len(objectives)}: {obj_name} ###")

        best_score = -np.inf
        best_combo = None

        for sel_idx, sel in enumerate(selections):
            for mut_idx, mut in enumerate(mutations):

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

                # Calculate evaluations for this combo
                evaluations_per_run = cfg.POP_SIZE * cfg.GENERATIONS
                combo_evaluations = evaluations_per_run * runs
                total_evaluations += combo_evaluations

                print(f"  Combo {sel_idx * len(mutations) + mut_idx + 1}/{len(selections) * len(mutations)}: "
                      f"{sel:<10} mut={mut:<5} ", end="")

                finals = []

                for r in range(runs):
                    random.seed(r)
                    np.random.seed(r)

                    best_fit, _, _, _ = run_ga(cfg, verbose=False)
                    finals.append(best_fit[-1])

                mean_final = np.mean(finals)

                print(f"→ {mean_final:.5f} "
                      f"({combo_evaluations:,} evals)")

                if mean_final > best_score:
                    best_score = mean_final
                    best_combo = (sel, mut)

        summary.append((obj_name, *best_combo, best_score))

    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time

    # ----- print summary -----
    print("\n" + "=" * 50)
    print("====== BEST COMBINATIONS ======")
    print(f"{'Objective':<12} {'Selection':<12} {'Mutation':<10} {'Score'}")

    for row in summary:
        print(f"{row[0]:<12} {row[1]:<12} {row[2]:<10} {row[3]:.5f}")

    # ----- statistics -----
    print("\n" + "=" * 50)
    print("====== STUDY STATISTICS ======")
    print(f"Total combinations tested: {len(objectives) * len(selections) * len(mutations)}")
    print(f"Total GA runs: {len(objectives) * len(selections) * len(mutations) * runs}")
    print(f"Total objective evaluations: {total_evaluations:,}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Time per GA run: {total_time / (len(objectives) * len(selections) * len(mutations) * runs):.3f} seconds")
    print(f"Evaluations per second: {total_evaluations / total_time:,.0f}")

    return summary


#==================== Gradient Descent Section ===========
def gd_single_run(cfg):
    x, fval, history, nfe = gradient_descent(
        cfg,
        alpha=cfg.GD_ALPHA,
        max_nfe=cfg.NFE
    )

    plt.figure()
    plt.plot(history)
    plt.title("Gradient Descent - Single Run Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Best Objective Value")
    plt.grid(True)
    plt.show()

    print("GD final value:", fval)
    print("GD evaluations:", nfe)


def gd_statistics(cfg):
    results = [
        gradient_descent(cfg, alpha=cfg.GD_ALPHA, max_nfe=cfg.NFE)[1]
        for _ in range(cfg.RUNS)
    ]

    mean = np.mean(results)
    std = np.std(results)

    print("\n====== GD 50-Run Statistics ======")
    print("Mean:", mean)
    print("Std:", std)

    return mean, std

def gd_alpha_sweep(cfg):

    alphas = [1e-4, 1e-3, 1e-2, 1e-1]
    results = []

    print("Alpha Sensitivity Study")

    for a in alphas:
        _, val, _, _ = gradient_descent(
            cfg,
            alpha=a,
            max_nfe=cfg.NFE
        )
        results.append((a, val))
        print("alpha =", a, "final =", val)

    return results

def plot_alpha_sweep(results):
    alphas = [r[0] for r in results]
    values = [r[1] for r in results]

    plt.figure()
    plt.plot(alphas, values, marker='o')
    plt.xscale("log")
    plt.title("GD Step Size Sensitivity")
    plt.xlabel("Alpha")
    plt.ylabel("Final Objective Value")
    plt.grid(True)
    plt.show()


def ga_vs_gd_comparison(cfg):
    print("\n====== GA vs GD COMPARISON ======")
    # --- GA ---
    _, _,best_obj_hist, _ = run_ga(cfg)
    ga_best = best_obj_hist[-1]
    # --- GD ---
    _, gd_best, _, _ = gradient_descent( cfg, alpha=cfg.GD_ALPHA, max_nfe=cfg.NFE )
    print(f"Objective: {cfg.OBJECTIVE_NAME}")
    print(f"GA Final Best: {ga_best:.6f}")
    print(f"GD Final Best: {gd_best:.6f}")



def ga_vs_gd_statistics(cfg):

    ga_results = []
    gd_results = []

    for _ in range(cfg.RUNS):
        _, ga_val, _, _ = run_ga(cfg)
        _, gd_val, _, _ = gradient_descent(
            cfg,
            alpha=cfg.GD_ALPHA,
            max_nfe=cfg.NFE
        )
        ga_results.append(ga_val)
        gd_results.append(gd_val)

    print(f"HERE {gd_results}")
    print("\n===== GA vs GD (50-run stats) =====")
    print("Algorithm   Mean        Std")
    print("--------------------------------")
    print(f"GA          {np.mean(ga_results):.6f}   {np.std(ga_results):.6f}")
    print(f"GD          {np.mean(gd_results):.6f}   {np.std(gd_results):.6f}")


