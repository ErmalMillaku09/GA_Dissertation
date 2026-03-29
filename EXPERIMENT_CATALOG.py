"""
COMPREHENSIVE EXPERIMENT CATALOG
All experiment types you can run. Copy-paste into main.py as needed.
"""

from experiments import *
from config import GAConfig
from plots import *
from portfolio_optimization import RealPortfolioOptimizer
from gradient_descent import gradient_descent, multi_start_gradient_descent
from core import run_ga
import numpy as np


# =========================================================
# GRADIENT DESCENT EXPERIMENTS
# =========================================================

def exp_gd_single_run():
    """Single GD run on default objective with visualization."""
    cfg = GAConfig()
    # cfg.NFE = 10000
    # cfg.OBJECTIVE_NAME = "rastrigin"
    x, fval, history, nfe = run_gd_single(cfg)
    plot_gd_convergence(history, title=f"GD Single Run Convergence | {cfg.OBJECTIVE_NAME.capitalize()}")
    print(f"GD final value: {fval:.6f}")
    print(f"GD evaluations: {nfe}")


def exp_gd_statistics():
    """GD statistics over 50 runs."""
    cfg = GAConfig()
    gd_finals, gd_histories = run_gd_statistics_with_histories(cfg)

    # Plot convergence and distribution (similar to GA)
    plot_gd_statistics(gd_histories, cfg)
    print("\n====== GD 50-Run Statistics ======")
    print(f"Mean: {np.mean(gd_finals):.6f}")
    print(f"Std: {np.std(gd_finals):.6f}")
    print(f"Min: {np.min(gd_finals):.6f}")
    print(f"Max: {np.max(gd_finals):.6f}")


def exp_gd_alpha_sensitivity():
    """GD step size (alpha) sensitivity analysis."""
    cfg = GAConfig()
    cfg.OBJECTIVE_NAME="ackley"
    alpha_results = run_gd_alpha_sweep(cfg, alphas=[1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0,2.0])
    plot_gd_alpha_sweep(alpha_results, cfg)
    print("\n====== Alpha Sensitivity Results ======")
    for alpha, final_val in alpha_results:
        print(f"Alpha {alpha:.4f}: {final_val:.6f}")


def exp_gd_on_all_benchmarks():
    """GD performance on all benchmark functions."""
    benchmarks = ["sphere", "rastrigin", "ackley", "rosenbrock"]
    cfg = GAConfig(RUNS=30, NFE=5000)
    
    print("\n====== GD Performance Across Benchmarks ======")
    print(f"{'Benchmark':<15} {'Mean':<15} {'Std':<15} {'Best':<15}")
    print("-" * 60)
    
    for benchmark in benchmarks:
        cfg.OBJECTIVE_NAME = benchmark
        results = run_gd_statistics(cfg)
        print(f"{benchmark:<15} {np.mean(results):<15.6f} {np.std(results):<15.6f} {np.min(results):<15.6f}")


def exp_gd_convergence_comparison():
    """Compare GD convergence across multiple alpha values."""
    cfg = GAConfig(DIMENSION=5, BOUNDS=(-5, 5), OBJECTIVE_NAME="ackley", NFE=1000)
    
    alphas = [0.001, 0.01, 0.1]
    histories = []
    
    print("\n====== GD Convergence Comparison ======")
    for alpha in alphas:
        _, fval, history, _ = gradient_descent(cfg, alpha=alpha, max_nfe=1000)
        histories.append(history)
        print(f"Alpha {alpha}: Final value = {fval:.6f}, Evaluations = {len(history)}")
    
    # Plot all convergence curves
    plt.figure(figsize=(10, 6))
    for i, (alpha, history) in enumerate(zip(alphas, histories)):
        plt.plot(history, label=f"α={alpha}", linewidth=2)
    plt.xlabel("Function Evaluations")
    plt.ylabel("Objective Value")
    plt.yscale("log")
    plt.title("GD Convergence: Alpha Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================================================
# TUNED GRADIENT DESCENT EXPERIMENTS
# =========================================================

def exp_tuned_gd_single_run():
    """Single tuned GD run with visualization."""
    cfg = GAConfig()
    x, fval, nfe = run_tuned_gd_single(cfg, n_starts=10)
    print(f"Tuned GD final value: {fval:.6f}")
    print(f"Tuned GD evaluations: {nfe}")


def exp_tuned_gd_statistics():
    """Tuned GD statistics over multiple runs."""
    cfg = GAConfig(RUNS=30)
    tuned_gd_finals = run_tuned_gd_statistics(cfg, n_starts=10)
    print("\n====== Tuned GD 30-Run Statistics ======")
    print(f"Mean: {np.mean(tuned_gd_finals):.6f}")
    print(f"Std: {np.std(tuned_gd_finals):.6f}")
    print(f"Min: {np.min(tuned_gd_finals):.6f}")
    print(f"Max: {np.max(tuned_gd_finals):.6f}")


def exp_ga_vs_tuned_gd():
    """Compare GA vs Tuned GD over multiple runs."""
    cfg = GAConfig(OBJECTIVE_NAME="rosenbrock", RUNS=50)
    run_ga_vs_tuned_gd_statistics(cfg, n_starts=10)


def exp_tuned_gd_on_all_benchmarks():
    """Tuned GD performance on all benchmark functions."""
    benchmarks = ["sphere", "rastrigin", "ackley", "rosenbrock"]
    cfg = GAConfig(RUNS=50, NFE=5000)
    
    print("\n====== Tuned GD Performance Across Benchmarks ======")
    print(f"{'Benchmark':<15} {'Mean':<15} {'Std':<15} {'Best':<15}")
    print("-" * 60)
    
    for benchmark in benchmarks:
        cfg.OBJECTIVE_NAME = benchmark
        results = run_tuned_gd_statistics(cfg, n_starts=10)
        print(f"{benchmark:<15} {np.mean(results):<15.6f} {np.std(results):<15.6f} {np.min(results):<15.6f}")


def exp_tuned_gd_vs_basic_gd():
    """Compare tuned GD vs basic GD."""
    cfg = GAConfig(RUNS=20, OBJECTIVE_NAME="rosenbrock")
    
    print("\n====== Tuned GD vs Basic GD ======")
    
    # Basic GD
    basic_results = run_gd_statistics(cfg)
    
    # Tuned GD
    tuned_results = run_tuned_gd_statistics(cfg, n_starts=10)
    
    print(f"Basic GD  - Mean: {np.mean(basic_results):.6f}, Std: {np.std(basic_results):.6f}")
    print(f"Tuned GD  - Mean: {np.mean(tuned_results):.6f}, Std: {np.std(tuned_results):.6f}")
    print(f"Improvement: {((np.mean(basic_results) - np.mean(tuned_results)) / np.mean(basic_results) * 100):.1f}%")


# =========================================================
# GENETIC ALGORITHM EXPERIMENTS
# =========================================================

def exp_ga_single_run():
    """Single GA run with convergence plot."""
    cfg = GAConfig(GENERATIONS=100)
    best_fit, avg_fit, best_obj, avg_obj = run_ga(cfg, verbose=True)
    plot_objective_history(best_obj, avg_obj, cfg)
    print(f"\nGA final best objective: {best_obj[-1]:.6f}")


def exp_ga_statistics():
    """GA statistics over multiple runs."""
    cfg = GAConfig(RUNS=30)
    all_fit, all_obj = run_experiment(cfg, runs=30)
    print("\n====== GA 30-Run Statistics ======")
    print(f"Mean final objective: {np.mean(all_obj[:, -1]):.6f}")
    print(f"Std: {np.std(all_obj[:, -1]):.6f}")
    print(f"Best: {np.min(all_obj[:, -1]):.6f}")

    # Plot performance for objective and fitness
    plot_objective_statistics(all_obj, cfg)
    plot_statistics(all_fit, np.mean(all_fit, axis=0), np.std(all_fit, axis=0), cfg)


def exp_ga_vs_random_search():
    """Compare GA against random search."""
    cfg = GAConfig(GENERATIONS=100, RUNS=30)
    ga_runs, rand_runs = compare_ga_vs_random(cfg, runs=30)
    
    plot_fitness_comparison(ga_runs, rand_runs, cfg)
    print("\n====== GA vs Random Search ======")
    print(f"GA best (final): {np.mean(ga_runs[:, -1]):.6f}")
    print(f"Random best (final): {np.mean(rand_runs[:, -1]):.6f}")
    print(f"GA advantage: {(np.mean(rand_runs[:, -1]) / np.mean(ga_runs[:, -1])):.2f}x better")


def exp_ga_on_all_benchmarks():
    """GA performance on all benchmark functions."""
    benchmarks = ["sphere", "rastrigin", "ackley", "rosenbrock"]
    cfg = GAConfig(GENERATIONS=200, RUNS=20)
    
    results = compare_objectives(benchmarks, base_cfg=cfg, runs=20)
    print("\n====== GA Performance Across Benchmarks ======")
    print("See plot for convergence comparison")

    # Visualize mean convergence for all objectives
    plot_objective_comparison(results, cfg)

def exp_ga_parameter_sweep():
    """Sweep GA Parms.
            Example:
        run_parameter_sweep("SELECTION_METHOD", ["roulette","tournament","ranking"])
        run_parameter_sweep("MUTATION_RATE", [0.05, 0.1, 0.2])
        run_parameter_sweep("POP_SIZE", [30, 50, 100])"""
    cfg = GAConfig(GENERATIONS=100, OBJECTIVE_NAME="ackley")
    
    run_parameter_sweep("SELECTION_METHOD", ["roulette","tournament","ranking"], cfg,20)
    print("\n====== Mutation Rate Sweep Complete ======")
    print("Results stored in 'results' dict")


def exp_ga_selection_comparison():
    """Compare different selection methods."""
    cfg = GAConfig(GENERATIONS=100, RUNS=20)
    selection_methods = ["roulette", "tournament", "ranking"]
    
    print("\n====== Selection Method Comparison ======")
    print(f"{'Method':<15} {'Mean':<15} {'Std':<15}")
    print("-" * 45)
    
    for method in selection_methods:
        cfg.SELECTION_METHOD = method
        results = run_experiment(cfg, runs=20)
        final_values = results[1][:, -1]  # best_obj
        print(f"{method:<15} {np.mean(final_values):<15.6f} {np.std(final_values):<15.6f}")


# =========================================================
# ALGORITHM COMPARISON EXPERIMENTS
# =========================================================

def exp_ga_vs_gd_single():
    """Single GA vs GD comparison."""
    cfg = GAConfig(DIMENSION=5, GENERATIONS=100, NFE=5000)
    ga_best, gd_best = run_ga_vs_gd_comparison(cfg)
    print(f"\nGA best: {ga_best:.6f}")
    print(f"GD best: {gd_best:.6f}")
    print(f"Winner: {'GA' if ga_best < gd_best else 'GD'} by {abs(ga_best - gd_best):.6f}")


def exp_ga_vs_gd_statistics():
    """GA vs GD over 50 runs each."""
    cfg = GAConfig(OBJECTIVE_NAME="rastrigin", RUNS=50, NFE=10000)
    ga_finals, gd_finals = run_ga_vs_gd_statistics(cfg)

    # ga_finals and gd_finals are now 1D arrays of final values (scalars)
    # No need to extract [:, -1] anymore

    plot_gd_vs_ga_comparison(ga_finals, gd_finals, cfg)
    print("\n====== GA vs GD (50 runs) ======")
    print(f"GA  - Mean: {np.mean(ga_finals):.6f}, Best: {np.min(ga_finals):.6f}")
    print(f"GD  - Mean: {np.mean(gd_finals):.6f}, Best: {np.min(gd_finals):.6f}")


def exp_algorithm_on_difficult_landscape():
    """Test basic GA vs basic GD on Rastrigin (difficult multi-modal landscape).
    
    This demonstrates why population-based methods (GA) outperform 
    local search methods (basic GD) on multi-modal optimization problems.
    Basic GD gets trapped in local minima, while GA explores multiple solutions.
    """
    cfg = GAConfig(
        OBJECTIVE_NAME="rastrigin",
        DIMENSION=10,
        GENERATIONS=200,
        POP_SIZE=100,
        RUNS=20,
        NFE=20000
    )
    
    print("\n====== Rastrigin (Difficult Landscape) ======")
    print("Comparing basic GA vs basic GD (single random start)")
    print("GA uses population-based search, GD uses local gradient descent")
    
    # GA
    ga_results = run_experiment(cfg, runs=50)
    ga_best = np.mean(ga_results[1][:, -1])
    
    # GD (basic, single start)
    gd_results = run_gd_statistics(cfg)
    gd_best = np.mean(gd_results)
    
    print(f"GA final value: {ga_best:.6f}")
    print(f"GD final value: {gd_best:.6f}")
    print(f"GA outperforms GD by {abs(ga_best - gd_best):.6f}")
    print("\nNote: GD gets stuck in local minima (~90) because Rastrigin has many.")
    print("GA explores multiple solutions simultaneously, finding better regions.")


# =========================================================
# PARAMETER SWEEP EXPERIMENTS
# =========================================================

def exp_population_size_sweep():
    """Sweep population size impact."""
    cfg = GAConfig(GENERATIONS=100)
    pop_sizes = [10, 20, 50, 100, 200]
    
    results = run_parameter_sweep("POP_SIZE", pop_sizes, base_cfg=cfg, runs=15)
    print("\n====== Population Size Sweep Complete ======")


def exp_crossover_rate_sweep():
    """Sweep crossover rate impact."""
    cfg = GAConfig(GENERATIONS=100)
    crossover_rates = [0.5, 0.7, 0.9, 0.95]
    
    results = run_parameter_sweep("CROSSOVER_RATE", crossover_rates, base_cfg=cfg, runs=15)
    print("\n====== Crossover Rate Sweep Complete ======")


def exp_tournament_size_sweep():
    """Sweep tournament selection size."""
    cfg = GAConfig(GENERATIONS=100, SELECTION_METHOD="tournament")
    tournament_sizes = [2, 3, 5, 10]
    
    results = run_parameter_sweep("TOURNAMENT_K", tournament_sizes, base_cfg=cfg, runs=15)
    print("\n====== Tournament Size Sweep Complete ======")


def exp_elitism_comparison():
    """Compare with and without elitism."""
    cfg = GAConfig(GENERATIONS=100, RUNS=20)
    
    print("\n====== Elitism Comparison ======")
    
    # With elitism
    cfg.USE_ELITISM = True
    results_elite = run_experiment(cfg, runs=20)
    
    # Without elitism
    cfg.USE_ELITISM = False
    results_no_elite = run_experiment(cfg, runs=20)
    
    print(f"With elitism    - Mean: {np.mean(results_elite[1][:, -1]):.6f}")
    print(f"Without elitism - Mean: {np.mean(results_no_elite[1][:, -1]):.6f}")


# =========================================================
# MULTI-OBJECTIVE & FULL STUDY EXPERIMENTS
# =========================================================

def exp_full_ga_study():
    """Full factorial GA study across multiple parameters."""
    cfg = GAConfig()
    
    objectives = ["sphere", "rastrigin", "ackley"]
    selections = ["roulette", "tournament", "ranking"]
    mutations = [0.1, 0.15, 0.2]
    
    summary = full_ga_study(objectives, selections, mutations, base_cfg=cfg, runs=10)
    print("\n====== Full GA Study Complete ======")


def exp_scalability_dimension():
    """Test GA/GD scalability across dimensions."""
    dimensions = [2, 5, 10, 20]
    cfg = GAConfig(OBJECTIVE_NAME="sphere", RUNS=20)
    
    print("\n====== Scalability Test (Sphere) ======")
    print(f"{'Dimension':<12} {'GA Mean':<15} {'GD Mean':<15}")
    print("-" * 42)
    
    for dim in dimensions:
        cfg.DIMENSION = dim
        cfg.BOUNDS = (-5.0, 5.0)
        
        # GA
        ga_results = run_experiment(cfg, runs=20)
        ga_mean = np.mean(ga_results[1][:, -1])
        
        # GD
        gd_results = run_gd_statistics(cfg)
        gd_mean = np.mean(gd_results)
        
        print(f"{dim:<12} {ga_mean:<15.6f} {gd_mean:<15.6f}")


# =========================================================
# PORTFOLIO OPTIMIZATION EXPERIMENTS
# =========================================================

def exp_portfolio_gd_variance():
    """GD optimization of portfolio variance."""
    print("\n====== Portfolio GD: Minimum Variance ======")
    
    portfolio = RealPortfolioOptimizer(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        start_date='2020-01-01', end_date='2024-12-31'
    )
    
    cfg = GAConfig(
        DIMENSION=len(portfolio.tickers),
        BOUNDS=(-2, 2),
        OBJECTIVE_NAME="portfolio",
        GD_ALPHA=0.01,
        NFE=5000,
        RUNS=30
    )
    
    results = []
    for _ in range(30):
        _, fval, _, _ = gradient_descent(cfg, alpha=0.01, max_nfe=5000, portfolio_optimizer=portfolio)
        results.append(fval)
    
    results = np.array(results)
    print(f"Mean portfolio variance: {np.mean(results):.6f}")
    print(f"Std: {np.std(results):.6f}")
    print(f"Best: {np.min(results):.6f}")


def exp_portfolio_gd_sharpe():
    """GD optimization of Sharpe ratio."""
    print("\n====== Portfolio GD: Maximum Sharpe Ratio ======")
    
    portfolio = RealPortfolioOptimizer(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        start_date='2020-01-01', end_date='2024-12-31'
    )
    
    cfg = GAConfig(
        DIMENSION=len(portfolio.tickers),
        BOUNDS=(-2, 2),
        OBJECTIVE_NAME="portfolio",
        GD_ALPHA=0.01,
        NFE=5000,
        RUNS=30
    )
    cfg.PORTFOLIO_OBJECTIVE = 'sharpe'
    
    results = []
    for _ in range(30):
        _, fval, _, _ = gradient_descent(cfg, alpha=0.01, max_nfe=5000, portfolio_optimizer=portfolio)
        results.append(fval)
    
    results = np.array(results)
    print(f"Mean Sharpe ratio: {np.mean(results):.6f}")
    print(f"Std: {np.std(results):.6f}")
    print(f"Best: {np.max(results):.6f}")  # Max because negative (minimization)


def exp_portfolio_ga_variance():
    """GA optimization of portfolio variance."""
    print("\n====== Portfolio GA: Minimum Variance ======")
    
    portfolio = RealPortfolioOptimizer(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        start_date='2020-01-01', end_date='2024-12-31'
    )
    
    cfg = GAConfig(
        DIMENSION=len(portfolio.tickers),
        BOUNDS=(-2, 2),
        POP_SIZE=100,
        GENERATIONS=100,
        OBJECTIVE_NAME="portfolio",
        RUNS=20
    )
    
    results = []
    for r in range(20):
        np.random.seed(r)
        _, _, best_obj, _ = run_ga(cfg, custom_objective=portfolio.variance_objective)
        results.append(best_obj[-1])
    
    results = np.array(results)
    print(f"Mean portfolio variance: {np.mean(results):.6f}")
    print(f"Std: {np.std(results):.6f}")
    print(f"Best: {np.min(results):.6f}")


def exp_portfolio_ga_vs_gd():
    """Full GA vs GD comparison on portfolio optimization."""
    print("\n====== Portfolio: GA vs GD Full Comparison ======")
    
    portfolio = RealPortfolioOptimizer(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        start_date='2020-01-01', end_date='2024-12-31'
    )
    
    cfg = GAConfig(
        DIMENSION=len(portfolio.tickers),
        BOUNDS=(-2, 2),
        POP_SIZE=100,
        GENERATIONS=100,
        GD_ALPHA=0.01,
        NFE=10000,
        OBJECTIVE_NAME="portfolio",
        RUNS=20
    )
    
    objectives = [
        ("Minimum Variance", portfolio.variance_objective, False),
        ("Maximum Sharpe Ratio", portfolio.sharpe_objective, True)
    ]
    
    print(f"\n{'Objective':<25} {'GA Mean':<15} {'GD Mean':<15}")
    print("-" * 55)
    
    for obj_name, obj_func, use_sharpe in objectives:
        # Set portfolio objective for GD
        cfg.PORTFOLIO_OBJECTIVE = 'sharpe' if use_sharpe else 'variance'
        
        # GA
        ga_results = []
        for r in range(20):
            np.random.seed(r)
            _, _, best_obj, _ = run_ga(cfg, custom_objective=obj_func)
            ga_results.append(best_obj[-1])
        
        # GD
        gd_results = []
        for r in range(20):
            np.random.seed(r)
            _, fval, _, _ = gradient_descent(cfg, alpha=0.01, max_nfe=10000, portfolio_optimizer=portfolio)
            gd_results.append(fval)
        
        ga_mean = np.mean(ga_results)
        gd_mean = np.mean(gd_results)
        
        # For Sharpe ratio, convert back to positive (since we minimize -sharpe)
        if use_sharpe:
            ga_mean = -ga_mean
            gd_mean = gd_mean
        
        print(f"{obj_name:<25} {ga_mean:<15.6f} {gd_mean:<15.6f}")


def exp_portfolio_tuned_gd_variance():
    """Tuned GD optimization of portfolio variance."""
    print("\n====== Portfolio Tuned GD: Minimum Variance ======")
    
    portfolio = RealPortfolioOptimizer(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        start_date='2020-01-01', end_date='2024-12-31'
    )
    
    cfg = GAConfig(
        DIMENSION=len(portfolio.tickers),
        BOUNDS=(-2, 2),
        OBJECTIVE_NAME="portfolio",
        NFE=5000,
        RUNS=20
    )
    
    results = []
    for _ in range(20):
        _, fval, _ = multi_start_gradient_descent(cfg, n_starts=5, alpha_init=0.01, 
                                                max_nfe=5000, portfolio_optimizer=portfolio)
        results.append(fval)
    
    results = np.array(results)
    print(f"Mean portfolio variance: {np.mean(results):.6f}")
    print(f"Std: {np.std(results):.6f}")
    print(f"Best: {np.min(results):.6f}")


def exp_portfolio_tuned_gd_sharpe():
    """Tuned GD optimization of Sharpe ratio."""
    print("\n====== Portfolio Tuned GD: Maximum Sharpe Ratio ======")
    
    portfolio = RealPortfolioOptimizer(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        start_date='2020-01-01', end_date='2024-12-31'
    )
    
    cfg = GAConfig(
        DIMENSION=len(portfolio.tickers),
        BOUNDS=(-2, 2),
        OBJECTIVE_NAME="portfolio",
        PORTFOLIO_OBJECTIVE='sharpe',
        NFE=5000,
        RUNS=20
    )
    
    results = []
    for _ in range(20):
        _, fval, _ = multi_start_gradient_descent(cfg, n_starts=5, alpha_init=0.01, 
                                                max_nfe=5000, portfolio_optimizer=portfolio)
        results.append(fval)
    
    results = np.array(results)
    print(f"Mean Sharpe ratio: {-np.mean(results):.6f}")
    print(f"Std: {np.std(results):.6f}")
    print(f"Best: {-np.max(results):.6f}")  # Max because negative (minimization)


def exp_portfolio_ga_vs_tuned_gd():
    """Compare GA vs tuned GD on portfolio optimization."""
    print("\n====== Portfolio: GA vs Tuned GD Comparison ======")
    
    portfolio = RealPortfolioOptimizer(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        start_date='2020-01-01', end_date='2024-12-31'
    )
    
    cfg = GAConfig(
        DIMENSION=len(portfolio.tickers),
        BOUNDS=(-2, 2),
        POP_SIZE=50,
        GENERATIONS=50,
        OBJECTIVE_NAME="portfolio",
        NFE=5000,
        RUNS=15
    )
    
    objectives = [
        ("Minimum Variance", portfolio.variance_objective, False),
        ("Maximum Sharpe Ratio", portfolio.sharpe_objective, True)
    ]
    
    print(f"\n{'Objective':<25} {'GA Mean':<15} {'Tuned GD Mean':<15}")
    print("-" * 60)
    
    for obj_name, obj_func, use_sharpe in objectives:
        # Set portfolio objective for tuned GD
        cfg.PORTFOLIO_OBJECTIVE = 'sharpe' if use_sharpe else 'variance'
        
        # GA
        ga_results = []
        for r in range(15):
            np.random.seed(r)
            _, _, best_obj, _ = run_ga(cfg, custom_objective=obj_func)
            ga_results.append(best_obj[-1])
        
        # Tuned GD
        gd_results = []
        for r in range(15):
            np.random.seed(r)
            _, gd_val, _ = multi_start_gradient_descent(cfg, n_starts=5, alpha_init=0.01, 
                                                      max_nfe=5000, portfolio_optimizer=portfolio)
            gd_results.append(gd_val)
        
        ga_mean = np.mean(ga_results)
        gd_mean = np.mean(gd_results)
        
        print(f"{obj_name:<25} {ga_mean:<15.6f} {gd_mean:<15.6f}")


def exp_portfolio_different_sectors():
    """Portfolio optimization on diverse stocks (different sectors)."""
    print("\n====== Portfolio: Different Sectors ======")
    
    portfolio = RealPortfolioOptimizer(
        tickers=['JPM', 'XOM', 'PG', 'JNJ', 'DIS'],  # Finance, Energy, Consumer, Healthcare, Media
        start_date='2020-01-01', end_date='2024-12-31'
    )
    
    cfg = GAConfig(
        DIMENSION=len(portfolio.tickers),
        BOUNDS=(-2, 2),
        OBJECTIVE_NAME="portfolio",
        NFE=5000,
        RUNS=15
    )
    
    results = []
    for _ in range(15):
        _, fval, _, _ = gradient_descent(cfg, alpha=0.01, max_nfe=5000, portfolio_optimizer=portfolio)
        results.append(fval)
    
    results = np.array(results)
    print(f"Mean variance (diverse portfolio): {np.mean(results):.6f}")
    print(f"Compare with tech portfolio results from earlier")




# =========================================================
# ADVANCED EXPERIMENTS
# =========================================================

def exp_gradient_descent_with_decay():
    """GD with learning rate decay schedule."""
    cfg = GAConfig(DIMENSION=5, BOUNDS=(-5, 5), OBJECTIVE_NAME="ackley", NFE=1000)
    
    print("\n====== GD with Decay Schedule ======")
    
    # Constant alpha
    _, const_val, const_history, _ = gradient_descent(cfg, alpha=0.05, max_nfe=1000)
    
    # With manual decay simulation
    x = np.random.uniform(-5, 5, 5)
    nfe = 0
    decay_history = []
    
    from benchmarks import get_objective
    from gradient_descent import grad_ackley
    
    f = get_objective("ackley")
    
    while nfe < 1000:
        fx = f(x.reshape(1, -1))[0]
        decay_history.append(fx)
        
        alpha_t = 0.1 / (1.0 + 0.01 * nfe)
        g = grad_ackley(x)
        x = x - alpha_t * g
        nfe += 1
    
    print(f"Constant α=0.05: final value = {const_val:.6f}")
    print(f"Decay schedule: final value = {decay_history[-1]:.6f}")


def exp_noise_robustness():
    """Test algorithm robustness to noisy objectives."""
    cfg = GAConfig(DIMENSION=5, BOUNDS=(-5, 5), OBJECTIVE_NAME="sphere", RUNS=20)
    
    print("\n====== Noise Robustness ======")
    print("(Requires modification to add noise to objectives)")
    print("Consider adding: objective_noisy = objective + np.random.normal(0, noise_std)")


def exp_comparison_table():
    """Generate comprehensive comparison table."""
    benchmarks = ["sphere", "rastrigin", "ackley", "rosenbrock"]
    cfg = GAConfig(DIMENSION=5, GENERATIONS=100, RUNS=20, NFE=10000)
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("=" * 100)
    print(f"{'Objective':<15} {'GA Mean':<15} {'GA Std':<15} {'GA Best':<15} {'GD Mean':<15} {'GD Std':<15} {'GD Best':<15}")
    print("-" * 100)
    
    for benchmark in benchmarks:
        cfg.OBJECTIVE_NAME = benchmark
        
        # GA
        ga_results = run_experiment(cfg, runs=20)
        ga_final = ga_results[1][:, -1]
        
        # GD
        gd_results = run_gd_statistics(cfg)
        
        print(f"{benchmark:<15} {np.mean(ga_final):<15.6f} {np.std(ga_final):<15.6f} {np.min(ga_final):<15.6f} "
              f"{np.mean(gd_results):<15.6f} {np.std(gd_results):<15.6f} {np.min(gd_results):<15.6f}")


# =========================================================
# RUN EXPERIMENTS - COPY FUNCTION CALLS BELOW
# =========================================================

if __name__ == "__main__":
    pass
    # Uncomment and run any of the experiments below:
    # exp_tuned_gd_on_all_benchmarks()
    # --- Gradient Descent ---
    #exp_gd_single_run()
    # exp_gd_statistics()
    #exp_gd_alpha_sensitivity()
    # exp_gd_on_all_benchmarks()
    #exp_gd_convergence_comparison()
    
    # --- Genetic Algorithm ---
    # exp_ga_single_run()
    # exp_ga_statistics()
    # exp_ga_vs_random_search()
    # exp_ga_on_all_benchmarks()
    # exp_ga_parameter_sweep()
    # exp_ga_selection_comparison()
    
    # --- Algorithm Comparison ---
    # exp_ga_vs_gd_single()
    #exp_ga_vs_gd_statistics()
    #exp_algorithm_on_difficult_landscape()
    exp_ga_vs_tuned_gd()
    #exp_tuned_gd_vs_basic_gd()
    # exp_comparison_table()
#### *** HERE Continue TESTING OTHER EXPERIMENTS AS NEEDED *** ####


    # --- Parameter Sweeps ---
    # exp_population_size_sweep()
    # exp_crossover_rate_sweep()
    # exp_tournament_size_sweep()
    # exp_elitism_comparison()
    
    # --- Full Studies ---
    # exp_full_ga_study()
    # exp_scalability_dimension()
    # exp_comparison_table()
    
    # --- Portfolio Optimization ---
    # exp_portfolio_gd_variance()
    # exp_portfolio_gd_sharpe()
    # exp_portfolio_ga_variance()
    # exp_portfolio_ga_vs_gd()
    # exp_portfolio_different_sectors()
    
    # --- Advanced ---
    # exp_gradient_descent_with_decay()
    # exp_noise_robustness()
    # exp_portfolio_tuned_gd_variance()
    # exp_portfolio_tuned_gd_sharpe()
