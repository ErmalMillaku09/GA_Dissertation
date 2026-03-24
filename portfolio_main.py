# portfolio_main.py
import numpy as np
import matplotlib.pyplot as plt
from portfolio_optimization import RealPortfolioOptimizer
from gradient_descent import gradient_descent
from core import run_ga
from config import GAConfig
import time


def tune_step_size(portfolio, alphas=None, nfe=5000):
    """
    Find best step size for GD on the variance objective.
    """
    if alphas is None:
        alphas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

    print("\n=== Tuning GD Step Size for Portfolio ===")
    cfg = GAConfig(
        DIMENSION=len(portfolio.tickers),
        BOUNDS=(-2, 2),
        OBJECTIVE_NAME="portfolio",
        NFE=nfe
    )
    best_alpha = None
    best_val = np.inf
    for a in alphas:
        _, fval, _, _ = gradient_descent(
            cfg, alpha=a, max_nfe=nfe, portfolio_optimizer=portfolio
        )
        print(f"α = {a:.6f} → final variance = {fval:.6f}")
        if fval < best_val:
            best_val = fval
            best_alpha = a
    print(f"Best α = {best_alpha:.6f} (variance = {best_val:.6f})")
    return best_alpha

def run_portfolio_comparison():
    """
    Run complete portfolio optimization comparison with custom objective functions.
    """
    print("\n" + "=" * 70)
    print("PORTFOLIO OPTIMIZATION WITH REAL MARKET DATA")
    print("=" * 70)

    # 1. Initialize portfolio with real data
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    portfolio = RealPortfolioOptimizer(tickers, '2020-01-01', '2024-12-31')
    best_alpha = tune_step_size(portfolio, nfe=5000)

    # 2. Configuration
    cfg = GAConfig(
        DIMENSION=len(tickers),
        BOUNDS=(-2, 2),
        POP_SIZE=100,
        GENERATIONS=100,  # 100*100 = 10,000 evaluations
        OBJECTIVE_NAME="portfolio",
        GD_ALPHA=0.01,
        RUNS=30,
        NFE=10000
    )

    # Add portfolio objective flag
    cfg.PORTFOLIO_OBJECTIVE = 'variance'  # or 'sharpe'

    # 3. Run comparison
    objectives = [
        ("Minimum Variance", 'variance'),
        ("Maximum Sharpe Ratio", 'sharpe')
    ]

    results = {}

    for obj_name, obj_type in objectives:
        print(f"\n{'=' * 50}")
        print(f"OBJECTIVE: {obj_name}")
        print('=' * 50)

        cfg.PORTFOLIO_OBJECTIVE = obj_type

        # Select correct objective function
        if obj_type == 'variance':
            portfolio_obj = portfolio.variance_objective
        else:
            portfolio_obj = portfolio.sharpe_objective

        # Run GA with custom objective
        ga_finals = []
        ga_times = []
        print(f"Running GA ({cfg.RUNS} runs)...")
        for i in range(cfg.RUNS):
            start = time.time()
            _, _, best_obj, _ = run_ga(cfg, custom_objective=portfolio_obj)
            ga_times.append(time.time() - start)
            ga_finals.append(best_obj[-1])
            if (i + 1) % 10 == 0:
                print(f"  GA run {i + 1}/{cfg.RUNS} complete")

        # Run GD
        gd_finals = []
        gd_times = []
        print(f"Running GD ({cfg.RUNS} runs)...")
        for i in range(cfg.RUNS):
            start = time.time()
            _, fval, _, _ = gradient_descent(
                cfg,
                alpha=cfg.GD_ALPHA,
                max_nfe=cfg.NFE,
                portfolio_optimizer=portfolio
            )
            gd_times.append(time.time() - start)
            gd_finals.append(fval)
            if (i + 1) % 10 == 0:
                print(f"  GD run {i + 1}/{cfg.RUNS} complete")

        # Store results
        results[obj_name] = {
            'GA': {'values': np.array(ga_finals), 'times': np.array(ga_times)},
            'GD': {'values': np.array(gd_finals), 'times': np.array(gd_times)}
        }

        # Print statistics
        print(f"\n{obj_name} Results ({cfg.RUNS} runs):")
        print("-" * 60)
        print(f"{'Algorithm':<12} {'Mean':<15} {'Std':<15} {'Best':<15} {'Time(s)':<10}")
        print("-" * 60)
        print(f"{'GA':<12} {np.mean(ga_finals):<15.6f} {np.std(ga_finals):<15.6f} "
              f"{np.min(ga_finals):<15.6f} {np.mean(ga_times):<10.3f}")
        print(f"{'GD':<12} {np.mean(gd_finals):<15.6f} {np.std(gd_finals):<15.6f} "
              f"{np.min(gd_finals):<15.6f} {np.mean(gd_times):<10.3f}")

    return results, portfolio


def plot_portfolio_results(results, portfolio):
    """
    Create publication-ready plots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # Plot 1: Price history
    ax1 = axes[0, 0]
    prices_normalized = portfolio.data.div(portfolio.data.iloc[0]) * 100
    prices_normalized.plot(ax=ax1)
    ax1.set_title("Stock Price History (Normalized to 100)")
    ax1.set_ylabel("Normalized Price")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=8)

    # Plot 2: Returns distribution
    ax2 = axes[0, 1]
    portfolio.returns.hist(ax=ax2, bins=50, alpha=0.7)
    ax2.set_title("Daily Returns Distribution")
    ax2.set_xlabel("Daily Return")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Correlation heatmap
    ax3 = axes[0, 2]
    corr_matrix = portfolio.returns.corr()
    im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(portfolio.tickers)))
    ax3.set_yticks(range(len(portfolio.tickers)))
    ax3.set_xticklabels(portfolio.tickers, rotation=45)
    ax3.set_yticklabels(portfolio.tickers)
    ax3.set_title("Return Correlations")
    plt.colorbar(im, ax=ax3)

    # Plot 4: GA vs GD boxplot
    ax4 = axes[1, 0]
    data_to_plot = []
    labels = []
    colors = []

    for obj_name in results:
        data_to_plot.append(results[obj_name]['GA']['values'])
        data_to_plot.append(results[obj_name]['GD']['values'])
        labels.append(f"GA\n{obj_name[:12]}")
        labels.append(f"GD\n{obj_name[:12]}")
        colors.extend(['lightblue', 'lightgreen'])

    bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
    ax4.set_ylabel("Objective Value")
    ax4.set_title("GA vs GD: 30-Run Comparison")
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Plot 5: Convergence comparison
    ax5 = axes[1, 1]

    # Get one representative run
    cfg = GAConfig(DIMENSION=len(portfolio.tickers), BOUNDS=(-2, 2),
                   POP_SIZE=100, GENERATIONS=100, OBJECTIVE_NAME="portfolio")

    # Run GA
    import benchmarks
    benchmarks.OBJECTIVE = portfolio.variance_objective
    _, _, ga_history, _ = run_ga(cfg)

    # Run GD
    _, _, gd_history, _ = gradient_descent(cfg, alpha=0.01, max_nfe=10000,
                                           portfolio_optimizer=portfolio)

    ga_gens = np.arange(len(ga_history))
    gd_gens = np.arange(len(gd_history)) / cfg.POP_SIZE

    ax5.plot(ga_gens, ga_history, 'b-', linewidth=2, label='GA')
    ax5.plot(gd_gens, gd_history, 'r--', linewidth=2, label='GD')
    ax5.set_xlabel("Generation Equivalent")
    ax5.set_ylabel("Objective Value")
    ax5.set_title("Convergence Comparison")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')

    # Plot 6: Optimal weights
    ax6 = axes[1, 2]

    # Get optimal weights from GD
    x, fval, _, _ = gradient_descent(cfg, alpha=0.01, max_nfe=10000,
                                     portfolio_optimizer=portfolio)
    exp_x = np.exp(x)
    weights = exp_x / np.sum(exp_x)

    # Sort by weight
    sorted_idx = np.argsort(weights)[::-1]
    sorted_tickers = [portfolio.tickers[i] for i in sorted_idx]
    sorted_weights = weights[sorted_idx]

    bars = ax6.bar(range(len(sorted_tickers)), sorted_weights, color='green', alpha=0.7)
    ax6.set_xticks(range(len(sorted_tickers)))
    ax6.set_xticklabels(sorted_tickers, rotation=45)
    ax6.set_xlabel("Assets")
    ax6.set_ylabel("Weight")
    ax6.set_title("Optimal Portfolio Weights\n(Minimum Variance)")
    ax6.grid(True, alpha=0.3, axis='y')

    plt.show()


if __name__ == "__main__":
    results, portfolio = run_portfolio_comparison()
    plot_portfolio_results(results, portfolio)