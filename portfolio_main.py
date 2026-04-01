# portfolio_main.py
import numpy as np
import matplotlib.pyplot as plt
from portfolio_optimization import RealPortfolioOptimizer
from gradient_descent import gradient_descent
from core import run_ga
from config import GAConfig
import time


def tune_step_size_separate(portfolio, objective_type='variance', alphas=None, nfe=5000, trials=5):
    """
    Tune GD step size for a specific portfolio objective.
    """
    if alphas is None:
        alphas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

    print(f"\n=== Tuning GD Step Size for {objective_type.upper()} ===")

    cfg = GAConfig(
        DIMENSION=len(portfolio.tickers),
        BOUNDS=(-2, 2),
        OBJECTIVE_NAME="portfolio",
        NFE=nfe
    )
    cfg.PORTFOLIO_OBJECTIVE = objective_type

    best_alpha = None
    best_score = np.inf

    for a in alphas:
        vals = []
        for _ in range(trials):
            _, fval, _, _ = gradient_descent(
                cfg,
                alpha=a,
                max_nfe=nfe,
                portfolio_optimizer=portfolio
            )
            vals.append(fval)

        mean_val = np.mean(vals)
        print(f"α = {a:.6f} -> mean final objective = {mean_val:.6f}")

        if mean_val < best_score:
            best_score = mean_val
            best_alpha = a

    print(f"Best α for {objective_type} = {best_alpha:.6f}")
    return best_alpha

def run_portfolio_comparison(tickers=None):
    """
    Run portfolio optimization comparison with proper reporting.
    """
    print("\n" + "=" * 70)
    print("PORTFOLIO OPTIMIZATION WITH REAL MARKET DATA")
    print("=" * 70)

    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    portfolio = RealPortfolioOptimizer(tickers, '2020-01-01', '2024-12-31')

    # Use actual number of valid assets (after filtering)
    actual_dimension = len(portfolio.tickers)

    cfg = GAConfig(
        DIMENSION=actual_dimension,
        BOUNDS=(-2, 2),
        POP_SIZE=100,
        GENERATIONS=100,
        OBJECTIVE_NAME="portfolio",
        RUNS=30,
        NFE=10000
    )

    objectives = [
        ("Minimum Variance", "variance"),
        ("Maximum Sharpe Ratio", "sharpe")
    ]

    results = {}

    for obj_name, obj_type in objectives:
        print(f"\n{'=' * 50}")
        print(f"OBJECTIVE: {obj_name}")
        print('=' * 50)

        cfg.PORTFOLIO_OBJECTIVE = obj_type
        gd_alpha = tune_step_size_separate(portfolio, objective_type=obj_type, nfe=3000, trials=3)

        if obj_type == 'variance':
            portfolio_obj = portfolio.variance_objective
        else:
            portfolio_obj = portfolio.sharpe_objective

        ga_objectives, ga_times = [], []
        ga_returns, ga_vols, ga_sharpes = [], [], []
        ga_best_x = None
        ga_best_obj = np.inf
        ga_best_history = None

        print(f"Running GA ({cfg.RUNS} runs)...")
        for i in range(cfg.RUNS):
            start = time.time()
            best_x, _, best_obj_history, _ = run_ga(cfg, custom_objective=portfolio_obj)
            elapsed = time.time() - start

            final_obj = float(best_obj_history[-1])
            metrics = portfolio.portfolio_metrics_from_raw(best_x)

            ga_times.append(elapsed)
            ga_objectives.append(final_obj)
            ga_returns.append(metrics["return"])
            ga_vols.append(metrics["volatility"])
            ga_sharpes.append(metrics["sharpe"])

            if final_obj < ga_best_obj:
                ga_best_obj = final_obj
                ga_best_x = best_x.copy()
                ga_best_history = np.array(best_obj_history, copy=True)

            if (i + 1) % 10 == 0:
                print(f"  GA run {i + 1}/{cfg.RUNS} complete")

        gd_objectives, gd_times = [], []
        gd_returns, gd_vols, gd_sharpes = [], [], []
        gd_best_x = None
        gd_best_obj = np.inf
        gd_best_history = None

        print(f"Running GD ({cfg.RUNS} runs, alpha={gd_alpha:.6f})...")
        for i in range(cfg.RUNS):
            start = time.time()
            x, fval, gd_history, _ = gradient_descent(
                cfg,
                alpha=gd_alpha,
                max_nfe=cfg.NFE,
                portfolio_optimizer=portfolio
            )
            elapsed = time.time() - start

            metrics = portfolio.portfolio_metrics_from_raw(x)

            gd_times.append(elapsed)
            gd_objectives.append(float(fval))
            gd_returns.append(metrics["return"])
            gd_vols.append(metrics["volatility"])
            gd_sharpes.append(metrics["sharpe"])

            if fval < gd_best_obj:
                gd_best_obj = float(fval)
                gd_best_x = x.copy()
                gd_best_history = np.array(gd_history, copy=True)

            if (i + 1) % 10 == 0:
                print(f"  GD run {i + 1}/{cfg.RUNS} complete")

        results[obj_name] = {
            "objective_type": obj_type,
            "gd_alpha": gd_alpha,
            "GA": {
                "objective": np.array(ga_objectives),
                "time": np.array(ga_times),
                "return": np.array(ga_returns),
                "volatility": np.array(ga_vols),
                "sharpe": np.array(ga_sharpes),
                "best_x": ga_best_x,
                "best_history": ga_best_history,
            },
            "GD": {
                "objective": np.array(gd_objectives),
                "time": np.array(gd_times),
                "return": np.array(gd_returns),
                "volatility": np.array(gd_vols),
                "sharpe": np.array(gd_sharpes),
                "best_x": gd_best_x,
                "best_history": gd_best_history,
            }
        }

        print(f"\n{obj_name} Results ({cfg.RUNS} runs):")
        print("-" * 105)
        print(f"{'Algorithm':<10} {'Obj Mean':<12} {'Obj Std':<12} {'Best Obj':<12} "
              f"{'Ret Mean':<12} {'Vol Mean':<12} {'Sharpe Mean':<14} {'Time(s)':<10}")
        print("-" * 105)

        for alg in ["GA", "GD"]:
            r = results[obj_name][alg]
            print(f"{alg:<10} "
                  f"{np.mean(r['objective']):<12.6f} "
                  f"{np.std(r['objective']):<12.6f} "
                  f"{np.min(r['objective']):<12.6f} "
                  f"{np.mean(r['return']):<12.6f} "
                  f"{np.mean(r['volatility']):<12.6f} "
                  f"{np.mean(r['sharpe']):<14.6f} "
                  f"{np.mean(r['time']):<10.3f}")

    return results, portfolio

def plot_large_portfolio_summary(results, portfolio,
                                 objective_name="Minimum Variance",
                                 algorithm="GD",
                                 top_k=15):
    """
    Single meaningful plot for large portfolios:
    shows the top-k asset weights and aggregates the rest into 'Others'.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    best_x = results[objective_name][algorithm]["best_x"]
    weights = portfolio.softmax_weights(best_x)

    # sort descending
    sorted_idx = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_idx]
    sorted_tickers = [portfolio.tickers[i] for i in sorted_idx]

    # keep only top_k, group rest as "Others"
    top_weights = sorted_weights[:top_k]
    top_tickers = sorted_tickers[:top_k]
    others_weight = np.sum(sorted_weights[top_k:])

    if others_weight > 0:
        plot_weights = np.append(top_weights, others_weight)
        plot_labels = top_tickers + ["Others"]
    else:
        plot_weights = top_weights
        plot_labels = top_tickers

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(plot_labels)), plot_weights, alpha=0.8)
    plt.xticks(range(len(plot_labels)), plot_labels, rotation=45, ha="right")
    plt.ylabel("Portfolio Weight")
    plt.xlabel("Assets")
    plt.title(f"{objective_name} Portfolio Weights ({algorithm}, top {top_k} + Others)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

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

    # Plot 2: Returns distribution
    ax2 = axes[0, 1]
    for col in portfolio.returns.columns:
        ax2.hist(portfolio.returns[col], bins=50, alpha=0.5, label=col)

    ax2.set_title("Daily Returns Distribution")
    ax2.set_xlabel("Daily Return")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

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

    # Plot 4: Maximum Sharpe Ratio boxplot
    ax4 = axes[1, 0]
    sharpe_data = [
        -results["Maximum Sharpe Ratio"]["GA"]["objective"],
        -results["Maximum Sharpe Ratio"]["GD"]["objective"]
    ]
    ax4.boxplot(sharpe_data, labels=["GA", "GD"])
    ax4.set_title("Maximum Sharpe Ratio")
    ax4.set_ylabel("Sharpe Ratio")
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Convergence comparison using stored best histories
    ax5 = axes[1, 1]

    ga_history = results["Minimum Variance"]["GA"]["best_history"]
    gd_history = results["Minimum Variance"]["GD"]["best_history"]

    cfg = GAConfig(
        DIMENSION=len(portfolio.tickers),
        BOUNDS=(-2, 2),
        POP_SIZE=100,
        GENERATIONS=100,
        OBJECTIVE_NAME="portfolio"
    )
    if ga_history is None or gd_history is None:
        raise ValueError("Best histories were not stored in results.")

    ga_gens = np.arange(len(ga_history))
    gd_gens = np.arange(len(gd_history)) / cfg.POP_SIZE

    ax5.plot(ga_gens, ga_history, 'b-', linewidth=2, label='GA')
    ax5.plot(gd_gens, gd_history, 'r--', linewidth=2, label='GD')
    ax5.set_xlabel("Generation Equivalent")
    ax5.set_ylabel("Objective Value")
    ax5.set_title("Convergence Comparison\n(Best Minimum Variance Runs)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')

    # Plot 6: Optimal weights from stored best GD solution
    ax6 = axes[1, 2]

    best_x = results["Minimum Variance"]["GD"]["best_x"]
    weights = portfolio.softmax_weights(best_x)

    sorted_idx = np.argsort(weights)[::-1]
    sorted_tickers = [portfolio.tickers[i] for i in sorted_idx]
    sorted_weights = weights[sorted_idx]

    ax6.bar(range(len(sorted_tickers)), sorted_weights, color='green', alpha=0.7)
    ax6.set_xticks(range(len(sorted_tickers)))
    ax6.set_xticklabels(sorted_tickers, rotation=45)
    ax6.set_xlabel("Assets")
    ax6.set_ylabel("Weight")
    ax6.set_title("Optimal Portfolio Weights\n(Best GD Minimum Variance Run)")
    ax6.grid(True, alpha=0.3, axis='y')

    plt.show()

if __name__ == "__main__":
    results, portfolio = run_portfolio_comparison()
    plot_portfolio_results(results, portfolio)