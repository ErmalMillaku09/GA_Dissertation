from experiments import *
from config import GAConfig
from plots import *
import numpy as np

cfg = GAConfig()

if __name__ == "__main__":
    # ----- Gradient Descent single run -----
    x, fval, history, nfe = run_gd_single(cfg)
    plot_gd_convergence(history, title="GD Single Run Convergence")
    print("GD final value:", fval)
    print("GD evaluations:", nfe)

    # ----- GD statistics (50 runs) -----
    gd_finals = run_gd_statistics(cfg)
    print("\n====== GD 50-Run Statistics ======")
    print("Mean:", np.mean(gd_finals))
    print("Std:", np.std(gd_finals))

    # ----- Alpha sweep -----
    alpha_results = run_gd_alpha_sweep(cfg)
    plot_gd_alpha_sweep(alpha_results)

    # ----- GA vs GD single comparison -----
    ga_best, gd_best = run_ga_vs_gd_comparison(cfg)

    # ----- GA vs GD statistics (50 runs) -----
    ga_finals, gd_finals = run_ga_vs_gd_statistics(cfg)

    # Extract final values from GA runs
    ga_final_values = ga_finals[:, -1]  # last generation value

    plot_gd_vs_ga_comparison(ga_final_values, gd_finals, cfg)

    # Example: run GA on rastrigin (commented)
    # cfg = GAConfig(OBJECTIVE_NAME="rastrigin", GENERATIONS=100)
    # print(cfg)
    # best_fit, avg_fit, best_obj, avg_obj = run_ga(cfg, verbose=True)
    # plot_history(best_fit, avg_fit, cfg)
    # plot_objective_history(best_obj, avg_obj, cfg)

    # ===== REAL PORTFOLIO OPTIMIZATION =====
    print("\n" + "=" * 70)
    print("PORTFOLIO OPTIMIZATION WITH REAL DATA")
    print("=" * 70)
    print("This will download data from Yahoo Finance.")
    print("Make sure you have yfinance installed: pip install yfinance")

    try:
        import yfinance

        print("✓ yfinance is installed")

        # Run portfolio optimization
        from portfolio_main import run_portfolio_comparison, plot_portfolio_results

        results, portfolio = run_portfolio_comparison()
        plot_portfolio_results(results, portfolio)

    except ImportError:
        print("\n yfinance not installed. Run: pip install yfinance")
    except Exception as e:
        print(f"\n Error in portfolio optimization: {e}")
        import traceback
        traceback.print_exc()