import matplotlib.pyplot as plt
import numpy as np
from config import GAConfig

cfg = GAConfig()

# Setup plotting style

plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

# =========================================================
# ---------------- PLOTTING -------------------------------
# =========================================================
def plot_statistics(all_runs, mean, std, cfg):
    gens = np.arange(len(mean))
    final_vals = all_runs[:, -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---------- Convergence ----------
    for r in all_runs:
        ax1.plot(r, color="gray", alpha=0.05)

    ax1.plot(mean, linewidth=3, label="Mean best", color="C0")
    ax1.fill_between(gens, mean - std, mean + std, alpha=0.25, label="±1 std")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best fitness")
    ax1.set_yscale("log")

    ax1.set_title(
        f"GA Convergence\n"
        f"{cfg.OBJECTIVE_NAME} | {cfg.SELECTION_METHOD} | pop={cfg.POP_SIZE}"
    )

    ax1.legend()

    # ---------- Final distribution ----------
    ax2.boxplot(final_vals, widths=0.5)

    ax2.set_ylabel("Final best fitness")
    ax2.set_yscale("log")

    ax2.set_title(
        f"Final Distribution\n"
        f"mean={final_vals.mean():.3g}  std={final_vals.std():.3g}"
    )

    fig.suptitle(
        f"Genetic Algorithm Performance | {cfg.OBJECTIVE_NAME}",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def plot_history(best_history, avg_history):
    plt.figure(figsize=(8, 5))

    plt.plot(best_history, linewidth=2.8, label="Best", color="C0")
    plt.plot(avg_history, linewidth=2, linestyle="--", label="Average", color="C1")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    plt.title(
        f"Single Run Convergence | {cfg.OBJECTIVE_NAME} | {cfg.SELECTION_METHOD}"
    )

    plt.legend()
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def plot_fitness_comparison(ga_runs, rand_runs, cfg):
    gens = np.arange(cfg.GENERATIONS)

    ga_mean = np.mean(ga_runs, axis=0)
    ga_std = np.std(ga_runs, axis=0)

    rand_mean = np.mean(rand_runs, axis=0)
    rand_std = np.std(rand_runs, axis=0)

    plt.figure(figsize=(9, 6))

    plt.plot(ga_mean, linewidth=3, label="Genetic Algorithm", color="C0")
    plt.fill_between(gens, ga_mean - ga_std, ga_mean + ga_std, alpha=0.2)

    plt.plot(rand_mean, linewidth=3, linestyle="--", label="Random Search", color="C1")
    plt.fill_between(gens, rand_mean - rand_std, rand_mean + rand_std, alpha=0.2)

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.yscale("log")

    plt.title(f"GA vs Random Search | {cfg.OBJECTIVE_NAME}")
    plt.legend()
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()



def plot_objective_history(best_obj, avg_obj):
    plt.figure(figsize=(8, 5))

    plt.plot(best_obj, linewidth=2.8, label="Best", color="C0")
    plt.plot(avg_obj, linewidth=2, linestyle="--", label="Average", color="C1")

    plt.xlabel("Generation")
    plt.ylabel("Objective value")
    plt.yscale("log")

    plt.title(f"Objective Convergence | {cfg.OBJECTIVE_NAME}")

    plt.legend()
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def plot_objective_statistics(all_runs, cfg):
    mean = np.mean(all_runs, axis=0)
    std = np.std(all_runs, axis=0)
    gens = np.arange(len(mean))
    final_vals = all_runs[:, -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---------- Convergence ----------
    for r in all_runs:
        ax1.plot(r, color="gray", alpha=0.05)

    ax1.plot(mean, linewidth=3, label="Mean best", color="C0")
    ax1.fill_between(gens, mean - std, mean + std, alpha=0.25, label="±1 std")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Objective value")
    ax1.set_yscale("log")

    ax1.set_title(f"Objective Convergence | {cfg.OBJECTIVE_NAME}")
    ax1.legend()

    # ---------- Distribution ----------
    ax2.boxplot(final_vals, widths=0.5)

    ax2.set_ylabel("Final best objective")
    ax2.set_yscale("log")

    ax2.set_title(
        f"Final Distribution\n"
        f"mean={final_vals.mean():.3g}  std={final_vals.std():.3g}"
    )

    fig.suptitle(
        f"Genetic Algorithm Performance | {cfg.OBJECTIVE_NAME}",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
