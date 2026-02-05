import numpy as np
from config import GAConfig

cfg = GAConfig()
# =========================================================
# ---------------- OBJECTIVE FUNCTIONS --------------------
# =========================================================

class Benchmarks:
    """
    Benchmark suite for GA testing
    """

    @staticmethod
    def sphere(X):
        """Shifted Sphere | global minimum at 1"""
        return np.sum((X - 1.0) ** 2, axis=1)

    @staticmethod
    def rastrigin(X):
        """Rastrigin | many local minima"""
        A = 10
        n = X.shape[1]
        return A * n + np.sum(X ** 2 - A * np.cos(2 * np.pi * X), axis=1)

    @staticmethod
    def ackley(X):
        """Ackley function"""
        n = X.shape[1]
        s1 = np.sum(X ** 2, axis=1)
        s2 = np.sum(np.cos(2 * np.pi * X), axis=1)
        return (
                -20 * np.exp(-0.2 * np.sqrt(s1 / n))
                - np.exp(s2 / n)
                + 20 + np.e
        )

    @staticmethod
    def rosenbrock(X):
        """Rosenbrock valley"""
        return np.sum(
            100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2,
            axis=1
        )


def get_objective(name):
    return getattr(Benchmarks, name)


OBJECTIVE = get_objective(cfg.OBJECTIVE_NAME)


def fitness_from_objective(obj):
    return 1.0 / (1.0 + obj)

