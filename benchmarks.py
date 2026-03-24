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

    # Portfolio Optimization Section

    @staticmethod
    def portfolio(X):
        """
        Portfolio variance minimization.
        This is a placeholder - will be replaced by the real portfolio optimizer.
        X: raw variables (shape: pop_size x n_assets)
        """
        # Default: return a large value (will be overridden in experiments)
        return np.sum(X**2, axis=1) * 1000  # Large penalty

def get_objective(name):
    return getattr(Benchmarks, name)


OBJECTIVE = get_objective(cfg.OBJECTIVE_NAME)


def fitness_from_objective(obj):
    return 1.0 / (1.0 + obj)


def validate_portfolio_weights(weights, tolerance=1e-6):
    """
    Validate that weights sum to 1 and are non-negative.
    Returns True if valid, False otherwise.
    """
    if not isinstance(weights, np.ndarray) or weights.ndim != 1:
        return False
    if np.any(weights < -tolerance):
        return False
    return abs(np.sum(weights) - 1.0) < tolerance


# In benchmarks.py, add this class for portfolio optimization with real data

class PortfolioOptimizer:
    """
    Portfolio optimization with real market data.
    The covariance matrix is passed at initialization.
    """

    def __init__(self, Sigma, mean_returns=None):
        """
        Parameters:
        -----------
        Sigma : np.array
            Covariance matrix of asset returns
        mean_returns : np.array, optional
            Mean returns for each asset (for return constraints)
        """
        self.Sigma = Sigma
        self.mean_returns = mean_returns
        self.n_assets = len(Sigma)

    def variance_only(self, X):
        """
        Minimize portfolio variance (Markowitz).
        X: raw variables -> softmax -> weights
        """
        exp_X = np.exp(X)
        w = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        # Portfolio variance: w^T Sigma w
        return np.einsum('ij,jk,ik->i', w, self.Sigma, w)

    def variance_with_return(self, X, target_return=0.1):
        """
        Minimize variance subject to return >= target_return.
        Uses penalty method for constraint.
        """
        exp_X = np.exp(X)
        w = exp_X / np.sum(exp_X, axis=1, keepdims=True)

        variance = np.einsum('ij,jk,ik->i', w, self.Sigma, w)

        if self.mean_returns is not None:
            portfolio_return = np.sum(w * self.mean_returns, axis=1)
            # Penalty for not meeting target return
            penalty = 1000 * np.maximum(0, target_return - portfolio_return) ** 2
            return variance + penalty
        return variance

    def sharpe_ratio(self, X, risk_free_rate=0.02):
        """
        Maximize Sharpe ratio (negative for minimization).
        """
        exp_X = np.exp(X)
        w = exp_X / np.sum(exp_X, axis=1, keepdims=True)

        variance = np.einsum('ij,jk,ik->i', w, self.Sigma, w)
        std = np.sqrt(variance)

        if self.mean_returns is not None:
            portfolio_return = np.sum(w * self.mean_returns, axis=1)
            sharpe = (portfolio_return - risk_free_rate) / std
            return -sharpe  # Negative for minimization
        return variance  # Fall back to variance