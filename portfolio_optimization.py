# portfolio_optimization.py
import numpy as np
import yfinance as yf
import pandas as pd
from dataclasses import dataclass
from market_data import download_stock_data


@dataclass
class PortfolioData:
    """Container for portfolio data"""
    tickers: list
    prices: pd.DataFrame
    returns: pd.DataFrame
    Sigma: np.ndarray
    mean_returns: np.ndarray


class RealPortfolioOptimizer:
    """
    Portfolio optimization with real market data.
    This class provides objective functions that can be used by both GA and GD.
    """

    def __init__(self, tickers, start_date='2020-01-01', end_date='2024-12-31'):
        """
        Download data and initialize optimizer.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        # Download data
        self.tickers, self.Sigma, self.mean_returns, self.data = download_stock_data(self.tickers, self.start_date, self.end_date)

        # Compute returns
        self.returns = self.data.pct_change().dropna()

        self.n_assets = len(self.tickers)

        print(f"\nPortfolio Data Summary:")
        print(f"  Tickers: {tickers}")
        print(f"  Period: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
        print(f"  Trading days: {len(self.returns)}")
        print(f"  Annualized returns: {self.mean_returns}")
        print(f"n_assets: {self.n_assets}, mean_returns.shape: {self.mean_returns.shape}")


    # Objective functions for GA (expects 2D array pop_size x n_assets)

    def variance_objective(self, X):
        """
        Minimize portfolio variance.
        X: raw variables (pop_size x n_assets) -> softmax -> weights
        """
        X = np.asarray(X, dtype=float)
        X = X - np.max(X, axis=1, keepdims=True)  # numerical stability
        exp_X = np.exp(X)
        w = exp_X / np.sum(exp_X, axis=1, keepdims=True)

        # Portfolio variance: w^T Sigma w for each row
        # Using einsum for efficiency: (batch, i) @ (i, j) @ (batch, j) -> (batch,)
        return np.einsum('ij,jk,ik->i', w, self.Sigma, w)

    def sharpe_objective(self, X, risk_free_rate=0.02):
        """
        Maximize Sharpe ratio (negative for minimization).
        """
        X = np.asarray(X, dtype=float)
        X = X - np.max(X, axis=1, keepdims=True)  # numerical stability
        exp_X = np.exp(X)
        w = exp_X / np.sum(exp_X, axis=1, keepdims=True)

        variance = np.einsum('ij,jk,ik->i', w, self.Sigma, w)
        std = np.sqrt(np.maximum(variance, 0.0))
        portfolio_return = np.sum(w * self.mean_returns, axis=1)
        sharpe = (portfolio_return - risk_free_rate) / (std + 1e-8)  # Avoid division by zero

        return -sharpe  # Negative for minimization

    def softmax_weights(self, x):
        """Convert raw variables to valid portfolio weights."""
        x = np.asarray(x, dtype=float)
        x = x - np.max(x)  # numerical stability
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def portfolio_metrics_from_weights(self, w, risk_free_rate=0.02):
        """Return annualized portfolio metrics from weights."""
        w = np.asarray(w, dtype=float)
        portfolio_return = float(w @ self.mean_returns)
        variance = float(w @ self.Sigma @ w)
        volatility = float(np.sqrt(max(variance, 0.0)))
        sharpe = (portfolio_return - risk_free_rate) / (volatility + 1e-8)

        return {
            "return": portfolio_return,
            "variance": variance,
            "volatility": volatility,
            "sharpe": float(sharpe),
        }

    def portfolio_metrics_from_raw(self, x, risk_free_rate=0.02):
        """Return weights + metrics from raw optimization variables."""
        w = self.softmax_weights(x)
        metrics = self.portfolio_metrics_from_weights(w, risk_free_rate=risk_free_rate)
        metrics["weights"] = w
        return metrics

    # Gradient functions for GD (expects 1D array n_assets)

    def variance_gradient(self, x):
        """
        Gradient of portfolio variance w.r.t raw variables.
        x: 1D array of raw variables
        """
        x = np.asarray(x, dtype=float)
        x = x - np.max(x)
        exp_x = np.exp(x)
        sum_exp = np.sum(exp_x)
        w = exp_x / sum_exp

        # Gradient w.r.t weights: 2 * Sigma @ w
        grad_w = 2 * self.Sigma @ w

        # Jacobian of softmax: dw/dx = diag(w) - w w^T
        J = np.diag(w) - np.outer(w, w)

        return J @ grad_w

    def sharpe_gradient(self, x, risk_free_rate=0.02):
        """
        Gradient of negative Sharpe ratio w.r.t raw variables.
        """
        x = np.asarray(x, dtype=float)
        x = x - np.max(x)
        exp_x = np.exp(x)
        sum_exp = np.sum(exp_x)
        w = exp_x / sum_exp

        # Portfolio metrics
        portfolio_return = np.sum(w * self.mean_returns)
        variance = w @ self.Sigma @ w
        std = np.sqrt(variance)

        # Gradient of Sharpe ratio: ∇(R/sigma) = (∇R * sigma - R * ∇sigma) / sigma^2
        # Where ∇sigma = (1/sigma) * Sigma @ w

        grad_return = self.mean_returns  # dR/dw

        # Gradient of variance: 2 * Sigma @ w
        grad_variance = 2 * self.Sigma @ w
        grad_std = grad_variance / (2 * std + 1e-8)  # d(sigma)/dw

        # Jacobian of softmax
        J = np.diag(w) - np.outer(w, w)

        # Gradient w.r.t weights
        grad_w_sharpe = (grad_return * std - (portfolio_return - risk_free_rate) * grad_std) / (std ** 2 + 1e-8)
        # Convert to gradient w.r.t raw variables
        return J @ (-grad_w_sharpe)