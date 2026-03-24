# gradient_descent.py
import numpy as np
from benchmarks import get_objective

# =========================================================
# GRADIENTS FOR OBJECTIVES
# =========================================================

def grad_sphere(x):
    """Gradient of shifted sphere: f(x) = sum((x-1)^2)"""
    return 2 * (x - 1.0)

def grad_rosenbrock(x):
    """Gradient of Rosenbrock function."""
    n = x.shape[0]
    g = np.zeros_like(x)
    g[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    for j in range(1, n - 1):
        g[j] = (200 * (x[j] - x[j-1]**2) -
                400 * x[j] * (x[j+1] - x[j]**2) -
                2 * (1 - x[j]))
    g[n - 1] = 200 * (x[n-1] - x[n-2]**2)
    return g

def grad_rastrigin(x, A=10):
    """Gradient of Rastrigin function."""
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

def grad_ackley(x, a=20, b=0.2, c=2*np.pi):
    """Gradient of Ackley function (standard version, no shift)."""
    n = x.shape[0]
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    sqrt_term = np.sqrt(sum_sq / n)
    exp1 = np.exp(-b * sqrt_term)
    exp2 = np.exp(sum_cos / n)

    grad = np.zeros_like(x)
    for i in range(n):
        if sqrt_term != 0:
            term1 = (a * b / (n * sqrt_term)) * x[i] * exp1
        else:
            term1 = 0.0
        term2 = (c / n) * np.sin(c * x[i]) * exp2
        grad[i] = term1 + term2
    return grad


# =========================================================
# GRADIENT DESCENT ALGORITHM
# =========================================================

def gradient_descent(cfg, alpha=0.001, max_nfe=None, tol=1e-8, portfolio_optimizer=None):
    """
    Gradient descent with support for portfolio optimization.
    """
    dim = cfg.DIMENSION
    low, high = cfg.BOUNDS

    # Get objective function
    if cfg.OBJECTIVE_NAME == "portfolio" and portfolio_optimizer is not None:
        # Portfolio: use the passed optimizer
        f = lambda x: portfolio_optimizer.variance_objective(x.reshape(1, -1))[0]
        if hasattr(cfg, 'PORTFOLIO_OBJECTIVE') and cfg.PORTFOLIO_OBJECTIVE == 'sharpe':
            grad_f = lambda x: portfolio_optimizer.sharpe_gradient(x)
        else:
            grad_f = lambda x: portfolio_optimizer.variance_gradient(x)
    else:
        # Standard benchmarks
        f = get_objective(cfg.OBJECTIVE_NAME)
        grad_map = {
            "sphere": grad_sphere,
            "rosenbrock": grad_rosenbrock,
            "rastrigin": grad_rastrigin,
            "ackley": grad_ackley,
        }
        grad_f = grad_map.get(cfg.OBJECTIVE_NAME)
        if grad_f is None:
            raise ValueError(f"No gradient available for {cfg.OBJECTIVE_NAME}")

    # Initial point
    x = np.random.uniform(low, high, dim)
    nfe = 0
    history = []

    while True:
        if cfg.OBJECTIVE_NAME == "portfolio" and portfolio_optimizer is not None:
            fx = f(x)
        else:
            fx = f(x.reshape(1, -1))[0]

        history.append(fx)
        nfe += 1

        if max_nfe and nfe >= max_nfe:
            break

        g = grad_f(x)
        grad_norm = np.linalg.norm(g)
        
        if grad_norm < tol:
            break

        # Adaptive step size: reduce alpha if step too large
        step = alpha * g
        if np.linalg.norm(step) > 1.0:
            alpha = alpha * 0.5
        
        x = x - alpha * g

    return x, fx, history, nfe