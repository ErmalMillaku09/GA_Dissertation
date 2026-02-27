# gradient_descent.py
import numpy as np
import time
from benchmarks import get_objective

# =========================================================
# GRADIENTS FOR OBJECTIVES
# =========================================================

def grad_sphere(x):
    return 2 * (x - 1.0)

def grad_rosenbrock(x):
    n = x.shape[0]
    g = np.zeros_like(x)
    g[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    for j in range(1, n - 1):
        g[j] = (
            200*(x[j] - x[j - 1]**2)
            - 400 * x[j]*(x[j + 1] - x[j]**2)
            - 2*(1 - x[j])
        )
    g[n - 1] = 200 * (x[n - 1] - x[n - 2]**2)
    return g

def grad_rastrigin(x, A=10):
    return 2 * x + 2*np.pi*A*np.sin(2*np.pi*x)

def grad_ackley(x, a=20, b=0.2, c=2*np.pi):
    n = x.shape[0]
    sum_sq = np.sum((x-1.0)**2) # shift x->(x-1) like GA
    sum_cos = np.sum(np.cos(c*(x-1.0)))
    sqrt_term = np.sqrt(sum_sq/n)
    exp1 = np.exp(-b*sqrt_term)
    exp2 = np.exp(sum_cos/n)
    grad = np.zeros_like(x)
    for i in range(n):
        if sqrt_term != 0:
            term1 = (a*b/(n*sqrt_term)) * (x[i]-1.0) * exp1
        else:
            term1 = 0.0
        term2 = (c / n) * np.sin(c*(x[i]-1.0)) * exp2
        grad[i] = term1 + term2
    return grad


# =========================================================
# GRADIENT DESENT ALGORITHM
# =========================================================

def gradient_descent(
    cfg,
    alpha=0.001,
    max_nfe=None,
    tol=1e-8
):
    dim = cfg.DIMENSION
    low, high = cfg.BOUNDS
    f = get_objective(cfg.OBJECTIVE_NAME)
    # choose correct gradient
    grad_map = {
        "sphere": grad_sphere,
        "rosenbrock": grad_rosenbrock,
        "rastrigin": grad_rastrigin,
        "ackley": grad_ackley
    }
    grad_f = grad_map[cfg.OBJECTIVE_NAME]

    x = np.random.uniform(low, high, dim)
    nfe = 0
    history = []

    while True:
        fx = f(x.reshape(1, -1))[0]
        history.append(fx)
        nfe += 1
        if max_nfe and nfe >= max_nfe:
            break

        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break

        x = x - alpha*g

    return x, fx, history, nfe