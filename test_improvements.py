"""
Validation test suite for code logic improvements.
Tests: GA custom objectives, GD portfolio support, weight validation.
"""

import numpy as np
import sys
from config import GAConfig
from benchmarks import validate_portfolio_weights, Benchmarks, fitness_from_objective
from core import run_ga, initialize_population, evaluate_population
from gradient_descent import gradient_descent
from portfolio_optimization import RealPortfolioOptimizer


def test_fitness_objective_conversion():
    """Test fitness conversion from objective."""
    obj = np.array([0, 1, 10])
    fit = fitness_from_objective(obj)
    assert np.allclose(fit, np.array([1.0, 0.5, 1/11])), "Fitness conversion failed"
    print("[PASS] Test 1: Fitness conversion works correctly")


def test_validate_portfolio_weights():
    """Test portfolio weight validation."""
    # Valid weights
    w_valid = np.array([0.2, 0.3, 0.5])
    assert validate_portfolio_weights(w_valid) == True, "Valid weights rejected"
    
    # Invalid: negative
    w_neg = np.array([0.2, -0.1, 0.9])
    assert validate_portfolio_weights(w_neg) == False, "Negative weights accepted"
    
    # Invalid: sum != 1
    w_sum = np.array([0.2, 0.3, 0.4])
    assert validate_portfolio_weights(w_sum) == False, "Invalid sum accepted"
    
    print("[PASS] Test 2: Portfolio weight validation works")


def test_ga_custom_objective():
    """Test GA with custom objective function."""
    cfg = GAConfig(
        DIMENSION=3,
        BOUNDS=(-1, 1),
        POP_SIZE=10,
        GENERATIONS=5,
        OBJECTIVE_NAME="sphere"
    )
    
    # Define custom objective
    def custom_sphere(X):
        return np.sum((X - 0.5) ** 2, axis=1)
    
    # Run GA with custom objective
    best_fit, avg_fit, best_obj, avg_obj = run_ga(cfg, custom_objective=custom_sphere)
    
    # Check shapes
    assert len(best_fit) == cfg.GENERATIONS, "Best fit history shape wrong"
    assert len(best_obj) == cfg.GENERATIONS, "Best obj history shape wrong"
    
    # Check convergence
    assert best_obj[-1] <= best_obj[0], "GA did not converge"
    
    print("[PASS] Test 3: GA with custom objective works")


def test_gd_standard_benchmarks():
    """Test GD on standard benchmarks."""
    cfg = GAConfig(
        DIMENSION=5,
        BOUNDS=(-5, 5),
        OBJECTIVE_NAME="sphere",
        GD_ALPHA=0.01,
        NFE=1000
    )
    
    x, fval, history, nfe = gradient_descent(cfg, alpha=0.01, max_nfe=1000)
    
    # Check outputs
    assert x.shape == (cfg.DIMENSION,), "Solution shape wrong"
    assert isinstance(fval, (float, np.floating)), "Final value not scalar"
    assert len(history) > 0, "No history recorded"
    assert nfe > 0, "No evaluations performed"
    
    # Check convergence
    assert history[-1] <= history[0], "GD did not converge"
    
    print("[PASS] Test 4: GD on standard benchmarks works")


def test_gd_alpha_sweep():
    """Test GD with different step sizes."""
    cfg = GAConfig(DIMENSION=5, BOUNDS=(-5, 5), OBJECTIVE_NAME="sphere", NFE=500)
    
    alphas = [0.001, 0.01, 0.1]
    final_values = []
    
    for alpha in alphas:
        _, fval, _, _ = gradient_descent(cfg, alpha=alpha, max_nfe=500)
        final_values.append(fval)
    
    # At least one alpha should produce reasonable results
    assert min(final_values) < 1.0, "GD failed to optimize on sphere"
    
    print("[PASS] Test 5: GD alpha sweep works")


def test_population_evaluation_consistency():
    """Test that custom objective is properly used in population evaluation."""
    cfg = GAConfig(DIMENSION=3, BOUNDS=(-1, 1), POP_SIZE=5)
    
    pop = initialize_population(cfg)
    
    # Evaluate with default
    obj1, fit1 = evaluate_population(pop, cfg)
    
    # Evaluate with custom objective
    def custom_obj(X):
        return np.sum(X ** 4, axis=1)
    
    obj2, fit2 = evaluate_population(pop, cfg, custom_objective=custom_obj)
    
    # Results should differ
    assert not np.allclose(obj1, obj2), "Custom objective not applied"
    
    print("[PASS] Test 6: Custom objective properly evaluated")


def test_gradient_descent_structure():
    """Test GD returns all required outputs."""
    cfg = GAConfig(DIMENSION=3, BOUNDS=(-5, 5), OBJECTIVE_NAME="sphere", NFE=100)
    
    x, fval, history, nfe = gradient_descent(cfg, alpha=0.01, max_nfe=100)
    
    assert isinstance(x, np.ndarray) and x.shape == (3,), "x output invalid"
    assert isinstance(fval, (float, np.floating)), "fval output invalid"
    assert isinstance(history, list) and len(history) > 0, "history output invalid"
    assert isinstance(nfe, (int, np.integer)) and nfe > 0, "nfe output invalid"
    
    print("[PASS] Test 7: GD output structure correct")


def test_all_benchmarks_callable():
    """Test all benchmark objectives are callable."""
    benchmarks_list = ["sphere", "rastrigin", "ackley", "rosenbrock"]
    cfg = GAConfig(DIMENSION=5, BOUNDS=(-5, 5))
    
    for name in benchmarks_list:
        obj_func = Benchmarks.__dict__[name]
        X = np.random.uniform(-5, 5, (10, 5))
        
        try:
            result = obj_func(X)
            assert result.shape == (10,), f"Shape error for {name}"
            assert np.all(~np.isnan(result)), f"NaN in {name}"
        except Exception as e:
            raise AssertionError(f"Benchmark {name} failed: {e}")
    
    print("[PASS] Test 8: All benchmarks callable and valid")


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("RUNNING VALIDATION TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_fitness_objective_conversion,
        test_validate_portfolio_weights,
        test_ga_custom_objective,
        test_gd_standard_benchmarks,
        test_gd_alpha_sweep,
        test_population_evaluation_consistency,
        test_gradient_descent_structure,
        test_all_benchmarks_callable,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_func.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
