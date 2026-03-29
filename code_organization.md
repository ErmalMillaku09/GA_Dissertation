# Genetic Algorithm and Gradient Descent Code Organization

## Overview

This codebase implements a comprehensive comparison framework for Genetic Algorithms (GA) and Gradient Descent (GD) optimization methods. It includes benchmark functions, portfolio optimization with real market data, and extensive experimental capabilities.

## Module Architecture

### Core Modules

#### `config.py`
- **Purpose**: Configuration management using Python dataclasses
- **Key Classes**:
  - `GAConfig`: Central configuration class containing all algorithm parameters
- **Parameters**:
  - GA parameters: population size, generations, crossover/mutation rates
  - GD parameters: learning rate, max function evaluations
  - Objective settings: function name, bounds, dimensions
  - Portfolio-specific settings

#### `core.py`
- **Purpose**: Core GA implementation
- **Key Functions**:
  - `initialize_population()`: Creates initial random population
  - `evaluate_population()`: Evaluates fitness/objective for entire population
  - `evolve_one_generation()`: Performs one GA iteration (selection, crossover, mutation)
  - `run_ga()`: Main GA execution loop
  - `run_random_search()`: Baseline random search implementation

#### `benchmarks.py`
- **Purpose**: Benchmark objective functions and fitness transformations
- **Key Classes**:
  - `Benchmarks`: Static class with standard optimization functions
    - `sphere()`: Shifted sphere function
    - `rastrigin()`: Multi-modal function with many local minima
    - `ackley()`: Complex landscape with global minimum in a valley
    - `rosenbrock()`: Banana-shaped valley function
  - `PortfolioOptimizer`: Placeholder for portfolio optimization (replaced by real implementation)
- **Utilities**:
  - `get_objective()`: Factory function for objective functions
  - `fitness_from_objective()`: Converts minimization objectives to maximization fitness
  - `validate_portfolio_weights()`: Portfolio weight validation

### Algorithm Modules

#### `gradient_descent.py`
- **Purpose**: Gradient-based optimization implementations
- **Key Functions**:
  - `gradient_descent()`: Basic gradient descent with bounds and convergence checks
  - `multi_start_gradient_descent()`: Multi-start GD with adaptive learning rates
  - Individual gradient functions for each benchmark (e.g., `grad_sphere()`, `grad_ackley()`)

#### `operators.py`
- **Purpose**: GA genetic operators
- **Key Functions**:
  - `select()`: Parent selection (roulette, tournament, ranking)
  - `arithmetic_crossover()`: Linear combination crossover
  - `mutate()`: Gaussian mutation with bounds checking

### Portfolio Optimization

#### `portfolio_optimization.py`
- **Purpose**: Real-world portfolio optimization with market data
- **Key Classes**:
  - `PortfolioData`: Data container for market data
  - `RealPortfolioOptimizer`: Main portfolio optimization class
- **Features**:
  - Downloads real stock data using yfinance
  - Computes covariance matrices and returns
  - Implements variance minimization and Sharpe ratio maximization
  - Provides gradients for GD optimization

#### `market_data.py`
- **Purpose**: Market data utilities (currently minimal)

### Experimentation Framework

#### `experiments.py`
- **Purpose**: Statistical experiment runners and comparisons
- **Key Functions**:
  - `run_experiment()`: Multiple GA runs with statistics
  - `run_gd_statistics()`: Multiple GD runs with statistics
  - `compare_ga_vs_random()`: GA vs random search comparison
  - `run_parameter_sweep()`: Parameter sensitivity analysis
  - `compare_objectives()`: Multi-objective performance comparison

#### `EXPERIMENT_CATALOG.py`
- **Purpose**: Comprehensive experiment catalog with ready-to-run functions
- **Categories**:
  - **Gradient Descent Experiments**: Single runs, statistics, parameter sweeps
  - **Tuned GD Experiments**: Multi-start with adaptive learning
  - **GA Experiments**: Single runs, statistics, parameter comparisons
  - **Algorithm Comparisons**: GA vs GD, GA vs random search
  - **Portfolio Experiments**: Real market data optimization
  - **Advanced Experiments**: Learning rate decay, noise robustness

### Visualization and Utilities

#### `plots.py`
- **Purpose**: Plotting utilities for results visualization
- **Key Functions**:
  - `plot_statistics()`: Convergence plots with confidence intervals
  - `plot_history()`: Single run convergence visualization
  - `plot_fitness_comparison()`: Algorithm comparison plots
  - `plot_gd_convergence()`: GD-specific convergence plots

#### `main.py`
- **Purpose**: Main entry point with example usage
- **Features**:
  - Demonstrates basic GA and GD runs
  - Shows portfolio optimization integration
  - Includes error handling for missing dependencies

#### `test_improvements.py`
- **Purpose**: Validation tests for code correctness
- **Tests**:
  - Fitness/objective conversion validation
  - Portfolio weight validation
  - GA custom objective functionality
  - GD benchmark performance

## Data Flow and Execution

### Standard Benchmark Optimization

1. **Configuration**: `GAConfig` sets problem dimensions, bounds, objective function
2. **GA Execution**:
   - `run_ga()` → `initialize_population()` → `evaluate_population()`
   - Loop: `evolve_one_generation()` → evaluate → track statistics
   - Returns: fitness/objective histories across generations
3. **GD Execution**:
   - `gradient_descent()` → compute gradients → update parameters
   - Returns: final solution, objective value, convergence history

### Portfolio Optimization

1. **Data Loading**: `RealPortfolioOptimizer` downloads and processes market data
2. **Objective Selection**:
   - Variance minimization: `variance_objective()` + `variance_gradient()`
   - Sharpe maximization: `sharpe_objective()` + `sharpe_gradient()`
3. **Optimization**: Same GA/GD flow but with custom objectives
4. **Weight Conversion**: Raw variables → softmax → portfolio weights

### Experiment Execution

1. **Single Experiment**: Direct function call (e.g., `exp_ga_single_run()`)
2. **Statistical Experiments**: Multiple runs with `numpy` statistics
3. **Comparisons**: Parallel execution of different algorithms
4. **Visualization**: Automatic plotting of results

## Key Design Patterns

### Configuration-Driven Design
- All parameters centralized in `GAConfig`
- Easy modification without code changes
- Consistent parameter passing across modules

### Objective Function Abstraction
- `get_objective()` factory pattern for benchmark selection
- Custom objectives via function parameters
- Unified interface for GA and GD

### Fitness vs Objective Distinction
- **Objective**: What we minimize (e.g., function value, portfolio variance)
- **Fitness**: What we maximize (1/(1+objective) for minimization)
- GA maximizes fitness, GD minimizes objective

### Modular Experiment Framework
- Experiment functions are self-contained
- Easy to add new experiments
- Consistent output formatting

## Dependencies

- **numpy**: Numerical computations and arrays
- **matplotlib**: Plotting and visualization
- **yfinance**: Real market data (optional, for portfolio experiments)
- **pandas**: Data manipulation (via yfinance)

## Running Experiments

### Basic Usage
```python
from EXPERIMENT_CATALOG import exp_ga_single_run
exp_ga_single_run()  # Runs GA on default sphere function
```

### Custom Configuration
```python
from config import GAConfig
from EXPERIMENT_CATALOG import exp_ga_statistics

cfg = GAConfig(
    OBJECTIVE_NAME="rastrigin",
    DIMENSION=10,
    GENERATIONS=200,
    RUNS=30
)
# Modify experiments.py functions to accept cfg parameter
```

### Portfolio Optimization
```python
from EXPERIMENT_CATALOG import exp_portfolio_ga_vs_gd
exp_portfolio_ga_vs_gd()  # Requires yfinance installation
```

## Performance Considerations

- **GA**: Population-based, parallel evaluation, good for noisy/multi-modal functions
- **GD**: Fast convergence on smooth functions, sensitive to initialization
- **Portfolio**: High-dimensional optimization, real data dependencies
- **Memory**: Large populations or many runs require significant RAM

## Extension Points

- Add new benchmark functions in `benchmarks.py`
- Implement new selection/crossover operators in `operators.py`
- Add new gradient functions in `gradient_descent.py`
- Create new experiment functions in `EXPERIMENT_CATALOG.py`
- Extend portfolio objectives in `portfolio_optimization.py`
