from src.swarm_algo.abc import ABC
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

ROSENBROCK_DATA_FOLDER = "data/rosenbrock/"
NUM_FOOD_SOURCES = 50
NUM_GENERATIONS = 1000
LIMIT = 100


def rosenbrock_function(x):
    """
    Rosenbrock function (Banana function)
    f(x) = sum([100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2])
    Global minimum at f(1,1,...,1) = 0
    """
    n = len(x)
    total = 0
    for i in range(n - 1):
        total += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return total


def read_test_case(test_number: int) -> Tuple[int, np.ndarray, float]:
    """
    Read Rosenbrock test case
    Returns: dimension, input_vector, initial_value
    """
    filename = ROSENBROCK_DATA_FOLDER + f"test_{test_number:02d}.txt"

    with open(filename, "r") as f:
        lines = f.readlines()

    dimension = None
    initial_value = None
    input_vector = []

    reading_vector = False
    for line in lines:
        line = line.strip()
        if line.startswith("# Dimension:"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("# Initial value:"):
            # Extract value from "# Initial value: f(x) = 12345.67"
            initial_value = float(line.split("=")[1].strip())
        elif line.startswith("# Input vector:"):
            reading_vector = True
        elif reading_vector and line and not line.startswith("#"):
            input_vector.append(float(line))

    return dimension, np.array(input_vector), initial_value


def rosenbrock_abc(test_number: int, visualize: bool = False):
    """
    Solve Rosenbrock optimization using ABC algorithm
    """
    dimension, input_vector, initial_value = read_test_case(test_number)

    print(f"\n{'='*60}")
    print(f"Test Case {test_number}: Rosenbrock Function Optimization")
    print(f"{'='*60}")
    print(f"Dimension: {dimension}")
    print(f"Initial point: {input_vector}")
    print(f"Initial value: {rosenbrock_function(input_vector):.6f}")
    print(f"Global minimum: f([1,1,...,1]) = 0.0")

    # Initialize ABC
    abc = ABC(
        dimension=dimension,
        sn=NUM_FOOD_SOURCES,
        mcn=NUM_GENERATIONS,
        limit=LIMIT,
        lb=-5.0,
        ub=5.0,
    )

    abc.set_objective_function(rosenbrock_function)

    print(f"\nRunning ABC Algorithm...")
    print(
        f"Parameters: food_sources={NUM_FOOD_SOURCES}, generations={NUM_GENERATIONS}, limit={LIMIT}"
    )

    # Run ABC
    best_solution, best_fitness, fitness_history = abc.run(visualize=visualize)

    # Calculate actual function value
    best_value = rosenbrock_function(best_solution)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Best solution found: {best_solution}")
    print(f"Best value: {best_value:.6f}")
    print(f"Initial value: {initial_value:.6f}")
    print(f"Improvement: {initial_value - best_value:.6f} ({((initial_value - best_value)/initial_value*100) if initial_value != 0 else 100:.2f}%)")
    print(f"Distance from optimal (1,1,...,1): {np.linalg.norm(best_solution - 1.0):.6f}")
    print(f"Error from optimal: {abs(best_value - 0.0):.6f}")

    if best_value < 0.01:
        print(f"✓ Successfully found near-optimal solution!")
    elif best_value < 1.0:
        print(f"○ Found good solution")
    else:
        print(f"✗ Solution needs improvement")

    return best_solution, best_value, fitness_history


if __name__ == "__main__":
    # Test on different cases
    test_cases = [1, 5, 10]

    for test_num in test_cases:
        rosenbrock_abc(test_num, visualize=False)
        print("\n")

    # Visualize the last one
    print("Running with visualization...")
    rosenbrock_abc(10, visualize=True)
