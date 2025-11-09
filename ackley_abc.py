from src.swarm_algo.abc import ABC
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

ACKLEY_DATA_FOLDER = "data/ackley/"
NUM_FOOD_SOURCES = 50
NUM_GENERATIONS = 1000
LIMIT = 100


def ackley_function(x, a=20, b=0.2, c=2 * np.pi):
    """
    Ackley function - a widely used benchmark for optimization algorithms

    f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c * x_i))) + a + exp(1)

    Global minimum at f(0,0,...,0) = 0

    Args:
        x: Input vector
        a: First constant (default: 20)
        b: Second constant (default: 0.2)
        c: Third constant (default: 2π)

    Returns:
        Function value
    """
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    result = term1 + term2 + a + np.e

    return result


def read_test_case(test_number: int) -> Tuple[int, np.ndarray, float]:
    """
    Read Ackley test case
    Returns: dimension, input_vector, initial_value
    """
    filename = ACKLEY_DATA_FOLDER + f"test_{test_number:02d}.txt"

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


def ackley_abc(test_number: int, visualize: bool = False):
    """
    Solve Ackley optimization using ABC algorithm
    """
    dimension, input_vector, initial_value = read_test_case(test_number)

    print(f"\n{'='*60}")
    print(f"Test Case {test_number}: Ackley Function Optimization")
    print(f"{'='*60}")
    print(f"Dimension: {dimension}")
    print(f"Initial point: {input_vector}")
    print(f"Initial value: {ackley_function(input_vector):.6f}")
    print(f"Global minimum: f([0,0,...,0]) = 0.0")

    # Initialize ABC
    abc = ABC(
        dimension=dimension,
        sn=NUM_FOOD_SOURCES,
        mcn=NUM_GENERATIONS,
        limit=LIMIT,
        lb=-5.0,
        ub=5.0,
    )

    abc.set_objective_function(ackley_function)

    print(f"\nRunning ABC Algorithm...")
    print(
        f"Parameters: food_sources={NUM_FOOD_SOURCES}, generations={NUM_GENERATIONS}, limit={LIMIT}"
    )

    # Run ABC
    best_solution, best_fitness, fitness_history = abc.run(visualize=visualize)

    # Calculate actual function value
    best_value = ackley_function(best_solution)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Best solution found: {best_solution}")
    print(f"Best value: {best_value:.6f}")
    print(f"Initial value: {initial_value:.6f}")
    print(
        f"Improvement: {initial_value - best_value:.6f} ({((initial_value - best_value)/initial_value*100) if initial_value != 0 else 0:.2f}%)"
    )
    print(f"Distance from optimal (0,0,...,0): {np.linalg.norm(best_solution):.6f}")
    print(f"Error from optimal: {abs(best_value - 0.0):.6f}")

    # Accuracy evaluation
    if best_value < 0.01:
        print(f"✓ Successfully found near-optimal solution!")
    elif best_value < 1.0:
        print(f"○ Found good solution")
    elif best_value < 5.0:
        print(f"△ Found acceptable solution")
    else:
        print(f"✗ Solution needs improvement")

    return best_solution, best_value, fitness_history


def run_all_tests(visualize_last: bool = True):
    """
    Run all test cases and show summary
    """
    results = []

    print("\n" + "=" * 70)
    print("ACKLEY FUNCTION OPTIMIZATION - ABC ALGORITHM")
    print("=" * 70)

    # Run all 10 test cases
    for test_num in range(1, 11):
        best_solution, best_value, fitness_history = ackley_abc(
            test_num, visualize=False
        )
        results.append((test_num, best_value))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL TEST CASES")
    print("=" * 70)
    print(f"{'Test':<6} {'Best Value':<15} {'Status':<20}")
    print("-" * 70)

    for test_num, best_value in results:
        if best_value < 0.01:
            status = "✓ Near-optimal"
        elif best_value < 1.0:
            status = "○ Good"
        elif best_value < 5.0:
            status = "△ Acceptable"
        else:
            status = "✗ Needs improvement"

        print(f"{test_num:<6} {best_value:<15.6f} {status:<20}")

    avg_value = np.mean([v for _, v in results])
    print("-" * 70)
    print(f"Average best value: {avg_value:.6f}")

    # Visualize the last test case
    if visualize_last:
        print("\n" + "=" * 70)
        print("Running Test Case 10 with Visualization...")
        print("=" * 70)
        ackley_abc(10, visualize=True)


if __name__ == "__main__":
    # Option 1: Run a single test case
    # ackley_abc(test_number=1, visualize=True)

    # Option 2: Run all test cases
    run_all_tests(visualize_last=True)
