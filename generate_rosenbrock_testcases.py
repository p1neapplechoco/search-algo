import numpy as np
import os

# Rosenbrock function
def rosenbrock(x):
    """
    Rosenbrock function
    Global minimum: f(1, 1, ..., 1) = 0
    """
    term1 = x[1:] - x[:-1]**2
    term2 = x[:-1] - 1
    return np.sum(100 * term1**2 + term2**2)


def generate_rosenbrock_testcases():
    """
    Generate 10 test cases for Rosenbrock function
    """
    output_dir = "data/testcases/rosenbrock"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Rosenbrock Test Cases")
    print("=" * 60)
    
    testcases = []
    
    # Test case 1: Optimal solution
    x1 = np.ones(5)
    testcases.append(("Optimal (all ones)", x1, rosenbrock(x1)))
    
    # Test case 2: All zeros
    x2 = np.zeros(5)
    testcases.append(("All zeros", x2, rosenbrock(x2)))
    
    # Test case 3: Random in [-5, 5]
    x3 = np.random.uniform(-5, 5, 5)
    testcases.append(("Random [-5, 5]", x3, rosenbrock(x3)))
    
    # Test case 4: Random in [0, 2]
    x4 = np.random.uniform(0, 2, 5)
    testcases.append(("Random [0, 2]", x4, rosenbrock(x4)))
    
    # Test case 5: Close to optimal
    x5 = np.ones(5) + np.random.uniform(-0.1, 0.1, 5)
    testcases.append(("Near optimal", x5, rosenbrock(x5)))
    
    # Test case 6: All same value (2)
    x6 = np.full(5, 2.0)
    testcases.append(("All 2s", x6, rosenbrock(x6)))
    
    # Test case 7: Alternating 0 and 1
    x7 = np.array([0, 1, 0, 1, 0], dtype=float)
    testcases.append(("Alternating 0,1", x7, rosenbrock(x7)))
    
    # Test case 8: Negative values
    x8 = np.random.uniform(-5, 0, 5)
    testcases.append(("Random negative", x8, rosenbrock(x8)))
    
    # Test case 9: Large dimension (10D)
    x9 = np.ones(10)
    testcases.append(("10D optimal", x9, rosenbrock(x9)))
    
    # Test case 10: Large dimension random
    x10 = np.random.uniform(-5, 5, 10)
    testcases.append(("10D random", x10, rosenbrock(x10)))
    
    # Save and print results
    for i, (name, x, value) in enumerate(testcases, 1):
        # Save to file
        filename = f"{output_dir}/test_{i:02d}.txt"
        with open(filename, 'w') as f:
            f.write(f"# Test Case {i}: {name}\n")
            f.write(f"# Dimension: {len(x)}\n")
            f.write(f"# Expected value: {value:.6f}\n")
            f.write("# Input vector:\n")
            for val in x:
                f.write(f"{val:.6f}\n")
        
        # Print to console
        print(f"\nTest Case {i}: {name}")
        print(f"  Dimension: {len(x)}")
        print(f"  Input: {x}")
        print(f"  f(x) = {value:.6f}")
        print(f"  Saved to: {filename}")
    
    print("\n" + "=" * 60)
    print(f"Generated {len(testcases)} test cases in {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    generate_rosenbrock_testcases()
