import numpy as np
import os

# Ackley function
def ackley(x):
    """
    Ackley function
    Global minimum: f(0, 0, ..., 0) = 0
    Typically bounded: [-32.768, 32.768]
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    
    sum_sq_term = -b * np.sqrt(np.sum(x**2) / n)
    sum_cos_term = np.sum(np.cos(c * x)) / n
    
    return -a * np.exp(sum_sq_term) - np.exp(sum_cos_term) + a + np.exp(1)


def generate_ackley_testcases():
    """
    Generate 10 test cases for Ackley function
    """
    output_dir = "data/testcases/ackley"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Ackley Test Cases")
    print("=" * 60)
    
    testcases = []
    
    # Test case 1: Optimal solution (all zeros)
    x1 = np.zeros(5)
    testcases.append(("Optimal (all zeros)", x1, ackley(x1)))
    
    # Test case 2: All ones
    x2 = np.ones(5)
    testcases.append(("All ones", x2, ackley(x2)))
    
    # Test case 3: Random in [-32, 32]
    x3 = np.random.uniform(-32, 32, 5)
    testcases.append(("Random [-32, 32]", x3, ackley(x3)))
    
    # Test case 4: Random in [-10, 10]
    x4 = np.random.uniform(-10, 10, 5)
    testcases.append(("Random [-10, 10]", x4, ackley(x4)))
    
    # Test case 5: Close to optimal
    x5 = np.random.uniform(-0.5, 0.5, 5)
    testcases.append(("Near optimal", x5, ackley(x5)))
    
    # Test case 6: All same value (5)
    x6 = np.full(5, 5.0)
    testcases.append(("All 5s", x6, ackley(x6)))
    
    # Test case 7: Alternating positive/negative
    x7 = np.array([-5, 5, -5, 5, -5], dtype=float)
    testcases.append(("Alternating ±5", x7, ackley(x7)))
    
    # Test case 8: Large values
    x8 = np.random.uniform(20, 32, 5)
    testcases.append(("Large values [20, 32]", x8, ackley(x8)))
    
    # Test case 9: Large dimension (10D)
    x9 = np.zeros(10)
    testcases.append(("10D optimal", x9, ackley(x9)))
    
    # Test case 10: Large dimension random
    x10 = np.random.uniform(-32, 32, 10)
    testcases.append(("10D random", x10, ackley(x10)))
    
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
    generate_ackley_testcases()
