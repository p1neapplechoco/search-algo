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
    Generate 10 challenging test cases for Rosenbrock function
    """
    output_dir = "data/rosenbrock"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Rosenbrock Test Cases (EXTREME DIFFICULTY)")
    print("=" * 60)
    
    testcases = []
    
    # Test case 1: Extreme far from optimal - all -10s
    x1 = np.full(5, -10.0)
    testcases.append(("Extreme negative (all -10s)", x1, rosenbrock(x1)))
    
    # Test case 2: Extreme positive values
    x2 = np.full(5, 10.0)
    testcases.append(("Extreme positive (all 10s)", x2, rosenbrock(x2)))
    
    # Test case 3: Wide spread extreme negative
    x3 = np.random.uniform(-10, -5, 5)
    testcases.append(("Wide extreme negative [-10, -5]", x3, rosenbrock(x3)))
    
    # Test case 4: Wide spread extreme positive
    x4 = np.random.uniform(5, 10, 5)
    testcases.append(("Wide extreme positive [5, 10]", x4, rosenbrock(x4)))
    
    # Test case 5: Alternating extreme values
    x5 = np.array([-10, 10, -10, 10, -10])
    testcases.append(("Alternating -10/+10", x5, rosenbrock(x5)))
    
    # Test case 6: Random very wide range
    x6 = np.random.uniform(-10, 10, 5)
    testcases.append(("Very wide random [-10, 10]", x6, rosenbrock(x6)))
    
    # Test case 7: All zeros (still challenging)
    x7 = np.zeros(5)
    testcases.append(("All zeros", x7, rosenbrock(x7)))
    
    # Test case 8: 10D extreme negative
    x8 = np.random.uniform(-10, -5, 10)
    testcases.append(("10D extreme negative [-10, -5]", x8, rosenbrock(x8)))
    
    # Test case 9: 10D extreme positive
    x9 = np.random.uniform(5, 10, 10)
    testcases.append(("10D extreme positive [5, 10]", x9, rosenbrock(x9)))
    
    # Test case 10: 10D very wide random
    x10 = np.random.uniform(-10, 10, 10)
    testcases.append(("10D very wide [-10, 10]", x10, rosenbrock(x10)))
    
    # Save and print results
    for i, (name, x, value) in enumerate(testcases, 1):
        # Save to file
        filename = f"{output_dir}/test_{i:02d}.txt"
        with open(filename, 'w') as f:
            f.write(f"# Test Case {i}: {name}\n")
            f.write(f"# Dimension: {len(x)}\n")
            f.write(f"# Initial value: f(x) = {value:.6f}\n")
            f.write(f"# Optimal: f([1,1,...,1]) = 0.0\n")
            f.write("# Input vector:\n")
            for val in x:
                f.write(f"{val:.6f}\n")
        
        # Print to console
        print(f"\nTest Case {i}: {name}")
        print(f"  Dimension: {len(x)}")
        print(f"  Input: {x}")
        print(f"  Initial f(x) = {value:.6f}")
        print(f"  Saved to: {filename}")
    
    print("\n" + "=" * 60)
    print(f"Generated {len(testcases)} test cases in {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    generate_rosenbrock_testcases()
