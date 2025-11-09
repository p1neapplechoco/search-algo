from src.search_algo.bfs import BFS
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

KNAPSACK_DATA_FOLDER = "data/knapsack/"


def get_problem_infos(PROBLEM: int) -> Tuple[int, List[int], List[int], List[int]]:
    """
    Load knapsack problem data from files.

    Args:
        PROBLEM: Problem number (1-18)

    Returns:
        Tuple of (capacity, profits, weights, solution)
    """
    problem = f"p{PROBLEM:02d}"

    with open(KNAPSACK_DATA_FOLDER + problem + "_c.txt") as f:
        capacity = int(f.read())

    with open(KNAPSACK_DATA_FOLDER + problem + "_p.txt") as f:
        profits = [int(p) for p in f.readlines()]

    with open(KNAPSACK_DATA_FOLDER + problem + "_w.txt") as f:
        weights = [int(w) for w in f.readlines()]

    with open(KNAPSACK_DATA_FOLDER + problem + "_s.txt") as f:
        solution = [int(s) for s in f.readlines()]

    return capacity, profits, weights, solution


def calculate_knapsack_value(
    selection: Tuple[int, ...], profits: List[int], weights: List[int], capacity: int
) -> Tuple[int, int, bool]:
    """
    Calculate total profit and weight for a knapsack selection.

    Args:
        selection: Binary tuple indicating which items are selected
        profits: List of item profits
        weights: List of item weights
        capacity: Knapsack capacity

    Returns:
        Tuple of (total_profit, total_weight, is_valid)
    """
    total_profit = sum(
        p for i, p in enumerate(profits) if i < len(selection) and selection[i]
    )
    total_weight = sum(
        w for i, w in enumerate(weights) if i < len(selection) and selection[i]
    )
    is_valid = total_weight <= capacity

    return total_profit, total_weight, is_valid


def relativity_to_solution(
    ans: Tuple[int, ...], sol: List[int], profits: List[int]
) -> float:
    """
    Calculate accuracy compared to optimal solution.

    Args:
        ans: Algorithm's solution
        sol: Optimal solution
        profits: List of item profits

    Returns:
        Accuracy ratio (0.0 to 1.0)
    """
    ans_profit = sum(profits[i] for i in range(len(profits)) if i < len(ans) and ans[i])
    sol_profit = sum(profits[i] for i in range(len(profits)) if sol[i])

    if sol_profit == 0:
        return 1.0 if ans_profit == 0 else 0.0

    return ans_profit / sol_profit


def knapsack_bfs(PROBLEM: int, visualize: bool = False, max_iterations: int = 1000000):
    """
    Solve knapsack problem using BFS algorithm.

    Args:
        PROBLEM: Problem number (1-18)
        visualize: Whether to show visualization
        max_iterations: Maximum iterations for BFS
    """
    capacity, profits, weights, solution = get_problem_infos(PROBLEM)
    num_items = len(profits)

    print(f"\n{'='*70}")
    print(f"Knapsack Problem {PROBLEM} - BFS Algorithm")
    print(f"{'='*70}")
    print(f"Items: {num_items}, Capacity: {capacity}")
    print(f"Theoretical worst case: 2^{num_items} = {2**num_items:,} states")

    # Calculate optimal solution value
    optimal_profit = sum(profits[i] for i in range(num_items) if solution[i])
    optimal_weight = sum(weights[i] for i in range(num_items) if solution[i])

    print(f"\nOptimal solution:")
    print(f"  Profit: {optimal_profit}")
    print(f"  Weight: {optimal_weight}/{capacity}")
    print(f"  Items: {sum(solution)}/{num_items}")

    # Define state representation: tuple of binary selection (0 or 1 for each item)
    # Start with empty knapsack
    start_state = tuple([0] * num_items)

    # Define get_neighbors function for knapsack
    def get_neighbors(state):
        """Generate all valid neighboring states by adding one more item."""
        neighbors = []
        current_weight = sum(weights[i] for i in range(len(state)) if state[i])

        for i in range(len(state)):
            # If item not selected, try adding it
            if state[i] == 0:
                # Check if adding this item would exceed capacity
                if current_weight + weights[i] <= capacity:
                    # Create new state with this item added
                    new_state = list(state)
                    new_state[i] = 1
                    neighbors.append(tuple(new_state))

        return neighbors

    # Define fitness function (negative profit to minimize)
    def fitness_function(state):
        """Calculate negative profit (we want to maximize profit, so minimize negative profit)."""
        total_profit = sum(profits[i] for i in range(len(state)) if state[i])
        return -total_profit

    # Define state_to_tuple function (already a tuple, but ensure consistency)
    def state_to_tuple(state):
        return tuple(state)

    # Solve using BFS
    print(f"\nRunning BFS...")
    start_time = time.time()

    bfs = BFS(start_state, get_neighbors, max_iterations)
    bfs.set_fitness_function(fitness_function)

    # Use search_best with depth limit (num_items is the max depth)
    best_state, best_fitness = bfs.search_best(
        max_depth=num_items, state_to_tuple=state_to_tuple
    )

    elapsed_time = time.time() - start_time

    # Extract results
    best_selection = best_state
    best_profit = -best_fitness  # Convert back to positive
    nodes_explored = bfs.visited_count

    # Calculate result metrics
    best_weight = sum(weights[i] for i in range(num_items) if best_selection[i])
    items_selected = sum(best_selection)
    accuracy = relativity_to_solution(best_selection, solution, profits)

    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"{'='*70}")
    print(f"BFS solution profit: {best_profit}")
    print(f"Optimal profit: {optimal_profit}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Weight used: {best_weight}/{capacity} ({best_weight/capacity*100:.1f}%)")
    print(f"Items selected: {items_selected}/{num_items}")
    print(f"Nodes explored: {nodes_explored:,}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")

    # Check if optimal
    if best_profit == optimal_profit:
        print(f"✓ Found optimal solution!")
    elif accuracy >= 0.95:
        print(f"○ Found near-optimal solution (95%+)")
    else:
        print(f"△ Solution could be improved")

    # Visualize BFS search tree (for small problems only)
    if visualize and num_items <= 10:
        print(f"\n📊 Visualizing BFS search tree...")
        print(f"   (Showing up to 100 nodes from the search space)")
        try:
            bfs.visualize_search_tree(max_nodes=100)
        except Exception as e:
            print(f"   Warning: Could not visualize search tree: {e}")

    # Show selected items
    if num_items <= 20:  # Only show for small problems
        print(f"\nSelected items:")
        for i in range(num_items):
            if best_selection[i]:
                print(f"  Item {i+1}: profit={profits[i]}, weight={weights[i]}")

    # Visualization
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Comparison bar chart
        ax1 = axes[0]
        categories = ["BFS Solution", "Optimal Solution"]
        values = [best_profit, optimal_profit]
        colors = ["#45B7D1" if best_profit == optimal_profit else "#FFA07A", "#2ECC71"]
        bars = ax1.bar(
            categories, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2
        )
        ax1.set_ylabel("Total Profit", fontsize=12)
        ax1.set_title(
            f"Knapsack Problem {PROBLEM} - Profit Comparison",
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        # Plot 2: Metrics comparison
        ax2 = axes[1]
        metrics = ["Accuracy\n(%)", "Weight\nUsage (%)", "Items\nSelected (%)"]
        bfs_metrics = [
            accuracy * 100,
            (best_weight / capacity) * 100,
            (items_selected / num_items) * 100,
        ]
        optimal_metrics = [
            100,
            (optimal_weight / capacity) * 100,
            (sum(solution) / num_items) * 100,
        ]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax2.bar(
            x - width / 2,
            bfs_metrics,
            width,
            label="BFS Solution",
            color="#45B7D1",
            alpha=0.7,
            edgecolor="black",
        )
        bars2 = ax2.bar(
            x + width / 2,
            optimal_metrics,
            width,
            label="Optimal",
            color="#2ECC71",
            alpha=0.7,
            edgecolor="black",
        )

        ax2.set_ylabel("Percentage (%)", fontsize=12)
        ax2.set_title("Metrics Comparison", fontsize=14, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_ylim(0, 110)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()

        # Save to file
        output_filename = f"knapsack_bfs_p{PROBLEM:02d}.png"
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"\n📊 Visualization saved to: {output_filename}")

        plt.show()


def run_small_problems():
    """
    Run BFS on small knapsack problems (recommended: problems with <= 20 items).
    """
    print("\n" + "=" * 70)
    print("KNAPSACK BFS - RUNNING SMALL PROBLEMS")
    print("=" * 70)
    print("\nNote: BFS has exponential complexity. Only small problems are feasible.")
    print("Recommended: Problems 1-7 (10-23 items)")

    # Test on problems with fewer items
    test_problems = [1, 2, 3, 4, 5]  # These have 10-23 items
    results = []

    for prob in test_problems:
        try:
            capacity, profits, weights, _ = get_problem_infos(prob)
            num_items = len(profits)

            # Skip if too many items
            if num_items > 23:
                print(f"\nSkipping Problem {prob} (too large: {num_items} items)")
                continue

            knapsack_bfs(prob, visualize=False, max_iterations=1000000)
            results.append(prob)

        except Exception as e:
            print(f"\nError on Problem {prob}: {e}")

    print("\n" + "=" * 70)
    print(f"Completed {len(results)} problems: {results}")
    print("=" * 70)


def demo_bfs_visualization():
    """
    Demo BFS visualization with a small custom problem.
    Creates a tiny knapsack problem to clearly show the BFS search tree.
    """
    print("\n" + "=" * 70)
    print("BFS SEARCH TREE VISUALIZATION DEMO")
    print("=" * 70)
    print("\nCreating a tiny knapsack problem to visualize BFS search tree...")

    # Small problem: 5 items
    capacity = 15
    profits = [10, 6, 12, 8, 5]
    weights = [5, 3, 7, 4, 2]
    num_items = len(profits)

    print(f"\nProblem setup:")
    print(f"  Capacity: {capacity}")
    print(f"  Items: {num_items}")
    for i in range(num_items):
        print(f"    Item {i+1}: profit={profits[i]}, weight={weights[i]}")

    # Define BFS components
    start_state = tuple([0] * num_items)

    def get_neighbors(state):
        neighbors = []
        current_weight = sum(weights[i] for i in range(len(state)) if state[i])
        for i in range(len(state)):
            if state[i] == 0:
                if current_weight + weights[i] <= capacity:
                    new_state = list(state)
                    new_state[i] = 1
                    neighbors.append(tuple(new_state))
        return neighbors

    def fitness_function(state):
        total_profit = sum(profits[i] for i in range(len(state)) if state[i])
        return -total_profit

    def state_to_tuple(state):
        return tuple(state)

    # Create BFS instance
    print(f"\nRunning BFS...")
    bfs = BFS(start_state, get_neighbors, max_iterations=10000)
    bfs.set_fitness_function(fitness_function)

    # Find best solution
    best_state, best_fitness = bfs.search_best(
        max_depth=num_items, state_to_tuple=state_to_tuple
    )

    best_profit = -best_fitness
    best_weight = sum(weights[i] for i in range(num_items) if best_state[i])

    print(f"\nBest solution found:")
    print(f"  Total profit: {best_profit}")
    print(f"  Total weight: {best_weight}/{capacity}")
    print(f"  Selected items: {[i+1 for i in range(num_items) if best_state[i]]}")
    print(f"  Nodes explored: {bfs.visited_count:,}")

    # Visualize search tree
    print(f"\n📊 Generating BFS search tree visualization...")
    print(f"   (Showing up to 50 nodes)")
    print(f"   Note: Each node represents a state (which items are selected)")
    print(f"   The tree shows how BFS explores different combinations")

    try:
        bfs.visualize_search_tree(max_nodes=50)
        print(f"\n✓ Search tree visualization displayed!")
    except ImportError:
        print(f"\n⚠ NetworkX not installed. Cannot visualize search tree.")
        print(f"   Install with: pip install networkx")
    except Exception as e:
        print(f"\n⚠ Could not visualize: {e}")


if __name__ == "__main__":
    # Option 1: Run single problem with visualization
    # Note: BFS can handle problems with ~20 items reasonably well
    # Larger problems may take significant time due to exponential complexity
    # knapsack_bfs(PROBLEM=1, visualize=True)

    # Option 2: Run multiple small problems
    # run_small_problems()

    # Option 3: Demo BFS search tree visualization
    demo_bfs_visualization()
