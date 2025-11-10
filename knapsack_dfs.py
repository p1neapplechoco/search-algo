from search_algo.dfs import DepthFirstSearch as DFS
from typing import List, Tuple
import numpy as np
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
    ans_profit = sum(profits[i] for i in range(
        len(profits)) if i < len(ans) and ans[i])
    sol_profit = sum(profits[i] for i in range(len(profits)) if sol[i])

    if sol_profit == 0:
        return 1.0 if ans_profit == 0 else 0.0

    return ans_profit / sol_profit


def knapsack_dfs(PROBLEM: int, visualize: bool = False, max_iterations: int = 1000000):
    """
    Solve knapsack problem using DFS algorithm.

    Args:
        PROBLEM: Problem number (1-18)
        visualize: Whether to show visualization
        max_iterations: Maximum iterations for DFS
    """
    capacity, profits, weights, solution = get_problem_infos(PROBLEM)
    num_items = len(profits)

    print(f"\n{'='*70}")
    print(f"Knapsack Problem {PROBLEM} - DFS Algorithm")
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

    # Define fitness function (positive profit for maximization)
    def fitness_function(state):
        """Calculate profit (DFS maximizes, so use positive profit)."""
        total_profit = sum(profits[i] for i in range(len(state)) if state[i])
        return total_profit

    # Solve using DFS
    print(f"\nRunning DFS...")
    start_time = time.time()

    dfs = DFS(start_state, get_neighbors)
    dfs.set_fitness_function(fitness_function)

    # Run DFS
    best_state, best_fitness = dfs.run(max_iterations=max_iterations)

    elapsed_time = time.time() - start_time

    # Extract results
    if best_state is None:
        best_selection = start_state
        best_profit = 0
    else:
        best_selection = best_state
        best_profit = best_fitness  # Already positive

    nodes_explored = len(dfs.visited)

    # Calculate result metrics
    best_weight = sum(weights[i]
                      for i in range(num_items) if best_selection[i])
    items_selected = sum(best_selection)
    accuracy = relativity_to_solution(best_selection, solution, profits)

    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"{'='*70}")
    print(f"DFS solution profit: {best_profit}")
    print(f"Optimal profit: {optimal_profit}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(
        f"Weight used: {best_weight}/{capacity} ({best_weight/capacity*100:.1f}%)")
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

    # Show selected items
    if num_items <= 20:  # Only show for small problems
        print(f"\nSelected items:")
        for i in range(num_items):
            if best_selection[i]:
                print(
                    f"  Item {i+1}: profit={profits[i]}, weight={weights[i]}")

    # Visualization
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Comparison bar chart
        ax1 = axes[0]
        categories = ["DFS Solution", "Optimal Solution"]
        values = [best_profit, optimal_profit]
        colors = ["#FF6B6B" if best_profit ==
                  optimal_profit else "#FFA07A", "#2ECC71"]
        bars = ax1.bar(
            categories, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2
        )
        ax1.set_ylabel("Total Profit", fontsize=12)
        ax1.set_title(
            f"Knapsack Problem {PROBLEM} - Profit Comparison (DFS)",
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
        dfs_metrics = [
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
            dfs_metrics,
            width,
            label="DFS Solution",
            color="#FF6B6B",
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
        output_filename = f"knapsack_dfs_p{PROBLEM:02d}.png"
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"\n📊 Visualization saved to: {output_filename}")

        plt.show()


def run_small_problems():
    """
    Run DFS on small knapsack problems (recommended: problems with <= 20 items).
    """
    print("\n" + "=" * 70)
    print("KNAPSACK DFS - RUNNING SMALL PROBLEMS")
    print("=" * 70)
    print("\nNote: DFS has exponential complexity. Only small problems are feasible.")
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
                print(
                    f"\nSkipping Problem {prob} (too large: {num_items} items)")
                continue

            knapsack_dfs(prob, visualize=False, max_iterations=1000000)
            results.append(prob)

        except Exception as e:
            print(f"\nError on Problem {prob}: {e}")

    print("\n" + "=" * 70)
    print(f"Completed {len(results)} problems: {results}")
    print("=" * 70)


def compare_bfs_dfs(PROBLEM: int, visualize: bool = True):
    """
    Compare BFS and DFS on the same problem with detailed metrics.
    """
    capacity, profits, weights, solution = get_problem_infos(PROBLEM)
    num_items = len(profits)

    print("\n" + "=" * 70)
    print(f"COMPARING BFS vs DFS - Problem {PROBLEM}")
    print("=" * 70)

    # Calculate optimal
    optimal_profit = sum(profits[i] for i in range(num_items) if solution[i])

    # Run BFS
    print("\n" + "🔵 RUNNING BFS...")
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

    from src.search_algo.bfs import BFS

    start_time_bfs = time.time()
    bfs = BFS(start_state, get_neighbors, max_iterations=1000000)
    bfs.set_fitness_function(
        lambda s: -sum(profits[i] for i in range(len(s)) if s[i]))
    best_state_bfs, best_fitness_bfs = bfs.search_best(
        max_depth=num_items, state_to_tuple=lambda s: tuple(s)
    )
    time_bfs = time.time() - start_time_bfs

    profit_bfs = -best_fitness_bfs
    nodes_bfs = bfs.visited_count

    print(
        f"  Profit: {profit_bfs}, Nodes: {nodes_bfs:,}, Time: {time_bfs:.2f}s")

    # Run DFS
    print("\n🔴 RUNNING DFS...")
    start_time_dfs = time.time()
    dfs = DFS(start_state, get_neighbors)
    dfs.set_fitness_function(lambda s: sum(
        profits[i] for i in range(len(s)) if s[i]))
    best_state_dfs, best_fitness_dfs = dfs.run(max_iterations=1000000)
    time_dfs = time.time() - start_time_dfs

    profit_dfs = best_fitness_dfs if best_state_dfs else 0
    nodes_dfs = len(dfs.visited)

    print(
        f"  Profit: {profit_dfs}, Nodes: {nodes_dfs:,}, Time: {time_dfs:.2f}s")

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'BFS':<20} {'DFS':<20} {'Winner'}")
    print("-" * 70)
    print(
        f"{'Profit':<20} {profit_bfs:<20} {profit_dfs:<20} {'🔵 BFS' if profit_bfs > profit_dfs else '🔴 DFS' if profit_dfs > profit_bfs else '🟰 Tie'}"
    )
    print(
        f"{'Nodes Explored':<20} {nodes_bfs:<20,} {nodes_dfs:<20,} {'🔵 BFS' if nodes_bfs < nodes_dfs else '🔴 DFS' if nodes_dfs < nodes_bfs else '🟰 Tie'}"
    )
    print(
        f"{'Time (seconds)':<20} {time_bfs:<20.2f} {time_dfs:<20.2f} {'🔵 BFS' if time_bfs < time_dfs else '🔴 DFS' if time_dfs < time_bfs else '🟰 Tie'}"
    )
    print(
        f"{'Accuracy':<20} {(profit_bfs/optimal_profit*100):<20.2f}% {(profit_dfs/optimal_profit*100):<20.2f}% {'🔵 BFS' if profit_bfs > profit_dfs else '🔴 DFS' if profit_dfs > profit_bfs else '🟰 Tie'}"
    )

    print("\n" + "=" * 70)
    print("KEY DIFFERENCES")
    print("=" * 70)
    print("  • BFS: Explores level-by-level (breadth-first)")
    print("  • DFS: Explores depth-first (goes deep before backtracking)")
    print("  • BFS: More memory-intensive (queue grows exponentially)")
    print("  • DFS: Less memory-intensive (stack depth = tree depth)")
    print("  • Both: Same worst-case time complexity O(2^n)")

    # Visualization
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Profit comparison
        ax1 = axes[0]
        algorithms = ["BFS", "DFS", "Optimal"]
        profits = [profit_bfs, profit_dfs, optimal_profit]
        colors = ["#45B7D1", "#FF6B6B", "#2ECC71"]
        bars = ax1.bar(
            algorithms, profits, color=colors, alpha=0.7, edgecolor="black", linewidth=2
        )
        ax1.set_ylabel("Total Profit", fontsize=12)
        ax1.set_title("Profit Comparison", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, profits):
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

        # Plot 2: Nodes explored
        ax2 = axes[1]
        algorithms = ["BFS", "DFS"]
        nodes = [nodes_bfs, nodes_dfs]
        colors = ["#45B7D1", "#FF6B6B"]
        bars = ax2.bar(
            algorithms, nodes, color=colors, alpha=0.7, edgecolor="black", linewidth=2
        )
        ax2.set_ylabel("Nodes Explored", fontsize=12)
        ax2.set_title("Search Efficiency", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, nodes):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:,}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        # Plot 3: Time comparison
        ax3 = axes[2]
        times = [time_bfs, time_dfs]
        bars = ax3.bar(
            algorithms, times, color=colors, alpha=0.7, edgecolor="black", linewidth=2
        )
        ax3.set_ylabel("Time (seconds)", fontsize=12)
        ax3.set_title("Runtime Comparison", fontsize=14, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, times):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}s",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        plt.tight_layout()

        # Save
        output_filename = f"bfs_vs_dfs_p{PROBLEM:02d}.png"
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"\n📊 Comparison visualization saved to: {output_filename}")

        plt.show()


if __name__ == "__main__":
    # Option 1: Run single problem with visualization
    # Note: DFS can handle problems with ~20 items reasonably well
    # Larger problems may take significant time due to exponential complexity
    knapsack_dfs(PROBLEM=13, visualize=True)

    # Option 2: Run multiple small problems
    # run_small_problems()

    # Option 3: Compare BFS vs DFS
    # compare_bfs_dfs(PROBLEM=11, visualize=True)
