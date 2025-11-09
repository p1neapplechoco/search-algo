"""
Knapsack Problem Solver using BFS (Breadth-First Search)

This module solves the 0/1 Knapsack problem using BFS algorithm.
BFS explores all possible combinations level by level to find the optimal solution.

Properties:
    - Complete: Will find optimal solution if one exists
    - Optimal: Guarantees finding the best solution
    - Time Complexity: O(2^n) in worst case
    - Space Complexity: O(2^n) - exponential space for queue

Note: BFS is not efficient for large knapsack problems due to exponential complexity.
For large problems, consider using dynamic programming or heuristic algorithms.
"""

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


class KnapsackBFS:
    """
    BFS solver for 0/1 Knapsack problem.

    State representation: tuple of (current_item_index, current_selection)
    - current_item_index: next item to consider
    - current_selection: tuple of 0/1 indicating which items are selected so far
    """

    def __init__(self, capacity: int, profits: List[int], weights: List[int]):
        """
        Initialize Knapsack BFS solver.

        Args:
            capacity: Maximum weight capacity
            profits: List of item profits
            weights: List of item weights
        """
        self.capacity = capacity
        self.profits = profits
        self.weights = weights
        self.num_items = len(profits)

        self.best_selection = tuple([0] * self.num_items)
        self.best_profit = 0
        self.nodes_explored = 0

    def solve(self) -> Tuple[Tuple[int, ...], int]:
        """
        Solve knapsack problem using BFS.

        Args:
            max_nodes: Maximum number of nodes to explore (to prevent excessive computation)

        Returns:
            Tuple of (best_selection, best_profit)
        """
        from collections import deque

        # State: (item_index, selection_tuple, current_profit, current_weight)
        initial_state = (0, tuple(), 0, 0)
        queue = deque([initial_state])

        print(f"Starting BFS search...")
        print(f"Items: {self.num_items}, Capacity: {self.capacity}")

        # Use tqdm for progress (based on nodes explored)
        pbar = tqdm(desc="BFS Progress", unit="nodes")

        while queue:
            item_idx, selection, profit, weight = queue.popleft()
            self.nodes_explored += 1
            pbar.update(1)

            # If we've made decision for all items
            if item_idx == self.num_items:
                if profit > self.best_profit:
                    self.best_profit = profit
                    self.best_selection = selection
                continue

            # Branch 1: Don't take current item
            new_selection_0 = selection + (0,)
            queue.append((item_idx + 1, new_selection_0, profit, weight))

            # Branch 2: Take current item (if it fits)
            new_weight = weight + self.weights[item_idx]
            if new_weight <= self.capacity:
                new_profit = profit + self.profits[item_idx]
                new_selection_1 = selection + (1,)
                queue.append((item_idx + 1, new_selection_1, new_profit, new_weight))

        pbar.close()

        return self.best_selection, self.best_profit


def knapsack_bfs(PROBLEM: int, visualize: bool = False):
    """
    Solve knapsack problem using BFS algorithm.

    Args:
        PROBLEM: Problem number (1-18)
        visualize: Whether to show visualization
        max_nodes: Maximum nodes to explore
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

    # Solve using BFS
    start_time = time.time()
    solver = KnapsackBFS(capacity, profits, weights)
    best_selection, best_profit = solver.solve()
    elapsed_time = time.time() - start_time

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
    print(f"Nodes explored: {solver.nodes_explored:,}")
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

            knapsack_bfs(prob, visualize=False, max_nodes=1000000)
            results.append(prob)

        except Exception as e:
            print(f"\nError on Problem {prob}: {e}")

    print("\n" + "=" * 70)
    print(f"Completed {len(results)} problems: {results}")
    print("=" * 70)


if __name__ == "__main__":
    # Option 1: Run single problem with visualization
    # Note: Problem 7 has 23 items - BFS will explore up to 2^23 = 8.4 million states
    # This is computationally expensive but feasible
    knapsack_bfs(PROBLEM=11, visualize=True)

    # Option 2: Run multiple small problems
    # run_small_problems()
