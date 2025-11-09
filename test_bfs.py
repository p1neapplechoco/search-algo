"""
Example usage of BFS (Breadth-First Search) algorithm

This demonstrates BFS for different types of problems:
1. Simple graph traversal
2. Shortest path finding
3. Grid-based pathfinding
4. State space search
"""

from src.search_algo.bfs import BFS
import numpy as np
from typing import List, Tuple


def example_1_simple_graph():
    """
    Example 1: Simple graph traversal
    Find path from node 0 to node 6
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Graph Traversal")
    print("=" * 70)

    # Define graph as adjacency list
    graph = {0: [1, 2], 1: [0, 3, 4], 2: [0, 5], 3: [1], 4: [1, 6], 5: [2], 6: [4]}

    # Define neighbor function
    def get_neighbors(node):
        return graph.get(node, [])

    # Initialize BFS
    bfs = BFS(start_state=0, get_neighbors=get_neighbors)

    # Search for node 6
    def goal_test(node):
        return node == 6

    path = bfs.search(goal_test)

    print(f"Graph: {graph}")
    print(f"Start: 0, Goal: 6")
    print(f"Path found: {path}")
    print(f"Path length: {len(path) if path else 'No path'}")
    print(f"Nodes visited: {bfs.visited_count}")


def example_2_grid_pathfinding():
    """
    Example 2: Grid-based pathfinding
    Find shortest path in a 2D grid with obstacles
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Grid Pathfinding (with obstacles)")
    print("=" * 70)

    # Define grid (0 = free, 1 = obstacle)
    grid = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    start = (0, 0)
    goal = (4, 4)

    print("Grid (0=free, 1=obstacle):")
    print(grid)
    print(f"Start: {start}, Goal: {goal}")

    # Define neighbor function for grid
    def get_neighbors(state):
        x, y = state
        neighbors = []
        # 4-directional movement (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if grid[nx, ny] == 0:  # Not an obstacle
                    neighbors.append((nx, ny))
        return neighbors

    # Initialize BFS
    bfs = BFS(start_state=start, get_neighbors=get_neighbors)

    # Search for goal
    def goal_test(state):
        return state == goal

    path = bfs.search(goal_test)

    if path:
        print(f"\nPath found with {len(path)} steps:")
        print(f"Path: {path}")
        print(f"Nodes visited: {bfs.visited_count}")

        # Visualize path on grid
        visual_grid = grid.copy().astype(str)
        visual_grid[visual_grid == "0"] = "."
        visual_grid[visual_grid == "1"] = "#"

        for i, (x, y) in enumerate(path):
            if i == 0:
                visual_grid[x, y] = "S"
            elif i == len(path) - 1:
                visual_grid[x, y] = "G"
            else:
                visual_grid[x, y] = "*"

        print("\nVisualized path (S=start, G=goal, *=path, #=obstacle):")
        for row in visual_grid:
            print(" ".join(row))
    else:
        print("No path found!")


def example_3_8_puzzle():
    """
    Example 3: 8-Puzzle problem (simplified)
    Find sequence of moves to reach goal configuration
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: 8-Puzzle Problem (3x3 sliding puzzle)")
    print("=" * 70)

    # State representation: tuple of 9 numbers (0 = empty space)
    start_state = (1, 2, 3, 4, 0, 5, 6, 7, 8)
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def print_puzzle(state):
        """Print puzzle in 3x3 format"""
        for i in range(0, 9, 3):
            row = state[i : i + 3]
            print("  ".join(["_" if x == 0 else str(x) for x in row]))

    print("Start state:")
    print_puzzle(start_state)
    print("\nGoal state:")
    print_puzzle(goal_state)

    # Define neighbor function (possible moves)
    def get_neighbors(state):
        neighbors = []
        state_list = list(state)
        zero_idx = state_list.index(0)

        # Convert to 2D coordinates
        row, col = zero_idx // 3, zero_idx % 3

        # Possible moves: up, down, left, right
        moves = []
        if row > 0:
            moves.append(-3)  # up
        if row < 2:
            moves.append(3)  # down
        if col > 0:
            moves.append(-1)  # left
        if col < 2:
            moves.append(1)  # right

        for move in moves:
            new_idx = zero_idx + move
            new_state = state_list[:]
            new_state[zero_idx], new_state[new_idx] = (
                new_state[new_idx],
                new_state[zero_idx],
            )
            neighbors.append(tuple(new_state))

        return neighbors

    # Initialize BFS
    bfs = BFS(
        start_state=start_state, get_neighbors=get_neighbors, max_iterations=10000
    )

    # Search for goal
    def goal_test(state):
        return state == goal_state

    print(f"\nSearching for solution...")
    path = bfs.search(goal_test)

    if path:
        print(f"\nSolution found in {len(path) - 1} moves!")
        print(f"Nodes visited: {bfs.visited_count}")
        print(f"\nShowing first 5 steps:")
        for i, state in enumerate(path[:5]):
            print(f"\nStep {i}:")
            print_puzzle(state)
    else:
        print("No solution found within iteration limit")


def example_4_optimization_problem():
    """
    Example 4: Using BFS for optimization
    Find state with minimal fitness value
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Optimization Problem (minimize sum of squares)")
    print("=" * 70)

    # State: tuple of integers
    start_state = (5, 5, 5)

    print(f"Start state: {start_state}")
    print(f"Objective: Minimize sum of squares (goal: (0,0,0) with value 0)")

    # Define neighbor function (increment/decrement each dimension)
    def get_neighbors(state):
        neighbors = []
        for i in range(len(state)):
            # Decrement
            if state[i] > 0:
                new_state = list(state)
                new_state[i] -= 1
                neighbors.append(tuple(new_state))
            # Increment (with limit)
            if state[i] < 10:
                new_state = list(state)
                new_state[i] += 1
                neighbors.append(tuple(new_state))
        return neighbors

    # Define fitness function
    def fitness_function(state):
        return sum(x**2 for x in state)

    # Initialize BFS
    bfs = BFS(start_state=start_state, get_neighbors=get_neighbors, max_iterations=5000)
    bfs.set_fitness_function(fitness_function)

    # Run BFS to find best state
    best_state, best_fitness = bfs.search_best(max_depth=10)

    print(f"\nStart fitness: {fitness_function(start_state)}")
    print(f"Best state found: {best_state}")
    print(f"Best fitness: {best_fitness}")
    print(f"Nodes visited: {bfs.visited_count}")


def example_5_word_ladder():
    """
    Example 5: Word Ladder problem
    Transform one word to another by changing one letter at a time
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Word Ladder Problem")
    print("=" * 70)

    # Small dictionary
    dictionary = {"cat", "cot", "cog", "dog", "dot", "bat", "bot", "bit", "big"}

    start_word = "cat"
    goal_word = "dog"

    print(f"Dictionary: {sorted(dictionary)}")
    print(f"Transform '{start_word}' to '{goal_word}'")

    # Define neighbor function
    def get_neighbors(word):
        neighbors = []
        for i in range(len(word)):
            for c in "abcdefghijklmnopqrstuvwxyz":
                if c != word[i]:
                    new_word = word[:i] + c + word[i + 1 :]
                    if new_word in dictionary:
                        neighbors.append(new_word)
        return neighbors

    # Initialize BFS
    bfs = BFS(start_state=start_word, get_neighbors=get_neighbors)

    # Search for goal
    def goal_test(word):
        return word == goal_word

    path = bfs.search(goal_test)

    if path:
        print(f"\nTransformation found in {len(path) - 1} steps:")
        for i, word in enumerate(path):
            print(f"  Step {i}: {word}")
        print(f"Nodes visited: {bfs.visited_count}")
    else:
        print("No transformation found!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BFS (BREADTH-FIRST SEARCH) ALGORITHM EXAMPLES")
    print("=" * 70)

    # Run all examples
    example_1_simple_graph()
    example_2_grid_pathfinding()
    example_3_8_puzzle()
    example_4_optimization_problem()
    example_5_word_ladder()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
