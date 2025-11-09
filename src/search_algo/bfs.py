import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Any, Callable, List, Set, Tuple, Optional


class BFS:
    """
    Breadth-First Search (BFS) algorithm for finding optimal solutions in discrete state spaces.
    """

    def __init__(
        self,
        start_state: Any,
        get_neighbors: Optional[Callable] = None,
        max_iterations: int = 10000,
    ):
        """
        Initialize BFS algorithm.

        Args:
            start_state: Initial state to begin search from
            get_neighbors: Function that returns list of neighbor states from current state
            max_iterations: Maximum number of iterations to prevent infinite loops
        """
        self.start_state = start_state
        self.get_neighbors_fn = get_neighbors
        self.max_iterations = max_iterations

        self.fitness_function = None
        self.visited_count = 0
        self.path_length = 0

    def set_fitness_function(self, fitness_function: Callable):
        """
        Set the fitness function for evaluating states.

        Args:
            fitness_function: Function that evaluates quality of a state (lower is better)
        """
        self.fitness_function = fitness_function

    def set_get_neighbors(self, get_neighbors: Callable):
        """
        Set the function to get neighbors of a state.

        Args:
            get_neighbors: Function that takes a state and returns list of neighbor states
        """
        self.get_neighbors_fn = get_neighbors

    def get_neighbors(self, state: Any) -> List[Any]:
        """
        Get all neighboring states from the current state.

        Args:
            state: Current state

        Returns:
            List of neighboring states
        """
        if self.get_neighbors_fn is None:
            raise ValueError(
                "get_neighbors function not set. Use set_get_neighbors() or pass it to __init__"
            )
        return self.get_neighbors_fn(state)

    def search(
        self, goal_test: Callable, state_to_tuple: Optional[Callable] = None
    ) -> Optional[List]:
        """
        Perform BFS to find the goal state.

        Args:
            goal_test: Function that returns True if state is a goal state
            state_to_tuple: Optional function to convert state to hashable tuple for visited set

        Returns:
            Path from initial state to goal state, or None if no path found
        """
        # Convert state to hashable type for visited set
        if state_to_tuple is None:
            state_to_tuple = lambda s: (
                tuple(s) if isinstance(s, (list, np.ndarray)) else s
            )

        # Initialize queue with (state, path) tuples
        queue = deque([(self.start_state, [self.start_state])])
        visited = set()
        visited.add(state_to_tuple(self.start_state))

        iterations = 0

        while queue and iterations < self.max_iterations:
            iterations += 1
            current_state, path = queue.popleft()

            # Check if goal reached
            if goal_test(current_state):
                self.visited_count = len(visited)
                self.path_length = len(path)
                return path

            # Explore neighbors
            try:
                neighbors = self.get_neighbors(current_state)
            except Exception as e:
                print(f"Error getting neighbors: {e}")
                continue

            for neighbor in neighbors:
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    queue.append((neighbor, path + [neighbor]))

        self.visited_count = len(visited)
        return None  # No path found

    def search_best(
        self, max_depth: Optional[int] = None, state_to_tuple: Optional[Callable] = None
    ) -> Tuple[Any, float]:
        """
        Perform BFS to find the best state (lowest fitness) up to max_depth.

        Args:
            max_depth: Maximum depth to explore (None for unlimited)
            state_to_tuple: Optional function to convert state to hashable tuple

        Returns:
            Tuple of (best_state, best_fitness)
        """
        if self.fitness_function is None:
            raise ValueError("Fitness function not set. Use set_fitness_function()")

        if state_to_tuple is None:
            state_to_tuple = lambda s: (
                tuple(s) if isinstance(s, (list, np.ndarray)) else s
            )

        queue = deque([(self.start_state, 0)])  # (state, depth)
        visited = set()
        visited.add(state_to_tuple(self.start_state))

        best_state = self.start_state
        best_fitness = self.fitness_function(self.start_state)

        iterations = 0

        while queue and iterations < self.max_iterations:
            iterations += 1
            current_state, depth = queue.popleft()

            # Evaluate current state
            fitness = self.fitness_function(current_state)
            if fitness < best_fitness:
                best_fitness = fitness
                best_state = current_state

            # Check depth limit
            if max_depth is not None and depth >= max_depth:
                continue

            # Explore neighbors
            try:
                neighbors = self.get_neighbors(current_state)
            except Exception as e:
                print(f"Error getting neighbors: {e}")
                continue

            for neighbor in neighbors:
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    queue.append((neighbor, depth + 1))

        self.visited_count = len(visited)
        return best_state, best_fitness

    def run(
        self, goal_value: float = 0.0, state_to_tuple: Optional[Callable] = None
    ) -> Tuple[Any, float]:
        """
        Run BFS algorithm to find state with fitness <= goal_value.

        Args:
            goal_value: Target fitness value (default: 0.0)
            state_to_tuple: Optional function to convert state to hashable tuple

        Returns:
            Tuple of (best_state, best_fitness)
        """
        if self.fitness_function is None:
            raise ValueError("Fitness function not set. Use set_fitness_function()")

        def goal_test(state):
            if self.fitness_function is None:
                return False
            return self.fitness_function(state) <= goal_value

        path = self.search(goal_test, state_to_tuple)

        if path is not None:
            best_state = path[-1]
            best_fitness = self.fitness_function(best_state)
            return best_state, best_fitness
        else:
            # If no path to goal, return best state found
            return self.search_best(state_to_tuple=state_to_tuple)

    def visualize_search_tree(self, max_nodes: int = 100):
        """
        Visualize the BFS search tree (for small state spaces).

        Args:
            max_nodes: Maximum number of nodes to visualize
        """
        try:
            import networkx as nx

            G = nx.DiGraph()
            queue = deque([(self.start_state, None, 0)])
            visited = set()
            node_count = 0

            state_to_tuple = lambda s: (
                tuple(s) if isinstance(s, (list, np.ndarray)) else s
            )

            while queue and node_count < max_nodes:
                current_state, parent, depth = queue.popleft()
                current_tuple = state_to_tuple(current_state)

                if current_tuple in visited:
                    continue

                visited.add(current_tuple)
                node_count += 1

                # Add node
                node_label = str(current_state)[:20]  # Truncate long labels
                G.add_node(node_label, depth=depth)

                if parent is not None:
                    G.add_edge(parent, node_label)

                # Add neighbors
                try:
                    neighbors = self.get_neighbors(current_state)
                    for neighbor in neighbors:
                        if state_to_tuple(neighbor) not in visited:
                            queue.append((neighbor, node_label, depth + 1))
                except:
                    continue

            # Draw graph
            pos = nx.spring_layout(G)
            plt.figure(figsize=(12, 8))
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color="lightblue",
                node_size=500,
                font_size=8,
            )
            plt.title(f"BFS Search Tree (showing {node_count} nodes)")
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("NetworkX not installed. Cannot visualize search tree.")
            print("Install with: pip install networkx")
