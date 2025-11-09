import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from math import gamma
from typing import Any, Callable, Optional


class DepthFirstSearch:
    def __init__(
        self,
        start_node: Any,
        get_neighbors: Callable[[Any], list[Any]],
    ):
        self.start_node = start_node
        self.get_neighbors = get_neighbors
        self.visited = set()
        self.stack = [start_node]

        self.fitness_function = None
        self.best_node = None
        self.best_fitness = float("-inf")

    def set_fitness_function(
        self,
        fitness_function: Callable[[Any], float],
    ):
        self.fitness_function = fitness_function

    def run(
        self,
        max_iterations: Optional[int] = 10000,
        visualize: bool = False,
    ):
        iterations = 0

        while self.stack and (max_iterations is None or iterations < max_iterations):
            current_node = self.stack.pop()

            if current_node not in self.visited:
                self.visited.add(current_node)

                # Evaluate fitness
                if self.fitness_function:
                    current_fitness = self.fitness_function(current_node)
                    if current_fitness > self.best_fitness:
                        self.best_fitness = current_fitness
                        self.best_node = current_node

                neighbors = self.get_neighbors(current_node)
                for neighbor in neighbors:
                    if neighbor not in self.visited:
                        self.stack.append(neighbor)
            iterations += 1

        if visualize:
            self.visualize_search_tree()

        return self.best_node, self.best_fitness

    def visualize_search_tree(self, max_nodes: int = 100):
        """
        Visualize the DFS search tree (for small state spaces).

        Args:
            max_nodes: Maximum number of nodes to visualize
        """
        try:
            import networkx as nx

            G = nx.DiGraph()
            stack = [(self.start_node, None, 0)]  # (node, parent, depth)
            visited = set()
            node_count = 0

            state_to_tuple = lambda s: (
                tuple(s) if isinstance(s, (list, np.ndarray)) else s
            )

            while stack and node_count < max_nodes:
                current_node, parent, depth = stack.pop()  # DFS: pop from end
                current_tuple = state_to_tuple(current_node)

                if current_tuple in visited:
                    continue

                visited.add(current_tuple)
                node_count += 1

                # Add node
                node_label = str(current_node)[:20]  # Truncate long labels
                G.add_node(node_label, depth=depth)

                if parent is not None:
                    G.add_edge(parent, node_label)

                # Add neighbors (in reverse order for DFS)
                try:
                    neighbors = self.get_neighbors(current_node)
                    for neighbor in neighbors:
                        neighbor_tuple = state_to_tuple(neighbor)
                        if neighbor_tuple not in visited:
                            stack.append((neighbor, node_label, depth + 1))
                except:
                    continue

            # Draw graph
            pos = nx.spring_layout(G, seed=42)
            plt.figure(figsize=(14, 10))

            # Color nodes by depth
            depths = [G.nodes[node].get("depth", 0) for node in G.nodes()]

            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color=depths,
                cmap="plasma",  # Different colormap for DFS
                node_size=800,
                font_size=7,
                font_weight="bold",
                edge_color="gray",
                arrowsize=15,
            )
            plt.title(
                f"DFS Search Tree (showing {node_count} nodes)",
                fontsize=16,
                fontweight="bold",
            )

            # Add colorbar to show depth
            sm = cm.ScalarMappable(
                cmap="plasma", norm=Normalize(vmin=min(depths), vmax=max(depths))
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label("Depth Level", rotation=270, labelpad=20)

            plt.tight_layout()

            # Save to file
            output_file = "dfs_search_tree.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"   Search tree saved to: {output_file}")

            plt.show()

        except ImportError:
            print("NetworkX not installed. Cannot visualize search tree.")
            print("Install with: pip install networkx")
