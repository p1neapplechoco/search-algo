import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
from copy import deepcopy
import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class Node:
    """Node in A* algorithm"""
    f_score: float
    g_score: float = field(compare=False)
    h_score: float = field(compare=False)
    state: Any = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)

    def __hash__(self):
        return hash(str(self.state))


class AStarProblem(ABC):
    @abstractmethod
    def h(self, state) -> float:
        """Estimate the distance from state to goal"""
        pass

    @abstractmethod
    def generate_initial_state(self) -> Any:
        """Initialize state"""
        pass

    @abstractmethod
    def get_neighbor(self, state) -> List[Any]:
        """Get the neighbors of current state"""
        pass

    @abstractmethod
    def is_terminate(self, state) -> bool:
        """Check if state is goal"""
        pass

    @abstractmethod
    def state_to_string(self, state: Any) -> str:
        pass


class AStar:
    def __init__(self, problem: AStarProblem, verbose=True):
        self.problem = problem
        self.verbose = verbose

    def solve(self):
        start_state = self.problem.generate_initial_state()
        h_start = self.problem.h(start_state)

        start_node = Node(
            f_score=h_start,
            g_score=0,
            h_score=h_start,
            state=start_state
        )

        open_list = [start_node]

        closed_set = set()

        g_scores = {self.problem.state_to_string(start_state): 0.0}

        nodes_expanded = 0
        max_queue_size = 1

        while open_list:

            current_node: Node = heapq.heappop(open_list)
            current_state_str = self.problem.state_to_string(
                current_node.state)

            if current_state_str in closed_set:
                continue

            closed_set.add(current_state_str)
            nodes_expanded += 1

            if self.problem.is_terminate(current_node.state):
                path = self._reconstruct_path(current_node)
                return path, current_node.g_score

            # Expand successors
            for neighbor_state, step_cost in self.problem.get_neighbor(current_node.state):
                neighbor_state_str = self.problem.state_to_string(
                    neighbor_state)

                if neighbor_state_str in closed_set:
                    continue

                tentative_g = current_node.g_score + step_cost

                if neighbor_state_str in g_scores and tentative_g >= g_scores[neighbor_state_str]:
                    continue

                h_score = self.problem.h(neighbor_state)
                f_score = tentative_g + h_score

                next_node = Node(
                    f_score=f_score,
                    g_score=tentative_g,
                    h_score=h_score,
                    state=neighbor_state,
                    parent=current_node
                )

                g_scores[neighbor_state_str] = tentative_g
                heapq.heappush(open_list, next_node)

            max_queue_size = max(max_queue_size, len(open_list))
        return None, float('inf')

    def _reconstruct_path(self, node: Node) -> List[Any]:
        """Reconstruct path từ goal về start"""
        path = []
        current = node
        while current is not None:
            path.append(current.state)
            current = current.parent
        return list(reversed(path))


class TSPProblem(AStarProblem):
    """Traveling Salesman Problem"""

    def __init__(self, cities: List[Tuple[float, float]], start_city: int = 0):
        """
        cities: List of (x, y) coordinates
        start_city: Index of starting city
        """
        self.cities = cities
        self.n = len(cities)
        self.start_city = start_city

        # Precompute distance matrix
        self.distances = self._compute_distances()

    def _compute_distances(self):
        colony3d = np.array(self.cities)[:, np.newaxis, :]
        colony3d_transform = np.array(self.cities)[np.newaxis, :, :]

        distance_matrix = np.sqrt(
            np.sum((colony3d - colony3d_transform) ** 2, axis=2)).tolist()
        return distance_matrix

    def generate_initial_state(self) -> Tuple[int, frozenset]:
        return (self.start_city, frozenset([self.start_city]))

    def is_terminate(self, state: Tuple[int, frozenset]) -> bool:
        """Goal khi đã visit tất cả các city và về start"""
        current_city, visited = state
        return len(visited) == self.n and current_city == self.start_city

    def get_neighbor(self, state: Tuple[int, frozenset]) -> List[Tuple[Any, float]]:
        """Lấy các thành phố kế tiếp có thể đi"""
        current_city, visited = state
        successors = []

        # Nếu đã visit hết, chỉ có thể về start
        if len(visited) == self.n:
            if current_city != self.start_city:
                next_state = (self.start_city, visited)
                cost = self.distances[current_city][self.start_city]
                successors.append((next_state, cost))
        else:
            # Đi đến các city chưa visit
            for next_city in range(self.n):
                if next_city not in visited:
                    next_visited = visited | {next_city}
                    next_state = (next_city, next_visited)
                    cost = self.distances[current_city][next_city]
                    successors.append((next_state, cost))

        return successors

    def h(self, state: Tuple[int, frozenset]) -> float:
        """MST heuristic - minimum spanning tree of unvisited cities"""
        current_city, visited = state
        unvisited = set(range(self.n)) - visited

        if not unvisited:
            # Nếu đã visit hết, return khoảng cách về start
            if current_city != self.start_city:
                return self.distances[current_city][self.start_city]
            return 0.0

        # Simple heuristic: nearest unvisited city + min distance back to start
        min_to_unvisited = min(
            self.distances[current_city][city] for city in unvisited)
        min_from_unvisited_to_start = min(
            self.distances[city][self.start_city] for city in unvisited)

        return min_to_unvisited + min_from_unvisited_to_start

    def state_to_string(self, state: Tuple[int, frozenset]) -> str:
        """Convert state to string"""
        current_city, visited = state
        return f"{current_city}:{sorted(visited)}"



