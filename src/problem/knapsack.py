from problem.problem import Problem
from typing import List, Tuple, Optional
import os


class Knapsack(Problem):
    """
    A binary Knapsack Problem implementation.
    """

    def __init__(self, PROBLEM_FOLDER: str, PROBLEM: int):
        super().__init__()
        self.capacity, self.items, self.weights, self.solution = (
            self.load_problem_infos(PROBLEM_FOLDER=PROBLEM_FOLDER, PROBLEM=PROBLEM)
        )

    def load_problem_infos(
        self, **kwargs
    ) -> Tuple[int, List[int], List[int], List[int]]:
        """
        kwargs:
            PROBLEM_FOLDER (str): folder path containing problem data files
            PROBLEM (int): problem number

        behavior:
            Reads PROBLEM_FOLDER/p{PROBLEM:02d}_c.txt for capacity
            Reads PROBLEM_FOLDER/p{PROBLEM:02d}_p.txt for item profits
            Reads PROBLEM_FOLDER/p{PROBLEM:02d}_w.txt for item weights
            Reads PROBLEM_FOLDER/p{PROBLEM:02d}_s.txt for solution

        Returns:
            capacity: int
            items: List[int]
            weights: List[int]
            solution: List[int]
        """
        if "PROBLEM_FOLDER" not in kwargs or "PROBLEM" not in kwargs:
            raise ValueError(
                "PROBLEM_FOLDER and PROBLEM must be provided in kwargs")

        PROBLEM_FOLDER = kwargs.get("PROBLEM_FOLDER")
        PROBLEM = kwargs.get("PROBLEM")

        if PROBLEM_FOLDER is None or PROBLEM is None:
            raise ValueError("PROBLEM_FOLDER and PROBLEM cannot be None")

        problem = f"p{PROBLEM:02d}"

        c_path = os.path.join(PROBLEM_FOLDER, problem + "_c.txt")
        with open(c_path) as f:
            capacity = int(f.read().strip())

        p_path = os.path.join(PROBLEM_FOLDER, problem + "_p.txt")
        with open(p_path) as f:
            items = [int(p.strip()) for p in f.readlines() if p.strip()]

        w_path = os.path.join(PROBLEM_FOLDER, problem + "_w.txt")
        with open(w_path) as f:
            weights = [int(w.strip()) for w in f.readlines() if w.strip()]

        s_path = os.path.join(PROBLEM_FOLDER, problem + "_s.txt")
        with open(s_path) as f:
            solution = [int(s.strip()) for s in f.readlines() if s.strip()]

        return capacity, items, weights, solution

    def relativity_to_solution(self, **kwargs) -> float:
        answer = kwargs.get("answer", None)

        if answer is None:
            raise ValueError("answer must be provided in kwargs")

        ans_profit = 0
        sol_profit = 0

        for i in range(len(self.items)):
            if answer[i]:
                ans_profit += self.items[i]
            if self.solution[i]:
                sol_profit += self.items[i]

        return ans_profit / sol_profit

    def calculate_fitness(self, answer) -> float:
        total_weight = 0
        total_profit = 0
        for i in range(len(self.items)):
            if answer[i]:
                total_weight += self.weights[i]
                total_profit += self.items[i]

        if total_weight > self.capacity:
            return 0.0

        return total_profit

    def get_actions(self) -> List[int]:
        return [0, 1]  # 0: not include item, 1: include item

    def get_neighbors(self, current_node: List[int]) -> List[List[int]]:
        neighbors = []
        current_weight = sum(
            self.weights[i] for i in range(len(current_node)) if current_node[i]
        )

        for i in range(len(current_node)):
            if current_node[i] == 0:
                if current_weight + self.weights[i] <= self.capacity:
                    new_state = list(current_node)
                    new_state[i] = 1
                    neighbors.append(tuple(new_state))
        return neighbors
