import numpy as np
from problem.problem import Problem
from typing import Tuple, List


class TSP(Problem):
    def __init__(self, PROBLEM_FOLDER: str, PROBLEM: int):
        super().__init__()

        self.distance_matrix, self.solution = self.load_problem_infos(
            PROBLEM_FOLDER=PROBLEM_FOLDER, PROBLEM=PROBLEM
        )

    def load_problem_infos(self, **kwargs) -> Tuple[np.ndarray, List[int]]:
        """
        kwargs:
            PROBLEM_FOLDER (str): folder path containing problem data files
            PROBLEM (int): problem number

        behavior:
            Reads PROBLEM_FOLDER/p{PROBLEM:02d}_d.txt for distance matrix
            Reads PROBLEM_FOLDER/p{PROBLEM:02d}_s.txt for solution path

        Returns:
            capacity: int
            items: List[int]
            weights: List[int]
            solution: List[int]
        """

        PROBLEM_FOLDER = kwargs.get("PROBLEM_FOLDER")
        PROBLEM = kwargs.get("PROBLEM")

        if PROBLEM_FOLDER is None or PROBLEM is None:
            raise ValueError("PROBLEM_FOLDER and PROBLEM cannot be None")

        problem = f"p{PROBLEM:02d}"

        # Read distance matrix
        with open(PROBLEM_FOLDER + problem + "_d.txt") as f:
            lines = f.readlines()
            distance_matrix = []
            for line in lines:
                row = [float(x) for x in line.split()]
                distance_matrix.append(row)
            distance_matrix = np.array(distance_matrix)

        # Read solution path
        with open(PROBLEM_FOLDER + problem + "_s.txt") as f:
            solution = [
                int(s.strip()) - 1 for s in f.readlines()
            ]  # Convert to 0-indexed

        return distance_matrix, solution

    def relativity_to_solution(self, answer: List[int]) -> float:
        ans_distance = 0

        for i in range(len(answer) - 1):
            ans_distance += self.distance_matrix[answer[i], answer[i + 1]]

        sol_distance = 0
        for i in range(len(self.solution) - 1):
            sol_distance += self.distance_matrix[self.solution[i], self.solution[i + 1]]

        return sol_distance / ans_distance

    def calculate_fitness(self, answer: List[int]) -> float:
        total_distance = 0

        for i in range(len(answer) - 1):
            total_distance += self.distance_matrix[answer[i], answer[i + 1]]

        return total_distance
