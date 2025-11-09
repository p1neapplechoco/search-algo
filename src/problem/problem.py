import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional


class Problem(ABC):
    def __init__(self) -> None:
        return

    @abstractmethod
    def get_problem_infos(self, **kwargs) -> Tuple:
        return ()

    @abstractmethod
    def relativity_to_solution(self, **kwargs) -> float:
        return 0.0

    @abstractmethod
    def calculate_fitness(self, **kwargs) -> float:
        return 0.0
