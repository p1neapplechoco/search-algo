from search_algo.a_star import AStarProblem, AStar
import numpy as np

class TSPProblem(AStarProblem):
    def __init__(self, distance_matrix: np.ndarray, verbose=True):
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.verbose = verbose

    def generate_initial_state(self):
        return np.random.choice(self.n_cities)
    
    def get_neighbor(self, node):
        return range(self.n_cities)
    
    def g(self, node, neighbor):
        return self.distance_matrix[node, neighbor]
    
    def h