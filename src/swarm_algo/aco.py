import numpy as np

# [[,].
#  [,],
#  [,],
#  [,]]
# 4,1,2    1,4,2
# ant_path
# [x1,x2],[y1,y2]


class AntColonyOptimizer:
    def __init__(self, colony, num_ant, iter, alpha, beta, rho, Q):
        self.colony = colony
        self.num_ant = num_ant
        self.iter = iter
        self.alpha = alpha  # pheromone importance
        self.beta = beta  # distance importance
        self.rho = rho  # evaporation rate
        self.zeta, self.distance_matrix = self.cal_distance()
        self.Q = Q
        self.theta_matrix = np.ones((len(self.colony), len(self.colony)))
        self.n_colonies = len(self.colony)

    def cal_distance(self):
        colony3d = self.colony[:, np.newaxis, :]
        colony3d_transform = self.colony[np.newaxis, :, :]

        distance_matrix = np.sqrt(np.sum((colony3d - colony3d_transform) ** 2, axis=2))
        zeta = np.where(distance_matrix > 0, 1.0 / distance_matrix, 0)
        return zeta, distance_matrix

    def path_to_ant_path(self, path):

        from_cities = np.array(path)
        to_cities = np.array(path[1:] + [path[0]])
        return (from_cities, to_cities)

    def fitness(self, ant_path):
        return np.sum(self.distance_matrix[ant_path[0], ant_path[1]])

    def update_pheromone(self, ant_paths):
        self.theta_matrix *= 1 - self.rho
        for path in ant_paths:
            delta = self.Q / self.distance_matrix[path[0], path[1]]
            self.theta_matrix[path[0], path[1]] += delta
            self.theta_matrix[path[1], path[0]] += delta

    def RWS(self, idx, visitted):
        theta = self.theta_matrix[idx].copy()
        eta = self.zeta[idx].copy()

        visitted = list(visitted)
        theta[visitted] = 0
        eta[visitted] = 0

        probs = theta**self.alpha * eta**self.beta
        prob_sum = np.sum(probs)
        if prob_sum == 0:
            unvisited = [i for i in range(self.n_colonies) if i not in visitted]
            if unvisited:
                return np.random.choice(unvisited)
            return None
        probs = probs / prob_sum

        next_city = np.random.choice(self.n_colonies, p=probs)
        return next_city

    def run(self, verbose=True):
        best_path = None
        best_fitness = float("inf")

        for i in range(self.iter):
            all_paths = []
            all_ant_paths = []
            all_fitness = []

            for ant in range(self.num_ant):
                cur_idx = np.random.randint(0, len(self.colony))
                visitted = {cur_idx}
                path = [cur_idx]

                while len(visitted) < self.n_colonies:
                    cur_idx = self.RWS(cur_idx, visitted)
                    if cur_idx is None:  # ✅ PHẢI CHECK
                        break
                    path.append(cur_idx)
                    visitted.add(cur_idx)

                if len(path) == self.n_colonies:
                    ant_path = self.path_to_ant_path(path)
                    fit = self.fitness(ant_path)

                    all_paths.append(path)
                    all_ant_paths.append(ant_path)
                    all_fitness.append(fit)

                    if fit < best_fitness:
                        best_fitness = fit
                        best_path = path.copy()

            if all_ant_paths:
                self.update_pheromone(all_ant_paths)

                if verbose and (i + 1) % 5 == 0:
                    print(f"Iteration {i+1:3d}: Best = {best_fitness:.2f}")
        return best_path, best_fitness
