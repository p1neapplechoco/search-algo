import numpy as np


# Problem
class Problem:
    def __init__(self, dimension, lb, ub):
        self.dimension = dimension
        self.lb = lb
        self.ub = ub

    def evaluate(self, solution):
        raise NotImplementedError


class Sphere(Problem):
    def evaluate(self, solution):
        return np.sum(solution ** 2)
    

class Rastrigin(Problem):
    def __init__(self, dimension, lb=-5.12, ub=5.12):
        super().__init__(dimension, lb, ub)
        self.A = 10

    def evaluate(self, solution):
        return self.A * self.dimension + np.sum(solution**2 - self.A * np.cos(2 * np.pi * solution))


class Rosenbrock(Problem):
    def __init__(self, dimension, lb=-5, ub=10):
        super().__init__(dimension, lb, ub)
        if dimension < 2:
             raise ValueError("Rosenbrock function requires at least 2 dimensions.")

    def evaluate(self, solution):
        term1 = solution[1:] - solution[:-1]**2
        term2 = solution[:-1] - 1
        return np.sum(100 * term1**2 + term2**2)


class Ackley(Problem):
    def __init__(self, dimension, lb=-32.768, ub=32.768):
        super().__init__(dimension, lb, ub)
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi

    def evaluate(self, solution):
        n = self.dimension
        sum_sq_term = -self.b * np.sqrt(np.sum(solution**2) / n)
        sum_cos_term = np.sum(np.cos(self.c * solution)) / n
        return -self.a * np.exp(sum_sq_term) - np.exp(sum_cos_term) + self.a + np.exp(1)


# Optimizer
class ABC:
    def __init__(self, problem: Problem, sn: int, mcn: int, limit: int):
        self.problem = problem
        self.sn = sn
        self.mcn = mcn
        self.limit = limit
        self.dimension = problem.dimension
        
        self.solutions = np.array([self.randomize_solution() for _ in range(sn)])
        self.fitness = np.array([self.calc_fitness(self.solutions[i]) for i in range(self.sn)])
        self.probs = np.zeros(self.sn)

    def randomize_solution(self):
        lb, ub = self.problem.lb, self.problem.ub
        return lb + np.random.rand(self.dimension) * (ub - lb)

    def calc_fitness(self, solution):
        value = self.problem.evaluate(solution)
        fitness = 1 / (1 + value) if value >= 0 else 1 + np.abs(value)
        return fitness
    
    def calc_probabilities(self):        
        maxfit = np.max(self.fitness) 
        self.probs = 0.1 + 0.9 * (self.fitness / maxfit)
    
    def local_search(self, i):
        idxs = np.delete(np.arange(self.sn), i)
        k = np.random.choice(idxs)
        j = np.random.randint(self.dimension)
        phi = np.random.uniform(-1, 1)

        new_solution = np.copy(self.solutions[i])
        new_solution[j] = new_solution[j] + phi * (new_solution[j] - self.solutions[k, j])
        new_solution[j] = np.clip(new_solution[j], self.problem.lb, self.problem.ub)
        
        return new_solution

    def run(self):
        cyc = 1
        trial = np.zeros(self.sn)
        global_best = {
            'solution': None,
            'fitness': -np.inf
        }
        
        while cyc < self.mcn:
            # Employed bees phase
            for i in range(self.sn):
                new_solution = self.local_search(i)
                new_fit = self.calc_fitness(new_solution)

                if new_fit > self.fitness[i]:
                    self.solutions[i] = new_solution
                    self.fitness[i] = new_fit
                    trial[i] = 0
                else:
                    trial[i] += 1
            
            self.calc_probabilities()
            
            # Onlooker bees phase
            i = 0
            t = 0
            while t < self.sn:
                if np.random.rand(1) < self.probs[i]:
                    t += 1
                    new_solution = self.local_search(i)
                    new_fit = self.calc_fitness(new_solution)

                    if new_fit > self.fitness[i]:
                        self.solutions[i] = new_solution
                        self.fitness[i] = new_fit
                        trial[i] = 0
                    else:
                        trial[i] += 1
                
                i = (i + 1) % self.sn
                
            best_idx = np.argmax(self.fitness)
            if self.fitness[best_idx] > global_best['fitness']:
                global_best['solution'] = np.copy(self.solutions[best_idx])
                global_best['fitness'] = self.fitness[best_idx]
            
            # Scout bee phase
            si = np.argmax(trial)
            if trial[si] > self.limit:
                self.solutions[si] = self.randomize_solution()
                self.fitness[si] = self.calc_fitness(self.solutions[si])
                trial[si] = 0
                
            cyc += 1

        return global_best['solution'], global_best['fitness'], self.problem.evaluate(global_best['solution'])


if __name__ == "__main__":
    # Khởi tạo các bài toán
    sphere_problem = Sphere(dimension=5, lb=-5.12, ub=5.12)
    rastrigin_problem = Rastrigin(dimension=5, lb=-5.12, ub=5.12)
    rosenbrock_problem = Rosenbrock(dimension=5, lb=-5, ub=10)
    ackley_problem = Ackley(dimension=5, lb=-32.768, ub=32.768)

    # Tạo một lời giải mẫu (ví dụ: toàn số 0)
    zero_solution = np.zeros(5)
    one_solution = np.ones(5) # Lời giải tối ưu cho Rosenbrock

    # Đánh giá lời giải
    print(f"Sphere(0,0,0,0,0) = {sphere_problem.evaluate(zero_solution)}") # Kết quả: 0.0
    print(f"Rastrigin(0,0,0,0,0) = {rastrigin_problem.evaluate(zero_solution)}") # Kết quả: 0.0
    print(f"Rosenbrock(0,0,0,0,0) = {rosenbrock_problem.evaluate(zero_solution)}") # Kết quả: 4.0 (cho 5 chiều)
    print(f"Rosenbrock(1,1,1,1,1) = {rosenbrock_problem.evaluate(one_solution)}") # Kết quả: 0.0
    print(f"Ackley(0,0,0,0,0) = {ackley_problem.evaluate(zero_solution)}") # Kết quả: Gần 0 (do sai số float)

    # --- Sử dụng với thuật toán ABC---
    print("\n--- Chạy ABC với Sphere ---")
    abc_sphere = ABC(sphere_problem, sn=30, mcn=1000, limit=50)
    best_sol, best_fit, best_val = abc_sphere.run()
    print(f"Solution: {np.round(best_sol, 5)}") # Nên gần vector 0
    print(f"Fitness: {best_fit}")             # Nên gần 1.0
    print(f"Sphere Value: {best_val}")      # Nên gần 0.0
    
    print("\n--- Chạy ABC với Rastrigin ---")
    abc_rastrigin = ABC(rastrigin_problem, sn=50, mcn=2000, limit=100) # Tăng thông số
    best_sol, best_fit, best_val = abc_rastrigin.run()
    print(f"Solution: {np.round(best_sol, 5)}") # Nên gần vector 0
    print(f"Fitness: {best_fit}")             # Nên gần 1.0
    print(f"Rastrigin Value: {best_val}")      # Nên gần 0.0

    print("\n--- Chạy ABC với Rosenbrock ---")
    abc_rosenbrock = ABC(rosenbrock_problem, sn=100, mcn=5000, limit=200) # Rosenbrock khó hơn, cần nhiều hơn
    best_sol, best_fit, best_val = abc_rosenbrock.run()
    print(f"Solution: {np.round(best_sol, 5)}") # Nên gần vector 1
    print(f"Fitness: {best_fit}")             # Nên gần 1.0
    print(f"Rosenbrock Value: {best_val}")      # Nên gần 0.0

    print("\n--- Chạy ABC với Ackley ---")
    abc_ackley = ABC(ackley_problem, sn=50, mcn=3000, limit=150)
    best_sol, best_fit, best_val = abc_ackley.run()
    print(f"Solution: {np.round(best_sol, 5)}") # Nên gần vector 0
    print(f"Fitness: {best_fit}")             # Nên gần 1.0
    print(f"Ackley Value: {best_val}")      # Nên gần 0.0
