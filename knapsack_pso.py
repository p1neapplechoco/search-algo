from src.swarm_algo.pso import ParticleSwarmOptimizer
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


KNAPSACK_DATA_FOLDER = "data/knapsack/"
MAX_POPULATION = 100
NUM_OF_GENERATIONS = 100


def get_problem_infos(PROBLEM: int) -> Tuple[int, List[int], List[int], List[int]]:
    problem = f"p{PROBLEM:02d}"

    with open(KNAPSACK_DATA_FOLDER + problem + "_c.txt") as f:
        capacity = int(f.read())

    with open(KNAPSACK_DATA_FOLDER + problem + "_p.txt") as f:
        items = [int(p) for p in f.readlines()]

    with open(KNAPSACK_DATA_FOLDER + problem + "_w.txt") as f:
        weights = [int(w) for w in f.readlines()]

    with open(KNAPSACK_DATA_FOLDER + problem + "_s.txt") as f:
        solution = [int(s) for s in f.readlines()]

    return capacity, items, weights, solution


def objective_solution_profit(ans: np.ndarray, items: List[int]) -> int:
    """Tổng lợi nhuận (profit) của lời giải nhị phân ans."""
    return int(np.dot(ans.astype(int), np.array(items, dtype=int)))


# --- Binary mapping + possible repair ---
def binarize_and_repair(x: np.ndarray,
                        capacity: int,
                        items: List[int],
                        weights: List[int]) -> np.ndarray:
    """
    - Binarize: threshold 0.5 on [0,1]^n => {0,1}^n
    - Repair: if overweight, remove items with 'bad' ratio (low profit/weight) first
    """
    x_bin = (x > 0.5).astype(int)
    w = np.array(weights, dtype=int)
    p = np.array(items, dtype=int)

    total_w = int(np.dot(x_bin, w))
    if total_w <= capacity:
        return x_bin

    # List of selected item indices
    chosen = np.where(x_bin == 1)[0].tolist()
    # Sort by profit/weight ratio (bad ones first) and remove until valid
    score = (p[chosen] / np.maximum(w[chosen], 1e-9))
    order = np.argsort(score)  # lowest first
    for idx in order:
        x_bin[chosen[idx]] = 0
        total_w -= int(w[chosen[idx]])
        if total_w <= capacity:
            break
    return x_bin


def knapsack_pso(PROBLEM: int):
    capacity, items, weights, solution = get_problem_infos(PROBLEM)
    num_items = len(items)

    # Target function for PSO (takes continuous vector x in [0,1]^n)
    def objective(x: np.ndarray) -> float:
        # Binarize and repair to ensure feasibility
        x_bin = binarize_and_repair(x, capacity, items, weights)
        # PSO => MIN problem => return -profit to MAX profit
        return -float(objective_solution_profit(x_bin, items))

    # [0,1] bounds for all dimensions
    bounds = [(0.0, 1.0)] * num_items

    # Initialize PSO 
    pso = ParticleSwarmOptimizer(
        objective=objective,
        bounds=bounds,
        n_particles=MAX_POPULATION,
        max_iters=NUM_OF_GENERATIONS,
        mode="constriction",      # or "inertia"
        phi1=2.05, phi2=2.05, chi=0.729,
        topology="ring", ring_neighbors=2,
        velocity_clamp=0.5,       # more stable for knapsack
        boundary_mode="clip",     # [0,1]
        seed=42,
        early_stopping_rounds=None,  # early stopping
        tol=0.0,
        enable_position_history=False,
    )

    # tqdm progress bar
    pbar = tqdm(total=NUM_OF_GENERATIONS, desc=f"PSO Knapsack p{PROBLEM:02d}")
    def _cb(t, X, V, Pbest, gbest_f):
        # gbest_f is MIN value => -gbest_f is current profit
        pbar.n = t + 1
        pbar.set_postfix_str(f"best_profit={int(-gbest_f)}")
        pbar.refresh()
    pso.callback = _cb

    best_x, best_f, info = pso.optimize()
    pbar.close()

    # Map best_x to final feasible binary solution
    best_ans = binarize_and_repair(best_x, capacity, items, weights)
    best_profit = objective_solution_profit(best_ans, items)
    optimal_profit = objective_solution_profit(np.array(solution, dtype=int), items)

    print(f"Problem {PROBLEM}: {num_items} items, capacity {capacity}")
    print(f"PSO solution profit : {best_profit}")
    print(f"Optimal solution    : {optimal_profit}")
    rel = (best_profit / max(1, optimal_profit)) * 100.0
    print(f"Accuracy            : {rel:.2f}%")
    print(f"Items selected      : {int(np.sum(best_ans))}/{num_items}")

    return best_ans, best_profit, info


if __name__ == "__main__":
    knapsack_pso(PROBLEM=18)