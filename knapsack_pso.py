# knapsack — PSO with decode adapter + visualization
from src.swarm_algo.pso import ParticleSwarmOptimizer
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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
    """Total profit of a binary knapsack solution."""
    return int(np.dot(ans.astype(int), np.array(items, dtype=int)))


# --------- Decode adapter (search-space x in [0,1]^n -> feasible {0,1}^n) ---------
def make_knapsack_decode(capacity: int, items: List[int], weights: List[int]):
    """
    Returns a decoder:
        x (float vector in [0,1]^n) -> x_bin (feasible 0/1 vector)
    Strategy:
      - threshold at 0.5 to get 0/1,
      - if overweight, drop items with lowest profit/weight first.
    """
    w = np.array(weights, dtype=float)
    p = np.array(items, dtype=float)
    ratio = p / np.maximum(w, 1e-12)

    def decode(x: np.ndarray) -> np.ndarray:
        x_bin = (x > 0.5).astype(int)
        total_w = float(x_bin @ w)
        if total_w <= capacity:
            return x_bin
        chosen = np.where(x_bin == 1)[0]
        # remove "bad" items (lowest p/w first) until feasible
        for j in chosen[np.argsort(ratio[chosen])]:
            x_bin[j] = 0
            total_w -= w[j]
            if total_w <= capacity:
                break
        return x_bin

    return decode


def knapsack_pso(PROBLEM: int, outdir: str = "visualization/knapsack_pso"):
    os.makedirs(outdir, exist_ok=True)

    capacity, items, weights, solution = get_problem_infos(PROBLEM)
    num_items = len(items)

    # 1) Build decode adapter
    decode = make_knapsack_decode(capacity, items, weights)

    # 2) Objective on PROBLEM-SPACE (already decoded to binary 0/1)
    def objective(x_bin: np.ndarray) -> float:
        # minimize negative profit == maximize profit
        return -float(objective_solution_profit(x_bin, items))

    # 3) Search-space bounds: [0,1]^n
    bounds = [(0.0, 1.0)] * num_items

    # 4) Initialize PSO with decode adapter + tracking for viz
    pso = ParticleSwarmOptimizer(
        objective=objective,        # objective expects decoded representation
        bounds=bounds,
        problem_type="subset",
        decode=decode,              # decode adapter for discrete problems
        n_particles=MAX_POPULATION,
        max_iters=NUM_OF_GENERATIONS,
        mode="constriction",
        phi1=2.05, phi2=2.05, chi=0.729,
        topology="ring", ring_neighbors=2,
        velocity_clamp=0.5,
        boundary_mode="clip",
        seed=42,
        early_stopping_rounds=None,
        tol=0.0,
        enable_position_history=False,
        track_positions=True,       # enable tracking swarm positions over time
        track_stride=1,             # log every iteration; increase this to save RAM
    )

    # Progress bar via callback
    pbar = tqdm(total=NUM_OF_GENERATIONS, desc=f"PSO Knapsack p{PROBLEM:02d}")
    def _cb(t, X, V, Pbest, gbest_f):
        # gbest_f is MIN value; -gbest_f is the current best profit
        pbar.n = t + 1
        pbar.set_postfix_str(f"best_profit={int(-gbest_f)}")
        pbar.refresh()
    pso.callback = _cb

    best_x, best_f, info = pso.optimize()
    pbar.close()

    # 5) Decode best found vector to final feasible answer
    best_ans: Optional[np.ndarray] = info.best_decoded if info.best_decoded is not None else decode(best_x)
    best_profit = objective_solution_profit(best_ans, items)
    optimal_profit = objective_solution_profit(np.array(solution, dtype=int), items)

    print(f"Problem {PROBLEM}: {num_items} items, capacity {capacity}")
    print(f"PSO solution profit : {best_profit}")
    print(f"Optimal solution    : {optimal_profit}")
    rel = (best_profit / max(1, optimal_profit)) * 100.0
    print(f"Accuracy            : {rel:.2f}%")
    print(f"Items selected      : {int(np.sum(best_ans))}/{num_items}")

    # ---------- Visualization ----------
    # (A) Convergence curve
    ax = pso.visualization.plot_convergence([info.history_best_f], labels=[f"PSO p{PROBLEM:02d}"], ylog=False)
    ax.set_title(f"Convergence — Knapsack p{PROBLEM:02d}")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"p{PROBLEM:02d}_convergence.png"), dpi=160)
    plt.close(fig)

    # (B) 2D animation of swarm (projected on first two variables in [0,1])
    # Vì knapsack search-space là [0,1]^n, ta trực quan hoá 2 trục đầu tiên.
    video_path = os.path.join(outdir, f"p{PROBLEM:02d}_swarm.mp4")   
    pso.visualization.animate_swarm_2d(
        outfile=video_path,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        fps=20,
        trail=True,   # draw trail of swarm centroid
        s=18
    )
    print(f"[viz] Saved convergence plot and animation to: {outdir}")

    return best_ans, best_profit, info


if __name__ == "__main__":
    knapsack_pso(PROBLEM=18)