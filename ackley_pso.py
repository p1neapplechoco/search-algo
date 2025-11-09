# ackley — Solve Ackley with PSO + Visualizations
import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from src.swarm_algo.pso import ParticleSwarmOptimizer  

ACKLEY_DATA_FOLDER = "data/ackley/"
NUM_PARTICLES = 50
NUM_GENERATIONS = 1000
BOUND_LB, BOUND_UB = -5.0, 5.0  # Ackley domain recommendation


# ----------------------- Ackley function (continuous benchmark) -----------------------
def ackley_function(x, a=20.0, b=0.2, c=2 * np.pi) -> float:
    """
    Ackley function (global min at x=0, f=0). x: np.ndarray shape (d,)
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    sum_sq = np.sum(x * x)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return float(term1 + term2 + a + np.e)


def read_test_case(test_number: int) -> Tuple[int, np.ndarray, float]:
    """
    Read Ackley test case: returns (dimension, initial_vector, initial_value).
    """
    filename = os.path.join(ACKLEY_DATA_FOLDER, f"test_{test_number:02d}.txt")
    with open(filename, "r") as f:
        lines = f.readlines()

    dimension = None
    initial_value = None
    input_vector = []
    reading_vector = False

    for line in lines:
        line = line.strip()
        if line.startswith("# Dimension:"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("# Initial value:"):
            initial_value = float(line.split("=")[1].strip())
        elif line.startswith("# Input vector:"):
            reading_vector = True
        elif reading_vector and line and not line.startswith("#"):
            input_vector.append(float(line))

    vec = np.array(input_vector, dtype=float)
    if initial_value is None:
        initial_value = ackley_function(vec)
    return dimension, vec, float(initial_value)


# ----------------------- PSO runner with visualization -----------------------
def ackley_pso(test_number: int, visualize: bool = True, outdir: str = "visualization/ackley_pso"):
    """
    Solve Ackley with PSO. Output images/videos if visualize=True.
    - Convergence PNG
    - 2D swarm animation (if d>=2)
    - 3D surface + swarm overlay (if d==2)
    """
    os.makedirs(outdir, exist_ok=True)

    # 1) Read test case
    d, x0, init_val = read_test_case(test_number)
    print(f"\n{'='*60}")
    print(f"Ackley — PSO | Test {test_number:02d}")
    print(f"{'-'*60}")
    print(f"Dimension       : {d}")
    print(f"Initial point   : {x0}")
    print(f"Initial value   : {ackley_function(x0):.6f}")
    print(f"Global optimum  : f(0,...,0) = 0.0")

    # 2) PSO setup — continuous, no decode
    bounds = [(BOUND_LB, BOUND_UB)] * d
    pso = ParticleSwarmOptimizer(
        objective=ackley_function,
        bounds=bounds,
        n_particles=NUM_PARTICLES,
        max_iters=NUM_GENERATIONS,
        mode="constriction",              # stable
        phi1=2.05, phi2=2.05, chi=0.729,
        topology="ring", ring_neighbors=2,
        velocity_clamp=0.2,               # 20% range
        boundary_mode="clip",
        seed=42,
        early_stopping_rounds=100, tol=1e-6,
        problem_type="continuous",
        decode=None,
        enable_position_history=False,    # only store best if needed
        track_positions=visualize,        # enable if need animate
        track_stride=1,
    )

    # 3) Run optimization
    best_x, best_f, info = pso.optimize()

    # 4) Log results
    print(f"{'-'*60}")
    print(f"Best solution   : {best_x}")
    print(f"Best value      : {best_f:.6f}")
    print(f"Distance to 0   : {np.linalg.norm(best_x):.6f}")
    print(f"Error vs optimum: {abs(best_f - 0.0):.6f}")

    # ----------------------- Visualization outputs -----------------------
    if visualize:
        # (A) Convergence curve
        ax = pso.visualization.plot_convergence(
            histories=[info.history_best_f],
            labels=[f"PSO (particles={NUM_PARTICLES}, iters={NUM_GENERATIONS})"],
            ylog=False
        )
        ax.set_title(f"Convergence — Ackley (Test {test_number:02d})")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"ackley_t{test_number:02d}_convergence.png"), dpi=160)
        plt.close(fig)

        # (B) 2D swarm animation (if d>=2: use first 2 coordinates; if d==1, skip)
        if d >= 2 and len(pso._positions_over_time) > 0:
            pso.visualization.animate_swarm_2d(
                outfile=os.path.join(outdir, f"ackley_t{test_number:02d}_swarm.mp4"), 
                bounds=[(BOUND_LB, BOUND_UB), (BOUND_LB, BOUND_UB)],
                fps=20,
                trail=True,
                s=18
            )

        # (C) 3D surface + overlay (chỉ ý nghĩa khi d==2)
        if d == 2:
            # 3D surface PNG
            fig3, ax3 = pso.visualization.plot_surface_3d(
                func=lambda v: ackley_function(np.array(v, dtype=float)),
                x_bounds=(BOUND_LB, BOUND_UB),
                y_bounds=(BOUND_LB, BOUND_UB),
                grid=160,
                elev=35, azim=-60
            )
            fig3.tight_layout()
            fig3.savefig(os.path.join(outdir, f"ackley_t{test_number:02d}_surface.png"), dpi=160)
            plt.close(fig3)

            # 3D overlay animation (nếu đã track positions)
            if len(pso._positions_over_time) > 0:
                pso.visualization.animate_swarm_on_surface(
                    outfile=os.path.join(outdir, f"ackley_t{test_number:02d}_surface_swarm.mp4"),
                    func=lambda v: ackley_function(np.array(v, dtype=float)),
                    x_bounds=(BOUND_LB, BOUND_UB),
                    y_bounds=(BOUND_LB, BOUND_UB),
                    fps=15
                )

        print(f"[viz] Saved plots/videos to: {outdir}")

    return best_x, best_f, info


# ----------------------- Batch runner  -----------------------
def run_all_tests(visualize_last: bool = True):
    results = []
    print("\n" + "=" * 72)
    print("ACKLEY FUNCTION OPTIMIZATION — PSO")
    print("=" * 72)

    for test_num in range(1, 11):
        best_x, best_f, _ = ackley_pso(test_num, visualize=False)
        results.append((test_num, best_f))

    print("\n" + "=" * 72)
    print("SUMMARY OF ALL TEST CASES")
    print("=" * 72)
    print(f"{'Test':<6} {'Best Value':<15} {'Status':<18}")
    print("-" * 72)

    for test_num, best_val in results:
        if best_val < 1e-2:
            status = "✓ Near-optimal"
        elif best_val < 1.0:
            status = "○ Good"
        elif best_val < 5.0:
            status = "△ Acceptable"
        else:
            status = "✗ Needs improvement"
        print(f"{test_num:<6} {best_val:<15.6f} {status:<18}")

    avg_val = float(np.mean([v for _, v in results]))
    print("-" * 72)
    print(f"Average best value: {avg_val:.6f}")

    if visualize_last:
        print("\n" + "=" * 72)
        print("Running last test with visualization...")
        print("=" * 72)
        ackley_pso(10, visualize=True)


if __name__ == "__main__":
    # Option 1: Run a single test case
    ackley_pso(test_number=1, visualize=True)

    # Option 2: Run all test cases
    # run_all_tests(visualize_last=True)