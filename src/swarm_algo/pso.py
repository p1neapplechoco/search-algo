# PARTICLE SWARM OPTIMIZATION ALGORITHM IMPLEMENTATION
"""
Ussage example
---------
from pso import ParticleSwarmOptimizer
def f(x: np.ndarray) -> float: ...
pso = ParticleSwarmOptimizer(f, bounds=[(-5,5), (-5,5)], mode="constriction", seed=42)
best_x, best_f, info = pso.optimize()

Gợi ý preset:
- mode="constriction" (phi1=phi2=2.05, chi=0.729) — ổn định, ít phải chỉnh.
- mode="inertia", w=(0.9->0.4), c1=c2=2.0, velocity_clamp=0.2 (20% miền tìm kiếm).

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Any, Literal
import numpy as np


BoundaryMode = Literal["clip", "reflect", "random"]
TopologyMode = Literal["gbest", "ring"]
PSOMode = Literal["constriction", "inertia"]


@dataclass
class PSOInfo:
    history_best_f: List[float]
    best_position: np.ndarray
    n_iter: int
    converged: bool
    bests_over_time: Optional[np.ndarray] = None  # (T, d) if enable_position_history=True


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization (PSO) with two classic variants:
    - 'constriction' (Clerc–Kennedy, 2002)
    - 'inertia' (Shi–Eberhart, 1998)

    Parameters
    ----------
    objective : Callable[[np.ndarray], float]
        Objective function f(x) -> scalar. Receives a vector of shape (d,).
    bounds : Iterable[Tuple[float, float]]
        Per-dimension (low, high) pairs. Must satisfy low < high.
    n_particles : int, default 40
        Number of particles in the swarm.
    max_iters : int, default 300
        Maximum number of iterations.
    mode : {'constriction', 'inertia'}, default 'constriction'
        Velocity update formulation to use.

    # --- Parameters for 'constriction' ---
    phi1 : float, default 2.05
    phi2 : float, default 2.05
    chi : Optional[float], default 0.729
        If None, χ is computed from (phi1 + phi2) using the Clerc–Kennedy formula.

    # --- Parameters for 'inertia' ---
    w : float | Tuple[float, float], default (0.9, 0.4)
        If a tuple, inertia weight is linearly annealed from w_start -> w_end.
    c1 : float, default 2.0
    c2 : float, default 2.0

    # --- Topology ---
    topology : {'gbest', 'ring'}, default 'gbest'
    ring_neighbors : int, default 2
        Number of neighbors on each side for 'ring' topology
        (total neighborhood size is 2*ring_neighbors + 1 including the particle itself).

    # --- Velocity & boundary handling ---
    velocity_clamp : None | float | Tuple[float, float] | np.ndarray, default 0.2
        - None: no velocity clamping
        - float: fraction of the search range per dimension (|v_j| <= frac * range_j)
        - tuple(float, float): symmetric (min_frac, max_frac)
        - ndarray of shape (d,): absolute caps for |v_j| per dimension
    boundary_mode : {'clip', 'reflect', 'random'}, default 'reflect'
        Strategy when a position exits the bounds.

    # --- Misc ---
    seed : Optional[int], default None
    early_stopping_rounds : Optional[int], default None
        Stop early if no improvement for K consecutive iterations.
    tol : float, default 0.0
        Minimal improvement to reset the early-stopping counter.
    enable_position_history : bool, default False
        Record the best position over time (useful for plotting).
    callback : Optional[Callable[[int, np.ndarray, np.ndarray, np.ndarray, float], Any]]
        Per-iteration hook: (t, X, V, pbest, gbest_f).
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: Iterable[Tuple[float, float]],
        *,
        n_particles: int = 40,
        max_iters: int = 300,
        mode: PSOMode = "constriction",
        # constriction
        phi1: float = 2.05,
        phi2: float = 2.05,
        chi: Optional[float] = 0.729,
        # inertia
        w: float | Tuple[float, float] = (0.9, 0.4),
        c1: float = 2.0,
        c2: float = 2.0,
        # topology
        topology: TopologyMode = "gbest",
        ring_neighbors: int = 2,
        # velocity/boundary
        velocity_clamp: None | float | Tuple[float, float] | np.ndarray = 0.2,
        boundary_mode: BoundaryMode = "reflect",
        # misc
        seed: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        tol: float = 0.0,
        enable_position_history: bool = False,
        callback: Optional[Callable[[int, np.ndarray, np.ndarray, np.ndarray, float], Any]] = None,
    ) -> None:
        self.objective = objective
        self.bounds = np.array(bounds, dtype=float)  # (d,2)
        assert self.bounds.ndim == 2 and self.bounds.shape[1] == 2, "`bounds` must be [(low,high), ...]"
        assert np.all(self.bounds[:, 0] < self.bounds[:, 1]), "Each (low,high) must have low < high"
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]
        self.range = self.high - self.low
        self.d = self.bounds.shape[0]

        self.n = int(n_particles)
        self.max_iters = int(max_iters)

        # Mode & params
        self.mode = mode
        self.phi1 = float(phi1)
        self.phi2 = float(phi2)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.chi = chi
        self.w = w

        # Topology
        self.topology = topology
        self.k = int(ring_neighbors)

        # Velocity clamp
        self.vclamp = self._build_vclamp(velocity_clamp)

        # Boundary handling
        self.boundary_mode: BoundaryMode = boundary_mode

        # Misc
        self.rng = np.random.default_rng(seed)
        self.early_stopping_rounds = early_stopping_rounds
        self.tol = float(tol)
        self.enable_position_history = enable_position_history
        self.callback = callback

        # Internal state
        self._X: Optional[np.ndarray] = None   # (n,d)
        self._V: Optional[np.ndarray] = None   # (n,d)
        self._pbest_pos: Optional[np.ndarray] = None
        self._pbest_val: Optional[np.ndarray] = None
        self._gbest_pos: Optional[np.ndarray] = None
        self._gbest_val: Optional[float] = None

    # ---------------------- Public API ----------------------
    def optimize(self) -> Tuple[np.ndarray, float, PSOInfo]:
        """Run PSO and return (best_x, best_f, info)"""
        self._initialize()

        history_best_f: List[float] = []
        bests_over_time: List[np.ndarray] = []

        no_improve = 0
        for t in range(self.max_iters):
            # Compute lbest/gbest for each particle
            lbest_pos = self._get_lbest_positions()

            # Time-varying parameters (w decreases if inertia tuple)
            w_t = self._current_inertia(t) if self.mode == "inertia" else None

            # Update velocity & position
            self._update_velocity(self._X, self._V, self._pbest_pos, lbest_pos, w_t)
            self._apply_velocity_clamp(self._V)
            self._X = self._X + self._V
            self._handle_boundaries(self._X, self._V)

            # Evaluate & update pbest/gbest
            fvals = self._evaluate_all(self._X)
            improved = fvals < self._pbest_val
            self._pbest_pos[improved] = self._X[improved]
            self._pbest_val[improved] = fvals[improved]

            g_idx = int(np.argmin(self._pbest_val))
            g_val = float(self._pbest_val[g_idx])
            if self._gbest_val is None or g_val < self._gbest_val - self.tol:
                self._gbest_val = g_val
                self._gbest_pos = self._pbest_pos[g_idx].copy()
                no_improve = 0
            else:
                no_improve += 1

            history_best_f.append(float(self._gbest_val))
            if self.enable_position_history:
                bests_over_time.append(self._gbest_pos.copy())

            if self.callback is not None:
                try:
                    self.callback(t, self._X, self._V, self._pbest_pos, self._gbest_val)
                except Exception:
                    pass

            if self.early_stopping_rounds is not None and no_improve >= self.early_stopping_rounds:
                break

        info = PSOInfo(
            history_best_f=history_best_f,
            best_position=self._gbest_pos.copy(),
            n_iter=len(history_best_f),
            converged=(self.early_stopping_rounds is not None and no_improve < self.early_stopping_rounds)
                      or (self.early_stopping_rounds is None),
            bests_over_time=np.vstack(bests_over_time) if (self.enable_position_history and bests_over_time) else None,
        )
        
        return self._gbest_pos.copy(), float(self._gbest_val), info

    # ---------------------- Core mechanics ----------------------
    def _initialize(self) -> None:
        # Initialize positions uniformly in [low, high], velocities = 0
        self._X = self.rng.uniform(self.low, self.high, size=(self.n, self.d))
        self._V = np.zeros((self.n, self.d), dtype=float)

        fvals = self._evaluate_all(self._X)
        self._pbest_pos = self._X.copy()
        self._pbest_val = fvals.copy()
        g_idx = int(np.argmin(fvals))
        self._gbest_pos = self._X[g_idx].copy()
        self._gbest_val = float(fvals[g_idx])

        # Normalize chi if needed (Clerc–Kennedy)
        if self.mode == "constriction":
            if self.chi is None:
                phi = self.phi1 + self.phi2
                assert phi > 4.0, "phi1+phi2 must be > 4 with constriction"
                self.chi = 2.0 / (phi - 2.0 + np.sqrt(phi**2 - 4.0 * phi))

    def _current_inertia(self, t: int) -> float:
        if isinstance(self.w, tuple):
            w_start, w_end = float(self.w[0]), float(self.w[1])
            T = max(self.max_iters - 1, 1)
            return w_start + (w_end - w_start) * (t / T)
        return float(self.w)

    def _get_lbest_positions(self) -> np.ndarray:
        if self.topology == "gbest":
            return np.tile(self._gbest_pos, (self.n, 1))
        # ring topology: each particle sees k neighbors on each side
        idx = np.arange(self.n)
        lbest = np.empty_like(self._pbest_pos)
        for i in range(self.n):
            neigh_idx = [(i + s) % self.n for s in range(-self.k, self.k + 1)]
            best_j = neigh_idx[int(np.argmin(self._pbest_val[neigh_idx]))]
            lbest[i] = self._pbest_pos[best_j]
        return lbest

    def _update_velocity(
        self,
        X: np.ndarray,
        V: np.ndarray,
        P: np.ndarray,
        L: np.ndarray,
        w_t: Optional[float],
    ) -> None:
        r1 = self.rng.random(size=V.shape)
        r2 = self.rng.random(size=V.shape)
        if self.mode == "constriction":
            # v = χ * [ v + phi1*r1*(P-X) + phi2*r2*(L-X) ]
            V[:] = V + self.phi1 * r1 * (P - X) + self.phi2 * r2 * (L - X)
            V[:] = self.chi * V  # type: ignore[operator]
        else:
            # v = w*v + c1*r1*(P-X) + c2*r2*(L-X)
            V[:] = w_t * V + self.c1 * r1 * (P - X) + self.c2 * r2 * (L - X)

    def _apply_velocity_clamp(self, V: np.ndarray) -> None:
        if self.vclamp is None:
            return
        vmax = self.vclamp
        np.clip(V, -vmax, vmax, out=V)

    def _handle_boundaries(self, X: np.ndarray, V: np.ndarray) -> None:
        if self.boundary_mode == "clip":
            np.clip(X, self.low, self.high, out=X)
            return

        out_low = X < self.low
        out_high = X > self.high
        out_any = out_low | out_high
        if not np.any(out_any):
            return

        if self.boundary_mode == "reflect":
            # Reflect positions and invert velocities at the boundaries
            X[out_low] = 2 * self.low[np.newaxis, :] - X[out_low]
            X[out_high] = 2 * self.high[np.newaxis, :] - X[out_high]
            V[out_any] *= -1.0
            # If still out of bounds (large step), clip to boundary
            np.clip(X, self.low, self.high, out=X)
        elif self.boundary_mode == "random":
            # Reinitialize randomly at the corresponding boundaries
            X[out_low] = self.rng.uniform(self.low, self.low + 0.1 * self.range)[np.newaxis, :][out_low]
            X[out_high] = self.rng.uniform(self.high - 0.1 * self.range, self.high)[np.newaxis, :][out_high]
            # slightly reduce velocity for stability
            V[out_any] *= 0.0
        else:
            raise ValueError(f"Unknown boundary_mode: {self.boundary_mode}")

    # ---------------------- Utils ----------------------
    def _build_vclamp(
        self, vc: None | float | Tuple[float, float] | np.ndarray
    ) -> Optional[np.ndarray]:
        if vc is None:
            return None
        if isinstance(vc, np.ndarray):
            assert vc.shape == (self.d,), "velocity_clamp ndarray must have shape (d,)"
            return np.abs(vc)
        if isinstance(vc, tuple):
            lo, hi = float(vc[0]), float(vc[1])
            frac = max(abs(lo), abs(hi))
            return frac * self.range
        # float: use as fraction of range
        frac = float(vc)
        return frac * self.range

    def _evaluate_all(self, X: np.ndarray) -> np.ndarray:
        # Vectorized evaluation over all particles
        fvals = np.empty((X.shape[0],), dtype=float)
        for i in range(X.shape[0]):
            fvals[i] = float(self.objective(X[i]))
        return fvals

    # ---------------------- Convenience ----------------------
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin function (benchmark)"""
        A = 10.0
        return A * x.size + np.sum(x * x - A * np.cos(2 * np.pi * x))

    @staticmethod
    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x * x))