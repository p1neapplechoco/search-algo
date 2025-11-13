# PARTICLE SWARM OPTIMIZATION ALGORITHM IMPLEMENTATION
"""
Usage patterns
--------------
Continuous (identity):
    pso = ParticleSwarmOptimizer(objective=f_cont, bounds=[(-5,5)]*d, decode=None)
    best_x, best_f, info = pso.optimize()

Discrete (via decode), for example, Knapsack:
    def knapsack_decode(x: np.ndarray) -> np.ndarray:
        # map x in [0,1]^n -> {0,1}^n with feasibility repair
        x_bin = (x > 0.5).astype(int)
        # drop items by low (profit/weight) until within capacity
        # (use pre-closed-over weights, profits, capacity)
        ...
        return x_bin

    def knapsack_objective(x_bin: np.ndarray) -> float:
        # minimize negative profit = -sum(profit[i]*x_bin[i])
        return -float(np.dot(profits, x_bin))

    pso = ParticleSwarmOptimizer(
        objective=knapsack_objective,
        bounds=[(0.0, 1.0)] * n_items,
        decode=knapsack_decode,
        ...
    )

Discrete (permutation via random-keys), e.g., TSP:
    def rk_decode(x: np.ndarray) -> np.ndarray:
        # returns a permutation (tour)
        return np.argsort(x)

    def tsp_objective(tour: np.ndarray) -> float:
        # total tour length
        return float(np.sum(dist[ tour, np.roll(tour, -1) ]))

    pso = ParticleSwarmOptimizer(
        objective=tsp_objective,
        bounds=[(0.0, 1.0)] * n_cities,
        decode=rk_decode,
        ...
    )
Notes
-----
- The optimizer always minimizes; for maximization, negate the objective
- `info.best_position` is the best raw vector in search space
- If `decode` is provided, `info.best_decoded` stores the corresponding decoded solution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Any, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl, imageio_ffmpeg # for saving as .mp4
mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
try:
    import seaborn as sns  # optional
except Exception:
    sns = None


BoundaryMode = Literal["clip", "reflect", "random"]
TopologyMode = Literal["gbest", "ring"]
PSOMode = Literal["constriction", "inertia"]
ProblemType = Literal["continuous", "binary", "subset", "permutation"]


@dataclass
class PSOInfo:
    history_best_f: List[float]
    best_position: np.ndarray # best in search space (raw x)
    n_iter: int
    converged: bool
    bests_over_time: Optional[np.ndarray] = None  # (T, d) if enable_position_history=True
    best_decoded: Optional[np.ndarray] = None  # decoded best (binary, permutation, ...)
    problem_type: Optional[str] = None # determined problem type

class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization (PSO) with two classic variants:
    - 'constriction' (Clerc–Kennedy, 2002)
    - 'inertia' (Shi–Eberhart, 1998)

    Parameters
    ----------
    objective : Callable[[np.ndarray], float]
        Objective function f(x) -> scalar. Receives a vector of shape (d,)
    bounds : Iterable[Tuple[float, float]]
        Per-dimension (low, high) pairs. Must satisfy low < high
    n_particles : int, default 40
        Number of particles in the swarm
    max_iters : int, default 300
        Maximum number of iterations.
    mode : {'constriction', 'inertia'}, default 'constriction'
        Velocity update formulation to use

    # --- Parameters for 'constriction' ---
    phi1 : float, default 2.05
    phi2 : float, default 2.05
    chi : Optional[float], default 0.729
        If None, chi is computed from (phi1 + phi2) using the Clerc–Kennedy formula

    # --- Parameters for 'inertia' ---
    w : float | Tuple[float, float], default (0.9, 0.4)
        If a tuple, inertia weight is linearly annealed from w_start -> w_end
    c1 : float, default 2.0
    c2 : float, default 2.0

    # --- Topology ---
    topology : {'gbest', 'ring'}, default 'gbest'
    ring_neighbors : int, default 2
        Number of neighbors on each side for 'ring' topology
        (total neighborhood size is 2*ring_neighbors + 1 including the particle itself)

    # --- Velocity & boundary handling ---
    velocity_clamp : None | float | Tuple[float, float] | np.ndarray, default 0.2
        - None: no velocity clamping
        - float: fraction of the search range per dimension (|v_j| <= frac * range_j)
        - tuple(float, float): symmetric (min_frac, max_frac)
        - ndarray of shape (d,): absolute caps for |v_j| per dimension
    boundary_mode : {'clip', 'reflect', 'random'}, default 'reflect'
        Strategy when a position exits the bounds

    # --- Misc ---
    seed : Optional[int], default None
    early_stopping_rounds : Optional[int], default None
        Stop early if no improvement for K consecutive iterations
    tol : float, default 0.0
        Minimal improvement to reset the early-stopping counter
    enable_position_history : bool, default False
        Record the best position over time (useful for plotting)
    callback : Optional[Callable[[int, np.ndarray, np.ndarray, np.ndarray, float], Any]]
        Per-iteration hook: (t, X, V, pbest, gbest_f)
    
    # -- Problem type --
     problem_type : {'continuous','binary','subset','permutation'}, default 'continuous'
        Declares the problem contract so the optimizer can validate configuration:
        - 'continuous'  : real-valued search; decode=None (or identity)
        - 'binary'      : solution in {0,1}^d (feature selection, ...). Requires a `decode`
        - 'subset'      : feasibility-constrained 0/1 vectors (knapsack, ...). Requires a `decode` that repairs
        - 'permutation' : solution is a permutation of [0..d-1] (TSP, ...). Requires a `decode` (random-keys, etc.)
    
    # -- Decoder (for discrete/combinatorial problems) ---   
    decode : Optional[Callable[[np.ndarray], np.ndarray]], default None
        Optional adapter that maps a search-space vector `x` (shape (d,))
        into a problem-space representation before evaluating `objective`.
        - Continuous problems: leave as None (identity).
        - Discrete/combinatorial: provide a decoder, e.g.:
            * Knapsack: x in [0,1]^n -> binarize+repair -> {0,1}^n
            * TSP (random-keys): x in [0,1]^n -> argsort(x) -> permutation
        The optimizer always minimizes `objective(decode(x) if decode else x)`.
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
        boundary_mode: BoundaryMode = "clip",
        # misc
        seed: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        tol: float = 0.0,
        enable_position_history: bool = False,
        callback: Optional[Callable[[int, np.ndarray, np.ndarray, np.ndarray, float], Any]] = None,
        # problem type
        problem_type: ProblemType = "continuous", 
        # decoder
        decode: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        # visualization params
        track_positions: bool = False,        # log all particles' positions over time
        track_stride: int = 1,                # log positions every k iterations (reduce RAM)
    ) -> None:
        self.objective = objective
        self.bounds = np.array(bounds, dtype=float)  # (d,2)
        assert self.bounds.ndim == 2 and self.bounds.shape[1] == 2, "`bounds` must be like this: [(low,high), ...]"
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
        
        # Problem type
        self.problem_type: ProblemType = problem_type
        # Decoder
        self.decode = decode
        # validate problem contract (suitable decode, bounds, ...)
        self._validate_problem_contract()
           
        # Visualization 
        self.track_positions = bool(track_positions)
        self.track_stride = int(track_stride)
        self._positions_over_time = []   # List[np.ndarray] of shape (n,d), save every track_stride iterations
        self.visualization = None        # will be set at end of __init__
        self.visualization = _PSOVisualizer(self)
        
        # Avoid velocity explosion
        if self.mode == "inertia" and (self.c1 + self.c2) > 4.0:
            raise ValueError("For inertia PSO, require c1 + c2 <= 4 for stability")

    # ----------- Validate problem type and config -----------
    def _validate_problem_contract(self) -> None:
        """
        Contract checks based on `problem_type`:
        - Ensure `decode` presence when needed
        - Ensure bounds make sense
        """
        pt = self.problem_type

        if pt == "continuous":
            # decode=None; bounds:any
            return

        # Discrete/combinatorial problems: require decode
        if self.decode is None:
            raise ValueError(
                f"problem_type='{pt}' requires a `decode(x)` adapter. "
                "Provide a mapping from search-space vector to problem-space solution"
            )

        # Recommended bounds for binary/permutation: [0,1]^d (random-keys / threshold)
        if pt in ("binary", "subset", "permutation"):
            if not np.allclose(self.low, 0.0) or not np.allclose(self.high, 1.0):
                # print warning only (no raise)
                print("[PSO][warn] For problem_type='binary/subset/permutation', "
                    "it is recommended to use bounds [(0.0, 1.0)]^d (random-keys / threshold)")
    
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
                
            # log positions for visualization
            if self.track_positions and (t % self.track_stride == 0):
                # float32 to save RAM
                self._positions_over_time.append(self._X.astype(np.float32).copy())

            if self.early_stopping_rounds is not None and no_improve >= self.early_stopping_rounds:
                break
            
        # check convergence for early stopping mode
        converged_flag = (
            self.early_stopping_rounds is not None and no_improve < self.early_stopping_rounds
        )
        
        # decode best position if decoder is provided
        best_decoded = None
        if self.decode is not None:
            try:
                best_decoded = self.decode(self._gbest_pos)
            except Exception:
                best_decoded = None  # don't let decoding crash the run

        # prepare info
        info = PSOInfo(
            history_best_f=history_best_f,
            best_position=self._gbest_pos.copy(),
            n_iter=len(history_best_f),
            converged=converged_flag,
            bests_over_time=np.vstack(bests_over_time) if (self.enable_position_history and bests_over_time) else None,
            best_decoded=best_decoded,
            problem_type=self.problem_type,
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
            # note: lbest được chọn từ pbest của láng giềng (ổn định hơn vị trí tức thời)
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
            # v = chi * [ v + phi1*r1*(P-X) + phi2*r2*(L-X) ]
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
            X[:] = np.where(out_low,  2*self.low  - X, X)
            X[:] = np.where(out_high, 2*self.high - X, X)
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
        if self.decode is None:
            for i in range(X.shape[0]):
                fvals[i] = float(self.objective(X[i]))
        else:
            # Decode per-particle (kept simple/explicit for clarity & debuggability)
            for i in range(X.shape[0]):
                xi = self.decode(X[i])
                fvals[i] = float(self.objective(xi))
        return fvals
    
    
# ---------------------- Visualization helper ----------------------
class _PSOVisualizer:
    """
    Attach-to instance visualizer for PSO.

    Features
    --------
    - plot_convergence(): lineplot of best-so-far (optionally compare methods)
    - plot_surface_3d(): 3D objective landscape (continuous problems)
    - animate_swarm_2d(): mp4/gif of swarm movement in 2D
    - animate_swarm_on_surface(): mp4/gif overlayed on 3D surface (2D domain)

    Notes
    -----
    - Video saving prefers FFMpegWriter (mp4). Falls back to PillowWriter (gif)
      if ffmpeg is unavailable. (Matplotlib animation docs) 
    - 3D surfaces use `plot_surface` from mplot3d. 
    - Convergence lineplot can use Seaborn if present; otherwise plain Matplotlib. 
    """
    def __init__(self, pso_obj):
        self.pso = pso_obj

    # ---------- 1) Convergence ----------
    def plot_convergence(self, histories, labels=None, ax=None, ylog=False):
        """
        Plot best-so-far curves.

        Parameters
        ----------
        histories : list of 1D arrays
            Each is a sequence of best_f over iterations. Pass [pso.info.history_best_f] for single plot.
        labels : list of str or None
        ax : matplotlib Axes or None
        ylog : bool, log-scale y-axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))
        if labels is None:
            labels = [f"run{i+1}" for i in range(len(histories))]

        x = np.arange(max(len(h) for h in histories))
        for h, lb in zip(histories, labels):
            y = np.asarray(h, dtype=float)
            if sns is not None:
                sns.lineplot(x=np.arange(len(y)), y=y, ax=ax, label=lb)
            else:
                ax.plot(np.arange(len(y)), y, label=lb)
        if ylog: ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best-so-far (objective)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # ---------- 2) 3D surface ----------
    def plot_surface_3d(self, func, x_bounds, y_bounds, grid=200, elev=30, azim=-60, cmap="viridis"):
        """
        Plot a 3D surface z = f([x,y]) over a rectangular domain.

        Parameters
        ----------
        func : Callable[[np.ndarray], float]
            Objective taking a 2D vector [x, y] -> scalar
        x_bounds, y_bounds : (low, high)
        grid : int
            Number of grid points per axis
        elev, azim : float
            View angles
        cmap : str
            Matplotlib colormap

        Returns
        -------
        fig, ax3d
        """
        xs = np.linspace(x_bounds[0], x_bounds[1], grid)
        ys = np.linspace(y_bounds[0], y_bounds[1], grid)
        X, Y = np.meshgrid(xs, ys)
        Z = np.empty_like(X, dtype=float)
        # vectorized eval (simple loop to avoid heavy memory)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = float(func(np.array([X[i, j], Y[i, j]], dtype=float)))

        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x,y)")
        return fig, ax

    # ---------- 3) Animation utils ----------
    def _choose_writer(self, fps=20):
        """Prefer MP4 via FFMpegWriter; fallback to GIF via PillowWriter"""
        try:
            Writer = animation.writers["ffmpeg"]   # may raise if ffmpeg missing
            return Writer(fps=fps, metadata=dict(artist="PSO"))
        except Exception:
            return animation.PillowWriter(fps=fps)

    def animate_swarm_2d(self, outfile, bounds, fps=20, trail=False, s=20, title="PSO Swarm (2D)"):
        """
        Animate 2D swarm from tracked positions (requires track_positions=True).

        Parameters
        ----------
        outfile : str
            Output path (.mp4 if ffmpeg available; falls back to .gif)
        bounds : [(x_low, x_high), (y_low, y_high)]
        fps : int
        trail : bool
            If True, draw fading trails of particles
        s : int
            Marker size
        """
        assert self.pso.d >= 2, "Need at least 2 dimensions for 2D animation"
        assert len(self.pso._positions_over_time) > 0, "Enable track_positions during optimize()"

        traj = self.pso._positions_over_time  # list of (n,d)
        xs = [P[:,0] for P in traj]
        ys = [P[:,1] for P in traj]

        fig, ax = plt.subplots(figsize=(6,6))
        scat = ax.scatter(xs[0], ys[0], s=s)
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

        if trail:
            line_hist, = ax.plot([], [], lw=1, alpha=0.5, color="tab:blue")

        def update(frame):
            scat.set_offsets(np.c_[xs[frame], ys[frame]])
            if trail:
                # draw convex hull-ish simple trail via connecting mean positions
                mean_xy = np.array([[np.mean(xs[t]), np.mean(ys[t])] for t in range(frame+1)])
                line_hist.set_data(mean_xy[:,0], mean_xy[:,1])
            return (scat,)

        ani = animation.FuncAnimation(fig, update, frames=len(xs), interval=1000/fps, blit=False)
        writer = self._choose_writer(fps=fps)
        ani.save(outfile, writer=writer)  
        plt.close(fig)

    def animate_swarm_on_surface(self, outfile, func, x_bounds, y_bounds, fps=20, cmap="viridis", title="PSO on 3D surface"):
        """
        Animate swarm moving over a 3D surface z=f(x,y). (For 2D-domain continuous problems.)

        Parameters
        ----------
        outfile : str
        func : Callable[[np.ndarray], float]
        x_bounds, y_bounds : (low, high)
        fps : int
        cmap : str
        """
        assert self.pso.d >= 2, "Need at least 2 dimensions for 3D overlay"
        assert len(self.pso._positions_over_time) > 0, "Enable track_positions during optimize()"

        # Prepare surface
        xs = np.linspace(x_bounds[0], x_bounds[1], 120)
        ys = np.linspace(y_bounds[0], y_bounds[1], 120)
        X, Y = np.meshgrid(xs, ys)
        Z = np.empty_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = float(func(np.array([X[i, j], Y[i, j]], dtype=float)))

        traj = self.pso._positions_over_time
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)  
        pts = ax.plot(traj[0][:,0], traj[0][:,1],
                      [func(traj[0][k,:2]) for k in range(traj[0].shape[0])],
                      "o", ms=3, color="k")[0]

        ax.set_xlim(*x_bounds); ax.set_ylim(*y_bounds)
        zmin, zmax = Z.min(), Z.max()
        ax.set_zlim(zmin, zmax)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")

        def update(frame):
            P = traj[frame]
            xs_, ys_ = P[:,0], P[:,1]
            zs_ = np.array([func(np.array([xs_[k], ys_[k]], dtype=float)) for k in range(len(xs_))])
            pts.set_data(xs_, ys_)
            pts.set_3d_properties(zs_)
            return (pts,)

        ani = animation.FuncAnimation(fig, update, frames=len(traj), interval=1000/fps, blit=False)
        writer = self._choose_writer(fps=fps)
        ani.save(outfile, writer=writer)  
        plt.close(fig)
