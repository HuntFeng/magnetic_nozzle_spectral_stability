# AVOIDING DISCRETIZATION ISSUES FOR NONLINEAR EIGENVALUE PROBLEMS
# by MATTHEW J. COLBROOK AND ALEX TOWNSEND, 2023
# https://arxiv.org/abs/2305.01691
#
# InfBeyn (Algorithm 2.1): Beyn's contour method applied to the operator itself.
# The NEP T(z)u = 0 is never discretized as a whole; only the linear BVPs
# T(z_k)u = v at the quadrature nodes are, and each of those is solved
# adaptively (Chebyshev collocation, doubling N until the Chebyshev coefficients
# are resolved). Functions (columns of the quasimatrices) are stored as
# Chebyshev coefficients on [-1, 1].
import os
import warnings
import traceback
import multiprocessing
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.linalg as la
from scipy.fft import dct
from threadpoolctl import threadpool_limits
from numpy.polynomial.chebyshev import chebval, chebvander
from numpy.polynomial.legendre import leggauss
from typing import Callable
from numpy.typing import NDArray


def _available_cpus() -> int:
    """CPUs this process may run on: respects Slurm/cgroup CPU masks, unlike os.cpu_count()."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


class Cheb:
    """Chebyshev points x_j = cos(pi j/N), j = 0..N (x[0] = 1, x[-1] = -1) and differentiation matrices."""

    def __init__(self, N: int) -> None:
        self.N = N
        x = np.cos(np.pi * np.arange(N + 1) / N)
        c = np.ones(N + 1)
        c[0] = c[-1] = 2
        c *= (-1) ** np.arange(N + 1)
        dX = x[:, None] - x[None, :]
        D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
        D -= np.diag(D.sum(axis=1))
        self.x = x
        self.I = np.eye(N + 1)
        self.D1 = D
        self.D2 = D @ D

        # Clenshaw-Curtis weights: sum_j w_j f(x_j) = integral_{-1}^{1} f dx for deg f <= N
        theta = np.pi * np.arange(N + 1) / N
        v = np.ones(N - 1)
        if N % 2 == 0:
            k = np.arange(1, N // 2)
            w_end = 1 / (N**2 - 1)
            v -= np.cos(N * theta[1:-1]) / (N**2 - 1)
        else:
            k = np.arange(1, (N - 1) // 2 + 1)
            w_end = 1 / N**2
        v -= (2 * np.cos(2 * np.outer(theta[1:-1], k)) / (4 * k**2 - 1)).sum(axis=1)
        self.w = np.concatenate(([w_end], 2 * v / N, [w_end]))

    @staticmethod
    def vals2coeffs(vals: NDArray) -> NDArray:
        """Values at Chebyshev points (along axis 0) -> Chebyshev coefficients."""
        N = vals.shape[0] - 1
        c = dct(vals, type=1, axis=0) / N
        c[0] /= 2
        c[-1] /= 2
        return c


def circle_contour(center: complex, radius: float, n_nodes: int = 64):
    """Trapezoidal rule on a circle. Returns nodes z_k and weights w_k ~ dz."""
    theta = 2 * np.pi * np.arange(n_nodes) / n_nodes
    nodes = center + radius * np.exp(1j * theta)
    weights = 1j * radius * np.exp(1j * theta) * 2 * np.pi / n_nodes
    return nodes, weights


class InfBeyn:
    def __init__(
        self,
        m: int = 30,
        n_func: int = 20,
        tol: float = 1e-12,
        n_min: int = 32,
        n_max: int = 2048,
        rank_tol: float = 1e-8,
        seed: int = 42,
    ) -> None:
        """
        Input:
            m: (default 30) number of terms in the Chebyshev series of the random test functions
            n_func: number of random test functions (m_eig + p, must exceed the number of eigenvalues in the contour)
            tol: relative tolerance on the Chebyshev tail for the adaptive BVP solves
            n_min, n_max: smallest / largest Chebyshev degree tried by the adaptive solver
            rank_tol: singular values of A0 below rank_tol * (scale of T^{-1}V) are treated as zero
            seed: seed for the random test functions
        """
        self.m = m
        self.n_func = n_func
        self.tol = tol
        self.n_min = n_min
        self.n_max = n_max
        self.rank_tol = rank_tol
        self.rng = np.random.default_rng(seed)
        self._cheb_cache: dict[int, Cheb] = {}
        self._psa_cache: dict[tuple[int, int], NDArray] = {}
        self._psa_poly_cache: dict[tuple[int, int], NDArray | None] = {}

    def set_contour(self, nodes: NDArray[np.complex128], weights: NDArray[np.complex128]):
        """Quadrature nodes z_k and weights w_k approximating integral_Gamma f(z) dz ~ sum_k w_k f(z_k)."""
        self.nodes = np.asarray(nodes)
        self.weights = np.asarray(weights)

    def set_problem(self, T: Callable, bc: Callable):
        """
        Set the Nonlinear Eigenvalue Problem (NEP), T(omega)u = 0 on [-1, 1]

        Input:
            T: T(z, cheb) -> (N+1)x(N+1) collocation matrix of the operator at z,
               built from cheb.x, cheb.I, cheb.D1, cheb.D2
            bc: bc(cheb) -> 2x(N+1) array, boundary functionals at x = +1 and x = -1;
                they replace the first and last rows of T (homogeneous BCs)
        """
        self.T = T
        self.bc = bc

    def solve(self):
        """
        Output:
            eigvals: eigenvalues inside the contour
            eigfuncs: Chebyshev coefficients of the eigenfunctions, one per column
        """
        V = self._generate_random_test_functions(self.n_func)
        return self._solve(self.nodes, self.weights, V)

    def _solve(self, nodes: NDArray, weights: NDArray, V: NDArray):
        """solve() on an explicit contour with explicit test functions V (no shared state)."""
        A0, A1, scale = self._compute_quasimatrices(V, nodes, weights)
        U, s, Vh, A1q = self._moments_svd(A0, A1)

        m = int(np.sum(s > self.rank_tol * scale))  # numerical rank = number of eigenvalues
        self.singular_values = s / scale
        if m == 0:
            return np.array([], dtype=complex), np.zeros((A0.shape[0], 0), dtype=complex)
        if m >= len(s):
            warnings.warn("A0 has full rank: increase n_func or shrink the contour")
        elif s[m - 1] < 1e4 * s[m]:
            warnings.warn(
                f"no clear gap in the singular values of A0 (s[{m - 1}]/s[{m}] = {s[m - 1] / s[m]:.1e}): "
                "the eigenvalue count is unreliable, shrink the contour or use scan()"
            )
        eigvals, eigfuncs = self._beyn_eig(A0, U, s, Vh, A1q, m)

        # eigenvalues outside the contour cannot come from Keldysh's expansion
        inside = self._inside_contour(eigvals, nodes)
        return eigvals[inside], eigfuncs[:, inside]

    @staticmethod
    def _moments_svd(A0: NDArray, A1: NDArray):
        """
        Economized SVD of the quasimatrix A0 in L2(-1, 1): sample on a Gauss-Legendre
        grid, sqrt(w) makes the l2 product the L2 product. Also returns A1 on that grid.
        """
        nq = A0.shape[0] + 1
        xq, wq = leggauss(nq)
        sw = np.sqrt(wq)[:, None]
        A0q = sw * chebval(xq, A0).T
        A1q = sw * chebval(xq, A1).T
        U, s, Vh = la.svd(A0q, full_matrices=False)
        return U, s, Vh, A1q

    @staticmethod
    def _beyn_eig(A0: NDArray, U: NDArray, s: NDArray, Vh: NDArray, A1q: NDArray, m: int):
        """Beyn's reduced eigenproblem for rank m: eigenvalues and eigenfunctions (Chebyshev coefficients)."""
        Um = U[:, :m]
        V0 = Vh[:m, :].conj().T
        # U^* A1 V0 x = lambda Sigma0 x
        eigvals, X = la.eig(Um.conj().T @ A1q @ V0, np.diag(s[:m]))
        # eigenfunctions u = U Sigma0 x = A0 V0 x
        return eigvals, A0 @ V0 @ X

    @staticmethod
    def _inside_contour(z: NDArray, nodes: NDArray) -> NDArray:
        """Winding number of the (closed, ordered) quadrature nodes around each z."""
        d = nodes[None, :] - np.asarray(z)[:, None]
        winding = np.angle(np.roll(d, -1, axis=1) / d).sum(axis=1) / (2 * np.pi)
        return np.abs(winding) > 0.5

    def refine(self, eigvals: NDArray, radius: float, n_nodes: int = 32, n_workers: int | None = None):
        """
        Rerun InfBeyn on a small circle around each eigenvalue (Sec. 2.2 of the paper).
        True eigenvalues reappear at the same place; numerical artifacts (e.g. from
        resolvents too large to evaluate in floating point) vanish or move.

        The circles are independent and are processed in parallel by forked worker
        processes (Linux; T and bc may be lambdas since nothing is pickled but arrays).
        The test functions are drawn up front, so the result does not depend on n_workers.

        Input:
            n_workers: number of processes (default: one per CPU available to this process); 1 runs serially
        Output:
            list of eigenvalues found inside each small circle
        """
        tasks = [
            (*circle_contour(lam, radius, n_nodes), self._generate_random_test_functions(self.n_func))
            for lam in eigvals
        ]
        n_workers = min(n_workers or _available_cpus(), len(tasks))
        if n_workers <= 1:
            return [self._solve(*task)[0] for task in tasks]
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_init_worker,
            initargs=(self,),
        ) as pool:
            return list(pool.map(_refine_one, tasks))

    def scan(
        self,
        region: tuple[float, float, float, float],
        n_edge: int = 32,
        max_depth: int = 6,
        gap_tol: float = 1e4,
        noise_tol: float = 1e-11,
        quad_tol: float = 1e-4,
        max_rank: int | None = None,
        backend: str = "auto",
        n_workers: int | None = None,
        comm=None,
        verbose: bool = False,
    ) -> "ScanResult":
        """
        All eigenvalues in the rectangle region = (re_min, re_max, im_min, im_max),
        by adaptive subdivision into rectangular tiles. Each tile runs InfBeyn on its
        boundary and is accepted only if
          - the singular values of A0 have a clear gap (s_m / s_{m+1} >= gap_tol,
            s_m > noise_tol relative to the resolvent scale): the count m is unambiguous,
          - m <= max_rank (default n_func // 2): enough random probes to spare,
          - the eigenvalues agree with those of the nested half rule (every other
            node, no extra solves) to quad_tol * tile size: the quadrature has converged,
          - every BVP solve on its boundary is resolved.
        Otherwise the tile is split into four, with cut lines moved away from the
        eigenvalue estimates. Tiles failing at max_depth are returned as unresolved
        (e.g. accumulating eigenvalues or continuous spectrum: inspect with pseudospectra()).

        The region actually scanned is enlarged by ~1.5% (unevenly) on every side, and
        the eigenvalues are then clipped back to region: a pole on a tile's edge spoils
        the quadrature however much the tile is refined, and the edges of region are
        often placed on such lines (e.g. Im = 0 when looking for unstable modes).

        Each edge uses Clenshaw-Curtis with n_edge + 1 nodes, and every BVP solve
        T(z)^{-1}V is cached: neighbouring tiles share their common edges. The
        accepted eigenvalues are polished by Newton's method (see polish()).

        All new BVP solves of one subdivision level, and all polishing steps, run in
        parallel on the chosen backend:
            "serial": this process
            "fork":   n_workers forked processes on this machine (default: one per CPU available to this process)
            "mpi":    all ranks of comm (default MPI.COMM_WORLD). SPMD: every rank runs
                      the same script and calls scan() with the same arguments; rank 0
                      drives the subdivision, the others only solve. Every rank gets
                      the result. Run with e.g. `mpiexec -n 64 python script.py`.
            "auto":   "mpi" if launched by an MPI launcher with more than one rank,
                      otherwise "fork" (or "serial" if n_workers == 1)

        Input:
            region: (re_min, re_max, im_min, im_max)
            n_edge: Clenshaw-Curtis degree on each edge (even)
            max_depth: maximum number of subdivisions of the initial tiles
            verbose: print the progress of each subdivision level (rank 0)
        Output:
            ScanResult (eigenvalues, eigenfunctions, accepted and unresolved tiles)
        """
        if n_edge % 2:
            raise ValueError("n_edge must be even (nested half rule)")
        ex = _make_backend(backend, self, n_workers, comm)
        result = None
        with ex:
            if ex.worker:
                ex.serve()
            else:
                result = self._scan(ex, region, n_edge, max_depth, gap_tol, noise_tol, quad_tol, max_rank, verbose)
        return ex.share(result)

    def _scan(self, ex, region, n_edge, max_depth, gap_tol, noise_tol, quad_tol, max_rank, verbose) -> "ScanResult":
        x0, x1, y0, y1 = map(float, region)
        V = self._generate_random_test_functions(self.n_func)
        pad = 0.015 * max(x1 - x0, y1 - y0)
        X0, X1, Y0, Y1 = x0 - 0.93 * pad, x1 + 1.07 * pad, y0 - 1.11 * pad, y1 + 0.97 * pad
        max_rank = max_rank or self.n_func // 2
        full, half = self._cheb(n_edge), self._cheb(n_edge // 2)

        # initial tiles as close to square as possible (thin tiles need more nodes)
        nx = max(1, round((X1 - X0) / (Y1 - Y0)))
        ny = max(1, round((Y1 - Y0) / (X1 - X0)))
        xs, ys = np.linspace(X0, X1, nx + 1), np.linspace(Y0, Y1, ny + 1)
        pending = [Tile(xs[i], xs[i + 1], ys[j], ys[j + 1]) for i in range(nx) for j in range(ny)]

        solutions: dict[complex, tuple[NDArray, bool]] = {}  # z -> T(z)^{-1}V, resolved
        segments: dict[tuple[complex, complex], tuple] = {}  # edge -> its integrals
        accepted, unresolved = [], []
        level = 0
        while pending:
            new_segs = {_segment(p, q)[0] for t in pending for p, q in t.edges()} - segments.keys()
            new_nodes = sorted(
                {z for key in new_segs for z in _segment_nodes(*key, full)[0]} - solutions.keys(),
                key=lambda z: (z.real, z.imag),
            )
            solutions.update(zip(new_nodes, ex.map(_task_bvp, new_nodes, V)))
            for key in new_segs:
                segments[key] = self._segment_integrals(key, full, half, solutions)

            nxt = []
            for t in pending:
                self._assess(t, segments, gap_tol, noise_tol, quad_tol, max_rank)
                if not t.reason:
                    accepted.append(t)
                elif t.depth >= max_depth or t.reason == "unresolved BVP":
                    unresolved.append(t)
                else:
                    nxt += t.split()
            if verbose:
                print(
                    f"scan level {level}: {len(pending)} tiles, {len(new_nodes)} new solves, "
                    f"{len(accepted)} accepted, {len(unresolved)} unresolved, {len(nxt)} to split",
                    flush=True,
                )
            pending = nxt
            level += 1

        # polish every estimate; allow it to move at most a tenth of its tile
        tasks = [(lam, u, 0.1 * t.size) for t in accepted for lam, u in zip(t.eigvals, t.eigfuncs.T)]
        polished = ex.map(_task_polish, tasks)
        estimates = np.array([lam for lam, _, _ in tasks], dtype=complex)
        eigvals = np.array([lam for lam, _, _ in polished], dtype=complex)
        ok = np.array([o for _, _, o in polished], dtype=bool)
        eigfuncs = [c for _, c, _ in polished]

        # merge duplicates (the same eigenvalue seen from both sides of an edge); two
        # distant estimates converging to the same eigenvalue means one of them was wrong
        keep = np.ones(len(eigvals), bool)
        for a in range(len(eigvals)):
            for b in range(a):
                if keep[b] and abs(eigvals[a] - eigvals[b]) <= 1e-8 * max(1, abs(eigvals[a])):
                    if abs(estimates[a] - estimates[b]) > 0.01 * tasks[a][2]:
                        warnings.warn(f"estimates {estimates[a]:.6g} and {estimates[b]:.6g} polish to the same eigenvalue")
                    keep[a] = False
                    break
        tol = 1e-8 * max(x1 - x0, y1 - y0)
        keep &= (x0 - tol <= eigvals.real) & (eigvals.real <= x1 + tol) & (y0 - tol <= eigvals.imag) & (eigvals.imag <= y1 + tol)
        if not np.all(ok):
            warnings.warn(f"{np.sum(~ok)} estimate(s) did not polish (see ScanResult.polished)")
        if unresolved:
            warnings.warn(f"{len(unresolved)} tile(s) unresolved at max_depth (see ScanResult.unresolved)")
        return ScanResult(
            eigvals=eigvals[keep],
            eigfuncs=[c for c, k in zip(eigfuncs, keep) if k],
            polished=ok[keep],
            estimates=estimates[keep],
            tiles=accepted,
            unresolved=unresolved,
            n_solves=len(solutions),
        )

    def _segment_integrals(self, key, full: Cheb, half: Cheb, solutions: dict) -> tuple:
        """
        Integrals of T(z)^{-1}V and z T(z)^{-1}V along the edge a -> b, with the
        Clenshaw-Curtis rule and with its nested half rule (every other node).
        Also the largest ||T(z)^{-1}V|| on the edge and whether all solves are resolved.
        """
        z, w = _segment_nodes(*key, full)
        wh = (key[1] - key[0]) / 2 * half.w
        sols = [solutions[zk] for zk in z]
        n = max(c.shape[0] for c, _ in sols)
        I = np.zeros((4, n, self.n_func), dtype=complex)  # I0, I1, half-rule I0, I1
        for k, (zk, (c, _)) in enumerate(zip(z, sols)):
            I[0, : c.shape[0]] += w[k] * c
            I[1, : c.shape[0]] += w[k] * zk * c
            if k % 2 == 0:
                I[2, : c.shape[0]] += wh[k // 2] * c
                I[3, : c.shape[0]] += wh[k // 2] * zk * c
        norm = max(la.norm(c) for c, _ in sols)
        return I, norm, all(r for _, r in sols)

    def _assess(self, t: "Tile", segments: dict, gap_tol, noise_tol, quad_tol, max_rank) -> None:
        """Beyn on the boundary of tile t; fills its estimates and diagnostics, t.reason = "" if accepted."""
        parts = [(segments[key], sign) for key, sign in (_segment(p, q) for p, q in t.edges())]
        if not all(ok for (_, _, ok), _ in parts):
            t.reason = "unresolved BVP"
            return
        n = max(I.shape[1] for (I, _, _), _ in parts)
        A = np.zeros((4, n, self.n_func), dtype=complex)
        for (I, _, _), sign in parts:
            A[:, : I.shape[1]] += sign * I
        A /= 2j * np.pi
        scale = max(norm for (_, norm, _), _ in parts) * t.perimeter / (2 * np.pi)

        U, s, Vh, A1q = self._moments_svd(A[0], A[1])
        t.singular_values = s / scale
        t.rank, t.gap = _gap_rank(t.singular_values, noise_tol)
        if t.rank == 0:
            return
        if t.gap < gap_tol:
            t.reason = "no rank gap"
            return
        if t.rank > max_rank:
            t.reason = "rank too large"
            return

        lam, F = self._beyn_eig(A[0], U, s, Vh, A1q, t.rank)
        lam_half, _ = self._beyn_eig(A[2], *self._moments_svd(A[2], A[3]), t.rank)
        # keep estimates slightly outside too: an eigenvalue on a shared edge may land
        # just outside both tiles; the duplicates are merged after polishing
        inside = t.contains(lam, pad=1e-3)
        t.eigvals, t.eigfuncs = lam[inside], F[:, inside]
        # compare the eigenvalues in and around the tile; leakage from far outside is irrelevant
        near = lam[t.contains(lam, pad=0.1)]
        if len(near):
            t.quad_err = np.max(np.min(np.abs(near[:, None] - lam_half[None, :]), axis=1)) / t.size
        if t.quad_err > quad_tol:
            t.reason = "quadrature"

    def polish(self, lam: complex, u: NDArray, max_move: float = np.inf, maxit: int = 30):
        """
        Newton's method on [T(lam)u; c^*u - 1] = 0 from an InfBeyn estimate (lam, u),
        on a Chebyshev grid doubled until lam is converged and u is resolved. A local
        discretization is safe here because InfBeyn has already located a true eigenvalue.

        Input:
            lam, u: eigenvalue estimate and Chebyshev coefficients of its eigenfunction
            max_move: the result is flagged as failed if it lands farther than this from lam
        Output:
            eigenvalue, Chebyshev coefficients of the eigenfunction, success flag
        """
        lam0 = lam
        N = self.n_min
        while N < len(u) - 1:
            N *= 2
        prev = None
        while N <= self.n_max:
            cheb = self._cheb(N)
            lam, vals = self._newton(lam, chebval(cheb.x, u), cheb, maxit)
            u = Cheb.vals2coeffs(vals)
            if prev is not None and abs(lam - prev) <= 1e-10 * max(1, abs(lam)) and np.all(self._resolved(u)):
                return lam, u, bool(abs(lam - lam0) <= max_move)
            prev = lam
            N *= 2
        return lam, u, False

    def _newton(self, lam: complex, u: NDArray, cheb: Cheb, maxit: int):
        """Newton on the collocated bordered system at fixed N. u: values on cheb.x."""
        u = u / la.norm(u)
        c = u.conj()  # normalization c^T u = 1
        n = len(u)
        J = np.zeros((n + 1, n + 1), dtype=complex)
        J[n, :n] = c
        step_prev = np.inf
        for _ in range(maxit):
            h = 1e-3 * max(1, abs(lam))
            T = lambda z: np.asarray(self.T(z, cheb), dtype=complex)
            dT = (8 * (T(lam + h) - T(lam - h)) - (T(lam + 2 * h) - T(lam - 2 * h))) / (12 * h)
            dT[[0, -1], :] = 0  # BCs do not depend on lam
            Tz = self._bvp_matrix(lam, cheb)
            J[:n, :n] = Tz
            J[:n, n] = dT @ u
            r = np.concatenate([Tz @ u, [c @ u - 1]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", la.LinAlgWarning)
                d = la.solve(J, -r)
            u, lam = u + d[:n], lam + d[n]
            step = abs(d[n])
            if step <= 1e-14 * max(1, abs(lam)) or step > 0.5 * step_prev:  # converged or stagnating
                break
            step_prev = step
        return lam, u

    def pseudospectra(
        self, z: NDArray, n: int = 64, n_range: int | None = None, n_workers: int | None = None
    ) -> tuple[NDArray, float]:
        """
        gamma_n(z, T) of eq. (2.10) on an array of points z (Sec. 2.3 of the paper).

        By Theorems 2.2-2.3, {z : gamma_n(z) < eps} is contained in the true
        eps-pseudospectrum of the operator T for every n (no spectral pollution),
        and converges to it as n -> infinity. Norms are L2(-1, 1) on both sides.

        Only the sigma_inf(T(z) P_n^*) term of (2.10) is computed: it already gives
        the inclusion; the adjoint term is needed for convergence only at interior
        points of a non-discrete spectrum.

        Input:
            z: points of the complex plane (any shape)
            n: degree of the polynomial trial space (functions satisfying the BCs)
            n_range: Chebyshev grid on which T(z)u is evaluated (default 2n, which
                     makes the L2 norms exact for constant coefficients)
            n_workers: number of processes (default: one per CPU available to this process); 1 runs serially
        Output:
            gamma: gamma_n(z), same shape as z
            floor: machine eps * max ||T(z) P_n^*||; gamma below floor is rounding noise
        """
        z = np.asarray(z, dtype=complex)
        smin, smax = self._gamma_many(z.ravel(), n, n_range or 2 * n, n_workers)
        return smin.reshape(z.shape), np.finfo(float).eps * smax.max()

    def verify(
        self,
        eigvals: NDArray,
        radius: float,
        ns=(32, 64, 128),
        n_halo: int = 16,
        n_range: int | None = None,
        n_workers: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Check computed eigenvalues with gamma_n (Sec. 2.3.1 of the paper).

        gamma_n(lam) = min ||T(lam)u|| / ||u|| over polynomials u of degree n that
        satisfy the BCs exactly, so lam is an exact eigenvalue of some T + E with
        ||E|| <= gamma_n(lam): lam is certified to lie in the gamma_n(lam)-pseudospectrum.
        A spurious eigenvalue keeps gamma_n(lam) = O(1) as n grows.

        A small gamma_n(lam) alone does not imply a true eigenvalue nearby: for
        strongly non-normal T the pseudospectrum fills whole regions. lam is a
        well-conditioned eigenvalue only if gamma_n is much larger on a small circle
        around it (halo >> gamma). Unlike gamma, a large halo is not a certificate
        (gamma_n is an upper bound), but it converges as n grows.

        Input:
            eigvals: computed eigenvalues
            radius: radius of the circle around each eigenvalue for the halo
            ns: increasing degrees of the trial space
            n_halo: number of points on the circle
            n_range: as in pseudospectra() (default 2n for each n)
            n_workers: as in pseudospectra()
        Output:
            gamma: len(eigvals) x len(ns), gamma_n(lam)
            floor: same shape, rounding floor of each gamma (see pseudospectra())
            halo: len(eigvals), min of gamma_n over the circle for n = ns[-1]
        """
        eigvals = np.atleast_1d(np.asarray(eigvals, dtype=complex))
        circle = radius * np.exp(2j * np.pi * np.arange(n_halo) / n_halo)
        gamma, floor = np.zeros((2, len(eigvals), len(ns)))
        for i, n in enumerate(ns):
            z = eigvals if n != ns[-1] else np.concatenate([eigvals, (eigvals[:, None] + circle).ravel()])
            smin, smax = self._gamma_many(z, n, n_range or 2 * n, n_workers)
            gamma[:, i] = smin[: len(eigvals)]
            floor[:, i] = np.finfo(float).eps * smax[: len(eigvals)]
        halo = smin[len(eigvals) :].reshape(len(eigvals), n_halo).min(axis=1)
        return gamma, floor, halo

    def _gamma_many(self, z: NDArray, n: int, n_range: int, n_workers: int | None):
        """_gamma at each point of the 1D array z, in parallel like refine()."""
        # build the cached operators before forking so workers inherit them
        self._psa_basis(n, n_range)
        self._psa_poly(n, n_range)
        n_workers = min(n_workers or _available_cpus(), len(z))
        chunks = [(n, n_range, c) for c in np.array_split(z, 4 * n_workers)]
        if n_workers <= 1:
            with threadpool_limits(1):  # small matrices: BLAS threads only add overhead
                out = [_gamma_chunk(task, self) for task in chunks]
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context("fork"),
                initializer=_init_worker,
                initargs=(self,),
            ) as pool:
                out = list(pool.map(_gamma_chunk, chunks))
        return np.concatenate([a for a, _ in out]), np.concatenate([b for _, b in out])

    def _psa_basis(self, n: int, n_range: int) -> NDArray:
        """
        Trial space of gamma_n: degree-n polynomials satisfying the BCs, as values on
        the n_range grid, with columns orthonormal in L2(-1, 1).
        """
        key = (n, n_range)
        if key not in self._psa_cache:
            if n_range < n:
                raise ValueError("n_range must be >= n")
            cn, cm = self._cheb(n), self._cheb(n_range)
            Z = la.null_space(np.asarray(self.bc(cn)))  # BCs hold exactly
            P = chebvander(cm.x, n) @ Cheb.vals2coeffs(np.eye(n + 1))  # n-grid -> n_range-grid
            Q = P @ Z
            # orthonormalize in L2 (Clenshaw-Curtis is exact for |u|^2 when n_range >= 2n)
            _, R = la.qr(np.sqrt(cm.w)[:, None] * Q, mode="economic")
            self._psa_cache[key] = la.solve_triangular(R, Q.T, trans="T").T  # Q R^{-1}
        return self._psa_cache[key]

    def _psa_matrix(self, z: complex, n: int, n_range: int) -> NDArray:
        """T(z) P_n^* as a matrix from L2 (orthonormal trial basis) to l2 (weighted grid values)."""
        cm = self._cheb(n_range)
        T = np.asarray(self.T(z, cm), dtype=complex)
        return np.sqrt(cm.w)[:, None] * (T @ self._psa_basis(n, n_range))

    def _psa_poly(self, n: int, n_range: int, max_degree: int = 4) -> NDArray | None:
        """
        If T(z) is a polynomial in z (degree <= max_degree), precompute
        T(z) P_n^* = sum_k z^k C_k with the C_k compressed onto their joint range:
        rows are an orthonormal basis of that range, which leaves the singular values
        unchanged (up to rounding). Each gamma_n then costs a small linear combination
        and the SVD of a (rank x n) instead of (n_range x n) matrix, and T is never
        rebuilt. Returns None (direct evaluation) if T is not such a polynomial.
        """
        key = (n, n_range)
        if key not in self._psa_poly_cache:
            K = max_degree + 1
            roots = np.exp(2j * np.pi * np.arange(K) / K)

            def coeffs(rho):  # interpolate on |z| = rho (DFT), exact for degree < K
                F = np.array([self._psa_matrix(rho * r, n, n_range) for r in roots])
                return np.einsum("jk,jab->kab", roots[:, None] ** -np.arange(K), F) / K / rho ** np.arange(K)[:, None, None]

            C = coeffs(1.0)
            norms = la.norm(C, axis=(1, 2))
            deg = int(np.max(np.nonzero(norms > 1e-13 * norms.max())[0]))
            rho = 1.0
            if deg > 0 and norms[0] > 0:
                # redo on the circle where z^k C_k are all of similar size: balanced rounding
                rho = (norms[0] / norms[deg]) ** (1 / deg)
                C = coeffs(rho)
            C = C[: deg + 1]
            ok = True
            for zt in rho * np.array([1.3 - 0.7j, -0.4 + 2.1j]):  # check the fit off the circle
                F = self._psa_matrix(zt, n, n_range)
                ok &= la.norm(F - np.einsum("k,kab->ab", zt ** np.arange(deg + 1), C)) <= 1e-10 * la.norm(F)
            if not ok:
                self._psa_poly_cache[key] = None
            else:
                S = np.hstack([c / la.norm(c) for c in C])
                U, sv, _ = la.svd(S, full_matrices=False)
                Q = U[:, sv > 1e-14 * np.sqrt(S.shape[0]) * sv[0]]
                self._psa_poly_cache[key] = np.einsum("ia,kib->kab", Q.conj(), C)
        return self._psa_poly_cache[key]

    def _gamma(self, z: complex, n: int, n_range: int) -> tuple[float, float]:
        """Smallest and largest singular values of T(z) P_n^* in L2 -> L2."""
        C = self._psa_poly(n, n_range)
        if C is None:
            A = self._psa_matrix(z, n, n_range)
        else:
            A = np.einsum("k,kab->ab", z ** np.arange(len(C)), C)
        s = la.svdvals(A)
        return s[-1], s[0]

    def _cheb(self, N: int) -> Cheb:
        if N not in self._cheb_cache:
            self._cheb_cache[N] = Cheb(N)
        return self._cheb_cache[N]

    def _generate_random_test_functions(self, n_func: int):
        """
        Generate random test functions from Gaussian process with slightly different kernel (Chebyshev)

        Input:
            n_func: number of functions to be generated
        Output:
            (m+1) x n_func Chebyshev coefficients
        """
        M = self.m
        k = np.arange(M + 1)
        std = np.exp(-0.02 * k**2)
        coeffs = self.rng.normal(size=(M + 1, n_func)) * std[:, None]
        return coeffs

    def _bvp_matrix(self, z: complex, cheb: Cheb) -> NDArray:
        """Collocation matrix of T(z) with the first and last rows replaced by the BCs."""
        T = np.array(self.T(z, cheb), dtype=complex)
        T[[0, -1], :] = self.bc(cheb)
        return T

    def _resolved(self, c: NDArray) -> NDArray:
        """Per column: Chebyshev tail below tol relative to the largest coefficient."""
        return np.abs(c[-8:]).max(axis=0) <= self.tol * np.abs(c).max(axis=0)

    def _solve_bvp(self, z: complex, V: NDArray) -> tuple[NDArray, bool]:
        """
        Adaptively solve T(z)U = V with homogeneous BCs. Double N until the
        Chebyshev coefficients of every column have decayed below tol.

        Input:
            V: Chebyshev coefficients of the right-hand sides
        Output:
            Chebyshev coefficients of the solutions, and whether they are resolved
        """
        N = self.n_min
        while True:
            cheb = self._cheb(N)
            T = self._bvp_matrix(z, cheb)
            rhs = chebval(cheb.x, V).T.astype(complex)
            rhs[[0, -1], :] = 0
            with warnings.catch_warnings():
                # collocation matrices are badly row-scaled (O(N^4) vs O(1) BC rows)
                warnings.simplefilter("ignore", la.LinAlgWarning)
                U = la.solve(T, rhs)
            c = Cheb.vals2coeffs(U)
            if np.all(self._resolved(c)):
                return c, True
            if 2 * N > self.n_max:
                tail = np.abs(c[-8:]).max()
                warnings.warn(f"BVP at z={z:.4g} not resolved with N={N} (tail {tail:.2e})")
                return c, False
            N *= 2

    def _compute_quasimatrices(self, V: NDArray[np.float64], nodes: NDArray, weights: NDArray):
        """
        Compute A0 and A1

        A0 = 1/(2*pi*i) integral_Gamma T(z)^{-1}V dz
        A1 = 1/(2*pi*i) integral_Gamma zT(z)^{-1}V dz

        where T(z) is the operator of the problem T(z)u = 0

        Output:
            A0, A1 as Chebyshev coefficients (padded to a common length) and the
            scale sum_k |w_k| ||T(z_k)^{-1}V|| / 2pi used as reference for the rank
        """
        sols = [self._solve_bvp(z, V)[0] for z in nodes]
        n = max(c.shape[0] for c in sols)

        A0 = np.zeros((n, V.shape[1]), dtype=complex)
        A1 = np.zeros((n, V.shape[1]), dtype=complex)
        scale = 0.0
        for z, w, c in zip(nodes, weights, sols):
            A0[: c.shape[0]] += w * c
            A1[: c.shape[0]] += w * z * c
            scale = max(scale, np.abs(w) * len(nodes) * la.norm(c, 2) / (2 * np.pi))

        A0 /= 2j * np.pi
        A1 /= 2j * np.pi
        return A0, A1, scale


# refine() workers: the solver reaches each forked process through the initializer
# (inherited, not pickled), so only the contour and test functions are sent per task
_worker_solver: InfBeyn | None = None


_worker_shared = None


def _init_worker(solver: InfBeyn, shared=None) -> None:
    global _worker_solver, _worker_shared
    _worker_solver = solver
    _worker_shared = shared
    # one BLAS thread per process, otherwise n_workers x n_cpu threads fight (~5x slower)
    threadpool_limits(1)


def _refine_one(task: tuple) -> NDArray:
    return _worker_solver._solve(*task)[0]


def _gamma_chunk(task: tuple, solver: InfBeyn | None = None) -> tuple[NDArray, NDArray]:
    n, n_range, z = task
    solver = solver or _worker_solver
    g = np.array([solver._gamma(zi, n, n_range) for zi in z]).reshape(-1, 2)
    return g[:, 0], g[:, 1]


# ---------------------------------------------------------------------------
# scan(): tiles, edges and parallel backends
# ---------------------------------------------------------------------------


@dataclass
class Tile:
    """Rectangle [x0, x1) x [y0, y1) of the complex plane visited by InfBeyn.scan()."""

    x0: float
    x1: float
    y0: float
    y1: float
    depth: int = 0
    eigvals: NDArray = field(default_factory=lambda: np.array([], dtype=complex))  # Beyn estimates inside
    eigfuncs: NDArray = field(default_factory=lambda: np.zeros((0, 0), dtype=complex))
    singular_values: NDArray | None = None  # of A0, relative to the resolvent scale
    rank: int = 0
    gap: float = np.inf
    quad_err: float = 0.0  # |full - half rule eigenvalues| / size
    reason: str = ""  # why the tile was rejected, "" if accepted

    @property
    def size(self) -> float:
        return max(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def perimeter(self) -> float:
        return 2 * (self.x1 - self.x0 + self.y1 - self.y0)

    def edges(self) -> list[tuple[complex, complex]]:
        """Counterclockwise directed edges."""
        c = [complex(self.x0, self.y0), complex(self.x1, self.y0), complex(self.x1, self.y1), complex(self.x0, self.y1)]
        return [(c[k], c[(k + 1) % 4]) for k in range(4)]

    def contains(self, z: NDArray, pad: float = 0.0) -> NDArray:
        """Half-open, so that an eigenvalue on a shared edge belongs to exactly one tile."""
        z = np.asarray(z)
        p = pad * self.size
        return (self.x0 - p <= z.real) & (z.real < self.x1 + p) & (self.y0 - p <= z.imag) & (z.imag < self.y1 + p)

    def split(self) -> list["Tile"]:
        """Four children, with the cut lines kept away from the current estimates."""
        xc = _cut(self.x0, self.x1, self.eigvals.real)
        yc = _cut(self.y0, self.y1, self.eigvals.imag)
        return [
            Tile(a, b, c, d, self.depth + 1)
            for a, b in ((self.x0, xc), (xc, self.x1))
            for c, d in ((self.y0, yc), (yc, self.y1))
        ]


@dataclass
class ScanResult:
    eigvals: NDArray  # polished eigenvalues in the region
    eigfuncs: list  # Chebyshev coefficients of each eigenfunction on [-1, 1]
    polished: NDArray  # bool: Newton converged close to the InfBeyn estimate
    estimates: NDArray  # the InfBeyn estimates the eigenvalues were polished from
    tiles: list  # accepted tiles
    unresolved: list  # tiles still failing the acceptance tests at max_depth
    n_solves: int  # number of BVP solves T(z)^{-1}V


def _cut(a: float, b: float, pts: NDArray) -> float:
    """
    Cut position near the middle of [a, b], at least 10% of a child away from pts.
    Slightly off-centre: eigenvalues often sit on symmetry lines (e.g. the real axis),
    which an exact bisection of a symmetric region would put right on the cut.
    """
    margin = 0.05 * (b - a)
    best, best_d = a + 0.47 * (b - a), -1.0
    for t in (0.47, 0.57, 0.37, 0.67, 0.43, 0.53, 0.33, 0.63):
        c = a + t * (b - a)
        d = np.min(np.abs(pts - c)) if len(pts) else np.inf
        if d >= margin:
            return c
        if d > best_d:
            best, best_d = c, d
    return best


def _segment(p: complex, q: complex) -> tuple[tuple[complex, complex], int]:
    """Canonical key of the edge p -> q (shared by both tiles adjacent to it) and its orientation."""
    return ((p, q), 1) if (p.real, p.imag) < (q.real, q.imag) else ((q, p), -1)


def _segment_nodes(a: complex, b: complex, rule: Cheb) -> tuple[list[complex], NDArray]:
    """Clenshaw-Curtis nodes and weights on a -> b; the corners are exactly a and b (shared with other edges)."""
    z = [complex(v) for v in (a + b) / 2 + (b - a) / 2 * rule.x]
    z[0], z[-1] = b, a
    return z, (b - a) / 2 * rule.w


def _gap_rank(s: NDArray, noise_tol: float) -> tuple[int, float]:
    """Rank at the largest gap s[m-1] / s[m] of the singular values above the noise floor."""
    if len(s) == 0 or s[0] <= noise_tol:
        return 0, np.inf
    below = np.maximum(np.append(s[1:], 0.0), noise_tol)
    gaps = np.where(s > noise_tol, s / below, 0.0)
    m = int(np.argmax(gaps))
    return m + 1, float(gaps[m])


def _task_bvp(solver: InfBeyn, z: complex, V: NDArray):
    return solver._solve_bvp(z, V)


def _task_polish(solver: InfBeyn, task: tuple, _=None):
    return solver.polish(*task)


def _run_task(task, item):
    return task(_worker_solver, item, _worker_shared)


class _SerialBackend:
    """Runs task(solver, item, shared) over a list of items. Tasks are module-level functions."""

    worker = False

    def __init__(self, solver: InfBeyn) -> None:
        self.solver = solver

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def map(self, task, items: list, shared=None) -> list:
        with threadpool_limits(1):  # small matrices: BLAS threads only add overhead
            return [task(self.solver, item, shared) for item in items]

    def share(self, result):
        return result


class _ForkBackend(_SerialBackend):
    """Forked worker processes on this machine; the pool lives for one scan()."""

    def __init__(self, solver: InfBeyn, n_workers: int) -> None:
        super().__init__(solver)
        self.n_workers = n_workers
        self.pool, self.shared = None, None

    def __exit__(self, *exc) -> bool:
        if self.pool is not None:
            self.pool.shutdown()
        return False

    def map(self, task, items: list, shared=None) -> list:
        if not items:
            return []
        if self.pool is None or shared is not self.shared:
            # shared data reaches the workers through fork, not through pickling per task
            if self.pool is not None:
                self.pool.shutdown()
            self.shared = shared
            self.pool = ProcessPoolExecutor(
                max_workers=self.n_workers,
                mp_context=multiprocessing.get_context("fork"),
                initializer=_init_worker,
                initargs=(self.solver, shared),
            )
        chunk = max(1, len(items) // (4 * self.n_workers))
        return list(self.pool.map(_run_task, [task] * len(items), items, chunksize=chunk))


class _MPIBackend(_SerialBackend):
    """
    SPMD over an MPI communicator: rank 0 broadcasts (task, items, shared), every rank
    (rank 0 included) runs items[rank::size], and rank 0 gathers the results. The
    other ranks sit in serve() until rank 0 leaves the with-block.
    """

    def __init__(self, solver: InfBeyn, comm=None) -> None:
        super().__init__(solver)
        from mpi4py import MPI

        self.comm = comm or MPI.COMM_WORLD
        self.rank, self.size = self.comm.Get_rank(), self.comm.Get_size()
        self.worker = self.rank != 0

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.worker:
            if exc is not None and self.size > 1:
                # the other ranks would wait forever for the next broadcast
                traceback.print_exception(exc)
                self.comm.Abort(1)
            self.comm.bcast(None, root=0)  # release serve()
        return False

    def _local(self, task, items, shared):
        try:
            with threadpool_limits(1):
                return [task(self.solver, item, shared) for item in items[self.rank :: self.size]]
        except Exception as e:  # sent back to rank 0 instead of deadlocking
            return e

    def map(self, task, items: list, shared=None) -> list:
        items = list(items)
        self.comm.bcast((task, items, shared), root=0)
        parts = self.comm.gather(self._local(task, items, shared), root=0)
        out = [None] * len(items)
        for r, part in enumerate(parts):
            if isinstance(part, Exception):
                raise RuntimeError(f"task failed on MPI rank {r}") from part
            out[r :: self.size] = part
        return out

    def serve(self) -> None:
        while (msg := self.comm.bcast(None, root=0)) is not None:
            self.comm.gather(self._local(*msg), root=0)

    def share(self, result):
        return self.comm.bcast(result, root=0)


def _launched_by_mpi() -> bool:
    for var in ("PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS"):
        if int(os.environ.get(var, "1")) > 1:
            return True
    return False


def _make_backend(backend: str, solver: InfBeyn, n_workers: int | None, comm):
    if backend == "auto":
        backend = "mpi" if comm is not None or _launched_by_mpi() else "fork"
    if backend == "mpi":
        return _MPIBackend(solver, comm)
    n_workers = n_workers or _available_cpus()
    if backend == "serial" or n_workers == 1:
        return _SerialBackend(solver)
    if backend == "fork":
        return _ForkBackend(solver, n_workers)
    raise ValueError(f"unknown backend {backend!r}")


def plot_pseudospectra(z, gamma, floor, eigvals=(), isolated=None, ax=None, n_levels: int = 10):
    """
    Filled contours of log10 gamma_n (as in Fig. 2.1 of the paper): the region
    darker than level eps is certified to lie in the eps-pseudospectrum.

    Input:
        z, gamma, floor: grid and output of InfBeyn.pseudospectra()
        eigvals: computed eigenvalues to overlay
        isolated: boolean mask from verify(), e.g. halo > 10 * gamma[:, -1];
                  isolated eigenvalues are drawn as circles, the others as crosses
        ax: matplotlib axes (default: current axes)
        n_levels: approximate number of contour levels; they span log10 gamma from
                  max(gamma) down to the rounding floor, in steps of 1/2 or whole decades
    Output:
        the contour set (pass to plt.colorbar)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

    ax = ax or plt.gca()
    # one-hue sequential ramp, dark = small gamma = deep inside the pseudospectrum
    ramp = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
    lg = np.log10(np.maximum(gamma, np.finfo(float).tiny))
    hi = lg.max()
    lo = max(lg.min(), np.log10(floor))
    step = max(0.5, np.ceil(2 * (hi - lo) / n_levels) / 2)
    levels = np.arange(np.floor(hi / step) * step, lo, -step)[::-1]
    cmap = LinearSegmentedColormap.from_list("psa", ramp)
    norm = BoundaryNorm(levels, cmap.N, extend="both")
    cs = ax.contourf(z.real, z.imag, lg, levels=levels, cmap=cmap, norm=norm, extend="both")
    ax.contour(z.real, z.imag, lg, levels=levels, colors="#fcfcfb", linewidths=0.4, alpha=0.6)

    eigvals = np.atleast_1d(np.asarray(eigvals, dtype=complex))
    if len(eigvals):
        isolated = np.ones(len(eigvals), bool) if isolated is None else np.asarray(isolated)
        ok, bad = eigvals[isolated], eigvals[~isolated]
        ax.plot(ok.real, ok.imag, "o", ms=8, mfc="#fcfcfb", mec="#0b0b0b", mew=1.2,
                ls="none", label="isolated eigenvalue")
        ax.plot(bad.real, bad.imag, "X", ms=9, mfc="#eb6834", mec="#0b0b0b", mew=0.8,
                ls="none", label="not isolated (artefact)")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.set_xlabel(r"Re $\lambda$")
    ax.set_ylabel(r"Im $\lambda$")
    return cs


if __name__ == "__main__":
    # constant velocity: w^2 v + 2i v0 w v' + (1 - v0^2) v'' = 0, v(+-1) = 0
    # exact eigenvalues: w_n = n pi (v0^2 - 1)/2, n = +-1, +-2, ...
    v0 = 1.5
    T = lambda z, cb: z**2 * cb.I + 2j * v0 * z * cb.D1 + (1 - v0**2) * cb.D2
    bc = lambda cb: cb.I[[0, -1]]
    exact = np.pi * (v0**2 - 1) / 2 * np.arange(1, 10)

    solver = InfBeyn()
    solver.set_problem(T, bc)
    for center, radius in [(7.0, 6.0), (14.6, 1.5), (-1.7 - 44.5j, 4.0), (52.5 + 8.8j, 3.0)]:
        solver.set_contour(*circle_contour(center, radius, 96))
        omega, _ = solver.solve()
        print(f"contour |z-({center})|={radius}: {np.sort_complex(omega)}")
        for lam, ref in zip(omega, solver.refine(omega, 0.3)):
            print(f"    refine {lam:.6f} -> {ref}")
        if len(omega):
            gamma, floor, halo = solver.verify(omega, radius=0.3)
            for lam, g, f, h in zip(omega, gamma, floor, halo):
                print(f"    verify {lam:.6f}: gamma_n = {g} (floor {f[-1]:.1e}), halo {h:.1e}")
    print("exact:", exact)
