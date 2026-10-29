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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.linalg as la
from scipy.fft import dct
from threadpoolctl import threadpool_limits
from numpy.polynomial.chebyshev import chebval, chebvander
from numpy.polynomial.legendre import leggauss
from typing import Callable
from numpy.typing import NDArray


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

        # economized SVD of the quasimatrix A0 in L2(-1, 1):
        # sample on a Gauss-Legendre grid, sqrt(w) makes the l2 product the L2 product
        nq = A0.shape[0] + 1
        xq, wq = leggauss(nq)
        sw = np.sqrt(wq)[:, None]
        A0q = sw * chebval(xq, A0).T
        A1q = sw * chebval(xq, A1).T
        U, s, Vh = la.svd(A0q, full_matrices=False)

        m = int(np.sum(s > self.rank_tol * scale))  # numerical rank = number of eigenvalues
        self.singular_values = s / scale
        if m == 0:
            return np.array([], dtype=complex), np.zeros((A0.shape[0], 0), dtype=complex)
        if m >= len(s):
            warnings.warn("A0 has full rank: increase n_func or shrink the contour")
        Um = U[:, :m]
        Sm = np.diag(s[:m])
        V0 = Vh[:m, :].conj().T

        # U^* A1 V0 x = lambda Sigma0 x
        eigvals, X = la.eig(Um.conj().T @ A1q @ V0, Sm)
        # eigenfunctions u = U Sigma0 x = A0 V0 x
        eigfuncs = A0 @ V0 @ X

        # eigenvalues outside the contour cannot come from Keldysh's expansion
        inside = self._inside_contour(eigvals, nodes)
        return eigvals[inside], eigfuncs[:, inside]

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
            n_workers: number of processes (default: one per CPU); 1 runs serially
        Output:
            list of eigenvalues found inside each small circle
        """
        tasks = [
            (*circle_contour(lam, radius, n_nodes), self._generate_random_test_functions(self.n_func))
            for lam in eigvals
        ]
        n_workers = min(n_workers or os.cpu_count() or 1, len(tasks))
        if n_workers <= 1:
            return [self._solve(*task)[0] for task in tasks]
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_init_worker,
            initargs=(self,),
        ) as pool:
            return list(pool.map(_refine_one, tasks))

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
            n_workers: number of processes (default: one per CPU); 1 runs serially
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
        n_workers = min(n_workers or os.cpu_count() or 1, len(z))
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

    def _solve_bvp(self, z: complex, V: NDArray):
        """
        Adaptively solve T(z)U = V with homogeneous BCs. Double N until the
        Chebyshev coefficients of every column have decayed below tol.

        Input:
            V: Chebyshev coefficients of the right-hand sides
        Output:
            Chebyshev coefficients of the solutions
        """
        N = self.n_min
        while True:
            cheb = self._cheb(N)
            T = np.array(self.T(z, cheb), dtype=complex)
            rhs = chebval(cheb.x, V).T.astype(complex)
            T[[0, -1], :] = self.bc(cheb)
            rhs[[0, -1], :] = 0
            with warnings.catch_warnings():
                # collocation matrices are badly row-scaled (O(N^4) vs O(1) BC rows)
                warnings.simplefilter("ignore", la.LinAlgWarning)
                U = la.solve(T, rhs)
            c = Cheb.vals2coeffs(U)
            tail = np.abs(c[-8:]).max(axis=0)
            if np.all(tail <= self.tol * np.abs(c).max(axis=0)):
                return c
            if 2 * N > self.n_max:
                warnings.warn(f"BVP at z={z:.4g} not resolved with N={N} (tail {tail.max():.2e})")
                return c
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
        sols = [self._solve_bvp(z, V) for z in nodes]
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


def _init_worker(solver: InfBeyn) -> None:
    global _worker_solver
    _worker_solver = solver
    # one BLAS thread per process, otherwise n_workers x n_cpu threads fight (~5x slower)
    threadpool_limits(1)


def _refine_one(task: tuple) -> NDArray:
    return _worker_solver._solve(*task)[0]


def _gamma_chunk(task: tuple, solver: InfBeyn | None = None) -> tuple[NDArray, NDArray]:
    n, n_range, z = task
    solver = solver or _worker_solver
    g = np.array([solver._gamma(zi, n, n_range) for zi in z]).reshape(-1, 2)
    return g[:, 0], g[:, 1]


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
