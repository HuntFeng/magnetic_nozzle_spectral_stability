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
import warnings
import numpy as np
import scipy.linalg as la
from scipy.fft import dct
from numpy.polynomial.chebyshev import chebval
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
        A0, A1, scale = self._compute_quasimatrices(V)

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
        inside = self._inside_contour(eigvals)
        return eigvals[inside], eigfuncs[:, inside]

    def _inside_contour(self, z: NDArray) -> NDArray:
        """Winding number of the (closed, ordered) quadrature nodes around each z."""
        d = self.nodes[None, :] - np.asarray(z)[:, None]
        winding = np.angle(np.roll(d, -1, axis=1) / d).sum(axis=1) / (2 * np.pi)
        return np.abs(winding) > 0.5

    def refine(self, eigvals: NDArray, radius: float, n_nodes: int = 32):
        """
        Rerun InfBeyn on a small circle around each eigenvalue (Sec. 2.2 of the paper).
        True eigenvalues reappear at the same place; numerical artifacts (e.g. from
        resolvents too large to evaluate in floating point) vanish or move.

        Output:
            list of eigenvalues found inside each small circle
        """
        nodes, weights = self.nodes, self.weights
        refined = []
        for lam in eigvals:
            self.set_contour(*circle_contour(lam, radius, n_nodes))
            refined.append(self.solve()[0])
        self.set_contour(nodes, weights)
        return refined

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

    def _compute_quasimatrices(self, V: NDArray[np.float64]):
        """
        Compute A0 and A1

        A0 = 1/(2*pi*i) integral_Gamma T(z)^{-1}V dz
        A1 = 1/(2*pi*i) integral_Gamma zT(z)^{-1}V dz

        where T(z) is the operator of the problem T(z)u = 0

        Output:
            A0, A1 as Chebyshev coefficients (padded to a common length) and the
            scale sum_k |w_k| ||T(z_k)^{-1}V|| / 2pi used as reference for the rank
        """
        sols = [self._solve_bvp(z, V) for z in self.nodes]
        n = max(c.shape[0] for c in sols)

        A0 = np.zeros((n, V.shape[1]), dtype=complex)
        A1 = np.zeros((n, V.shape[1]), dtype=complex)
        scale = 0.0
        for z, w, c in zip(self.nodes, self.weights, sols):
            A0[: c.shape[0]] += w * c
            A1[: c.shape[0]] += w * z * c
            scale = max(scale, np.abs(w) * len(self.nodes) * la.norm(c, 2) / (2 * np.pi))

        A0 /= 2j * np.pi
        A1 /= 2j * np.pi
        return A0, A1, scale


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
    print("exact:", exact)
