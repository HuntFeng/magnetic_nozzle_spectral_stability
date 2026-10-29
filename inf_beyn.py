# AVOIDING DISCRETIZATION ISSUES FOR NONLINEAR EIGENVALUE PROBLEMS
# by MATTHEW J. COLBROOK AND ALEX TOWNSEND, 2023
# https://arxiv.org/abs/2305.01691
import numpy as np
import scipy.linalg as la
from numpy.polynomial.chebyshev import chebval
from typing import Callable
from numpy.typing import NDArray


class InfBeyn:
    def __init__(self, n: int, m: int = 30) -> None:
        """
        Input:
            n: number of discretization points of the problem
            m: (default 30) number of terms in the series
        """
        self.n = n
        self.m = m
        self.x = np.linspace(-1, 1, n)

    def set_contour(self, nodes: NDArray[np.complex64], weights: NDArray[np.float64]):
        self.nodes = nodes
        self.weights = weights
        self.contour = zip(nodes, weights)

    def set_problem(self, T: Callable):
        """
        Set the Nonlinear Eigenval Problem (NEP), T(omega)u = 0

        Input:
            T: operator as a function of omega
        """
        self.T = T

    def solve(self):
        self.V = self._generate_random_test_functions(20)

    def _generate_random_test_functions(self, n_func: int):
        """
        Generate randomw test functions from Gaussian process with slightly different kernel (Chebyshev)

        Input:
            n_func: number of functions to be generated
        """
        rng = np.random.default_rng(42)

        M = self.m
        k = np.arange(M + 1)
        std = np.exp(-0.02 * k**2)
        coeffs = rng.normal(size=(n_func, M + 1)) * std
        funcs = np.array([chebval(self.x, c) for c in coeffs])
        return funcs

    def _compute_quasimatrices(self):
        """
        Compute A0 and A1

        A0 = 1/(2*pi*i) integral_Gamma T(z)^{-1}V dz
        A1 = 1/(2*pi*i) integral_Gamma zT(z)^{-1}V dz

        where T(z) is the operator of the problem T(z)u = 0
        """

        A0 = np.zeros((self.n, self.m), dtype=complex)
        A1 = np.zeros((self.n, self.m), dtype=complex)

        for z, w in self.contour:
            T = self.T(z)
            V = self.V
            lu, piv = la.lu_factor(T)
            U = la.lu_solve((lu, piv), V)
            A0 += w * U
            A1 += w * z * U

        A0 /= 2j * np.pi
        A1 /= 2j * np.pi

    def _compute_svd(self):
        """
        Compute m-truncated SVD of A0
        """
        pass

    def _solve_gep(self):
        """
        Solve mxm generalized eigenvalue problem
        """
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    inf_beyn = InfBeyn(200)
    funcs = inf_beyn._generate_random_test_functions(5)
    plt.figure()
    for i, f in enumerate(funcs):
        plt.plot(inf_beyn.x, f, label=f"{i+1} test function")
    plt.legend()
    plt.show()

