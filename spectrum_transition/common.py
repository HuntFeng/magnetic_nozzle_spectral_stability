"""Shared pieces for the spectrum-transition study (v_m through the sonic point)."""
import os
import sys
import numpy as np
import scipy.linalg as la
from scipy.special import lambertw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "nontransonic_flow"))
from inf_beyn import InfBeyn, Cheb  # noqa: E402
from nozzle_infbeyn import B_field, BM, velocity_profile, polyeig  # noqa: E402

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
REGION = (-0.5, 26.0, -3.0, 3.0)


# ----------------------------------------------------------------------------- profiles
def profile_branch(z, vm, k):
    """v0, v0', v0'' on a given Lambert branch k (0 subsonic, -1 supersonic), any vm."""
    z = np.asarray(z, dtype=float)
    B, dB, d2B = B_field(z)
    v0 = np.sqrt(-np.real(lambertw(-((B / BM) ** 2) * vm**2 * np.exp(-(vm**2)), k=k)))
    g = -dB / B
    dg = -(d2B * B - dB**2) / B**2
    q = v0**2 - 1
    dv0 = v0 * g / q
    d2v0 = dv0 * g / q + v0 * dg / q - 2 * v0**2 * g * dv0 / q**2
    return v0, dv0, d2v0


def sonic_slope():
    """v0'(0) of the transonic profiles: 2 s^2 = -B''(0)/B_m."""
    _, _, d2B = B_field(0.0)
    return np.sqrt(-d2B / (2 * BM))


def transonic_profile(z, accelerating=True, h=0.02, deg=8):
    """
    v0, v0', v0'' of the smooth transonic profile (v_m = 1): subsonic branch on one
    side of the throat, supersonic on the other. Near z = 0 both branch formulas
    lose digits (0/0), so |z| < h is replaced by a polynomial fit through both sides.
    """
    z = np.asarray(z, dtype=float)
    kl, kr = (0, -1) if accelerating else (-1, 0)
    out = [np.zeros_like(z) for _ in range(3)]
    left, right, mid = z < -h, z > h, np.abs(z) <= h
    for mask, k in ((left, kl), (right, kr)):
        if mask.any():
            for o, val in zip(out, profile_branch(z[mask], 1.0, k)):
                o[mask] = val
    if mid.any():
        zz = np.concatenate([np.linspace(-3 * h, -h, 25), np.linspace(h, 3 * h, 25)])
        vv = np.concatenate([profile_branch(zz[:25], 1.0, kl)[0], profile_branch(zz[25:], 1.0, kr)[0]])
        p = np.polynomial.Polynomial.fit(zz, vv, deg)
        out[0][mid] = p(z[mid])
        out[1][mid] = p.deriv(1)(z[mid])
        out[2][mid] = p.deriv(2)(z[mid])
    return out


# ----------------------------------------------------------------------------- operators
def coefficient_matrices(v0, dv0, d2v0, D1, D2):
    """A1, A0 of T(w) = w^2 I + w A1 + A0 (thesis Eq. 3.6) on a grid with derivative matrices D1, D2."""
    A1 = 2j * (v0[:, None] * D1 + np.diag(dv0))
    A0 = (
        (1 - v0**2)[:, None] * D2
        - ((3 * v0 + 1 / v0) * dv0)[:, None] * D1
        - np.diag((1 - 1 / v0**2) * dv0**2 + (v0 + 1 / v0) * d2v0)
    )
    return A1, A0


def nontransonic_problem(vm, bc="fixed_open"):
    """T(z, cheb), bc(cheb) of the full-nozzle problem at throat Mach vm != 1."""

    def T(z, cb):
        v0, dv0, d2v0 = velocity_profile(cb.x, vm, False)
        A1, A0 = coefficient_matrices(v0, dv0, d2v0, cb.D1, cb.D2)
        return z**2 * cb.I + z * A1 + A0

    if bc == "fixed_fixed":
        bcf = lambda cb: cb.I[[0, -1]]
    else:
        bcf = lambda cb: np.vstack([cb.D1[0], cb.I[-1]])  # cb.x[0] = +1 exit, cb.x[-1] = -1 entrance
    return T, bcf


class TransonicInfBeyn(InfBeyn):
    """
    InfBeyn for the accelerating transonic flow on the subsonic half [-1, 0].
    The sonic throat z = 0 is a regular singular point of Eq. 3.6; the regular
    solution is selected by imposing the ODE itself at z = 0 (its second-derivative
    term vanishes there, which is exactly the Frobenius condition c_1 = -b c_0 / a).
    So only one boundary row is replaced, v(-1) = 0; x in [-1, 1] maps to z = (x - 1)/2.
    """

    def _bvp_matrix(self, z, cheb):
        T = np.array(self.T(z, cheb), dtype=complex)
        T[-1, :] = self.bc(cheb)[0]
        return T

    def _solve_bvp(self, z, V):
        """As InfBeyn._solve_bvp, but only the last row (z = -1) is a boundary row: the
        first row is the ODE at the throat and keeps its right-hand side."""
        import warnings
        from numpy.polynomial.chebyshev import chebval
        N = self.n_min
        while True:
            cheb = self._cheb(N)
            T = self._bvp_matrix(z, cheb)
            rhs = chebval(cheb.x, V).T.astype(complex)
            rhs[-1, :] = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", la.LinAlgWarning)
                U = la.solve(T, rhs)
            c = Cheb.vals2coeffs(U)
            if np.all(self._resolved(c)):
                return c, True
            if 2 * N > self.n_max:
                return c, False
            N *= 2

    def _newton(self, lam, u, cheb, maxit):
        """As InfBeyn._newton, but the throat row keeps its omega-derivative."""
        import warnings
        u = u / la.norm(u)
        c = u.conj()
        n = len(u)
        J = np.zeros((n + 1, n + 1), dtype=complex)
        J[n, :n] = c
        step_prev = np.inf
        for _ in range(maxit):
            h = 1e-3 * max(1, abs(lam))
            Tf = lambda z: np.asarray(self.T(z, cheb), dtype=complex)
            dT = (8 * (Tf(lam + h) - Tf(lam - h)) - (Tf(lam + 2 * h) - Tf(lam - 2 * h))) / (12 * h)
            dT[-1, :] = 0
            Tz = self._bvp_matrix(lam, cheb)
            J[:n, :n] = Tz
            J[:n, n] = dT @ u
            r = np.concatenate([Tz @ u, [c @ u - 1]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", la.LinAlgWarning)
                d = la.solve(J, -r)
            u, lam = u + d[:n], lam + d[n]
            step = abs(d[n])
            if step <= 1e-14 * max(1, abs(lam)) or step > 0.5 * step_prev:
                break
            step_prev = step
        return lam, u

    def _psa_basis(self, n, n_range):
        # trial space: polynomials with v(-1) = 0 only
        key = (n, n_range)
        if key not in self._psa_cache:
            from numpy.polynomial.chebyshev import chebvander
            cn, cm = self._cheb(n), self._cheb(n_range)
            Z = la.null_space(np.asarray(self.bc(cn)))
            P = chebvander(cm.x, n) @ Cheb.vals2coeffs(np.eye(n + 1))
            Q = P @ Z
            _, R = la.qr(np.sqrt(cm.w)[:, None] * Q, mode="economic")
            self._psa_cache[key] = la.solve_triangular(R, Q.T, trans="T").T
        return self._psa_cache[key]


def transonic_problem(accelerating=True):
    """T(z, cheb), bc(cheb) on the half nozzle [-1, 0] (x -> z = (x-1)/2, d/dz = 2 d/dx)."""

    def T(z, cb):
        zz = (cb.x - 1) / 2
        v0, dv0, d2v0 = transonic_profile(zz, accelerating)
        v0[0], dv0[0], d2v0[0] = 1.0, dv0[0], d2v0[0]
        A1, A0 = coefficient_matrices(v0, dv0, d2v0, 2 * cb.D1, 4 * cb.D2)
        # exact sonic point: kill the second-derivative row at x = +1 (z = 0)
        A0[0] -= (1 - v0[0] ** 2) * 4 * cb.D2[0]
        return z**2 * cb.I + z * A1 + A0

    return T, lambda cb: cb.I[[-1]]


# ----------------------------------------------------------------------------- collocation (cheap continuation)
def collocation_eigs(vm, N=400, bc="fixed_open", mapped=False):
    """All eigenvalues of the discretized full-nozzle problem on N+1 Chebyshev points (optionally throat-clustered)."""
    cb = Cheb(N)
    zz, D1, D2 = mapped_grid(cb, map_parameter(vm)) if mapped else (cb.x, cb.D1, cb.D2)
    v0, dv0, d2v0 = velocity_profile(zz, vm, False)
    A1, A0 = coefficient_matrices(v0, dv0, d2v0, D1, D2)
    if bc == "fixed_fixed":
        s = slice(1, -1)
        w, _ = polyeig(A0[s, s], A1[s, s], np.eye(N - 1))
        return w
    E = np.zeros((N + 1, N - 1), dtype=complex)
    E[1:-1] = np.eye(N - 1)
    E[0] = -D1[0, 1:-1] / D1[0, 0]
    P = np.eye(N + 1)[1:-1]
    w, _ = polyeig(P @ A0 @ E, P @ A1 @ E, P @ E)
    return w


def in_window(w, region=REGION):
    x0, x1, y0, y1 = region
    return w[(w.real >= x0) & (w.real <= x1) & (w.imag >= y0) & (w.imag <= y1)]


# ----------------------------------------------------------------------------- throat-clustered coordinate map
def map_parameter(vm, floor=0.3):
    """
    L of the map z = sinh(L x)/sinh(L), which compresses x near the throat by L/sinh(L).
    The resolvent solutions oscillate there with wavelength ~ (1 - v0)/|omega|; empirically
    (tail of the Chebyshev coefficients at N = 256 for |omega| <= 26) L = 1 + 1.5 ln(1/|1 - v_m|)
    resolves them. Identity (L = 0) when |1 - v_m| >= floor.
    """
    eps = abs(1.0 - vm)
    if eps >= floor:
        return 0.0
    return 1.0 + 1.5 * np.log(1.0 / eps)


def mapped_grid(cb, L):
    """z-grid and d/dz, d^2/dz^2 matrices on the standard Chebyshev object cb through z = sinh(Lx)/sinh(L)."""
    if L == 0.0:
        return cb.x, cb.D1, cb.D2
    x = cb.x
    g1 = L * np.cosh(L * x) / np.sinh(L)
    g2 = L**2 * np.sinh(L * x) / np.sinh(L)
    z = np.sinh(L * x) / np.sinh(L)
    Dz = cb.D1 / g1[:, None]
    Dzz = cb.D2 / g1[:, None] ** 2 - (g2 / g1**3)[:, None] * cb.D1
    return z, Dz, Dzz


def mapped_problem(vm, bc="fixed_open", L=None):
    """Same NEP as nontransonic_problem but discretized on the throat-clustered grid."""
    L = map_parameter(vm) if L is None else L

    def T(z, cb):
        zz, D1, D2 = mapped_grid(cb, L)
        v0, dv0, d2v0 = velocity_profile(zz, vm, False)
        A1, A0 = coefficient_matrices(v0, dv0, d2v0, D1, D2)
        return z**2 * cb.I + z * A1 + A0

    if bc == "fixed_fixed":
        bcf = lambda cb: cb.I[[0, -1]]
    else:
        bcf = lambda cb: np.vstack([mapped_grid(cb, L)[1][0], cb.I[-1]])
    return T, bcf, L
