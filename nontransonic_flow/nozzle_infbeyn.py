"""
Spectral stability of non-transonic plasma flows in the magnetic nozzle, solved with
InfBeyn (Colbrook & Townsend 2023) instead of discretize-then-eig.

Polynomial eigenvalue problem (thesis Eq. 3.6):

    w^2 v + 2 i w (v0 v' + v0' v)
      + [(1 - v0^2) v'' - (3 v0 + 1/v0) v0' v' - (1 - 1/v0^2) v0'^2 v - (v0 + 1/v0) v0'' v] = 0

on z in [-1, 1], with Dirichlet  v(-1) = v(1) = 0   ("fixed_fixed")
            or fixed-open v(-1) = v'(1) = 0  ("fixed_open").

Equilibrium velocity (thesis Eq. 2.33), B(z) = B0 (1 + R exp(-(z/Delta)^2)):

    v0(z) = sqrt(-W_k(-(B/Bm)^2 vm^2 exp(-vm^2))),   k = 0 subsonic, k = -1 supersonic.

Usage:  uv run python nontransonic_flow/nozzle_infbeyn.py [case ...]
        (cases: constant_subsonic_fixed_fixed, ...; default: all eight)
"""
import os
import sys
import time
import warnings
import numpy as np
import scipy.linalg as la
from scipy.special import lambertw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from inf_beyn import InfBeyn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# magnetic field of the nozzle (same parameters as archive/nozzle.py)
B0, R, DELTA = 1.0, 1.5, 0.1 / 0.3
BM = B0 * (1 + R)

REGION = (-0.5, 26.0, -3.0, 3.0)  # scanned window of the complex omega plane
CASES = [
    f"{profile}_{bc}"
    for profile in ("constant_subsonic", "constant_supersonic", "subsonic", "supersonic")
    for bc in ("fixed_fixed", "fixed_open")
]


def B_field(z):
    e = np.exp(-((z / DELTA) ** 2))
    B = B0 * (1 + R * e)
    dB = B0 * R * e * (-2 * z / DELTA**2)
    d2B = B0 * R * e * (4 * z**2 / DELTA**4 - 2 / DELTA**2)
    return B, dB, d2B


def velocity_profile(z, vm: float, constant: bool):
    """v0, v0', v0'' at z. Derivatives from the equilibrium ODE (v0^2 - 1) v0' = -v0 B'/B."""
    z = np.asarray(z, dtype=float)
    if constant:
        return vm * np.ones_like(z), np.zeros_like(z), np.zeros_like(z)
    k = 0 if vm < 1 else -1
    B, dB, d2B = B_field(z)
    v0 = np.sqrt(-np.real(lambertw(-((B / BM) ** 2) * vm**2 * np.exp(-(vm**2)), k=k)))
    g = -dB / B
    dg = -(d2B * B - dB**2) / B**2
    q = v0**2 - 1
    dv0 = v0 * g / q
    d2v0 = dv0 * g / q + v0 * dg / q - 2 * v0**2 * g * dv0 / q**2
    return v0, dv0, d2v0


def parse_case(case: str):
    constant = case.startswith("constant_")
    vm = 0.5 if "subsonic" in case else 1.5
    bc = "fixed_open" if case.endswith("fixed_open") else "fixed_fixed"
    return vm, constant, bc


def make_problem(case: str):
    """T(z, cheb) and bc(cheb) for InfBeyn.set_problem."""
    vm, constant, bc = parse_case(case)

    def T(z, cb):
        v0, dv0, d2v0 = velocity_profile(cb.x, vm, constant)
        A1 = 2j * (v0[:, None] * cb.D1 + np.diag(dv0))
        A0 = (
            (1 - v0**2)[:, None] * cb.D2
            - ((3 * v0 + 1 / v0) * dv0)[:, None] * cb.D1
            - np.diag((1 - 1 / v0**2) * dv0**2 + (v0 + 1 / v0) * d2v0)
        )
        return z**2 * cb.I + z * A1 + A0

    # cheb.x[0] = +1 (exit), cheb.x[-1] = -1 (entrance)
    if bc == "fixed_fixed":
        bcf = lambda cb: cb.I[[0, -1]]
    else:
        bcf = lambda cb: np.vstack([cb.D1[0], cb.I[-1]])
    return T, bcf


def exact_eigenvalues(case: str, n_max: int = 40):
    """Thesis Eqs. (3.10), (3.12) for constant v0; None otherwise."""
    vm, constant, bc = parse_case(case)
    if not constant:
        return None
    n = np.arange(-n_max, n_max + 1)
    if bc == "fixed_fixed":
        n = n[n != 0]  # n = 0 gives w = 0 with the zero function (thesis Eq. 3.9): not an eigenvalue
        return (n * np.pi * (1 - vm**2) / 2).astype(complex)
    # v'(1) = 0 gives exp(-4 i w / (vm^2 - 1)) = (1 + vm) / (vm - 1): the right-hand side is
    # negative for vm < 1, which adds i pi to its logarithm (thesis Eq. 3.12 omits this; its
    # Fig. 3.2 already shows the half-integer modes).
    phase = n * np.pi / 2 if vm > 1 else (2 * n + 1) * np.pi / 4
    return (vm**2 - 1) * (phase - 0.25j * np.log(abs((vm - 1) / (vm + 1))))


# ---------------------------------------------------------------------------
# old technique: second-order central finite differences, discretize then eig
# ---------------------------------------------------------------------------


def fd_matrices(N: int):
    x = np.linspace(-1, 1, N)
    h = x[1] - x[0]
    D1 = (np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)) / (2 * h)
    D2 = (np.diag(np.ones(N - 1), 1) - 2 * np.eye(N) + np.diag(np.ones(N - 1), -1)) / h**2
    D1[0, :3] = np.array([-3, 4, -1]) / (2 * h)
    D1[-1, -3:] = np.array([1, -4, 3]) / (2 * h)
    D2[0, :4] = np.array([2, -5, 4, -1]) / h**2
    D2[-1, -4:] = np.array([-1, 4, -5, 2]) / h**2
    return x, D1, D2


def polyeig(A0, A1, A2):
    """Eigenvalues of (A0 + w A1 + w^2 A2) v = 0 by companion linearization."""
    n = A0.shape[0]
    C = np.block([[np.zeros((n, n)), np.eye(n)], [-A0, -A1]])
    D = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), A2]])
    w, X = la.eig(C, D)
    return w, X[:n]


def finite_difference(case: str, N: int = 101):
    """All eigenvalues of the FD-discretized problem (spurious ones included) on N points."""
    vm, constant, bc = parse_case(case)
    x, D1, D2 = fd_matrices(N)
    v0, dv0, d2v0 = velocity_profile(x, vm, constant)
    A1 = 2j * (v0[:, None] * D1 + np.diag(dv0))
    A0 = (
        (1 - v0**2)[:, None] * D2
        - ((3 * v0 + 1 / v0) * dv0)[:, None] * D1
        - np.diag((1 - 1 / v0**2) * dv0**2 + (v0 + 1 / v0) * d2v0)
    )
    A2 = np.eye(N).astype(complex)
    if bc == "fixed_fixed":
        # v(-1) = v(1) = 0: drop the end points
        s = slice(1, -1)
        w, X = polyeig(A0[s, s], A1[s, s], A2[s, s])
        V = np.pad(X, ((1, 1), (0, 0)))
    else:
        # v(-1) = 0: drop x[0]. v'(1) = 0 with the one-sided stencil (3 v_N - 4 v_{N-1} + v_{N-2}) / 2h = 0
        # eliminates v_N = (4 v_{N-1} - v_{N-2}) / 3
        E = np.eye(N)[:, 1:-1]
        E[-1, -1] = 4 / 3
        E[-1, -2] = -1 / 3
        P = np.eye(N)[1:-1]  # keep interior equations
        w, X = polyeig(P @ A0 @ E, P @ A1 @ E, P @ A2 @ E)
        V = E @ X
    return x, w, V


# ---------------------------------------------------------------------------


def run_case(case: str, verbose: bool = True):
    T, bc = make_problem(case)
    solver = InfBeyn(n_func=24)
    solver.set_problem(T, bc)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solver.scan(REGION, n_edge=32, max_depth=7, verbose=verbose)
    t_scan = time.time() - t0
    eigvals = result.eigvals
    order = np.argsort(eigvals.real)
    eigvals = eigvals[order]
    eigfuncs = [result.eigfuncs[i] for i in order]
    polished = result.polished[order]

    # certify: gamma_n(lam) small and a halo around it much larger
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gamma, floor, halo = solver.verify(eigvals, radius=0.25, ns=(64, 128, 256)) if len(eigvals) else (np.zeros((0, 3)),) * 2 + (np.zeros(0),)
    t_verify = time.time() - t0

    # pseudospectrum on a grid of the window
    re = np.linspace(REGION[0], REGION[1], 265)
    im = np.linspace(REGION[2], REGION[3], 61)
    Z = re[None, :] + 1j * im[:, None]
    t0 = time.time()
    gamma_grid, floor_grid = solver.pseudospectra(Z, n=128)
    t_psa = time.time() - t0

    unresolved = np.array([[t.x0, t.x1, t.y0, t.y1] for t in result.unresolved]).reshape(-1, 4)
    reasons = np.array([t.reason for t in result.unresolved])
    tiles = np.array([[t.x0, t.x1, t.y0, t.y1] for t in result.tiles]).reshape(-1, 4)
    n = max((len(c) for c in eigfuncs), default=0)
    F = np.zeros((n, len(eigfuncs)), dtype=complex)
    for j, c in enumerate(eigfuncs):
        F[: len(c), j] = c

    # old technique on the same problem, two resolutions
    fd = {}
    for N in (101, 201):
        x, w, V = finite_difference(case, N)
        fd[f"fd_x_N{N}"], fd[f"fd_omega_N{N}"], fd[f"fd_V_N{N}"] = x, w, V

    os.makedirs(DATA, exist_ok=True)
    np.savez(
        os.path.join(DATA, f"infbeyn_{case}.npz"),
        eigvals=eigvals,
        eigfuncs=F,
        polished=polished,
        gamma=gamma,
        floor=floor,
        halo=halo,
        psa_re=re,
        psa_im=im,
        psa_gamma=gamma_grid,
        psa_floor=floor_grid,
        tiles=tiles,
        unresolved=unresolved,
        unresolved_reasons=reasons,
        n_solves=result.n_solves,
        region=np.array(REGION),
        times=np.array([t_scan, t_verify, t_psa]),
        warnings=np.array([str(w.message) for w in caught if "fork()" not in str(w.message)]),
        **fd,
    )
    if verbose:
        print(f"\n== {case}: {len(eigvals)} eigenvalues, {result.n_solves} solves, "
              f"scan {t_scan:.0f}s, verify {t_verify:.0f}s, psa {t_psa:.0f}s")
        exact = exact_eigenvalues(case)
        for lam, g, h, ok in zip(eigvals, gamma[:, -1], halo, polished):
            err = f"  |err| {np.min(np.abs(exact - lam)):.1e}" if exact is not None else ""
            print(f"  {lam: .10f}  gamma {g:.1e}  halo {h:.1e}{err}  {'ok' if h > 10 * g and ok else 'CHECK'}")
        for box, r in zip(unresolved, reasons):
            print(f"  unresolved tile Re [{box[0]:.3g},{box[1]:.3g}] Im [{box[2]:.3g},{box[3]:.3g}]: {r}")
        for w in caught:
            if "fork()" not in str(w.message):
                print("  warning:", w.message)
    return eigvals


if __name__ == "__main__":
    cases = sys.argv[1:] or CASES
    for case in cases:
        run_case(case)
