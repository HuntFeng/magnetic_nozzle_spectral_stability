# All eigenvalues of the constant-velocity test problem (see inf_beyn.py __main__)
# in a rectangle, by InfBeyn.scan(), then certified by InfBeyn.verify().
#
#   python scan_demo.py                    # forked processes, one per available CPU
#   mpiexec -n 64 python scan_demo.py      # MPI (needs the `mpi` extra: uv sync --extra mpi)
#
# Under MPI every rank runs this whole script (SPMD): scan() must be called by all
# ranks with the same arguments, and returns the same result on each of them.
import warnings
import numpy as np
from inf_beyn import InfBeyn

v0 = 1.5
T = lambda z, cb: z**2 * cb.I + 2j * v0 * z * cb.D1 + (1 - v0**2) * cb.D2
bc = lambda cb: cb.I[[0, -1]]
exact = np.pi * (v0**2 - 1) / 2 * np.arange(1, 31)  # n = 1..30 lie in the region below

solver = InfBeyn()
solver.set_problem(T, bc)
result = solver.scan((0, 60, -5, 5), verbose=True)

try:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0

if rank == 0:
    eigvals = np.sort_complex(result.eigvals)
    print(f"{len(eigvals)} eigenvalues, {len(result.tiles)} tiles, {result.n_solves} BVP solves")
    for t in result.unresolved:
        print(f"  unresolved: Re [{t.x0:.3g}, {t.x1:.3g}] Im [{t.y0:.3g}, {t.y1:.3g}] ({t.reason})")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gamma, floor, halo = solver.verify(eigvals, radius=0.3, ns=(128, 256))  # n must resolve the most oscillatory eigenfunction
    for lam, g, h in zip(eigvals, gamma[:, -1], halo):
        err = np.min(np.abs(exact - lam))
        print(f"  {lam:.12f}  |error| {err:.1e}  gamma_n {g:.1e}  halo {h:.1e}  {'ok' if h > 10 * g else 'NOT ISOLATED'}")
