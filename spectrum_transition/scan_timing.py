"""InfBeyn full scans at several v_m: wall time, solves, eigenvalue count, max Chebyshev degree."""
import sys, time, warnings, json
import numpy as np
from common import *

vms = [float(v) for v in sys.argv[1:]] or [0.9, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.1]
bc = "fixed_open"
for vm in vms:
    T, bcf = nontransonic_problem(vm, bc)
    solver = InfBeyn(n_func=24, n_max=4096)
    solver.set_problem(T, bcf)
    # record the largest N the adaptive solver needed
    maxN = [0]
    orig = solver._solve_bvp
    def wrapped(z, V, orig=orig, maxN=maxN):
        c, ok = orig(z, V)
        maxN[0] = max(maxN[0], c.shape[0] - 1)
        return c, ok
    solver._solve_bvp = wrapped
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = solver.scan(REGION, n_edge=32, max_depth=7, verbose=False)
    dt = time.time() - t0
    lam = np.sort_complex(res.eigvals)
    # maxN measured in the parent only (workers are forked): recompute from eigenfunction lengths
    nmax_eig = max((len(c) for c in res.eigfuncs), default=0)
    msgs = [str(w.message) for w in caught if "fork()" not in str(w.message)]
    rec = dict(vm=vm, bc=bc, time=dt, n_solves=int(res.n_solves), n_eig=len(lam),
               n_unresolved=len(res.unresolved), unresolved=[(t.x0, t.x1, t.y0, t.y1, t.reason) for t in res.unresolved],
               n_cheb_eigfunc=nmax_eig, n_unpolished=int(np.sum(~res.polished)), warnings=msgs[:5])
    print(json.dumps(rec), flush=True)
    np.savez(os.path.join(DATA, f"scan_{bc}_vm{vm:.4f}.npz"), eigvals=lam, time=dt, n_solves=res.n_solves,
             tiles=np.array([[t.x0, t.x1, t.y0, t.y1] for t in res.tiles]).reshape(-1, 4),
             unresolved=np.array([[t.x0, t.x1, t.y0, t.y1] for t in res.unresolved]).reshape(-1, 4),
             eigfuncs=np.array([np.pad(c, (0, nmax_eig - len(c))) for c in res.eigfuncs]).reshape(-1, nmax_eig).T)
