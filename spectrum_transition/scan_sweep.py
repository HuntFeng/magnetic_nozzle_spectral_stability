"""
InfBeyn full scans of the fixed-open problem over a v_m grid on the throat-clustered grid
(mapped_problem). Records wall time, solves, eigenvalue count, max Chebyshev degree, and
certifies every eigenvalue with gamma_n and its halo. Usage: scan_sweep.py [vm ...]
"""
import sys, time, warnings, json
import numpy as np
from common import *

vms = [float(v) for v in sys.argv[1:]]
bc = "fixed_open"
for vm in vms:
    T, bcf, L = mapped_problem(vm, bc)
    solver = InfBeyn(n_func=24, n_max=4096)
    solver.set_problem(T, bcf)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = solver.scan(REGION, n_edge=32, max_depth=7, verbose=False, n_workers=int(os.environ.get("NW", "0")) or None)
    dt = time.time() - t0
    order = np.argsort(res.eigvals.real)
    lam = res.eigvals[order]
    funcs = [res.eigfuncs[i] for i in order]
    polished = res.polished[order]
    t1 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gamma, floor, halo = solver.verify(lam, radius=0.2, ns=(64, 128, 256)) if len(lam) else (np.zeros((0, 3)),) * 2 + (np.zeros(0),)
    dt_v = time.time() - t1
    nmax = max((len(c) for c in funcs), default=0)
    msgs = [str(w.message) for w in caught if "fork()" not in str(w.message)]
    rec = dict(vm=vm, L=L, time=dt, time_verify=dt_v, n_solves=int(res.n_solves), n_eig=len(lam),
               n_unresolved=len(res.unresolved), n_cheb_eigfunc=nmax, n_unpolished=int(np.sum(~polished)),
               n_uncertified=int(np.sum(halo <= 10 * gamma[:, -1])) if len(lam) else 0, warnings=msgs[:4])
    print(json.dumps(rec), flush=True)
    np.savez(os.path.join(DATA, f"sweep_{bc}_vm{vm:.4f}.npz"), eigvals=lam, polished=polished, gamma=gamma, halo=halo,
             L=L, time=dt, time_verify=dt_v, n_solves=res.n_solves,
             tiles=np.array([[t.x0, t.x1, t.y0, t.y1] for t in res.tiles]).reshape(-1, 4),
             unresolved=np.array([[t.x0, t.x1, t.y0, t.y1] for t in res.unresolved]).reshape(-1, 4),
             eigfuncs=np.array([np.pad(c, (0, nmax - len(c))) for c in funcs]).reshape(-1, nmax).T)
