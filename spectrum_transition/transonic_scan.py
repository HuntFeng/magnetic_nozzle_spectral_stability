"""InfBeyn on the accelerating transonic flow (half nozzle, regularity at the throat)."""
import time, warnings, json
import numpy as np
from common import *

REG = (-0.5, 22.0, -6.0, 3.0)
T, bcf = transonic_problem(True)
solver = TransonicInfBeyn(n_func=24, n_max=2048)
solver.set_problem(T, bcf)
t0 = time.time()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res = solver.scan(REG, n_edge=32, max_depth=7, verbose=True, n_workers=6)
dt = time.time() - t0
lam = np.sort_complex(res.eigvals)
print(f"{len(lam)} eigenvalues, {res.n_solves} solves, {dt:.0f}s, unresolved {len(res.unresolved)}")
for t in res.unresolved:
    print("  unresolved", (t.x0, t.x1, t.y0, t.y1), t.reason)
for w in caught:
    if "fork()" not in str(w.message):
        print("  warning:", w.message)
gamma, floor, halo = solver.verify(lam, radius=0.25, ns=(64, 128, 256), n_workers=4) if len(lam) else (np.zeros((0, 3)),) * 2 + (np.zeros(0),)
thesis = np.sort_complex(np.load(os.path.join(ROOT, "data", "shooting_accelerating.npz"))["omega"])
for l, g, h in zip(lam, gamma[:, -1], halo):
    d = np.min(np.abs(thesis - l))
    print(f"  {l: .8f}  gamma {g:.1e} halo {h:.1e}  nearest thesis {d:.2e}")
re = np.linspace(REG[0], REG[1], 226)
im = np.linspace(REG[2], REG[3], 91)
Z = re[None, :] + 1j * im[:, None]
gg, ff = solver.pseudospectra(Z, n=128, n_workers=4)
n = max(len(c) for c in res.eigfuncs)
F = np.array([np.pad(c, (0, n - len(c))) for c in res.eigfuncs]).T
np.savez(os.path.join(DATA, "transonic_accelerating.npz"), eigvals=lam, eigfuncs=F, gamma=gamma, halo=halo,
         psa_re=re, psa_im=im, psa_gamma=gg, psa_floor=ff, time=dt, n_solves=res.n_solves, region=np.array(REG),
         unresolved=np.array([[t.x0, t.x1, t.y0, t.y1] for t in res.unresolved]).reshape(-1, 4))
