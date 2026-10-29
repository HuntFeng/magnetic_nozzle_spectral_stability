"""
Cheap continuation of the spectrum in v_m with Chebyshev collocation (discretize-then-eig),
used only to measure how fast eigenvalues move and how dense they get as v_m -> 1,
i.e. to size the v_m grid for the InfBeyn study. Resolution N grows like 20/|1-v_m|.
"""
import sys, time
import numpy as np
from common import *

bc = sys.argv[1] if len(sys.argv) > 1 else "fixed_open"
vms = np.concatenate([
    np.arange(0.50, 0.90, 0.02), np.arange(0.90, 0.98, 0.005), np.arange(0.98, 0.9951, 0.0025), [0.997, 0.998, 0.999, 1.001, 1.002, 1.003],
    np.arange(1.005, 1.02, 0.0025), np.arange(1.02, 1.10, 0.005), np.arange(1.10, 1.501, 0.02)])
N_of = lambda vm: 300  # throat-clustered grid: 300 points resolve every case down to |1 - v_m| = 1e-3
rows = []
t0 = time.time()
for vm in vms:
    N = N_of(vm)
    t1 = time.time()
    w = in_window(collocation_eigs(vm, N, bc, mapped=True))
    # convergence check: compare with N + 100 (keep only eigenvalues reproduced to 1e-6)
    w2 = in_window(collocation_eigs(vm, N + 100, bc, mapped=True))
    d = np.abs(w[:, None] - w2[None, :]).min(axis=1) if len(w2) else np.full(len(w), np.inf)
    conv = w[d < 1e-6]
    rows.append((vm, N, conv))
    print(f"vm={vm:.4f} N={N} eigs in window: {len(w)} converged: {len(conv)} ({time.time() - t1:.1f}s)", flush=True)
np.savez(os.path.join(DATA, f"continuation_{bc}.npz"), vms=vms, N=np.array([r[1] for r in rows]),
         eigs=np.array([r[2] for r in rows], dtype=object), allow_pickle=True)
print(f"total {time.time() - t0:.0f}s")
