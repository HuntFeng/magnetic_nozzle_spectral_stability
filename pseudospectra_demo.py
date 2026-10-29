# Pseudospectra of the constant-velocity test problem (see inf_beyn.py __main__),
# with the InfBeyn eigenvalues overlaid and classified by InfBeyn.verify().
import sys
import numpy as np
import matplotlib.pyplot as plt
from inf_beyn import InfBeyn, circle_contour, plot_pseudospectra

v0 = 1.5
T = lambda z, cb: z**2 * cb.I + 2j * v0 * z * cb.D1 + (1 - v0**2) * cb.D2
bc = lambda cb: cb.I[[0, -1]]

solver = InfBeyn()
solver.set_problem(T, bc)
eigvals = []
for center, radius in [(7.0, 6.0), (14.6, 1.5), (-1.7 - 44.5j, 4.0)]:
    solver.set_contour(*circle_contour(center, radius, 96))
    eigvals.append(solver.solve()[0])
eigvals = np.concatenate(eigvals)
gamma, floor, halo = solver.verify(eigvals, radius=0.3, ns=(64, 128))
isolated = halo > 10 * gamma[:, -1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
views = [
    ("Near the real axis", 64, np.linspace(0, 17, 1021), np.linspace(-1, 1, 81)),
    ("Wide view: non-normal lower half-plane", 128, np.linspace(-20, 60, 160), np.linspace(-55, 15, 140)),
]
for ax, (title, n, re, im) in zip(axes, views):
    X, Y = np.meshgrid(re, im)
    z = X + 1j * Y
    g, f = solver.pseudospectra(z, n=n)
    inside = (eigvals.real >= re[0]) & (eigvals.real <= re[-1]) & (eigvals.imag >= im[0]) & (eigvals.imag <= im[-1])
    cs = plot_pseudospectra(z, g, f, eigvals[inside], isolated[inside], ax=ax)
    ax.set_title(rf"{title}, $n={n}$, rounding floor $\approx${f:.0e}", fontsize=10, loc="left")
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label(r"$\log_{10}\,\gamma_n(z)$  (darker = deeper in the pseudospectrum)")
fig.suptitle(rf"$\gamma_n$ pseudospectra, constant velocity $v_0={v0}$", fontsize=11)

if len(sys.argv) > 1:
    fig.savefig(sys.argv[1], dpi=150)
else:
    plt.show()
