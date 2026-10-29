"""
Why is only the ground mode of the subsonic (v_m = 0.5) fixed-open flow unstable?

(1) Energy budget. Multiply thesis Eq. 3.6 by conj(u) and integrate over [-1, 1] with
    u(-1) = u'(1) = 0:  a w^2 + b w + c = 0  with
        a = int |u|^2                                   (real)
        b = 2i int conj(u) (v0 u' + v0' u)
          = i [ int v0' |u|^2 + v0(1)|u(1)|^2 ] - 2 Im int v0 conj(u) u'
        c = -int (1 - v0^2)|u'|^2 - int (v0 + 1/v0) v0' conj(u) u'
            - int [ (1 - 1/v0^2) v0'^2 + (v0 + 1/v0) v0'' ] |u|^2
    The imaginary part of the quadratic gives
        Im w = -( Re w * Im b + Im c ) / ( 2 Re w * a + Re b ),
    i.e. three sources of growth/damping in the numerator:
        boundary (radiation through the open end):  Re w * v0(1)|u(1)|^2
        profile gradient:                           Re w * int v0' |u|^2
        convective coupling:                        -Im int (v0 + 1/v0) v0' conj(u) u'
(2) Continuation in the mirror ratio R from 0 (constant v0 = 0.5, all modes Im w = -0.206)
    to 1.5 (thesis profile), tracking every mode by Newton polishing.
(3) Continuation in the throat Mach number v_m at R = 1.5.
"""
import os
import sys
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval, chebder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nozzle_infbeyn as nz  # noqa: E402
from inf_beyn import InfBeyn, Cheb, _leggauss  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
CASE = "subsonic_fixed_open"
C = {"b": "#2a78d6", "o": "#eb6834", "a": "#1baf7a", "v": "#6250d6", "k": "#0b0b0b"}
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.6,
})
warnings.filterwarnings("ignore")


def budget(lam, coeffs, vm=0.5, nq=600):
    """Terms of the integrated quadratic for eigenpair (lam, Chebyshev coeffs)."""
    x, w = _leggauss(nq)
    c = coeffs[: np.max(np.nonzero(np.abs(coeffs) > 0)) + 1]
    u = chebval(x, c)
    du = chebval(x, chebder(c))
    u1 = chebval(1.0, c)
    v0, dv0, d2v0 = nz.velocity_profile(x, vm, False)
    I = lambda f: np.sum(w * f)
    a = I(np.abs(u) ** 2).real
    X = I(np.conj(u) * (v0 * du + dv0 * u))
    b = 2j * X
    c_ = (-I((1 - v0**2) * np.abs(du) ** 2) - I((v0 + 1 / v0) * dv0 * np.conj(u) * du)
          - I(((1 - 1 / v0**2) * dv0**2 + (v0 + 1 / v0) * d2v0) * np.abs(u) ** 2))
    residual = abs(a * lam**2 + b * lam + c_) / (abs(a * lam**2) + abs(b * lam) + abs(c_))
    wr = lam.real
    boundary = wr * nz.velocity_profile(1.0, vm, False)[0] * abs(u1) ** 2
    gradient = wr * I(dv0 * np.abs(u) ** 2).real
    coupling = -I((v0 + 1 / v0) * dv0 * np.conj(u) * du).imag
    denom = 2 * wr * a + b.real
    im_w = -(boundary + gradient + coupling) / denom
    return dict(a=a, b=b, c=c_, residual=residual, boundary=boundary, gradient=gradient,
                coupling=coupling, denom=denom, im_w=im_w, im_b_check=b.imag - (boundary + gradient) / wr)


def problem(vm):
    """T, bc of the fixed-open problem with throat Mach number vm (R read from nz.R)."""

    def T(z, cb):
        v0, dv0, d2v0 = nz.velocity_profile(cb.x, vm, False)
        A1 = 2j * (v0[:, None] * cb.D1 + np.diag(dv0))
        A0 = ((1 - v0**2)[:, None] * cb.D2 - ((3 * v0 + 1 / v0) * dv0)[:, None] * cb.D1
              - np.diag((1 - 1 / v0**2) * dv0**2 + (v0 + 1 / v0) * d2v0))
        return z**2 * cb.I + z * A1 + A0

    return T, lambda cb: np.vstack([cb.D1[0], cb.I[-1]])


def collocation_eigs(vm, N=200):
    """All eigenvalues of the fixed-open problem by Chebyshev collocation on N+1 points (R read from nz.R)."""
    cb = Cheb(N)
    v0, dv0, d2v0 = nz.velocity_profile(cb.x, vm, False)
    A1 = 2j * (v0[:, None] * cb.D1 + np.diag(dv0))
    A0 = ((1 - v0**2)[:, None] * cb.D2 - ((3 * v0 + 1 / v0) * dv0)[:, None] * cb.D1
          - np.diag((1 - 1 / v0**2) * dv0**2 + (v0 + 1 / v0) * d2v0))
    # eliminate u(x0 = 1) with u'(1) = 0 and drop x[-1] = -1 where u = 0
    E = np.zeros((N + 1, N - 1), dtype=complex)
    E[1:-1] = np.eye(N - 1)
    E[0] = -cb.D1[0, 1:-1] / cb.D1[0, 0]
    P = np.eye(N + 1)[1:-1]
    w, _ = nz.polyeig(P @ A0 @ E, P @ A1 @ E, P @ np.eye(N + 1) @ E)
    return w


def track(eigs, param_values, set_param):
    """Follow each eigenvalue as a parameter is varied: nearest collocation eigenvalue at each step."""
    paths = np.zeros((len(param_values), len(eigs)), dtype=complex)
    cur = np.array(eigs, dtype=complex)
    for i, p in enumerate(param_values):
        vm = set_param(p)
        w = collocation_eigs(vm)
        cur = np.array([w[np.argmin(np.abs(w - lam))] for lam in cur])
        paths[i] = cur
    return paths


if __name__ == "__main__":
    d = np.load(os.path.join(HERE, "data", f"infbeyn_{CASE}.npz"))
    lam, F = d["eigvals"], d["eigfuncs"]
    n_show = 8

    # ---------------------------------------------------------------- (1) energy budget
    print("energy budget, subsonic v_m = 0.5, fixed-open (numerator terms of Im w; negative = damping)")
    print(f"{'n':>2} {'omega':>22} {'resid':>8} {'boundary':>10} {'gradient':>10} {'coupling':>10} {'denom':>9} {'Im w (budget)':>14} {'Im w':>9}")
    rows = []
    for k in range(n_show):
        r = budget(lam[k], F[:, k])
        rows.append(r)
        print(f"{k + 1:>2} {lam[k].real:>10.5f}{lam[k].imag:+.5f}i {r['residual']:>8.1e} {-r['boundary']:>10.4f} "
              f"{-r['gradient']:>10.4f} {-r['coupling']:>10.4f} {r['denom']:>9.4f} {r['im_w']:>14.5f} {lam[k].imag:>9.5f}")

    fig, ax = plt.subplots(figsize=(5.2, 2.8), layout="constrained")
    n = np.arange(1, n_show + 1)
    wdt = 0.26
    for j, (key, lab, col) in enumerate((("boundary", "radiation through open end", C["o"]),
                                         ("gradient", r"$\mathrm{Re}\,\omega\int v_0'|\tilde v|^2$", C["a"]),
                                         ("coupling", r"$-\mathrm{Im}\int(v_0+1/v_0)v_0'\bar{\tilde v}\tilde v'$", C["v"]))):
        vals = [-r[key] / r["denom"] for r in rows]
        ax.bar(n + (j - 1) * wdt, vals, wdt, color=col, label=lab, edgecolor="#fcfcfb", linewidth=0.5)
    ax.plot(n, lam[:n_show].imag, "o", color=C["k"], ms=4, label=r"$\mathrm{Im}\,\omega$ (sum)", zorder=5)
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_xlabel("mode number")
    ax.set_ylabel(r"contribution to $\mathrm{Im}\,\omega$")
    ax.set_title("$v_m=0.5$, fixed-open: energy budget of each mode", loc="left")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(FIG, "subsonic_fixed_open_budget.pdf"))
    plt.close(fig)

    # eigenfunction shapes against the profile, modes 1 and 2
    z = np.linspace(-1, 1, 801)
    v0, dv0, _ = nz.velocity_profile(z, 0.5, False)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), layout="constrained")
    for k, ax in enumerate(axes):
        c = F[:, k]
        c = c[: np.max(np.nonzero(np.abs(c) > 0)) + 1]
        u = chebval(z, c)
        u /= np.abs(u).max()
        ax.plot(z, np.abs(u) ** 2, color=C["b"], label=r"$|\tilde v|^2$")
        ax.plot(z, dv0 / np.abs(dv0).max(), color=C["o"], ls="--", lw=1.2, label=r"$v_0'$ (scaled)")
        ax.axhline(0, color="#52514e", lw=0.6)
        ax.set_title(f"mode {k + 1}: $\\omega={lam[k].real:.3f}{lam[k].imag:+.3f}i$", fontsize=9)
        ax.set_xlabel("$z$")
    axes[0].legend(frameon=False)
    fig.savefig(os.path.join(FIG, "subsonic_fixed_open_modeshapes.pdf"))
    plt.close(fig)

    # ---------------------------------------------------------------- (2) continuation in R
    n_track = 6
    R_values = np.linspace(1.5, 0.0, 61)
    def set_R(R):
        nz.R = R
        nz.BM = nz.B0 * (1 + R)  # keep the throat velocity at v_m = 0.5 while the bump shrinks
        return 0.5

    paths_R = track(lam[:n_track], R_values, set_R)
    nz.R, nz.BM = 1.5, 2.5
    exact0 = nz.exact_eigenvalues("constant_subsonic_fixed_open")
    print("\ncontinuation in R (mirror ratio): eigenvalues at R = 0 vs exact constant-velocity values")
    for k in range(n_track):
        lam0 = paths_R[-1, k]
        print(f"  mode {k + 1}: R=1.5 {lam[k]:.5f} -> R=0 {lam0:.5f}, nearest exact {exact0[np.argmin(np.abs(exact0 - lam0))]:.5f}")
    # R at which mode 1 crosses Im = 0
    im1 = paths_R[:, 0].imag
    idx = np.where(np.diff(np.sign(im1)) != 0)[0]
    R_cross = [np.interp(0, [im1[i + 1], im1[i]], [R_values[i + 1], R_values[i]]) for i in idx]
    print(f"  mode 1 crosses Im w = 0 at R = {R_cross}")

    # ---------------------------------------------------------------- (3) continuation in v_m
    vm_values = np.concatenate([np.linspace(0.5, 0.95, 46), np.linspace(0.5, 0.1, 41)[1:]])
    paths_vm = np.full((len(vm_values), n_track), np.nan + 0j)
    for idxs in (np.arange(46), np.arange(46, len(vm_values))):  # upward and downward from 0.5
        paths_vm[idxs] = track(lam[:n_track], vm_values[idxs], lambda v: v)
    vm_sorted = np.argsort(vm_values)
    print("\ncontinuation in v_m (throat Mach), R = 1.5: Im w of modes 1-3")
    for i in vm_sorted[::5]:
        print(f"  v_m = {vm_values[i]:.2f}: " + "  ".join(f"{paths_vm[i, k].imag:+.4f}" for k in range(3)))

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), layout="constrained")
    cols = [C["b"], C["o"], C["a"], C["v"], "#eda100", "#e87ba4"]
    for k in range(n_track):
        axes[0].plot(R_values, paths_R[:, k].imag, color=cols[k], label=f"mode {k + 1}")
        axes[1].plot(vm_values[vm_sorted], paths_vm[vm_sorted, k].imag, color=cols[k])
    for ax in axes:
        ax.axhline(0, color="#52514e", lw=0.6)
        ax.set_ylabel(r"$\mathrm{Im}\,\omega$")
    axes[0].set_xlabel("mirror ratio $R$ ($v_m=0.5$)")
    axes[1].set_xlabel("throat Mach number $v_m$ ($R=1.5$)")
    axes[0].set_title("profile amplitude", loc="left")
    axes[1].set_title("throat velocity", loc="left")
    axes[0].legend(frameon=False, ncol=2, fontsize=7)
    fig.suptitle("subsonic, fixed-open: growth rates under continuation", fontsize=10)
    fig.savefig(os.path.join(FIG, "subsonic_fixed_open_continuation.pdf"))
    plt.close(fig)
    np.savez(os.path.join(HERE, "data", "subsonic_fixed_open_continuation.npz"),
             R_values=R_values, paths_R=paths_R, vm_values=vm_values, paths_vm=paths_vm)
