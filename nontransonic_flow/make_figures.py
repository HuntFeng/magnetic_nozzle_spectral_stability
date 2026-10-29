"""Figures and LaTeX tables from data/infbeyn_*.npz (this directory) and ../data/*.npz (thesis runs)."""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nozzle_infbeyn import CASES, DATA, HERE, REGION, exact_eigenvalues, parse_case, velocity_profile  # noqa: E402

sys.path.insert(0, os.path.join(HERE, ".."))
from inf_beyn import plot_pseudospectra  # noqa: E402

OLD = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.6,
})
C = {"infbeyn": "#2a78d6", "fd": "#eb6834", "collocation": "#1baf7a", "galerkin": "#6250d6", "exact": "#0b0b0b"}
TEXT = "#0b0b0b"
MATCH = 0.1  # an old eigenvalue within this distance of an InfBeyn one is "the same mode"


def title(case):
    vm, constant, bc = parse_case(case)
    v = f"constant $v_0={vm}$" if constant else f"$v_m={vm}$"
    return f"{v}, {'Dirichlet' if bc == 'fixed_fixed' else 'fixed-open'}"


def old_eigs(case):
    """All eigenvalues of the thesis-era runs in the scanned window: {method: omega}."""
    out = {}
    for method in ("collocation", "galerkin"):
        f = os.path.join(OLD, f"{method}_{case}.npz")
        if os.path.exists(f):
            out[method] = np.load(f)["omega"]
    return out


def in_window(w, region=REGION, pad=0.0):
    x0, x1, y0, y1 = region
    return w[(w.real >= x0 - pad) & (w.real <= x1 + pad) & (w.imag >= y0 - pad) & (w.imag <= y1 + pad)]


def eigenfunction_values(coeffs, z):
    c = coeffs[: np.max(np.nonzero(np.abs(coeffs) > 0)) + 1]
    v = chebval(z, c)
    k = np.argmax(np.abs(v))
    return v / v[k]  # largest value real and = 1


def fmt(x, nd=6):
    s = f"{x.real:.{nd}f}"
    s += f"{x.imag:+.{nd}f}i"
    return s


def fmt_e(x):
    if not np.isfinite(x):
        return "--"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


# --------------------------------------------------------------------------- velocity profiles
fig, ax = plt.subplots(figsize=(4.2, 2.6))
z = np.linspace(-1, 1, 801)
for vm, lab, col in ((0.5, "subsonic, $v_m=0.5$", C["infbeyn"]), (1.5, "supersonic, $v_m=1.5$", C["fd"])):
    v0 = velocity_profile(z, vm, False)[0]
    ax.plot(z, v0, color=col, label=lab)
    ax.axhline(vm, color=col, ls=":", lw=1, label=f"constant $v_0={vm}$")
ax.axhline(1, color="#52514e", lw=0.8, ls="--")
ax.text(-0.98, 1.03, "sonic", color="#52514e", fontsize=7)
ax.set_xlabel("$z$")
ax.set_ylabel("$v_0$ (Mach)")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
fig.savefig(os.path.join(FIG, "velocity_profiles.pdf"))
plt.close(fig)

# --------------------------------------------------------------------------- per case
summary_rows = []
cmp_rows = []
for case in CASES:
    d = np.load(os.path.join(DATA, f"infbeyn_{case}.npz"), allow_pickle=True)
    lam = d["eigvals"]
    gamma, halo, polished = d["gamma"], d["halo"], d["polished"]
    exact = exact_eigenvalues(case)
    exact_w = in_window(exact, pad=0.5) if exact is not None else None
    old = old_eigs(case)
    fd = {N: d[f"fd_omega_N{N}"] for N in (101, 201)}

    # ---- spectrum on the pseudospectrum
    Z = d["psa_re"][None, :] + 1j * d["psa_im"][:, None]
    fig, ax = plt.subplots(figsize=(6.4, 2.4), layout="constrained")
    cs = plot_pseudospectra(Z, d["psa_gamma"], d["psa_floor"], lam, isolated=halo > 10 * gamma[:, -1], ax=ax, n_levels=16)
    if exact_w is not None:
        ax.plot(exact_w.real, exact_w.imag, "x", ms=5, color=C["exact"], mew=1, ls="none", label="exact")
    ax.legend(loc="upper right", frameon=True, fontsize=7)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}\gamma_n(\omega)$")
    ax.set_xlabel(r"Re $\omega$")
    ax.set_ylabel(r"Im $\omega$")
    ax.set_title(f"{title(case)}: InfBeyn eigenvalues and pseudospectrum")
    ax.grid(False)
    fig.savefig(os.path.join(FIG, f"{case}_spectrum.pdf"))
    plt.close(fig)

    # ---- comparison with the old discretize-then-eig methods
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw={"width_ratios": [1.6, 1]}, layout="constrained")
    series = [("FD, $N=101$", fd[101], C["fd"], "^", 5), ("FD, $N=201$", fd[201], C["fd"], "v", 5)]
    if "collocation" in old:
        series.append(("collocation, $N=101$", old["collocation"], C["collocation"], "s", 4.5))
    if "galerkin" in old:
        series.append(("Galerkin, 30 basis", old["galerkin"], C["galerkin"], "D", 4))
    for ax, (region, zoom) in zip(axes, ((REGION, False), (None, True))):
        if zoom:
            # zoom on the imaginary parts of the first modes
            y = lam.imag
            ylo, yhi = y.min() - 0.1, y.max() + 0.1
            region = (REGION[0], 14, ylo, yhi)
        for name, w, col, mk, ms in series:
            w = in_window(w, region)
            ax.plot(w.real, w.imag, mk, ms=ms, color=col, mfc="none", mew=1.1, ls="none", label=name)
        if exact_w is not None:
            ax.plot(exact_w.real, exact_w.imag, "x", ms=6, color=C["exact"], mew=1, ls="none", label="exact")
        ax.plot(lam.real, lam.imag, "o", ms=3.5, color=C["infbeyn"], ls="none", label="InfBeyn")
        ax.set_xlim(region[0], region[1])
        ax.set_ylim(region[2], region[3])
        ax.set_xlabel(r"Re $\omega$")
        ax.set_ylabel(r"Im $\omega$")
        ax.axhline(0, color="#52514e", lw=0.6)
    axes[0].set_title("scanned window", loc="left")
    axes[1].set_title("zoom on the first modes", loc="left")
    axes[0].legend(frameon=True, fontsize=6.5, loc="lower right", ncol=2)
    fig.suptitle(title(case), ha="center", fontsize=10)
    fig.savefig(os.path.join(FIG, f"{case}_comparison.pdf"))
    plt.close(fig)

    # ---- eigenfunctions of the first three modes
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.1), sharey=True, layout="constrained")
    zz = np.linspace(-1, 1, 1001)
    for k, ax in enumerate(axes):
        if k >= len(lam):
            ax.axis("off")
            continue
        v = eigenfunction_values(d["eigfuncs"][:, k], zz)
        ax.plot(zz, v.real, color=C["infbeyn"], label=r"Re $\tilde v$")
        ax.plot(zz, v.imag, color=C["infbeyn"], ls="--", lw=1.2, label=r"Im $\tilde v$")
        ax.set_title(f"$\\omega={lam[k].real:.3f}{lam[k].imag:+.3f}i$", fontsize=8)
        ax.set_xlabel("$z$")
        ax.set_xlim(-1, 1)
    axes[0].set_ylabel(r"$\tilde v$")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"{title(case)}: first three eigenfunctions", fontsize=10)
    fig.savefig(os.path.join(FIG, f"{case}_eigenfunctions.pdf"))
    plt.close(fig)

    # ---- distances old -> InfBeyn for the matched modes, and counts of spurious ones
    n_show = min(10, len(lam))
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    dist = {}
    spurious = {}
    for name, w, col, mk, ms in series:
        ww = in_window(w)
        dmat = np.abs(lam[:, None] - ww[None, :]) if len(ww) else np.full((len(lam), 1), np.inf)
        dist[name] = dmat.min(axis=1)
        spurious[name] = int(np.sum(dmat.min(axis=0) > MATCH)) if len(ww) else 0
        ax.plot(np.arange(1, n_show + 1), np.maximum(dist[name][:n_show], 1e-16), mk, ms=ms, color=col,
                mfc="none", mew=1.1, ls="none", label=name)
    if exact is not None:
        err = np.min(np.abs(exact[None, :] - lam[:, None]), axis=1)
        ax.plot(np.arange(1, n_show + 1), np.maximum(err[:n_show], 1e-16), "o", ms=3.5, color=C["infbeyn"],
                ls="none", label="InfBeyn vs exact")
    ax.set_yscale("log")
    ax.set_xlabel("mode number (by Re $\\omega$)")
    ax.set_ylabel(r"$|\omega_{\rm method}-\omega_{\rm ref}|$")
    ax.set_title(title(case), loc="left")
    ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(os.path.join(FIG, f"{case}_errors.pdf"))
    plt.close(fig)

    # ---- tables
    with open(os.path.join(TAB, f"{case}_eigs.tex"), "w") as f:
        cols = "rlllll" if exact is not None else "rllll"
        f.write("\\begin{tabular}{" + cols + "}\n\\toprule\n")
        f.write("$n$ & $\\omega_n$ (InfBeyn) & $\\gamma_{256}(\\omega_n)$ & halo & polished"
                + (" & $|\\omega_n-\\omega_n^{\\rm exact}|$" if exact is not None else "") + "\\\\\n\\midrule\n")
        for k in range(n_show):
            row = f"{k + 1} & ${fmt(lam[k], 8)}$ & {fmt_e(gamma[k, -1])} & {fmt_e(halo[k])} & {'yes' if polished[k] else 'no'}"
            if exact is not None:
                row += f" & {fmt_e(np.min(np.abs(exact - lam[k])))}"
            f.write(row + "\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(TAB, f"{case}_compare.tex"), "w") as f:
        names = [s[0] for s in series]
        f.write("\\begin{tabular}{r" + "l" * len(names) + "}\n\\toprule\n$n$ & " + " & ".join(names) + "\\\\\n\\midrule\n")
        for k in range(min(8, len(lam))):
            f.write(f"{k + 1} & " + " & ".join(fmt_e(dist[n][k]) for n in names) + "\\\\\n")
        f.write("\\midrule\nin window & " + " & ".join(str(len(in_window(s[1]))) for s in series) + "\\\\\n")
        f.write("spurious & " + " & ".join(str(spurious[n]) for n in names) + "\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    nsp = [spurious[n] for n in names]
    summary_rows.append((case, len(lam), int(d["n_solves"]), d["times"], lam.imag.max(), lam[np.argmax(lam.imag)],
                         (gamma[:, -1] / halo).max(), nsp))

with open(os.path.join(TAB, "summary.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrll}\n\\toprule\n")
    f.write("case & BC & \\#eig & BVP solves & scan [s] & $\\max\\,\\mathrm{Im}\\,\\omega$ (mode) & verdict\\\\\n\\midrule\n")
    for case, n, ns, t, mx, lm, _, _ in summary_rows:
        vm, constant, bc = parse_case(case)
        prof = f"constant $v_0={vm}$" if constant else f"$v_m={vm}$"
        verdict = "unstable" if mx > 1e-9 else "stable"
        f.write(f"{prof} & {'Dirichlet' if bc == 'fixed_fixed' else 'fixed-open'} & {n} & {ns} & {t[0]:.0f} & "
                f"${mx:.4f}$ (${fmt(lm, 3)}$) & {verdict}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(TAB, "spurious.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
    f.write("case & BC & InfBeyn & FD $N{=}101$ & FD $N{=}201$ & collocation & Galerkin\\\\\n\\midrule\n")
    for case, n, _, _, _, _, _, nsp in summary_rows:
        vm, constant, bc = parse_case(case)
        prof = f"constant $v_0={vm}$" if constant else f"$v_m={vm}$"
        f.write(f"{prof} & {'Dirichlet' if bc == 'fixed_fixed' else 'fixed-open'} & 0 & " + " & ".join(map(str, nsp)) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("done")
