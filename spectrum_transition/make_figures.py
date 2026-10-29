"""Figures and tables for the spectrum-transition readiness report. Skips what is not computed yet."""
import glob, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from numpy.polynomial.chebyshev import chebval
from common import *
import nozzle_infbeyn as nz

sys.path.insert(0, ROOT)
from inf_beyn import plot_pseudospectra  # noqa

TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.6,
})
fmt_e = lambda x: "--" if not np.isfinite(x) else "$%s\\times10^{%d}$" % tuple((lambda m, e: (m, int(e)))(*f"{x:.1e}".split("e")))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf")); plt.close(fig)


# ----------------------------------------------------------------- 1. profiles through the sonic point
z = np.linspace(-1, 1, 2001)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), layout="constrained")
ax = axes[0]
cmap_b = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#0d366b"])
cmap_o = LinearSegmentedColormap.from_list("o", ["#fbd9c9", "#8a3410"])
for i, vm in enumerate([0.5, 0.8, 0.9, 0.95, 0.99]):
    ax.plot(z, velocity_profile(z, vm, False)[0], color=cmap_b(0.25 + 0.75 * i / 4), lw=1.2)
for i, vm in enumerate([1.5, 1.2, 1.1, 1.05, 1.01]):
    ax.plot(z, velocity_profile(z, vm, False)[0], color=cmap_o(0.25 + 0.75 * i / 4), lw=1.2)
ax.plot(z, transonic_profile(z, True)[0], color=INK, lw=1.8, label="accelerating, $v_m=1$")
ax.axhline(1, color=INK2, lw=0.6, ls="--")
ax.set_xlabel("$z$"); ax.set_ylabel("$v_0$"); ax.set_title("profiles: $v_m=0.5\\to0.99$ (blue), $1.5\\to1.01$ (orange)", loc="left", fontsize=8.5)
ax.legend(frameon=False, loc="upper left")
ax = axes[1]
zz = np.linspace(-0.1, 0.1, 2001)
for i, vm in enumerate([0.9, 0.95, 0.99]):
    ax.plot(zz, 1 - velocity_profile(zz, vm, False)[0], color=cmap_b(0.4 + 0.6 * i / 2), lw=1.2, label=f"$v_m={vm}$")
ax.plot(zz, 1 - transonic_profile(zz, True)[0], color=INK, lw=1.8, label="$v_m=1$ (slope $\\pm2.32$)")
ax.set_yscale("symlog", linthresh=1e-3); ax.set_xlabel("$z$"); ax.set_ylabel("$1-v_0$")
ax.set_title("the subsonic family develops a cusp, not the smooth transonic profile", loc="left", fontsize=8.5)
ax.legend(frameon=False, fontsize=7)
save(fig, "profiles_near_sonic")

# ----------------------------------------------------------------- 2. resolution needed with / without the map
meas_plain = {0.5: 256, 0.9: 1024, 0.95: 2048, 0.99: np.nan}  # BVP degree reached by the adaptive solver (nan: unresolved at 4096)
fig, ax = plt.subplots(figsize=(4.6, 2.7), layout="constrained")
eps = np.array([0.5, 0.1, 0.05, 0.01])
ax.plot(eps, [meas_plain[v] for v in (0.5, 0.9, 0.95, 0.99)], "o", color=C[1], ms=6, label="plain Chebyshev (measured)")
ax.plot([0.01], [4096], "^", color=C[1], ms=7, mfc="none", label="unresolved at $N=4096$")
e = np.logspace(-4, 0, 50); ax.plot(e, 100 / e, color=C[1], lw=0.8, ls=":", label="$N\\approx100/|1-v_m|$")
meas_map = {0.5: 256, 0.2: 128, 0.1: 128, 0.05: 256, 0.01: 256, 1e-3: 512, 1e-4: 1024}
ax.plot(list(meas_map), list(meas_map.values()), "s", color=C[0], ms=5.5, label="throat-clustered grid (measured)")
ax.set_xscale("log"); ax.set_yscale("log", base=2); ax.invert_xaxis()
ax.set_xlabel("$|1-v_m|$"); ax.set_ylabel("Chebyshev degree $N$ per BVP solve")
ax.set_title("resolution of $T(\\omega)^{-1}V$, $|\\omega|\\leq 26$", loc="left")
ax.legend(frameon=False, fontsize=7, loc="upper left")
save(fig, "resolution_vs_vm")

# ----------------------------------------------------------------- 3. timing tables
plain = [json.loads(l) for l in open(os.path.join(DATA, "scan_timing.log")) if l.startswith("{")] if os.path.exists(os.path.join(DATA, "scan_timing.log")) else []
sweep = [json.loads(l) for l in open(os.path.join(DATA, "sweep.log")) if l.startswith("{")] if os.path.exists(os.path.join(DATA, "sweep.log")) else []
base = np.load(os.path.join(ROOT, "nontransonic_flow", "data", "infbeyn_subsonic_fixed_open.npz"))
with open(os.path.join(TAB, "timing.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrrrr}\n\\toprule\n$v_m$ & grid & scan [s] & BVP solves & max $N$ & \\#eig & unpolished & unresolved tiles\\\\\n\\midrule\n")
    f.write(f"0.5 & plain (thesis run) & {base['times'][0]:.0f} & {int(base['n_solves'])} & {base['eigfuncs'].shape[0]-1} & {len(base['eigvals'])} & {int(np.sum(~base['polished']))} & {len(base['unresolved'])}\\\\\n")
    for r in plain:
        f.write(f"{r['vm']} & plain & {r['time']:.0f} & {r['n_solves']} & {r['n_cheb_eigfunc']-1} & {r['n_eig']} & {r['n_unpolished']} & {r['n_unresolved']}\\\\\n")
    f.write("\\midrule\n")
    for r in sweep:
        f.write(f"{r['vm']} & mapped, $L={r['L']:.1f}$ & {r['time']:.0f} & {r['n_solves']} & {r['n_cheb_eigfunc']-1} & {r['n_eig']} & {r['n_unpolished']} & {r['n_unresolved']}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------- 4. sweep: spectrum vs v_m
files = sorted(glob.glob(os.path.join(DATA, "sweep_fixed_open_vm*.npz")))
if files:
    S = [(float(re.search(r"vm([0-9.]+)\.npz", f).group(1)), np.load(f)) for f in files]
    sub = [(v, d) for v, d in S if v < 1]; sup = [(v, d) for v, d in S if v > 1]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.0), sharex=True, layout="constrained")
    for ax, group, cmap, lab in ((axes[0], sub, cmap_b, "subsonic $v_m\\to1^-$"), (axes[1], sup, cmap_o, "supersonic $v_m\\to1^+$")):
        if not group:
            ax.set_title(lab + " (not computed yet)", loc="left"); continue
        eps_ = np.array([abs(1 - v) for v, _ in group])
        norm = Normalize(np.log10(eps_.min()) - 0.3, np.log10(eps_.max()) + 0.3)
        for v, d in group:
            lam = d["eigvals"]; ok = d["halo"] > 10 * d["gamma"][:, -1]
            col = cmap(1 - norm(np.log10(abs(1 - v))))
            ax.plot(lam[ok].real, lam[ok].imag, "o", ms=4, color=col, mec=INK, mew=0.3, ls="none", label=f"$v_m={v}$")
            ax.plot(lam[~ok].real, lam[~ok].imag, "x", ms=5, color=col, ls="none")
        if os.path.exists(os.path.join(DATA, "transonic_accelerating.npz")):
            t = np.load(os.path.join(DATA, "transonic_accelerating.npz"))
            ax.plot(t["eigvals"].real, t["eigvals"].imag, "*", ms=9, color=INK, ls="none", label="transonic (InfBeyn)")
        ax.axhline(0, color=INK2, lw=0.6); ax.set_ylabel("Im $\\omega$"); ax.set_title(lab, loc="left")
        ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="lower right")
    axes[1].set_xlabel("Re $\\omega$"); axes[1].set_xlim(REGION[0], REGION[1])
    save(fig, "sweep_spectra")

    # growth rate of the first modes vs v_m, tracking by nearest neighbour
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), layout="constrained")
    for ax, group, lab in ((axes[0], sub, "subsonic side"), (axes[1], sup, "supersonic side")):
        if not group:
            continue
        group = sorted(group, key=lambda t: -abs(1 - t[0]))
        v0_, d0 = group[0]; lam0 = d0["eigvals"][d0["eigvals"].real > 0.05]
        n_track = min(6, len(lam0)); cur = lam0[:n_track].copy()
        paths = []
        for v, d in group:
            lam = d["eigvals"]; cur = np.array([lam[np.argmin(np.abs(lam - c))] for c in cur]); paths.append(cur)
        paths = np.array(paths); e_ = np.array([abs(1 - v) for v, _ in group])
        for k in range(n_track):
            ax.plot(e_, paths[:, k].imag, "o-", ms=3, color=C[k], label=f"mode {k+1}")
        ax.set_xscale("log"); ax.invert_xaxis(); ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xlabel("$|1-v_m|$"); ax.set_ylabel("Im $\\omega$"); ax.set_title(lab, loc="left"); ax.legend(frameon=False, fontsize=6.5, ncol=2)
    save(fig, "sweep_growth_rates")

# ----------------------------------------------------------------- 5. collocation continuation: speed of eigenvalues, grid estimate
for bc in ("fixed_open", "fixed_fixed"):
    fn = os.path.join(DATA, f"continuation_{bc}.npz")
    if not os.path.exists(fn):
        continue
    d = np.load(fn, allow_pickle=True); vms, eigs = d["vms"], d["eigs"]
    _, keep = np.unique(np.round(vms, 6), return_index=True); vms, eigs = vms[keep], eigs[keep]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), layout="constrained")
    ax = axes[0]
    for v, w in zip(vms, eigs):
        w = np.asarray(w, dtype=complex)
        col = cmap_b(0.3 + 0.7 * (v - 0.5) / 0.5) if v < 1 else cmap_o(0.3 + 0.7 * (1.5 - v) / 0.5)
        ax.plot(w.real, w.imag, ".", ms=2.5, color=col, ls="none")
    ax.axhline(0, color=INK2, lw=0.6); ax.set_xlabel("Re $\\omega$"); ax.set_ylabel("Im $\\omega$"); ax.set_xlim(REGION[0], REGION[1])
    if bc == "fixed_fixed":
        ax.set_ylim(-0.5, 0.5)
    ax.set_title(f"{'fixed-open' if bc == 'fixed_open' else 'Dirichlet'}, $v_m=0.5\\ldots1.5$", loc="left", fontsize=9)
    # speed |d omega / d vm| and min spacing on each side
    ax = axes[1]
    for side, mask, col in (("$v_m<1$", vms < 1, C[0]), ("$v_m>1$", vms > 1, C[1])):
        vv = vms[mask]; ww = [np.asarray(w, dtype=complex) for w in eigs[mask]]
        speed, gap = [], []
        for i in range(len(vv) - 1):
            a, b = ww[i], ww[i + 1]
            if len(a) == 0 or len(b) == 0:
                speed.append(np.nan); gap.append(np.nan); continue
            move = np.abs(b[:, None] - a[None, :]).min(axis=1)  # each new eigenvalue to the nearest old one
            speed.append(np.median(move) / abs(vv[i + 1] - vv[i]))
            dd = np.abs(a[:, None] - a[None, :]); np.fill_diagonal(dd, np.inf); gap.append(dd.min())
        vmid = 0.5 * (vv[1:] + vv[:-1])
        ax.plot(np.abs(1 - vmid), speed, "o-", ms=3, color=col, label=f"median $|d\\omega/dv_m|$, {side}")
        ax.plot(np.abs(1 - vmid), gap, "s--", ms=3, color=col, mfc="none", label=f"min eigenvalue spacing, {side}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlabel("$|1-v_m|$")
    ax.set_title("how fast the eigenvalues move", loc="left"); ax.legend(frameon=False, fontsize=6.5)
    save(fig, f"continuation_{bc}")

# ----------------------------------------------------------------- 6. transonic: InfBeyn vs Frobenius shooting
fn = os.path.join(DATA, "transonic_accelerating.npz")
fr = np.load(os.path.join(DATA, "frobenius_refined.npz"))
if os.path.exists(fn):
    t = np.load(fn); lam = t["eigvals"]
    Z = t["psa_re"][None, :] + 1j * t["psa_im"][:, None]
    fig, ax = plt.subplots(figsize=(6.6, 3.0), layout="constrained")
    cs = plot_pseudospectra(Z, t["psa_gamma"], t["psa_floor"], lam, isolated=t["halo"] > 10 * t["gamma"][:, -1], ax=ax, n_levels=14)
    ax.plot(fr["thesis"].real, fr["thesis"].imag, "x", ms=7, color=C[1], mew=1.5, ls="none", label="thesis Frobenius + RK4 shooting")
    ax.plot(fr["refined"].real, fr["refined"].imag, "+", ms=8, color=C[3], mew=1.5, ls="none", label="careful shooting (series start, DOP853)")
    ax.legend(loc="upper right", frameon=True, fontsize=6.5)
    fig.colorbar(cs, ax=ax, pad=0.02).set_label(r"$\log_{10}\gamma_n(\omega)$")
    ax.set_xlabel("Re $\\omega$"); ax.set_ylabel("Im $\\omega$"); ax.grid(False)
    ax.set_title("accelerating transonic flow, half nozzle $[-1,0]$ with the regularity condition at the throat", loc="left", fontsize=8.5)
    save(fig, "transonic_spectrum")
    # eigenfunctions
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1), sharey=True, layout="constrained")
    xx = np.linspace(-1, 1, 801); zz = (xx - 1) / 2
    for k, ax in enumerate(axes):
        if k >= len(lam):
            ax.axis("off"); continue
        c = t["eigfuncs"][:, k]; u = chebval(xx, c); u /= u[np.argmax(np.abs(u))]
        ax.plot(zz, u.real, color=C[0], label="Re"); ax.plot(zz, u.imag, color=C[0], ls="--", lw=1.2, label="Im")
        ax.set_title(f"$\\omega={lam[k].real:.3f}{lam[k].imag:+.3f}i$", fontsize=8); ax.set_xlabel("$z$")
    axes[0].legend(frameon=False); axes[0].set_ylabel("$\\tilde v$")
    save(fig, "transonic_eigenfunctions")
    with open(os.path.join(TAB, "transonic.tex"), "w") as f:
        f.write("\\begin{tabular}{rlllll}\n\\toprule\n$n$ & $\\omega_n$ (InfBeyn) & $\\gamma_{256}$ & halo & thesis shooting & $|\\Delta|$\\\\\n\\midrule\n")
        for k, l in enumerate(lam):
            j = np.argmin(np.abs(fr["thesis"] - l)); dth = abs(fr["thesis"][j] - l)
            th = f"${fr['thesis'][j].real:.4f}{fr['thesis'][j].imag:+.4f}i$" if dth < 1 else "--"
            f.write(f"{k+1} & ${l.real:.6f}{l.imag:+.6f}i$ & {fmt_e(t['gamma'][k,-1])} & {fmt_e(t['halo'][k])} & {th} & {fmt_e(dth) if dth < 1 else '--'}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
with open(os.path.join(TAB, "frobenius.tex"), "w") as f:
    f.write("\\begin{tabular}{rll}\n\\toprule\n$n$ & thesis $\\omega_n$ & careful shooting from it\\\\\n\\midrule\n")
    for k, (a, b) in enumerate(zip(fr["thesis"], fr["refined"])):
        f.write(f"{k+1} & ${a.real:.4f}{a.imag:+.4f}i$ & ${b.real:.4f}{b.imag:+.4f}i$ \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("figures done")
