"""
Animation of the fixed-open spectrum as v_m crosses 1 (subsonic -> supersonic).
One frame per collocation-continuation sample (data/continuation_fixed_open.npz),
with the certified InfBeyn sweep eigenvalues overlaid where they exist.
Needs an ffmpeg binary: uv run --with imageio-ffmpeg python make_movie.py
"""
import glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from common import *

try:
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True})

d = np.load(os.path.join(DATA, "continuation_fixed_open.npz"), allow_pickle=True)
vms, eigs = d["vms"], d["eigs"]
_, keep = np.unique(np.round(vms, 6), return_index=True)
vms, eigs = vms[keep], [np.asarray(w, dtype=complex) for w in eigs[keep]]
sweep = {round(float(re.search(r"vm([0-9.]+)\.npz", f).group(1)), 4): np.load(f)
         for f in glob.glob(os.path.join(DATA, "sweep_fixed_open_vm*.npz"))}
TRAIL, HOLD, HOLD_CERT = 4, 12, 6  # ghost frames; frames held at start / sonic crossing / end; at certified sweep points

# frame sequence: hold the first frame, pause on both sides of the sonic point, hold the end
i_cross = int(np.searchsorted(vms, 1.0))
run = lambda r: [i for i in r for _ in range(HOLD_CERT if round(float(vms[i]), 4) in sweep else 1)]
seq = [0] * HOLD + run(range(i_cross)) + [i_cross - 1] * HOLD + [i_cross] * HOLD + run(range(i_cross, len(vms))) + [len(vms) - 1] * HOLD

fig = plt.figure(figsize=(10, 5.6), dpi=150)
ax = fig.add_axes([0.08, 0.30, 0.82, 0.62])
bar = fig.add_axes([0.08, 0.10, 0.82, 0.06])
ax.set_xlim(REGION[0], REGION[1]); ax.set_ylim(-0.45, 1.85)
ax.axhline(0, color=INK2, lw=0.8)
ax.annotate("unstable", (1, 0), xycoords=("axes fraction", "data"), xytext=(6, 4), textcoords="offset points", va="bottom", color=INK2, fontsize=8, annotation_clip=False)
ax.annotate("stable", (1, 0), xycoords=("axes fraction", "data"), xytext=(6, -4), textcoords="offset points", va="top", color=INK2, fontsize=8, annotation_clip=False)
ax.set_xlabel(r"Re $\omega$"); ax.set_ylabel(r"Im $\omega$")
ghosts = [ax.plot([], [], "o", ms=5, ls="none", alpha=0.5 * (1 - (k + 1) / (TRAIL + 1)), mew=0)[0] for k in range(TRAIL)]
cur, = ax.plot([], [], "o", ms=8, ls="none", mec="white", mew=1.0)
cert, = ax.plot([], [], "o", ms=13, ls="none", mfc="none", mec=INK, mew=1.1)
ax.legend([plt.Line2D([], [], marker="o", ms=8, ls="none", color=INK2, mec="white"), plt.Line2D([], [], marker="o", ms=13, ls="none", mfc="none", mec=INK)],
          ["collocation, $N=300$ (converged to $10^{-6}$)", "InfBeyn, certified (only at sweep points)"], loc="upper right", frameon=False, fontsize=8.5)
title = ax.set_title("", loc="left", fontsize=12)
info = ax.text(0.99, 0.80, "", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK2)

# v_m progress strip, linear in v_m, sonic point marked
bar.set_xlim(vms[0], vms[-1]); bar.set_ylim(0, 1); bar.set_yticks([]); bar.grid(False)
for s in ("left", "right", "top"):
    bar.spines[s].set_visible(False)
bar.axvspan(vms[0], 1, color=BLUE, alpha=0.15, lw=0); bar.axvspan(1, vms[-1], color=ORANGE, alpha=0.15, lw=0)
bar.axvline(1, color=INK, lw=1.2)
bar.plot(vms, np.full_like(vms, 0.5), "|", color=INK2, ms=6, alpha=0.5)
bar.set_xlabel(r"throat Mach number $v_m$  (ticks: computed samples)")
marker, = bar.plot([], [], "v", color=INK, ms=10, clip_on=False)


def color(v):
    return BLUE if v < 1 else ORANGE


def draw(n):
    i = seq[n]
    v, w = vms[i], eigs[i]
    cur.set_data(w.real, w.imag); cur.set_color(color(v)); cur.set_markeredgecolor("white")
    for k, g in enumerate(ghosts):
        j = i - k - 1
        same_side = j >= 0 and (vms[j] < 1) == (v < 1)
        g.set_data((eigs[j].real, eigs[j].imag) if same_side else ([], []))
        g.set_color(color(v))
    s = sweep.get(round(float(v), 4))
    if s is not None:
        ok = s["halo"] > 10 * s["gamma"][:, -1]
        cert.set_data(s["eigvals"][ok].real, s["eigvals"][ok].imag)
    else:
        cert.set_data([], [])
    regime = "subsonic" if v < 1 else "supersonic"
    title.set_text(rf"fixed-open nozzle spectrum, $v_m = {v:.4f}$  ({regime}, $|1-v_m| = {abs(1 - v):.3g}$)")
    title.set_color(color(v))
    info.set_text(f"{len(w)} eigenvalues in window\nmax Im $\\omega$ = {w.imag.max():+.3f}")
    marker.set_data([v], [1.15])
    return [cur, cert, title, info, marker, *ghosts]


anim = FuncAnimation(fig, draw, frames=len(seq), blit=False)
out = os.path.join(FIG, "spectrum_transition.mp4")
anim.save(out, writer=FFMpegWriter(fps=8, bitrate=4000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
print(out, len(seq), "frames")
