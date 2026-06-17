"""Compose the four publication figures for the PRX Quantum manuscript.

Reuses the verified datasets in ``publication_ready/`` and the optimization
record in ``invDesResults/full_seed/``; no cloud simulations. Outputs:
  figures/fig1_device.pdf        platform schematic + final device + cross-section
  figures/fig2_optimization.pdf  loss trace + intermediate design snapshots
  figures/fig3_fields_bands.pdf  field cross-sections + Bloch band structure
  figures/fig4_performance.pdf   metric trajectories + design lineage + trim

    PYTHONPATH=. python paper/make_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Ellipse, Rectangle, FancyBboxPatch
import numpy as np

from inverse_design import config as cfg
from inverse_design import record as record_mod
from inverse_design.visualization import draw_layout_topview
from publication_ready import paths
from publication_ready.lib import datasets, layout_ops

# ── publication style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "savefig.dpi": 400,
    "figure.dpi": 200,
})

COL = 3.4          # single-column width (in)
WIDE = 7.05        # two-column-spanning width (in)
F_D2_THZ = cfg.F_TARGET / 1e12
HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
RECORD_DIR = cfg.RESULTS_ROOT / "full_seed"


def panel(ax, label, x=0.012, y=0.97, color="k"):
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="top", ha="left", color=color,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return out


# ── data loaders ───────────────────────────────────────────────────────────

def load_final_layout():
    fc = datasets.read_json(paths.FINAL_CANDIDATE_JSON)
    par = layout_ops.CavityParametrization(n_cells=fc["n_cells"],
                                           context=fc["context"])
    layout = layout_ops.layout_from_dict(fc["layout"], par)
    return layout, dict(fc["context"]), par, fc


def load_record():
    return record_mod.load_record(RECORD_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — platform schematic + final device + cross-section
# ═══════════════════════════════════════════════════════════════════════════

def _draw_atom(ax, x, y, r=0.15, color="#7a52c4", z=8):
    """An atom drawn as a fuzzy electron-cloud blob (layered translucent rings
    around a small nucleus)."""
    for rr, a in ((1.9, 0.10), (1.45, 0.15), (1.05, 0.24), (0.65, 0.42)):
        ax.add_patch(plt.Circle((x, y), r * rr, facecolor=color, edgecolor="none",
                                alpha=a, zorder=z))
    ax.add_patch(plt.Circle((x, y), r * 0.22, facecolor="#2e1d59",
                            edgecolor="none", zorder=z + 1))


def _draw_photon(ax, x0, y0, x1, y1, n=4, amp=0.12, color="#e2861c", lw=1.7,
                 z=5, arrow=True):
    """A 'wiggling photon': a sine wave from (x0,y0) to (x1,y1) tapered at the
    ends, with an arrowhead at the tip."""
    L = float(np.hypot(x1 - x0, y1 - y0))
    if L == 0:
        return
    t = np.linspace(0, 1, 120)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L           # along the path
    px, py = -uy, ux                                 # perpendicular
    env = np.clip(np.sin(np.pi * t) * 1.4, 0, 1)     # taper both ends
    w = amp * env * np.sin(2 * np.pi * n * t)
    X = x0 + ux * t * L + px * w
    Y = y0 + uy * t * L + py * w
    ax.plot(X, Y, color=color, lw=lw, zorder=z, solid_capstyle="round")
    if arrow:
        ax.add_patch(FancyArrowPatch((X[-3], Y[-3]), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=9, lw=lw, color=color, zorder=z))


def draw_schematic(ax):
    """Concept of the interconnect: a tweezer-trapped atom (electron cloud) above
    a suspended PhC cavity maps its qubit state onto a flying photon that is
    collected into fiber and sent to a remote node; an array of such cavities,
    over which atoms are transported, scales the link."""
    ax.set_xlim(0, 11.7)
    ax.set_ylim(0, 5.55)
    ax.set_aspect("equal")
    ax.axis("off")

    # suspended substrate with the etched undercut
    ax.add_patch(Rectangle((0, 0.35), 2.15, 4.3, facecolor="#cdc6ba",
                           edgecolor="#7a7366", hatch="////", lw=0.8, zorder=1))
    ax.text(1.05, 0.85, "Si\nsubstrate", ha="center", va="center", fontsize=7)
    ax.annotate("etched\nundercut", xy=(2.15, 3.5), xytext=(1.05, 2.55),
                fontsize=6.0, ha="center", va="center", color="#4a4438",
                arrowprops=dict(arrowstyle="->", lw=0.5, color="#5a5346"))

    finger_y = [1.05, 2.45, 3.85]
    fh = 0.40
    xL, xR, xd = 1.75, 7.0, 3.95
    for yc in finger_y:
        # cavity-mode glow at the defect
        for rr, a in ((0.95, 0.12), (0.62, 0.20), (0.34, 0.32)):
            ax.add_patch(Ellipse((xd, yc), rr, rr * 0.6, facecolor="#ffcf57",
                                 edgecolor="none", alpha=a, zorder=2))
        ax.add_patch(FancyBboxPatch((xL, yc - fh / 2), xR - xL, fh,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor="#9bbbd6", edgecolor="#3f5d77",
                                    lw=0.8, zorder=3))
        for x in np.linspace(xL + 0.7, xR - 0.25, 21):
            r = 0.08 if abs(x - xd) > 0.5 else 0.045
            ax.add_patch(Ellipse((x, yc), 0.13, 2 * r, facecolor="white",
                                 edgecolor="#3f5d77", lw=0.35, zorder=4))
        # the cavity maps the atomic state onto a photon that leaves the waveguide
        _draw_photon(ax, xd + 0.35, yc, xR + 0.95, yc, n=4, amp=0.10)
    ax.text(xd, finger_y[0] - 0.42, "cavity mode", fontsize=6.0, ha="center",
            va="top", color="#a9780a")
    ax.text(6.5, finger_y[0] - 0.42, "emitted photon", fontsize=6.0,
            ha="center", va="top", color="#c4730f")
    ax.annotate("SiN photonic-crystal cavity", xy=(5.4, finger_y[1] - 0.2),
                xytext=(5.4, finger_y[1] - 0.82), fontsize=6.1, ha="center",
                va="top", color="#27435a",
                arrowprops=dict(arrowstyle="->", lw=0.5, color="#27435a"))

    # tweezer-trapped electron-cloud atoms above the TOP finger (open space above
    # leaves room for the labels); the middle finger feeds the flying qubit below.
    ycm = finger_y[1]
    yact = finger_y[2]
    atom_x = [xd, xd + 1.05, xd + 2.1]
    ax.plot([xd, xd], [yact + 0.16, yact + 0.42], color="#7a52c4", ls=":", lw=0.9,
            zorder=5)
    for i, x in enumerate(atom_x):
        _draw_atom(ax, x, yact + 0.55, r=0.15 if i == 0 else 0.115)
    ax.annotate("trapped atom (electron cloud)", xy=(xd, yact + 0.66),
                xytext=(xd + 0.15, yact + 1.32), fontsize=6.2, color="#4a2f8a",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.55, color="#4a2f8a"))
    ax.add_patch(FancyArrowPatch((atom_x[1] + 0.35, yact + 0.55),
                                 (atom_x[-1] + 0.55, yact + 0.55), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.9, color="#555", zorder=7))
    ax.text(atom_x[-1] + 0.6, yact + 0.55, "transport", ha="left", va="center",
            fontsize=6.0, color="#444")

    # collection: photons -> fiber -> flying photonic qubit -> remote node
    xf = 8.2
    ax.plot([xf, xf], [finger_y[0] - 0.25, finger_y[-1] + 0.25], color="#555",
            lw=2.4, solid_capstyle="round", zorder=4)
    ax.text(xf + 0.16, finger_y[-1] - 0.05, "fiber", fontsize=6.2, ha="left",
            va="center", color="#444")
    _draw_photon(ax, xf + 0.12, ycm, 11.0, ycm, n=4, amp=0.15, color="#d24b4b",
                 lw=1.9)
    ax.text(11.15, ycm + 0.5, "flying\nphotonic\nqubit", fontsize=6.4,
            ha="left", va="center", color="#a82c2c")
    ax.annotate("to remote node /\nquantum network", xy=(11.0, ycm),
                xytext=(9.7, finger_y[0] - 0.5), fontsize=5.9, ha="center",
                va="top", color="#777",
                arrowprops=dict(arrowstyle="->", lw=0.5, color="#999"))


def fig1():
    layout, ctx, par, fc = load_final_layout()
    pos = np.asarray(layout["positions"])
    fig = plt.figure(figsize=(WIDE, 5.3))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.55, 0.5, 1.0],
                          hspace=0.5, wspace=0.32)

    ax_s = fig.add_subplot(gs[0, :])
    draw_schematic(ax_s)
    panel(ax_s, "a", y=0.99)

    # (b) full device top view
    ax_b = fig.add_subplot(gs[1, :])
    draw_layout_topview(layout, ctx, ax_b)
    ax_b.set_title("")
    ax_b.annotate("output port ($-x$)", xy=(pos[0] + 0.3, 0),
                  xytext=(pos[0] + 3.0, ctx["width"] * 2.0), fontsize=6.6,
                  ha="center", arrowprops=dict(arrowstyle="->", lw=0.7, color="C3"))
    ax_b.annotate("defect (atom site)", xy=(0, 0),
                  xytext=(0, ctx["width"] * 2.2), fontsize=6.6, ha="center",
                  arrowprops=dict(arrowstyle="->", lw=0.7, color="C0"))
    ax_b.text(pos[-1] * 0.6, -ctx["width"] * 2.0, "back mirror", fontsize=6.3,
              ha="center", color="#444")
    ax_b.set_xlabel(r"$x$ ($\mu$m)", labelpad=1)
    ax_b.set_ylabel(r"$y$", labelpad=1)
    panel(ax_b, "b")

    # (c) defect-region zoom
    ax_c = fig.add_subplot(gs[2, :2])
    draw_layout_topview(layout, ctx, ax_c)
    ax_c.set_title("")
    ax_c.set_xlim(-2.2, 2.2)
    ax_c.set_ylim(-0.42, 0.42)
    ax_c.set_xlabel(r"$x$ ($\mu$m)")
    ax_c.set_ylabel(r"$y$ ($\mu$m)")
    ax_c.set_title("graded defect (free-form holes)", fontsize=7.5)
    panel(ax_c, "c")

    # (d) yz cross-section with the atom height
    ax_d = fig.add_subplot(gs[2, 2])
    w, t = ctx["width"], ctx["thickness"]
    ax_d.add_patch(Rectangle((-w / 2, -t / 2), w, t, facecolor="#9bbbd6",
                             edgecolor="#3f5d77", lw=1.0))
    za = t / 2 + 0.25
    ax_d.add_patch(plt.Circle((0, za), 0.028, facecolor="#d24b4b", edgecolor="k",
                              lw=0.4))
    ax_d.annotate("", xy=(0, za), xytext=(0, t / 2),
                  arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.3"))
    ax_d.text(0.04, (za + t / 2) / 2, "250 nm", fontsize=6, va="center")
    ax_d.text(0, -t / 2 - 0.05, "500 nm wide", fontsize=6, ha="center", va="top")
    ax_d.text(-w / 2 - 0.03, 0, "150 nm", fontsize=6, ha="right", va="center",
              rotation=90)
    ax_d.text(0, za + 0.05, "atom", fontsize=6, ha="center", va="bottom",
              color="#a82c2c")
    ax_d.set_xlim(-0.42, 0.42)
    ax_d.set_ylim(-0.18, 0.46)
    ax_d.set_aspect("equal")
    ax_d.set_xlabel(r"$y$ ($\mu$m)")
    ax_d.set_ylabel(r"$z$ ($\mu$m)")
    ax_d.set_title("cross-section", fontsize=7.5)
    panel(ax_d, "d")

    _save(fig, "fig1_device.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — loss trace + intermediate design snapshots
# ═══════════════════════════════════════════════════════════════════════════

def fig2():
    rec = load_record()
    iters = [e["params"]["iteration"] for e in rec]
    eta_tilde = [e["perf"]["proxy"]["eta_tilde"] for e in rec]   # surrogate eta minimized
    stages = [e["params"].get("stage", "") for e in rec]
    bnd = next((iters[i] for i in range(1, len(stages))
                if stages[i] != stages[i - 1]), None)
    total_h = sum(e["time"]["duration_s"] for e in rec) / 3600.0

    # verified (non-gradient) diagnostics
    diag = [(e["params"]["iteration"], e["perf"]["diagnostic"])
            for e in rec if e["perf"].get("diagnostic")]
    d_it = np.array([it for it, _ in diag])
    d_eta = np.array([d["eta_atom"] for _, d in diag])
    best = np.maximum.accumulate(d_eta)
    ETA_BASE = 0.083                                          # hand-tuned baseline (design_1)

    fig = plt.figure(figsize=(WIDE, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.2], hspace=0.16)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.24)
    gs_bot = gs[1].subgridspec(3, 1, hspace=0.22)

    # (a) verified interface efficiency vs iteration (top-left)
    axa = fig.add_subplot(gs_top[0, 0])
    axa.axhline(ETA_BASE, ls=":", color="0.5", lw=1,
                label=f"baseline ({ETA_BASE:.2f})")
    axa.plot(d_it, best, "-", color="C3", lw=2.4, zorder=3, label="best so far")
    axa.plot(d_it, d_eta, "o-", color="C2", ms=4, lw=0.8, alpha=0.7, zorder=2,
             label=r"verified $\eta$")
    if bnd is not None:
        axa.axvline(bnd - 0.5, color="0.6", ls="--", lw=0.9)
        axa.text(bnd - 0.6, 0.9, "layout", ha="right", va="top", fontsize=6.3,
                 color="0.45")
        axa.text(bnd + 0.3, 0.9, "shapes", ha="left", va="top", fontsize=6.3,
                 color="0.45")
    axa.annotate(f"{d_eta[0]:.2f}", (d_it[0], d_eta[0]), textcoords="offset points",
                 xytext=(7, 2), fontsize=7, color="C2")
    axa.annotate(f"{d_eta[-1]:.2f}", (d_it[-1], d_eta[-1]), textcoords="offset points",
                 xytext=(-20, 3), fontsize=7, color="C3")
    axa.set_ylim(0, 0.95)
    axa.set_ylabel(r"verified efficiency $\eta$")
    axa.set_xlabel("iteration")
    axa.set_title(f"optimization time {total_h:.1f} h", fontsize=7.5)
    axa.legend(fontsize=6.0, loc="lower right", framealpha=0.9)
    panel(axa, "a")

    # (b) verified loss -log(eta): best-so-far falls monotonically to the design
    axb = fig.add_subplot(gs_top[0, 1])
    d_loss = -np.log(d_eta)
    best_loss = np.minimum.accumulate(d_loss)
    axb.semilogy(iters, -np.log(np.clip(eta_tilde, 1e-4, None)), color="0.78",
                 lw=0.7, zorder=1, label=r"proxy $-\log\tilde\eta$")
    axb.semilogy(d_it, d_loss, "o", color="C2", ms=3.5, alpha=0.7, zorder=2,
                 label=r"verified $-\log\eta$")
    axb.semilogy(d_it, best_loss, "-", color="C3", lw=2.4, zorder=3,
                 label="best so far")
    if bnd is not None:
        axb.axvline(bnd - 0.5, color="0.6", ls="--", lw=0.9)
    axb.annotate(f"{d_loss[0]:.1f}", (d_it[0], d_loss[0]), textcoords="offset points",
                 xytext=(5, -1), fontsize=7, color="C2")
    axb.annotate(f"{best_loss[-1]:.2f}", (d_it[-1], best_loss[-1]),
                 textcoords="offset points", xytext=(-20, 4), fontsize=7, color="C3")
    axb.set_xlabel("iteration")
    axb.set_ylabel(r"objective $-\log\eta$")
    axb.legend(fontsize=5.6, loc="upper right", framealpha=0.9, labelspacing=0.3)
    axb.set_ylim(0.1, 30)
    panel(axb, "b")

    # (c)-(e) full-width structure snapshots: the hole schedule evolving
    snap_iters = [iters[0], iters[len(iters) // 3], iters[-1]]
    from inverse_design.visualization import layout_from_entry
    by_iter = {e["params"]["iteration"]: e for e in rec}
    for r, it in enumerate(snap_iters):
        ax = fig.add_subplot(gs_bot[r])
        lay, c = layout_from_entry(by_iter[it])
        draw_layout_topview(lay, c, ax)
        ax.set_aspect("auto")
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(y=0.16)             # headroom so the label clears the device
        ax.text(0.006, 0.93, f"({'cde'[r]}) iter {it}", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.2, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="0.7", lw=0.4))
    _save(fig, "fig2_optimization.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — field cross-sections + band structure
# ═══════════════════════════════════════════════════════════════════════════

def _field_panel(ax, fields, tag, comp, cmap, label, signed=False,
                 show_xlabel=True):
    A = np.asarray(fields[f"{tag}_{comp}"])
    dims = [str(x) for x in fields[f"{tag}_dims"]]
    c0 = np.asarray(fields[f"{tag}_{dims[0]}"])
    c1 = np.asarray(fields[f"{tag}_{dims[1]}"])
    kw = dict(cmap=cmap, shading="auto", rasterized=True)
    if signed:
        vmax = float(np.nanmax(np.abs(A)))
        kw.update(vmin=-vmax, vmax=vmax)
    im = ax.pcolormesh(c0, c1, A.T, **kw)
    if show_xlabel:
        ax.set_xlabel(rf"${dims[0]}$ ($\mu$m)")
    ax.set_ylabel(rf"${dims[1]}$ ($\mu$m)")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    return im


def _true_box(ax):
    """Set the axes box aspect to the true data aspect (height/width). Unlike
    set_aspect('equal'), this plays well with constrained_layout + gridspec."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_box_aspect(abs(y1 - y0) / abs(x1 - x0))


def fig3():
    fields = dict(np.load(paths.DATA_DIR / "fields_final.npz", allow_pickle=True))
    bands_f = datasets.read_json(paths.DATA_DIR / "bands_final.json")
    za = float(fields["atom_height_um"])

    fig = plt.figure(figsize=(WIDE, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.55], hspace=0.16)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.30)
    gs_bot = gs[1].subgridspec(3, 1, hspace=0.10)

    # (a) defect cross-section yz at x=0 (near-square) -- top left
    ax = fig.add_subplot(gs_top[0, 0])
    _field_panel(ax, fields, "vertical_x", "Eabs", "magma", r"$|E|$ (arb.)")
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.55, 0.62)
    _true_box(ax)
    ax.axhline(za, color="cyan", ls="--", lw=0.9)
    ax.plot(0.0, za, marker="*", ms=10, color="cyan", mec="k", mew=0.5, zorder=6)
    ax.text(-0.82, za + 0.03, "atom", color="cyan", fontsize=6.5, va="bottom")
    ax.set_title(r"defect cut ($x=0$)", fontsize=7.5)
    panel(ax, "a", color="w")

    # (b) Bloch band structure -- top right
    axb = fig.add_subplot(gs_top[0, 1])

    # (c) beam midplane xy |E| -- full width, true aspect
    axc = fig.add_subplot(gs_bot[0])
    _field_panel(axc, fields, "midplane_z", "Eabs", "magma", r"$|E|$ (arb.)",
                 show_xlabel=False)
    axc.set_xlim(-2.6, 2.6)
    axc.set_ylim(-0.5, 0.5)
    _true_box(axc)
    # atom sits at x=0 but 250 nm ABOVE this plane: mark the column, not a point
    axc.axvline(0.0, color="cyan", ls=":", lw=0.8, zorder=5)
    axc.plot(0.0, 0.0, marker="*", ms=9, color="cyan", mec="k", mew=0.5, zorder=6)
    axc.text(0.08, 0.46, "atom column\n($250$ nm above)", color="cyan",
             fontsize=5.6, ha="left", va="top")
    # the mode is over-coupled to the -x output, so the envelope leans that way
    axc.annotate("to output ($-x$)", xy=(-2.45, 0.36), xytext=(-1.05, 0.36),
                 color="w", fontsize=6.0, va="center",
                 arrowprops=dict(arrowstyle="->", color="w", lw=1.0))
    axc.set_title(r"beam midplane ($z=0$)", fontsize=7.5)
    panel(axc, "c", color="w")

    # (d) longitudinal xz |E| + atom height -- full width
    axd = fig.add_subplot(gs_bot[1], sharex=axc)
    _field_panel(axd, fields, "vertical_y", "Eabs", "magma", r"$|E|$ (arb.)",
                 show_xlabel=False)
    axd.set_ylim(-0.55, 0.62)
    _true_box(axd)
    axd.axhline(za, color="cyan", ls="--", lw=0.9)
    axd.plot(0.0, za, marker="*", ms=10, color="cyan", mec="k", mew=0.5, zorder=6)
    axd.text(0.08, za + 0.04, "atom", color="cyan", fontsize=6.5, va="bottom")
    axd.set_title(r"longitudinal ($y=0$),  $|E|$", fontsize=7.5)
    panel(axd, "d", color="w")

    # (e) longitudinal xz Re(Ey) -- the standing-wave sign structure
    axe = fig.add_subplot(gs_bot[2], sharex=axc)
    _field_panel(axe, fields, "vertical_y", "Ey", "RdBu_r", r"$\mathrm{Re}\,E_y$",
                 signed=True)
    axe.set_ylim(-0.55, 0.62)
    _true_box(axe)
    axe.axhline(za, color="0.25", ls="--", lw=0.7)
    axe.plot(0.0, za, marker="*", ms=9, color="yellow", mec="k", mew=0.5, zorder=6)
    axe.set_title(r"longitudinal ($y=0$),  $\mathrm{Re}\,E_y$ (standing wave)",
                  fontsize=7.5)
    panel(axe, "e")

    # band structure (mirror + defect cells) -- panel (b).
    # Each Bloch unit-cell ring-down returns several resonances per k (plus
    # spurious near-DC fits); we trace the LOWEST physical band of each cell --
    # the dielectric band whose zone-edge maximum opens the mirror gap. The
    # defect cell's band edge is pushed up to D2, localizing the cavity mode.
    ax = axb

    def lowest_band(rec, lo=80.0, hi=400.0):
        ks, fs = [], []
        for k, fr in zip(rec["ks"], rec["freqs"]):
            cand = [f / 1e12 for f in fr if lo < f / 1e12 < hi]
            if cand:
                ks.append(k)
                fs.append(min(cand))
        return np.asarray(ks), np.asarray(fs)

    # mirror photonic band gap: between the two lowest physical X-point modes
    gap_lo = gap_hi = None
    if "final_mirror" in bands_f:
        xpt = sorted(f / 1e12 for f in bands_f["final_mirror"]["freqs"][-1]
                     if f / 1e12 > 50.0)
        if len(xpt) >= 2:
            gap_lo, gap_hi = xpt[0], xpt[1]
            ax.axhspan(gap_lo, gap_hi, color="C0", alpha=0.12, zorder=1)

    for key, color, lab in [("final_mirror", "C0", "mirror cell band"),
                            ("final_defect", "C1", "defect cell band")]:
        if key not in bands_f:
            continue
        kk, ff = lowest_band(bands_f[key])
        ax.plot(kk, ff, "o-", color=color, ms=4, lw=1.5, zorder=4, label=lab)

    if "final_mirror" in bands_f:
        a_mir = bands_f["final_mirror"]["a"]
        kk = np.linspace(0, 0.5, 60)
        ax.plot(kk, kk * cfg.C0_UM / a_mir * 1e-12, "-", color="0.55", lw=0.7,
                alpha=0.8, zorder=2, label="light line")
    ax.axhline(F_D2_THZ, color="C3", ls="--", lw=1.2, zorder=5, label="Rb D2")

    if gap_lo is not None:
        ax.text(0.035, 0.5 * (gap_lo + gap_hi), "mirror band gap", color="C0",
                fontsize=6.2, va="center", ha="left", zorder=6)
        ax.annotate("defect edge\nlifted to D2", xy=(0.5, gap_lo + 18),
                    xytext=(0.275, 0.5 * (gap_lo + gap_hi)), color="C1",
                    fontsize=5.6, ha="left", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="->", color="C1", lw=0.8))
    ax.set_xlim(0, 0.5)
    ax.set_ylim(150, 440)
    ax.set_xlabel(r"$k_x$ ($2\pi/a$)")
    ax.set_ylabel("frequency (THz)")
    ax.set_title("Bloch band structure", fontsize=7.5)
    ax.legend(loc="lower right", ncol=1, fontsize=5.8, framealpha=0.92,
              handletextpad=0.4, labelspacing=0.3)
    panel(ax, "b")
    _save(fig, "fig3_fields_bands.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — performance evolution + lineage + trim
# ═══════════════════════════════════════════════════════════════════════════

def fig4():
    rec = load_record()
    base = datasets.read_json(RECORD_DIR / "baseline.json")["baseline_perf"]
    diag = [(e["params"]["iteration"], e["perf"]["diagnostic"])
            for e in rec if e["perf"].get("diagnostic")]
    its = [i for i, _ in diag]

    def col(key):
        return [d[key] for _, d in diag]

    fig = plt.figure(figsize=(WIDE, 5.1))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.15], hspace=0.55,
                          wspace=0.5)

    metrics = [("eta_atom", r"$\eta$", 1.0, "C0"),
               ("beta_wg", r"$\beta=\kappa_{\rm wg}/\kappa_{\rm tot}$", 1.0, "C1"),
               ("C_atom", r"$C$", 1.0, "C2"),
               ("Q", r"$Q$", 1.0, "C4")]
    for j, (key, lab, sc, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, j])
        ax.plot(its, np.asarray(col(key)) * sc, "o-", ms=2.5, color=color)
        bl = ax.axhline(base[key] * sc, ls="--", lw=0.9, color="C3",
                        label="baseline")
        ax.set_xlabel("iteration")
        ax.set_ylabel(lab)
        if key == "Q":
            ax.set_yscale("log")
        if j == 0:
            ax.legend([bl], ["baseline"], loc="lower right", fontsize=6,
                      framealpha=0.85, handlelength=1.4)
        panel(ax, "abcd"[j])

    # (e) lineage bars
    rows = datasets.read_csv(paths.TABLES_DIR / "comparison_table.csv")
    order = ["design_1", "invDes(design_1)", "seed", "invDes(seed)", "final"]
    short = {"design_1": "base", "invDes(design_1)": "inv(base)", "seed": "seed",
             "invDes(seed)": "inv(seed)", "final": "final"}
    by = {r["design"]: r for r in rows}
    designs = [d for d in order if d in by]
    cols = [plt.cm.viridis(i / (len(designs) - 1)) for i in range(len(designs))]
    ax = fig.add_subplot(gs[1, :2])
    x = np.arange(len(designs))
    ax.bar(x, [by[d]["eta_atom"] for d in designs], color=cols)
    ax.axhline(0.75, ls=":", color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short[d] for d in designs], rotation=20, ha="right",
                       fontsize=6.3)
    ax.set_ylabel(r"$\eta$")
    ax.set_title("design lineage", fontsize=7.5)
    panel(ax, "e")

    ax2 = fig.add_subplot(gs[1, 2:])
    ax2.bar(x, [by[d]["beta_wg"] for d in designs], color=cols)
    ax2.axhline(0.80, ls=":", color="0.4", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([short[d] for d in designs], rotation=20, ha="right",
                        fontsize=6.3)
    ax2.set_ylabel(r"$\beta$")
    ax2.set_title("waveguide branching ratio", fontsize=7.5)
    panel(ax2, "f")

    # (g) resonance trim onto D2
    tg = datasets.read_csv(paths.DATA_DIR / "tuning_global_scale.csv")
    tg = sorted(tg, key=lambda r: r["value"])
    s_pct = [(r["value"] - 1.0) * 100 for r in tg]      # in-plane scale change (%)
    det = [r["detuning_GHz"] for r in tg]
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(s_pct, det, "o-", color="C0", ms=3)
    ax.axhline(0, color="C3", ls="--", lw=0.9)
    sel = next((r for r in tg if r.get("is_selected")), None)
    if sel:
        xs = (sel["value"] - 1.0) * 100
        ax.scatter([xs], [sel["detuning_GHz"]], marker="*", s=130, color="C3",
                   zorder=5)
        ax.annotate("on D2", xy=(xs, 0), xytext=(xs, 120), fontsize=6.3,
                    ha="center", arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("in-plane scale change (%)")
    ax.set_ylabel("detuning from D2 (GHz)")
    ax.set_title("resonance trim", fontsize=7.5)
    panel(ax, "g")

    # (h) overcoupling Pareto: beta vs C colored by eta
    oc = datasets.read_csv(paths.DATA_DIR / "output_coupling_sweep.csv")
    ax = fig.add_subplot(gs[2, 2:])
    bv = [r["beta_wg"] for r in oc]
    cv = [r["C_atom"] for r in oc]
    ev = [r["eta_atom"] for r in oc]
    scat = ax.scatter(bv, cv, c=ev, cmap="viridis", s=55, zorder=3,
                      edgecolor="k", lw=0.4)
    ax.set_yscale("log")
    ax.axhline(20, ls=":", color="0.4", lw=0.8)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$C$")
    ax.set_title("over-coupling trade-off", fontsize=7.5)
    cb = fig.colorbar(scat, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\eta$", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    panel(ax, "h")

    _save(fig, "fig4_performance.pdf")


def main():
    fig1()
    fig2()
    fig3()
    fig4()
    print("[fig] all four figures written to", FIGDIR)


if __name__ == "__main__":
    main()
