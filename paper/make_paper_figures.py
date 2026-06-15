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

def draw_schematic(ax):
    """Top-down concept: suspended PhC fingers beyond an etched substrate;
    tweezer-trapped atoms transported over a series of cavities, transduced to
    flying photonic qubits collected into fiber."""
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0.2, 6.0)
    ax.axis("off")

    # substrate (left), etched/undercut edge at x=2.4
    ax.add_patch(Rectangle((0, 0.2), 2.4, 5.8, facecolor="#cdc6ba",
                           edgecolor="#7a7366", hatch="////", lw=0.8, zorder=1))
    ax.text(1.2, 0.75, "Si\nsubstrate", ha="center", va="center", fontsize=7)
    ax.annotate("etched undercut", xy=(2.4, 4.0), xytext=(1.45, 5.55),
                fontsize=6.3, ha="center", va="center", color="#5a5346",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#5a5346"))

    finger_y = [1.25, 3.0, 4.75]
    fh = 0.5
    for yc in finger_y:
        ax.add_patch(FancyBboxPatch((1.9, yc - fh / 2), 6.7, fh,
                                    boxstyle="round,pad=0.0,rounding_size=0.05",
                                    facecolor="#9bbbd6", edgecolor="#3f5d77",
                                    lw=0.8, zorder=2))
        xs = np.linspace(2.9, 8.1, 22)
        for x in xs:
            r = 0.09 if abs(x - 5.5) > 0.55 else 0.055
            ax.add_patch(Ellipse((x, yc), 0.15, 2 * r, facecolor="white",
                                 edgecolor="#3f5d77", lw=0.4, zorder=3))
    ax.text(8.6, finger_y[-1] + 0.42, "SiN PhC fingers", fontsize=6.8,
            ha="right", va="bottom", color="#27435a")

    # trapped atoms at the defect sites of the middle finger
    yc = finger_y[1]
    atom_x = [4.4, 5.5, 6.6]
    for x in atom_x:
        ax.add_patch(plt.Circle((x, yc + 0.42), 0.115, facecolor="#d24b4b",
                                edgecolor="k", lw=0.4, zorder=6))
    ax.annotate("trapped atoms", xy=(4.4, yc + 0.3), xytext=(4.4, yc - 0.62),
                fontsize=6.5, color="#a82c2c", ha="center", va="top",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#a82c2c"))

    # transport: atom row slides along the cavity series (arrow above middle finger)
    ax.add_patch(FancyArrowPatch((4.2, yc + 1.05), (6.9, yc + 1.05),
                                 arrowstyle="-|>", mutation_scale=11,
                                 lw=1.2, color="#444", zorder=7))
    ax.text(5.55, yc + 1.2, "transport over cavities", ha="center", va="bottom",
            fontsize=6.5, color="#333")

    # transduction: photon from each finger end -> bus -> fiber -> flying qubit
    for yc2 in finger_y:
        ax.add_patch(FancyArrowPatch((8.7, yc2), (9.2, yc2), arrowstyle="-|>",
                                     mutation_scale=8, lw=1.0, color="#2c7fb8",
                                     zorder=6))
    ax.plot([9.2, 9.2], [finger_y[0], finger_y[-1]], color="#2c7fb8", lw=1.4)
    ax.add_patch(FancyArrowPatch((9.2, finger_y[1]), (9.7, finger_y[1]),
                                 arrowstyle="-|>", mutation_scale=11, lw=1.7,
                                 color="#2c7fb8"))
    ax.text(9.78, finger_y[1], "flying\nphotonic\nqubits", ha="left",
            va="center", fontsize=6.6, color="#1f6391")


def fig1():
    layout, ctx, par, fc = load_final_layout()
    pos = np.asarray(layout["positions"])
    fig = plt.figure(figsize=(WIDE, 4.7))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 0.55, 1.0],
                          hspace=0.55, wspace=0.32)

    ax_s = fig.add_subplot(gs[0, :])
    draw_schematic(ax_s)
    panel(ax_s, "a")

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
    losses = [e["loss"] for e in rec]
    stages = [e["params"].get("stage", "") for e in rec]
    # stage A->B boundary
    bnd = next((iters[i] for i in range(1, len(stages))
                if stages[i] != stages[i - 1]), None)
    total_h = sum(e["time"]["duration_s"] for e in rec) / 3600.0

    fig = plt.figure(figsize=(WIDE, 2.7), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.0])

    ax = fig.add_subplot(gs[:, 0])
    ax.plot(iters, losses, "o-", ms=3, color="C0")
    if bnd is not None:
        ax.axvline(bnd - 0.5, color="0.5", ls="--", lw=0.9)
        ax.text(bnd - 1, max(losses) * 0.97, "layout", ha="right", va="top",
                fontsize=7, color="0.4")
        ax.text(bnd, max(losses) * 0.97, "  +shapes", ha="left", va="top",
                fontsize=7, color="0.4")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"objective $\mathcal{L}$")
    ax.set_title(f"optimization time {total_h:.1f} h", fontsize=8)
    panel(ax, "a")

    # intermediate snapshots stacked on the right
    snap_iters = [iters[0], iters[len(iters) // 3], iters[-1]]
    from inverse_design.visualization import layout_from_entry
    by_iter = {e["params"]["iteration"]: e for e in rec}
    for r, it in enumerate(snap_iters):
        ax = fig.add_subplot(gs[r, 1])
        lay, c = layout_from_entry(by_iter[it])
        draw_layout_topview(lay, c, ax)
        ax.set_aspect("auto")          # fill the cell (thumbnail of the schedule)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.99, 0.84, f"iter {it}", transform=ax.transAxes, ha="right",
                va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
        panel(ax, "bcd"[r])
    _save(fig, "fig2_optimization.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — field cross-sections + band structure
# ═══════════════════════════════════════════════════════════════════════════

def _field_panel(ax, fields, tag, comp, cmap, label):
    A = np.asarray(fields[f"{tag}_{comp}"])
    dims = [str(x) for x in fields[f"{tag}_dims"]]
    c0 = np.asarray(fields[f"{tag}_{dims[0]}"])
    c1 = np.asarray(fields[f"{tag}_{dims[1]}"])
    im = ax.pcolormesh(c0, c1, A.T, cmap=cmap, shading="auto", rasterized=True)
    ax.set_xlabel(rf"${dims[0]}$ ($\mu$m)")
    ax.set_ylabel(rf"${dims[1]}$ ($\mu$m)")
    ax.set_aspect("equal")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    return im


def fig3():
    fields = dict(np.load(paths.DATA_DIR / "fields_final.npz", allow_pickle=True))
    bands_f = datasets.read_json(paths.DATA_DIR / "bands_final.json")
    za = float(fields["atom_height_um"])

    fig = plt.figure(figsize=(WIDE, 5.7), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.85, 1.0, 1.25], wspace=0.05)

    # (a) midplane xy, full width (the long standing-wave mode)
    ax = fig.add_subplot(gs[0, :])
    _field_panel(ax, fields, "midplane_z", "Eabs", "magma", r"$|E|$ (arb.)")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-0.55, 0.55)
    ax.set_title(r"beam midplane ($z=0$)", fontsize=7.5)
    panel(ax, "a", color="w")

    # (b) longitudinal xz at y=0, atom height marked
    ax = fig.add_subplot(gs[1, 0])
    _field_panel(ax, fields, "vertical_y", "Eabs", "magma", r"$|E|$ (arb.)")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-0.7, 0.75)
    ax.axhline(za, color="cyan", ls="--", lw=0.9)
    ax.text(-2.5, za + 0.04, "atom", color="cyan", fontsize=6.5, va="bottom")
    ax.set_title(r"longitudinal ($y=0$)", fontsize=7.5)
    panel(ax, "b", color="w")

    # (c) defect cross-section yz at x=0
    ax = fig.add_subplot(gs[1, 1])
    _field_panel(ax, fields, "vertical_x", "Eabs", "magma", r"$|E|$ (arb.)")
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.7, 0.75)
    ax.axhline(za, color="cyan", ls="--", lw=0.9)
    ax.text(-0.82, za + 0.04, "atom", color="cyan", fontsize=6.5, va="bottom")
    ax.set_title(r"defect cut ($x=0$)", fontsize=7.5)
    panel(ax, "c", color="w")

    # band structure (mirror + defect cells)
    ax = fig.add_subplot(gs[2, :])
    cells = [("final_mirror", "C0", "mirror cell"),
             ("final_defect", "C1", "defect cell")]
    for key, color, lab in cells:
        if key not in bands_f:
            continue
        r = bands_f[key]
        for k, fr in zip(r["ks"], r["freqs"]):
            ax.scatter([k] * len(fr), np.asarray(fr) / 1e12, s=9, color=color,
                       alpha=0.8, zorder=3, label=lab)
            lab = None
    # mirror band-gap shading from the mirror X-point edges
    if "final_mirror" in bands_f:
        xf = sorted(np.asarray(bands_f["final_mirror"]["freqs"][-1]) / 1e12)
        if len(xf) >= 2:
            ax.axhspan(xf[0], xf[1], color="C0", alpha=0.10, zorder=1,
                       label="mirror gap")
        a_mir = bands_f["final_mirror"]["a"]
        kk = np.linspace(0, 0.5, 60)
        ax.plot(kk, kk * cfg.C0_UM / a_mir * 1e-12, "k-", lw=0.8, alpha=0.5,
                label="light line")
    ax.axhline(F_D2_THZ, color="C3", ls="--", lw=1.1, label="Rb D2 (384.23 THz)")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(330, 470)
    ax.set_xlabel(r"$k_x$ ($2\pi/a$)")
    ax.set_ylabel("frequency (THz)")
    ax.legend(loc="upper left", ncol=2, fontsize=6, framealpha=0.85)
    panel(ax, "d")
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
