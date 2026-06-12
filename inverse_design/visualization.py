"""Required result plots, structure-evolution GIF, and GDS export.

All outputs land in the run's directory under ``invDesResults/``:

* ``loss_history.png``        — loss vs iteration (+ total optimization time)
* ``perf_history.png``        — physical parameters vs iteration, with dashed
                                horizontal lines at the baseline values
* ``cross_sections_final.png``— z=0 / y=0 / x=x_src cross sections
* ``fields_final.png``        — |E| and Ey at resonance (verification sim)
* ``structure_evolution.gif`` — z=0 cross section per iteration (PIL)
* ``final_design.gds``        — gdsfactory export of the optimized geometry
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from inverse_design import config as cfg
from inverse_design.parametrization import CavityParametrization
from inverse_design.record import reconstruct_layout


# ──────────────────────────────────────────────────────────────────────────
# Geometry drawing helpers
# ──────────────────────────────────────────────────────────────────────────

def draw_layout_topview(layout, context, ax, title=None):
    """z=0 cross section: beam slab with hole polygons."""
    pos = np.asarray(layout["positions"])
    width = context["width"]
    x0 = pos[0] - 0.5
    x1 = pos[-1] + 0.5
    ax.add_patch(plt.Rectangle((x0, -width / 2), x1 - x0, width,
                               facecolor="#888", edgecolor="none"))
    for i, v in enumerate(layout["vertices"]):
        v = np.asarray(v)
        ax.fill(v[:, 0] + pos[i], v[:, 1], facecolor="white", edgecolor="k",
                linewidth=0.2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-width, width)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    if title:
        ax.set_title(title, fontsize=9)


def layout_from_entry(entry):
    layout = reconstruct_layout(entry)
    par = CavityParametrization(n_cells=entry["params"]["n_cells"],
                                context=entry["params"]["context"])
    verts = par.hole_vertices({
        "sx": np.asarray(layout["sx"]), "sy": np.asarray(layout["sy"]),
        "rho_holes": np.asarray(layout["rho_holes"])})
    layout["vertices"] = [np.asarray(v) for v in verts]
    return layout, entry["params"]["context"]


# ──────────────────────────────────────────────────────────────────────────
# Required plots
# ──────────────────────────────────────────────────────────────────────────

def plot_loss_history(record, out_dir: Path, total_time_s=None):
    iters = [e["params"]["iteration"] for e in record]
    losses = [e["loss"] for e in record]
    lphys = [e["perf"]["proxy"].get("L_phys") for e in record]

    fig, ax = plt.subplots(figsize=(7, 4.5), tight_layout=True)
    ax.plot(iters, losses, "o-", label="total loss")
    if all(v is not None for v in lphys):
        ax.plot(iters, lphys, "s--", ms=3, label=r"$-\log\tilde\eta$ (physics term)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend()
    if total_time_s is None:
        total_time_s = sum(e["time"]["duration_s"] for e in record)
    ax.set_title(f"Objective vs iteration — total optimization time "
                 f"{total_time_s / 3600:.2f} h ({total_time_s:.0f} s)")
    fig.savefig(Path(out_dir) / "loss_history.png", dpi=200)
    plt.close(fig)


_PERF_PANELS = [
    ("f_res_Hz", "f_res (THz)", 1e-12),
    ("Q", "Q (resonance finder)", 1.0),
    ("beta_wg", r"$\kappa_{wg}/\kappa_{tot}$", 1.0),
    ("Vmode_lambda_n3", r"V$_{mode}$ ((λ/n)³)", 1.0),
    ("purcell", "Purcell factor", 1.0),
    ("g_2pi_Hz_at_250nm", "g/2π @250 nm (GHz)", 1e-9),
    ("C_atom", "C (atom @250 nm)", 1.0),
    ("eta_atom", "η (atom @250 nm)", 1.0),
]


def plot_perf_history(record, baseline_perf: dict, out_dir: Path):
    diag = [(e["params"]["iteration"], e["perf"]["diagnostic"])
            for e in record if "diagnostic" in e.get("perf", {})]
    prox = [(e["params"]["iteration"], e["perf"]["proxy"]) for e in record
            if "proxy" in e.get("perf", {})]

    fig, axes = plt.subplots(4, 2, figsize=(11, 12), tight_layout=True)
    for ax, (key, label, scale) in zip(axes.ravel(), _PERF_PANELS):
        if diag:
            its = [i for i, d in diag if key in d]
            vals = [d[key] * scale for _, d in diag if key in d]
            if its:
                ax.plot(its, vals, "o-", color="C0", label="optimized (verified)")
        if key == "f_res_Hz" and prox:
            ax.plot([i for i, p in prox], [p["f_hat"] * scale for _, p in prox],
                    ".", ms=3, alpha=0.5, color="C2", label="proxy f^")
        if key == "beta_wg" and prox:
            ax.plot([i for i, p in prox], [p["beta"] for _, p in prox],
                    ".", ms=3, alpha=0.5, color="C2", label="proxy β")
        if key in baseline_perf and baseline_perf[key] is not None:
            ax.axhline(baseline_perf[key] * scale, ls="--", color="C3",
                       label="baseline (design_1)")
        if key == "f_res_Hz":
            ax.axhline(cfg.F_TARGET * scale, ls=":", color="k", alpha=0.6,
                       label="384.2 THz target")
        ax.set_xlabel("iteration")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    fig.suptitle("Physical parameters vs iteration (dashed = current best design)")
    fig.savefig(Path(out_dir) / "perf_history.png", dpi=200)
    plt.close(fig)


def plot_cross_sections(final_entry, out_dir: Path, sim=None):
    """Hole-polygon top view + (if sim given) eps cross sections."""
    layout, context = layout_from_entry(final_entry)

    if sim is not None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), tight_layout=True)
        draw_layout_topview(layout, context, axes[0],
                            title="Optimal design — hole polygons (z=0)")
        sim.plot_eps(z=0.0, ax=axes[1])
        axes[1].set_title("eps, z=0 plane")
        sim.plot_eps(y=0.0, ax=axes[2])
        axes[2].set_title("eps, y=0 plane")
    else:
        fig, ax = plt.subplots(figsize=(14, 3.2), tight_layout=True)
        draw_layout_topview(layout, context, ax,
                            title="Optimal design — hole polygons (z=0)")
    fig.savefig(Path(out_dir) / "cross_sections_final.png", dpi=200)
    plt.close(fig)


def plot_fields(sim_data, out_dir: Path, fname="fields_final.png"):
    """|E| and Ey on available field profile monitors of a completed sim."""
    names = [n for n in ("profile_z", "Field_Profile_Monitor_z",
                         "Field_Profile_Monitor_y")
             if n in sim_data.monitor_data]
    if not names:
        print("[plot_fields] no profile monitor found")
        return
    fig, axes = plt.subplots(len(names), 2, figsize=(14, 4 * len(names)),
                             tight_layout=True, squeeze=False)
    for r, name in enumerate(names):
        mon = sim_data[name]
        comps = {}
        for c in ("Ex", "Ey", "Ez"):
            arr = getattr(mon, c, None)
            if arr is None:
                continue
            if "t" in arr.dims:
                arr = arr.isel(t=-1)
            if "f" in arr.dims:
                arr = arr.isel(f=0)
            comps[c] = arr.squeeze()
        Eabs = np.sqrt(sum(np.abs(v) ** 2 for v in comps.values()))
        for col, (data, label, cmap) in enumerate(
                [(Eabs, "|E|", "magma"),
                 (np.real(comps.get("Ey")), "Re(Ey)", "RdBu")]):
            ax = axes[r][col]
            data.plot(ax=ax, cmap=cmap, add_colorbar=True)
            ax.set_title(f"{label} — {name}")
            ax.set_aspect("equal")
    fig.savefig(Path(out_dir) / fname, dpi=200)
    plt.close(fig)


def make_structure_gif(record, out_dir: Path, fname="structure_evolution.gif",
                       max_frames=120, duration_ms=200):
    entries = record
    if len(entries) > max_frames:
        idx = np.linspace(0, len(entries) - 1, max_frames).astype(int)
        entries = [record[i] for i in idx]

    frames = []
    for e in entries:
        layout, context = layout_from_entry(e)
        fig, ax = plt.subplots(figsize=(12, 2.6), tight_layout=True)
        draw_layout_topview(
            layout, context, ax,
            title=f"iter {e['params']['iteration']} — loss {e['loss']:.3f}")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P"))

    if not frames:
        print("[gif] no frames")
        return
    frames[0].save(Path(out_dir) / fname, save_all=True,
                   append_images=frames[1:], duration=duration_ms, loop=0)


def export_gds(final_entry, out_dir: Path, name="final_design"):
    """GDS via gdsfactory using the same polygon vertices as the simulation."""
    try:
        import gdsfactory as gf
        try:
            gf.get_active_pdk()
        except Exception:
            gf.gpdk.PDK.activate()   # gdsfactory >= 9 requires an active PDK
    except ImportError:
        print("[gds] gdsfactory not available; skipping GDS export")
        return None
    layout, context = layout_from_entry(final_entry)
    pos = np.asarray(layout["positions"])
    width = context["width"]

    c = gf.Component()
    x0, x1 = pos[0] - 2.0, pos[-1] + 2.0
    c.add_polygon([(x0, -width / 2), (x1, -width / 2),
                   (x1, width / 2), (x0, width / 2)], layer=(1, 0))
    for i, v in enumerate(layout["vertices"]):
        v = np.asarray(v)
        c.add_polygon([(float(x + pos[i]), float(y)) for x, y in v], layer=(2, 0))
    path = Path(out_dir) / f"{name}.gds"
    c.write_gds(str(path))
    return path


def generate_all(record, baseline_perf, out_dir: Path, final_sim=None,
                 final_sim_data=None, total_time_s=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_history(record, out_dir, total_time_s=total_time_s)
    plot_perf_history(record, baseline_perf, out_dir)
    plot_cross_sections(record[-1], out_dir, sim=final_sim)
    if final_sim_data is not None:
        plot_fields(final_sim_data, out_dir)
    make_structure_gif(record, out_dir)
    export_gds(record[-1], out_dir)
