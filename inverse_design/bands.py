"""TE band structures of nanobeam unit cells via Tidy3D Bloch sims.

MPB is not installed in tidyEnv, so this uses the same approach as the legacy
`Cavity_simulation.bandstructure_tidy` (random dipoles in a Bloch-periodic
unit cell + ResonanceFinder per k), but with a *suspended nanobeam* unit cell
(finite-width beam, PML in y and z) rather than a laterally periodic slab,
deterministic seeded dipoles, and non-interactive batch execution.

Usage (from repo root, tidyEnv):
    PYTHONPATH=. python -m inverse_design.bands          # compute + save
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import tidy3d as td
from tidy3d.plugins.resonance import ResonanceFinder

from inverse_design import config as cfg
from inverse_design.builder import resolve_medium

C0 = td.constants.C_0

F_MIN, F_MAX = 280e12, 480e12        # analysis band (Hz)
N_K = 9                              # k points, Gamma -> X inclusive
RUN_TIME = 3e-12
N_DIPOLES = 5
N_MONITORS = 2


def unit_cell_simulation(a_cell, hole_vertices, context, kx, rng):
    """One Bloch unit-cell simulation at normalized kx (units of 2pi/a)."""
    width = context["width"]
    thickness = context["thickness"]
    medium = resolve_medium(context["medium"])
    pad = 1.3

    beam = td.Structure(
        geometry=td.PolySlab(
            vertices=[[-a_cell, width / 2], [a_cell, width / 2],
                      [a_cell, -width / 2], [-a_cell, -width / 2]],
            axis=2, slab_bounds=(-thickness / 2, thickness / 2)),
        medium=medium, name="beam")
    structures = [beam]
    if hole_vertices is not None:
        structures.append(td.Structure(
            geometry=td.PolySlab(vertices=hole_vertices, axis=2,
                                 slab_bounds=(-thickness / 2, thickness / 2)),
            medium=td.Medium(permittivity=1.0), name="hole"))

    freq0 = 0.5 * (F_MIN + F_MAX)
    fwidth = 0.35 * (F_MAX - F_MIN)

    sources = []
    pos = rng.uniform([-a_cell / 2, -width / 2, -thickness / 2],
                      [a_cell / 2, width / 2, thickness / 2], [N_DIPOLES, 3])
    phases = rng.uniform(0, 2 * np.pi, N_DIPOLES)
    for i in range(N_DIPOLES):
        sources.append(td.PointDipole(
            center=tuple(pos[i]), polarization="Ey",
            source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth,
                                         phase=phases[i]),
            name=f"dip_{i}"))

    monitors = []
    mpos = rng.uniform([-a_cell / 2, -width / 2, -thickness / 2],
                       [a_cell / 2, width / 2, thickness / 2], [N_MONITORS, 3])
    for i in range(N_MONITORS):
        monitors.append(td.FieldTimeMonitor(
            center=tuple(mpos[i]), size=(0, 0, 0), start=5 / fwidth,
            fields=["Ex", "Ey", "Ez"], name=f"mon_{i}"))

    return td.Simulation(
        center=(0, 0, 0),
        size=(a_cell, width + 2 * pad, thickness + 2 * pad),
        structures=structures, sources=sources, monitors=monitors,
        run_time=RUN_TIME, shutoff=0.0,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.bloch(kx),  # in units of 2pi/size_x
            y=td.Boundary.pml(), z=td.Boundary.pml()),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=15),
        medium=td.Medium(permittivity=1.0),
    )


def compute_bands(cells: dict, context, out_path: Path,
                  folder_name="invdes_bands"):
    """cells: {name: (a_cell, hole_vertices or None)} -> saved results dict."""
    rng = np.random.default_rng(7)
    ks = np.linspace(0.0, 0.5, N_K)

    sims = {}
    for name, (a_cell, verts) in cells.items():
        cell_rng = np.random.default_rng(rng.integers(1 << 31))
        for i, k in enumerate(ks):
            sims[f"{name}__k{i}"] = unit_cell_simulation(
                a_cell, verts, context, k, np.random.default_rng(
                    cell_rng.integers(1 << 31)))

    batch = td.web.Batch(simulations=sims, folder_name=folder_name,
                         verbose=True)
    batch_data = batch.run(path_dir=str(out_path / "bands_data"))

    rf = ResonanceFinder(freq_window=(F_MIN, F_MAX))
    results = {name: {"a": cells[name][0], "ks": ks.tolist(), "freqs": []}
               for name in cells}
    for name in cells:
        for i in range(N_K):
            sd = batch_data[f"{name}__k{i}"]
            data = rf.run(signals=sd.data)
            data = data.where(data.amplitude > 1e-3 * float(data.amplitude.max()),
                              drop=True)
            data = data.where(data.error < 0.3, drop=True)
            results[name]["freqs"].append(np.abs(data.freq.to_numpy()).tolist())

    # merge with any previously computed cells rather than overwriting
    pkl = out_path / "bands.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            merged = pickle.load(f)
    else:
        merged = {}
    merged.update(results)
    with open(pkl, "wb") as f:
        pickle.dump(merged, f)
    with open(out_path / "bands.json", "w") as f:
        json.dump(merged, f, indent=1)
    return results


def plot_bands(results: dict, out_path: Path, titles: dict = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.6),
                             tight_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        r = results[name]
        a = r["a"]
        for k, freqs in zip(r["ks"], r["freqs"]):
            ax.scatter([k] * len(freqs), np.asarray(freqs) / 1e12, s=14,
                       color="C0", alpha=0.85)
        kk = np.linspace(0, 0.5, 50)
        ax.plot(kk, kk * C0 / a * 1e-12, "k-", lw=1, alpha=0.5,
                label="light line")
        ax.fill_between(kk, kk * C0 / a * 1e-12, 480, color="gray", alpha=0.15)
        ax.axhline(cfg.F_TARGET / 1e12, color="C3", ls="--", lw=1.2,
                   label="384.2 THz (Rb D2)")
        ax.set_xlim(0, 0.5)
        ax.set_ylim(F_MIN / 1e12, F_MAX / 1e12)
        ax.set_xlabel(r"$k_x$  ($2\pi/a$)")
        ax.set_title((titles or {}).get(name, name) + f"\n(a = {a*1000:.0f} nm)",
                     fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("frequency (THz)")
            ax.legend(fontsize=8, loc="lower right")
    path = out_path / "bands.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    from inverse_design import record as rm
    from inverse_design.visualization import layout_from_entry
    from main_code.crystal import crystal_polygon_2d

    out_path = cfg.RESULTS_ROOT / "full" / "report"
    out_path.mkdir(parents=True, exist_ok=True)

    rec = rm.load_record(cfg.RESULTS_ROOT / "full")
    layout, context = layout_from_entry(rec[-1])
    a = np.asarray(layout["a"])

    # representative cells of the optimized design
    i_def = 35     # defect-center cell (even N_defect: first right of center)
    i_mir = 70     # back (right) mirror region
    cells = {
        "opt_defect": (float(a[i_def]), np.asarray(layout["vertices"][i_def])),
        "opt_mirror": (float(a[i_mir]), np.asarray(layout["vertices"][i_mir])),
        "base_defect": (0.29, crystal_polygon_2d(
            dict(cfg.BASELINE_CONTEXT), [0.20, 0.20], 0.29)),
        "base_mirror": (0.35, crystal_polygon_2d(
            dict(cfg.BASELINE_CONTEXT), [0.20, 0.30], 0.35)),
    }
    results = compute_bands(cells, dict(cfg.BASELINE_CONTEXT), out_path)
    titles = {
        "opt_defect": "Optimized: defect-center cell",
        "opt_mirror": "Optimized: back-mirror cell",
        "base_defect": "Baseline: defect cell",
        "base_mirror": "Baseline: back-mirror cell",
    }
    print("bands ->", plot_bands(results, out_path, titles))


if __name__ == "__main__":
    main()
