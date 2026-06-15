"""Final-design geometry package: final_design.{json,csv,gds}.

Self-contained, fabrication-facing description of the trimmed cavity: per-hole
positions/lattice/sizes, coordinate conventions, material indices, target
frequency, and output-port location, plus a GDS layout matching the simulation
polygons exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inverse_design import config as cfg

from publication_ready.lib import datasets


def _sections(n_cells):
    order = [("left_taper", n_cells["N_left_taper"]),
             ("left_mirror(output)", n_cells["N_left_mirror"]),
             ("defect", n_cells["N_defect"]),
             ("right_mirror(back)", n_cells["N_right_mirror"]),
             ("right_taper", n_cells["N_right_taper"])]
    labels = []
    for name, n in order:
        labels += [name] * n
    return labels


def per_hole_rows(layout, n_cells):
    labels = _sections(n_cells)
    rows = []
    pos = np.asarray(layout["positions"])
    a = np.asarray(layout["a"])
    sx = np.asarray(layout["sx"])
    sy = np.asarray(layout["sy"])
    for i in range(len(pos)):
        rows.append({
            "hole_index": i, "section": labels[i] if i < len(labels) else "",
            "x_center_um": float(pos[i]), "lattice_a_nm": float(a[i] * 1e3),
            "sx_nm": float(sx[i] * 1e3), "sy_nm": float(sy[i] * 1e3),
        })
    return rows


PER_HOLE_COLUMNS = ["hole_index", "section", "x_center_um", "lattice_a_nm",
                    "sx_nm", "sy_nm"]


def export_gds(layout, context, path):
    try:
        import gdsfactory as gf
        try:
            gf.get_active_pdk()
        except Exception:
            gf.gpdk.PDK.activate()
    except Exception as exc:
        print(f"[geom] GDS export skipped: {exc}")
        return None
    pos = np.asarray(layout["positions"])
    width = context["width"]
    c = gf.Component()
    x0, x1 = pos[0] - 2.0, pos[-1] + 2.0
    c.add_polygon([(x0, -width / 2), (x1, -width / 2),
                   (x1, width / 2), (x0, width / 2)], layer=(1, 0))
    for i, v in enumerate(layout["vertices"]):
        v = np.asarray(v)
        c.add_polygon([(float(x + pos[i]), float(y)) for x, y in v], layer=(2, 0))
    c.write_gds(str(path))
    return path


def export_package(layout, context, n_cells, out_dir, perf=None, name="final_design"):
    out_dir = Path(out_dir)
    rows = per_hole_rows(layout, n_cells)
    datasets.write_csv(out_dir / f"{name}.csv", rows, PER_HOLE_COLUMNS)

    meta = {
        "n_cells": dict(n_cells),
        "beam_width_um": context["width"],
        "beam_thickness_um": context["thickness"],
        "material": str(context.get("medium")),
        "n_SiN_at_f0": None,
        "coordinate_convention": "x along beam, defect center at x=0; output port at -x (left)",
        "output_port": cfg.OUTPUT_FACE,
        "target_frequency_THz": cfg.F_TARGET / 1e12,
        "polarization": context.get("polarization"),
        "per_hole": rows,
        "perf": perf,
    }
    datasets.write_json(out_dir / f"{name}.json", meta, with_convention=True)
    gds = export_gds(layout, context, out_dir / f"{name}.gds")
    return {"csv": out_dir / f"{name}.csv", "json": out_dir / f"{name}.json",
            "gds": gds}
