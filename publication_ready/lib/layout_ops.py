"""Pure-numpy transforms on detached cavity layout dicts.

A *layout* is the dict produced by ``inverse_design.visualization.layout_from_entry``
or ``CavityParametrization.detach(par.layout(theta))``::

    {"positions": (N,), "a": (N,), "sx": (N,), "sy": (N,),
     "rho_holes": (N, N_FREE_RHO), "vertices": [ (N_pts, 2), ... ]}

All lengths are μm; ``positions`` are hole centers with the defect pinned at
x=0. ``vertices`` are hole-centered polygon coordinates (∝ sx, sy). The verifier
(`diagnostics.verification_sim_from_layout`) consumes ``positions`` + ``vertices``
for geometry and ``a`` + ``sx`` + ``sy`` for domain/bookkeeping, so every
transform keeps all of them mutually consistent.

These functions never trace autograd and never touch ``inverse_design``'s
modules — they only call the read-only ``CavityParametrization`` helpers
(``positions``, ``hole_vertices``) to re-pin positions and rebuild vertices.
"""

from __future__ import annotations

import copy

import numpy as np

from inverse_design import config as cfg
from inverse_design import record as record_mod
from inverse_design.parametrization import CavityParametrization
from inverse_design.visualization import layout_from_entry

F_D2 = cfg.F_TARGET


# ── loading the design we trim ─────────────────────────────────────────────

def load_layout_from_record(record_dir) -> tuple:
    """Return (layout, context, par) for the final entry of a record dir."""
    rec = record_mod.load_record(record_dir)
    entry = rec[-1]
    layout, context = layout_from_entry(entry)
    par = CavityParametrization(n_cells=entry["params"]["n_cells"],
                                context=entry["params"]["context"])
    return _as_float_layout(layout), dict(context), par


def load_seed_layout() -> tuple:
    """Return (layout, context, par) for the band-engineered seed design."""
    import json
    seed_path = cfg.RESULTS_ROOT.parent / "seeds" / "seed_780nm.json"
    with open(seed_path) as f:
        seed = json.load(f)
    parameters = {
        k: {"lattice": float(v["lattice"]),
            "geometry_params": np.asarray(v["geometry_params"], dtype=float)}
        for k, v in seed["parameters"].items()}
    par = CavityParametrization(n_cells=seed["n_cells"],
                                context=dict(cfg.BASELINE_CONTEXT))
    theta = par.init_theta_from_baseline(parameters=parameters,
                                         n_cells=seed["n_cells"])
    layout = CavityParametrization.detach(par.layout(theta))
    return _as_float_layout(layout), dict(par.context), par


def _as_float_layout(layout: dict) -> dict:
    out = {
        "positions": np.asarray(layout["positions"], dtype=float),
        "a": np.asarray(layout["a"], dtype=float),
        "sx": np.asarray(layout["sx"], dtype=float),
        "sy": np.asarray(layout["sy"], dtype=float),
        "rho_holes": np.asarray(layout["rho_holes"], dtype=float),
        "vertices": [np.asarray(v, dtype=float) for v in layout["vertices"]],
    }
    return out


def _clone(layout: dict) -> dict:
    return {
        "positions": layout["positions"].copy(),
        "a": layout["a"].copy(),
        "sx": layout["sx"].copy(),
        "sy": layout["sy"].copy(),
        "rho_holes": layout["rho_holes"].copy(),
        "vertices": [v.copy() for v in layout["vertices"]],
    }


def rebuild_vertices(layout: dict, par: CavityParametrization) -> list:
    """Rebuild hole-centered vertices from (sx, sy, rho_holes)."""
    verts = par.hole_vertices({
        "sx": np.asarray(layout["sx"], dtype=float),
        "sy": np.asarray(layout["sy"], dtype=float),
        "rho_holes": np.asarray(layout["rho_holes"], dtype=float),
    })
    return [np.asarray(v, dtype=float) for v in verts]


def _defect_slice(n_cells: dict) -> slice:
    i0 = n_cells["N_left_taper"] + n_cells["N_left_mirror"]
    return slice(i0, i0 + n_cells["N_defect"])


def _output_mirror_slice(n_cells: dict) -> slice:
    """The output (−x, left) mirror cells — the overcoupling lever."""
    i0 = n_cells["N_left_taper"]
    return slice(i0, i0 + n_cells["N_left_mirror"])


# ── trim / tuning transforms ───────────────────────────────────────────────

def scale_layout(layout: dict, s: float) -> dict:
    """Global conformal in-plane scale by ``s`` (positions, a, sx, sy, vertices).

    f_res responds ≈ f/s; thickness/width are fixed by ``context`` and are not
    scaled here (the in-plane-only trim). rho_holes is dimensionless → unchanged.
    """
    out = _clone(layout)
    out["positions"] = layout["positions"] * s
    out["a"] = layout["a"] * s
    out["sx"] = layout["sx"] * s
    out["sy"] = layout["sy"] * s
    out["vertices"] = [v * s for v in layout["vertices"]]
    return out


def scale_defect_region(layout: dict, par: CavityParametrization,
                        s_d: float) -> dict:
    """Scale only the defect-cell block (lattice + sizes); mirrors held fixed.

    Positions are recomputed from the modified per-cell ``a`` via the cumulative
    sum and re-pinned at the defect center (exactly ``CavityParametrization.
    positions``), so the length change distributes symmetrically and the source/
    atom monitors stay on the antinode.
    """
    out = _clone(layout)
    sl = _defect_slice(par.n_cells)
    out["a"][sl] = layout["a"][sl] * s_d
    out["sx"][sl] = layout["sx"][sl] * s_d
    out["sy"][sl] = layout["sy"][sl] * s_d
    out["positions"] = np.asarray(par.positions(out["a"]), dtype=float)
    out["vertices"] = rebuild_vertices(out, par)
    return out


def scale_output_mirror(layout: dict, par: CavityParametrization,
                        m_out: float) -> dict:
    """Scale the output (left, −x) mirror hole sizes by ``m_out``.

    Larger holes → stronger per-cell reflectivity → lower κ_wg → lower β;
    smaller → weaker mirror → higher κ_wg → higher β (more overcoupled). Hole
    centers (positions) do not move; only sizes/vertices change.
    """
    out = _clone(layout)
    sl = _output_mirror_slice(par.n_cells)
    out["sx"][sl] = layout["sx"][sl] * m_out
    out["sy"][sl] = layout["sy"][sl] * m_out
    out["vertices"] = rebuild_vertices(out, par)
    return out


# ── fabrication perturbation ───────────────────────────────────────────────

def perturb_layout(layout: dict, par: CavityParametrization,
                   etch_nm: float = 0.0, lattice_nm: float = 0.0) -> dict:
    """Apply etch-bias (hole radius) and lattice-constant biases.

    ``etch_nm`` is a uniform radial edge bias: every hole diameter changes by
    ``2·etch`` (positive = larger holes). ``lattice_nm`` adds to every per-cell
    period; positions are recomputed and re-pinned. Width/thickness biases are
    applied by the driver through ``context`` (not the layout), see
    ``context_with_width``.
    """
    out = _clone(layout)
    if etch_nm:
        d = 2.0 * etch_nm * 1e-3            # nm radius → μm diameter
        out["sx"] = np.maximum(layout["sx"] + d, 1e-3)
        out["sy"] = np.maximum(layout["sy"] + d, 1e-3)
    if lattice_nm:
        out["a"] = layout["a"] + lattice_nm * 1e-3
        out["positions"] = np.asarray(par.positions(out["a"]), dtype=float)
    out["vertices"] = rebuild_vertices(out, par)
    return out


def context_with_width(context: dict, width_nm_bias: float = 0.0) -> dict:
    """Context with a beam-width bias (μm). width_nm_bias in nm."""
    ctx = dict(context)
    ctx["width"] = float(context["width"]) + width_nm_bias * 1e-3
    return ctx


# ── output-cell-count rebuild (contingency only) ───────────────────────────

def rebuild_with_output_cells(layout: dict, par: CavityParametrization,
                              n_left_mirror: int) -> tuple:
    """Rebuild the layout with a different OUTPUT (left) mirror cell count.

    Returns (new_layout, new_par). The output mirror is grown/shrunk by
    replicating/truncating its unit cells (template = current output-mirror
    cells, ordered outermost→innermost); all other sections are preserved and
    the defect center is re-pinned at x=0. This is the coarse overcoupling lever
    used only if the hole-size sweep cannot meet the β/η gate, because changing
    the hole count alters the spline anchor schedule.
    """
    nc = dict(par.n_cells)
    n_old = nc["N_left_mirror"]
    sl = _output_mirror_slice(nc)
    a_t = layout["a"][sl]
    sx_t = layout["sx"][sl]
    sy_t = layout["sy"][sl]
    rho_t = layout["rho_holes"][sl]

    def _resample(arr, n_new):
        if n_new == n_old:
            return arr.copy()
        if n_new < n_old:                  # drop innermost (defect-adjacent) cells
            return arr[:n_new].copy()
        # extend outward by repeating the outermost cell
        extra = n_new - n_old
        pad = np.repeat(arr[:1], extra, axis=0)
        return np.concatenate([pad, arr], axis=0)

    a_new_mir = _resample(a_t, n_left_mirror)
    sx_new_mir = _resample(sx_t, n_left_mirror)
    sy_new_mir = _resample(sy_t, n_left_mirror)
    rho_new_mir = _resample(rho_t, n_left_mirror)

    i0 = nc["N_left_taper"]
    a_new = np.concatenate([layout["a"][:i0], a_new_mir, layout["a"][sl.stop:]])
    sx_new = np.concatenate([layout["sx"][:i0], sx_new_mir, layout["sx"][sl.stop:]])
    sy_new = np.concatenate([layout["sy"][:i0], sy_new_mir, layout["sy"][sl.stop:]])
    rho_new = np.concatenate([layout["rho_holes"][:i0], rho_new_mir,
                              layout["rho_holes"][sl.stop:]], axis=0)

    nc["N_left_mirror"] = int(n_left_mirror)
    new_par = CavityParametrization(n_cells=nc, context=dict(par.context))

    new_layout = {"a": a_new, "sx": sx_new, "sy": sy_new, "rho_holes": rho_new}
    new_layout["positions"] = np.asarray(new_par.positions(a_new), dtype=float)
    new_layout["vertices"] = rebuild_vertices(new_layout, new_par)
    return new_layout, new_par


# ── analytic retune ────────────────────────────────────────────────────────

def first_order_retune_factor(f_res_hz: float, f_target_hz: float = F_D2) -> float:
    """Global-scale factor that maps f_res onto f_target to first order (f∝1/s)."""
    return float(f_res_hz) / float(f_target_hz)


# ── compact (de)serialization (vertices rebuilt, not stored) ───────────────

def layout_to_dict(layout: dict) -> dict:
    """Serializable layout WITHOUT vertices (rebuilt from sx, sy, rho_holes)."""
    return {
        "positions": np.asarray(layout["positions"], dtype=float).tolist(),
        "a": np.asarray(layout["a"], dtype=float).tolist(),
        "sx": np.asarray(layout["sx"], dtype=float).tolist(),
        "sy": np.asarray(layout["sy"], dtype=float).tolist(),
        "rho_holes": np.asarray(layout["rho_holes"], dtype=float).tolist(),
    }


def layout_from_dict(d: dict, par: CavityParametrization) -> dict:
    """Inverse of ``layout_to_dict``: rebuild a full layout (with vertices)."""
    layout = {
        "positions": np.asarray(d["positions"], dtype=float),
        "a": np.asarray(d["a"], dtype=float),
        "sx": np.asarray(d["sx"], dtype=float),
        "sy": np.asarray(d["sy"], dtype=float),
        "rho_holes": np.asarray(d["rho_holes"], dtype=float),
    }
    layout["vertices"] = rebuild_vertices(layout, par)
    return layout
